#!/usr/bin/env python3
"""
==============================================================================
FloodResQ — PART 05: Validation, Testing, Case Study & Hand-off
Teammate 3 (GIS / Damage / Connectivity)           Track B: Trishuli flood
==============================================================================

WHAT THIS PART DOES
-------------------
Parts 01-04 produce numbers (damaged roads, cut-off settlements).  Part 05
answers: "CAN WE TRUST THEM, AND CAN SOMEONE ELSE USE THEM?"

    A. CONTRACT CHECKS     Are the files from Parts 01-04 well-formed?
    B. INDEPENDENT ORACLE  Re-compute damage + cut-off with different code and
                           compare with the pipeline (N-version checking).
    C. METAMORPHIC TESTS   Properties that must hold even without ground truth
                           (more flood => never less damage, etc.).
    D. ACCURACY            Precision / recall / F1 / IoU vs. a reference layer,
                           with spatial tolerance and confidence intervals.
    E. SENSITIVITY         How fragile is the answer to flood-mask uncertainty?
    F. CASE STUDY          Headline numbers, map PNG, Markdown write-up.
    G. HAND-OFF            Single GeoPackage, GeoJSON, manifest with hashes,
                           README + data dictionary, ZIP bundle.

HOW TO RUN
----------
    python part05_validation_handoff.py selftest     # no data needed (synthetic)
    python part05_validation_handoff.py run --data-dir data/processed \
                                            --out-dir outputs/part05

DEPENDENCIES:  geopandas>=0.14, shapely>=2.0, networkx, numpy, pandas, matplotlib

!! ASSUMPTION ABOUT PARTS 01-04 !!
Only the *outputs* of earlier parts are touched, never their functions.  File
names are in `Config`; column names are auto-detected through the *_ALIASES
lists.  If your files are named differently, edit ONLY `Config` / the aliases.

------------------------------------------------------------------------------
CLEVER TRICKS USED (not obvious at first glance) — each is also marked
"TRICK n" next to the code:
------------------------------------------------------------------------------
 TRICK 1  Coordinate-keyed graph nodes (`_node_key`).  The oracle builds its
          graph from rounded endpoint coordinates, NOT from the u/v ids of
          Part 02.  So an id bug in Part 02 can't silently cancel itself out.
 TRICK 2  Prepared geometry (`shapely.prepared.prep`) makes "does this line hit
          this huge flood polygon" ~10-100x faster; essential for 100k+ edges.
 TRICK 3  Nodes are ALWAYS added to the oracle graph; only unflooded EDGES are
          added.  A node whose every edge is flooded stays as an isolated
          component (otherwise it would vanish and be mis-snapped).
 TRICK 4  "Flood-INDUCED" cut-off = cut_after & ~cut_before.  Villages already
          unreachable before the flood (bad OSM data) are not blamed on it.
 TRICK 5  Metamorphic testing: no ground truth needed.  Nested floods must give
          nested damage; translating the world, shuffling rows or reversing
          line direction must not change the answer.
 TRICK 6  Two-sided spatial tolerance.  Satellite labels are misaligned by
          10-30 m, so exact matching punishes you unfairly.  Precision uses
          "predicted feature within tol of reference"; recall uses the
          reverse — they are NOT computed on the same set.
 TRICK 7  Bootstrap of a yes/no metric = Binomial(n, p_hat)/n.  Identical in
          distribution to resampling rows, but O(B) memory instead of O(B*n).
 TRICK 8  Canonical content hash.  GeoPackage bytes change every write
          (timestamps), so file hashes are useless for "is this the same data".
          We hash sorted attributes + WKB snapped to a fixed grid instead.
 TRICK 9  Sanity bounding box in WGS84.  Swapped lat/lon or a wrong CRS lands
          your data in the ocean; one bbox test catches the whole class.
 TRICK 10 Stability classes in the sensitivity sweep: a settlement cut off in
          100% of buffer scenarios is a SAFE priority; one that flips is
          "fragile" and should be ground-verified.
==============================================================================
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import platform
import subprocess
import sys
import unittest
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree

log = logging.getLogger("floodresq.part05")

# ============================================================================
# 0. CONFIGURATION
# ============================================================================


@dataclass
class Config:
    """Every path / threshold lives here so nothing is hard-coded below."""

    data_dir: Path = Path("data/processed")
    out_dir: Path = Path("outputs/part05")

    # Outputs of earlier parts (edit to match your actual file names)
    edges_file: str = "road_edges.gpkg"            # Part 02
    damage_file: str = "damaged_edges.gpkg"        # Part 03
    settlements_file: str = "settlements.gpkg"     # Part 01
    hubs_file: str = "hubs.gpkg"                   # Part 04 (optional)
    flood_file: str = "flood_extent.gpkg"          # flood mask polygons
    connectivity_file: str = "connectivity.csv"    # Part 04
    reference_file: str = "reference_damage.gpkg"  # optional ground truth

    snap_tol_m: float = 300.0       # settlement -> nearest road node
    match_tol_m: float = 30.0       # tolerance for accuracy matching (TRICK 6)
    flood_buffers_m: Tuple[float, ...] = (0, 10, 20, 50, 100)
    robust_share: float = 0.8       # >= this share of scenarios => "robust"
    bootstrap_n: int = 2000
    seed: int = 42
    # TRICK 9: (min_lon, min_lat, max_lon, max_lat) — Nepal with margin
    sanity_bbox_wgs84: Tuple[float, float, float, float] = (79.0, 26.0, 89.0, 31.0)

    event_name: str = "Trishuli River flood, August 2026"
    flood_source: str = "<fill in: e.g. Sentinel-1 SAR change detection, dates>"
    team: str = "FloodResQ — Teammate 3 (GIS / Damage / Connectivity)"


# Column-name aliases: first match (case-insensitive) wins.
EDGE_ID_ALIASES = ["edge_id", "eid", "id", "osmid", "uv_key"]
SETTLEMENT_ID_ALIASES = ["settlement_id", "sid", "id", "osm_id", "osmid", "name"]
FLAG_ALIASES = ["flooded", "is_flooded", "damaged", "is_damaged", "hit"]
CUTOFF_ALIASES = ["is_cut_off", "cut_off", "cutoff", "is_cutoff", "isolated"]
POP_ALIASES = ["population", "pop", "pop_est"]

# ============================================================================
# 1. SMALL UTILITIES: report object, column picking, loading, CRS helpers
# ============================================================================


@dataclass
class Check:
    name: str
    passed: bool
    severity: str = "error"     # "error" blocks hand-off, "warn" doesn't
    detail: str = ""


class Report:
    """Collects PASS/FAIL checks; the single source of truth for 'are we OK?'."""

    def __init__(self) -> None:
        self.checks: List[Check] = []

    def add(self, name: str, passed: bool, detail: str = "", severity: str = "error") -> bool:
        passed = bool(passed)
        self.checks.append(Check(name, passed, severity, detail))
        level = logging.INFO if passed else (logging.ERROR if severity == "error" else logging.WARNING)
        log.log(level, "%s | %s %s", "PASS" if passed else "FAIL", name, detail)
        return passed

    def ok(self) -> bool:
        """True when no *error*-level check failed (warnings are allowed)."""
        return all(c.passed or c.severity != "error" for c in self.checks)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.checks])

    def summary(self) -> str:
        n = len(self.checks)
        bad = [c for c in self.checks if not c.passed]
        errs = sum(c.severity == "error" for c in bad)
        return f"{n - len(bad)}/{n} checks passed, {errs} errors, {len(bad) - errs} warnings"


def pick_column(df: pd.DataFrame, aliases: Sequence[str], required: bool = True) -> Optional[str]:
    """Return the real column name matching the first alias (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    if required:
        raise KeyError(f"None of {list(aliases)} found in columns {list(df.columns)}")
    return None


def load_layer(path: Path):
    """Load .csv -> DataFrame, anything else -> GeoDataFrame; None if missing."""
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return gpd.read_file(path)


def metric_crs_of(gdf: gpd.GeoDataFrame):
    """A metre-based CRS for `gdf` (itself if projected, else its UTM zone)."""
    return gdf.estimate_utm_crs() if gdf.crs.is_geographic else gdf.crs


def align_to(gdf: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    """Reproject only when needed (length/buffer maths must be in metres)."""
    return gdf if gdf.crs == crs else gdf.to_crs(crs)


# ============================================================================
# 2. SECTION A — CONTRACT CHECKS ("is the file even sane?")
# ============================================================================


def check_layer_contract(report: Report, name: str, gdf: gpd.GeoDataFrame,
                         geom_types: Iterable[str], id_col: Optional[str],
                         bbox_wgs84: Tuple[float, float, float, float]) -> None:
    """
    Generic health check for one spatial layer: non-empty, CRS set, no
    null/empty/invalid geometry, expected geometry types, unique ids, and
    inside the sanity bbox (TRICK 9).
    """
    if not report.add(f"{name}: non-empty", len(gdf) > 0, f"{len(gdf)} rows"):
        return
    if not report.add(f"{name}: CRS defined", gdf.crs is not None):
        return
    bad_geom = gdf.geometry.isna() | gdf.geometry.is_empty
    report.add(f"{name}: no null/empty geometry", not bad_geom.any(), f"{int(bad_geom.sum())} bad")
    good = gdf.geometry[~bad_geom]
    report.add(f"{name}: valid geometry", good.is_valid.all(),
               f"{int((~good.is_valid).sum())} invalid", severity="warn")
    found = set(good.geom_type)
    report.add(f"{name}: geometry types", found <= set(geom_types), f"found {sorted(found)}")
    if id_col:
        dups = int(gdf[id_col].duplicated().sum())
        report.add(f"{name}: unique '{id_col}'", dups == 0, f"{dups} duplicates", severity="warn")

    # TRICK 9: one bbox test catches swapped lat/lon and wrong-CRS mistakes.
    minx, miny, maxx, maxy = good.to_crs(4326).total_bounds
    bx0, by0, bx1, by1 = bbox_wgs84
    inside = (minx >= bx0) and (miny >= by0) and (maxx <= bx1) and (maxy <= by1)
    report.add(f"{name}: inside sanity bbox", inside,
               f"bounds lon[{minx:.2f},{maxx:.2f}] lat[{miny:.2f},{maxy:.2f}]")


# ============================================================================
# 3. SECTION B — INDEPENDENT ORACLE (re-compute with different code)
# ============================================================================
# Idea (N-version programming): if two separately written implementations
# agree, a bug would need to be identical in both — very unlikely.


def oracle_flooded_edges(edges: gpd.GeoDataFrame, flood: Optional[gpd.GeoDataFrame],
                         min_frac: float = 0.0) -> pd.Series:
    """
    Brute-force damage: an edge is flooded if it touches the flood polygons.

    min_frac > 0 additionally requires that fraction of the edge LENGTH to be
    inside the flood (ignores edges that merely graze the mask).
    Returns a bool Series with the same index as `edges`.
    """
    out = pd.Series(False, index=edges.index, dtype=bool)
    if flood is None or len(flood) == 0:
        return out
    flood = align_to(flood, edges.crs)
    union = unary_union(flood.geometry.values)       # one multipolygon
    if union.is_empty:
        return out
    prepared = prep(union)                            # TRICK 2
    geoms = edges.geometry.values
    hit = np.fromiter((prepared.intersects(g) for g in geoms), dtype=bool, count=len(geoms))
    if min_frac > 0:
        for i in np.flatnonzero(hit):                 # exact maths only where needed
            g = geoms[i]
            hit[i] = g.length > 0 and g.intersection(union).length / g.length >= min_frac
    out[:] = hit
    return out


def _node_key(coord) -> Tuple[float, float]:
    """TRICK 1: identify a graph node by its rounded (x, y), never by an id."""
    return (round(float(coord[0]), 2), round(float(coord[1]), 2))


def resolve_hubs(settlements: gpd.GeoDataFrame, hubs: Optional[gpd.GeoDataFrame],
                 pop_col: Optional[str]) -> gpd.GeoDataFrame:
    """
    Hubs = places that count as 'reachable safety' (hospital, district HQ...).
    Fallback if no hub layer exists: the most populous settlement.
    !! Match this with the definition you used in Part 04 !!
    """
    if hubs is not None and len(hubs) > 0:
        return hubs
    if pop_col:
        return settlements.loc[[settlements[pop_col].idxmax()]]
    return settlements.iloc[[0]]


def oracle_cutoff(edges: gpd.GeoDataFrame, flooded: pd.Series,
                  settlements: gpd.GeoDataFrame, hubs: gpd.GeoDataFrame,
                  snap_tol_m: float = 300.0) -> pd.Series:
    """
    Post-flood cut-off status per settlement (True = cannot reach any hub).

    Steps: build undirected graph -> connected components -> snap hubs and
    settlements to nearest node -> cut off iff component has no hub.
    Settlements farther than `snap_tol_m` from any road node count as cut off.
    """
    flooded = flooded.reindex(edges.index).fillna(False).astype(bool)
    G = nx.Graph()
    for geom, bad in zip(edges.geometry.values, flooded.values):
        a, b = _node_key(geom.coords[0]), _node_key(geom.coords[-1])
        G.add_node(a)                                  # TRICK 3: nodes always
        G.add_node(b)
        if not bad:                                    # ...edges only if intact
            G.add_edge(a, b)

    keys = list(G.nodes)
    node_pts = shapely.points(np.array(keys, dtype=float))
    tree = STRtree(node_pts)

    def snap(gdf: gpd.GeoDataFrame):
        pts = np.asarray(align_to(gdf, edges.crs).geometry.representative_point().to_numpy())
        idx = tree.nearest(pts)
        return idx, shapely.distance(pts, node_pts[idx])

    comp_of: Dict[Tuple[float, float], int] = {}
    for cid, comp in enumerate(nx.connected_components(G)):
        for n in comp:
            comp_of[n] = cid

    hub_idx, hub_d = snap(hubs)
    hub_comps = {comp_of[keys[i]] for i, d in zip(hub_idx, hub_d) if d <= snap_tol_m}

    s_idx, s_d = snap(settlements)
    cut = [(d > snap_tol_m) or (comp_of[keys[i]] not in hub_comps) for i, d in zip(s_idx, s_d)]
    return pd.Series(cut, index=settlements.index, dtype=bool)


def induced_cutoff(cutoff_fn: Callable, edges, flooded: pd.Series, settlements, hubs) -> pd.Series:
    """
    TRICK 4: cut-off CAUSED by the flood = cut after AND NOT cut before.
    Settlements that were already isolated on the intact network (OSM gaps)
    are excluded, so we never over-claim flood impact.
    """
    none = pd.Series(False, index=edges.index, dtype=bool)
    before = cutoff_fn(edges, none, settlements, hubs)
    after = cutoff_fn(edges, flooded, settlements, hubs)
    return after & ~before


def cohen_kappa(a, b) -> float:
    """Chance-corrected agreement between two boolean arrays (1 = perfect)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    if pe == 1.0:
        return 1.0 if po == 1.0 else float("nan")
    return float((po - pe) / (1 - pe))


def pipeline_edge_flags(edges: gpd.GeoDataFrame, edge_id: str, damage) -> Optional[pd.Series]:
    """
    Convert Part 03's output into a bool Series aligned with `edges`.
    Handles both layouts: (a) all edges with a flag column, (b) only the
    damaged edges listed (presence = damaged).
    """
    if damage is None:
        return None
    flag = pick_column(damage, FLAG_ALIASES, required=False)
    did = pick_column(damage, [edge_id] + EDGE_ID_ALIASES, required=False)
    if flag and did:
        m = damage.set_index(did)[flag].astype(bool).to_dict()
        return edges[edge_id].map(m).fillna(False).astype(bool)
    if flag and len(damage) == len(edges):
        return pd.Series(damage[flag].astype(bool).values, index=edges.index)
    if did:
        return edges[edge_id].isin(set(damage[did])).astype(bool)
    return None


def compare_to_oracle(report: Report, cfg: Config, edges, edge_id, settlements, sid,
                      flood, hubs, pop_col, damage, connectivity) -> Dict[str, pd.Series]:
    """
    Run the oracle and compare with Part 03 / Part 04 outputs.
    Writes disagreement CSVs so you can inspect exactly which ids differ.
    """
    results: Dict[str, pd.Series] = {}
    oracle_flags = oracle_flooded_edges(edges, flood)
    results["oracle_flooded"] = oracle_flags
    pipe_flags = pipeline_edge_flags(edges, edge_id, damage)
    if pipe_flags is not None:
        agree = float((pipe_flags == oracle_flags).mean())
        kappa = cohen_kappa(pipe_flags, oracle_flags)
        report.add("Part03 vs oracle: edge agreement >= 99%", agree >= 0.99,
                   f"agreement={agree:.4f}, kappa={kappa:.3f}")
        diff = edges.loc[pipe_flags != oracle_flags, [edge_id]].assign(
            pipeline=pipe_flags[pipe_flags != oracle_flags],
            oracle=oracle_flags[pipe_flags != oracle_flags])
        diff.to_csv(cfg.out_dir / "disagree_edges.csv", index=False)
        report.add("Part03 flagged subset of edges", pipe_flags.index.isin(edges.index).all())
    else:
        report.add("Part03 output readable", False, "no flag/id column found", severity="warn")

    hubs_r = resolve_hubs(settlements, hubs, pop_col)
    flags_for_oracle = pipe_flags if pipe_flags is not None else oracle_flags
    fn = lambda e, f, s, h: oracle_cutoff(e, f, s, h, cfg.snap_tol_m)
    oracle_cut = induced_cutoff(fn, edges, oracle_flags, settlements, hubs_r)
    results["oracle_cutoff"] = oracle_cut

    if connectivity is not None:
        c_id = pick_column(connectivity, [sid] + SETTLEMENT_ID_ALIASES, required=False)
        c_cut = pick_column(connectivity, CUTOFF_ALIASES, required=False)
        if c_id and c_cut:
            m = connectivity.set_index(c_id)[c_cut].astype(bool).to_dict()
            pipe_cut = settlements[sid].map(m)
            report.add("Part04: covers all settlements", pipe_cut.notna().all(),
                       f"{int(pipe_cut.isna().sum())} missing", severity="warn")
            pipe_cut = pipe_cut.fillna(False).astype(bool)
            agree = float((pipe_cut == oracle_cut).mean())
            report.add("Part04 vs oracle: cut-off agreement >= 95%", agree >= 0.95,
                       f"agreement={agree:.4f}, kappa={cohen_kappa(pipe_cut, oracle_cut):.3f}")
            bad = pipe_cut != oracle_cut
            settlements.loc[bad, [sid]].assign(pipeline=pipe_cut[bad], oracle=oracle_cut[bad]) \
                .to_csv(cfg.out_dir / "disagree_settlements.csv", index=False)
            results["pipeline_cutoff"] = pipe_cut
            # cross-part invariants
            report.add("Invariant: cut-off ids exist in settlements",
                       set(connectivity[c_id][connectivity[c_cut].astype(bool)]) <= set(settlements[sid]))
            if pipe_flags is not None and not pipe_flags.any():
                report.add("Invariant: no damage => no cut-off", not pipe_cut.any())
        else:
            report.add("Part04 output readable", False, "id/cut-off column missing", severity="warn")
    return results


# ============================================================================
# 4. SECTION C — SYNTHETIC SCENARIO + METAMORPHIC TESTS
# ============================================================================


def make_synthetic_scenario(n: int = 5, spacing: float = 100.0,
                            origin: Tuple[float, float] = (500000.0, 3100000.0)):
    """
    A 5x5 street grid in UTM 45N (EPSG:32645) with a KNOWN answer.

        y=4  D . . . B          hub at (0,0); settlements A,B,C,D,E
        y=3  . . . . .          a vertical flood band covers x in [150, 250]
        y=2  . . . C .          => columns x>=2 are severed from the hub
        y=1  . E . . .          => expected cut off: A, B, C   (D, E are safe)
        y=0  H . . . A
             x=0 1 2 3 4
    """
    ox, oy = origin
    lines = []
    for i in range(n):
        for j in range(n):
            p = (ox + i * spacing, oy + j * spacing)
            if i + 1 < n:
                lines.append(LineString([p, (ox + (i + 1) * spacing, oy + j * spacing)]))
            if j + 1 < n:
                lines.append(LineString([p, (ox + i * spacing, oy + (j + 1) * spacing)]))
    edges = gpd.GeoDataFrame({"edge_id": range(len(lines))}, geometry=lines, crs="EPSG:32645")

    spots = {"A": (4, 0), "B": (4, 4), "C": (3, 2), "D": (0, 4), "E": (1, 1)}
    settlements = gpd.GeoDataFrame(
        {"settlement_id": list(spots), "population": [100, 200, 300, 400, 500]},
        geometry=[Point(ox + i * spacing, oy + j * spacing) for i, j in spots.values()],
        crs="EPSG:32645")
    hubs = gpd.GeoDataFrame({"hub_id": ["H"]}, geometry=[Point(ox, oy)], crs="EPSG:32645")
    flood = gpd.GeoDataFrame({"fid": [1]},
                             geometry=[box(ox + 150, oy - 50, ox + 250, oy + (n - 1) * spacing + 50)],
                             crs="EPSG:32645")
    return edges, settlements, hubs, flood, {"A", "B", "C"}


def _shift(gdf: gpd.GeoDataFrame, dx: float, dy: float) -> gpd.GeoDataFrame:
    return gdf.set_geometry(gdf.geometry.translate(dx, dy))


def run_metamorphic_suite(report: Report,
                          damage_fn: Callable = oracle_flooded_edges,
                          cutoff_fn: Callable = oracle_cutoff) -> None:
    """
    TRICK 5 — metamorphic tests. We don't need ground truth, only RELATIONS
    between runs.  Plug your REAL Part 03 / Part 04 code in through the two
    callables to test it the same way:

        damage_fn(edges, flood_gdf)                      -> bool Series (index = edges)
        cutoff_fn(edges, flooded, settlements, hubs)     -> bool Series (index = settlements)
    """
    edges, sett, hubs, flood, expected = make_synthetic_scenario()
    ox, oy = 500000.0, 3100000.0
    ind = lambda e, f, s, h: induced_cutoff(cutoff_fn, e, f, s, h)

    # 1. known answer
    fl = damage_fn(edges, flood)
    cut = ind(edges, fl, sett, hubs)
    got = set(sett.loc[cut, "settlement_id"])
    report.add("Metamorphic: known-answer scenario", got == expected, f"got {sorted(got)}, want {sorted(expected)}")

    # 2. empty flood
    empty = flood.iloc[0:0]
    fl0 = damage_fn(edges, empty)
    report.add("Metamorphic: empty flood => nothing flooded", not fl0.any())
    report.add("Metamorphic: empty flood => nobody cut off", not ind(edges, fl0, sett, hubs).any())

    # 3. flood everything
    big = flood.set_geometry([box(ox - 1000, oy - 1000, ox + 2000, oy + 2000)])
    fl_all = damage_fn(edges, big)
    report.add("Metamorphic: total flood => all edges flooded", fl_all.all())
    report.add("Metamorphic: total flood => all settlements cut off",
               ind(edges, fl_all, sett, hubs).all())

    # 4. monotonicity over NESTED floods
    prev_f, prev_c, mono = None, None, True
    for w in (10, 60, 110, 160, 300):
        g = flood.set_geometry([box(ox + 200 - w, oy - 50, ox + 200 + w, oy + 450)])
        f = damage_fn(edges, g)
        c = ind(edges, f, sett, hubs)
        if prev_f is not None:
            mono &= not (prev_f & ~f).any() and not (prev_c & ~c).any()
        prev_f, prev_c = f, c
    report.add("Metamorphic: bigger flood never reduces damage / cut-off", mono)

    # 5. translation invariance
    dx, dy = 12345.0, -6789.0
    e2, s2, h2, f2 = _shift(edges, dx, dy), _shift(sett, dx, dy), _shift(hubs, dx, dy), _shift(flood, dx, dy)
    c2 = ind(e2, damage_fn(e2, f2), s2, h2)
    report.add("Metamorphic: translation invariance", c2.equals(cut))

    # 6. row-order invariance
    shuf = edges.sample(frac=1.0, random_state=1)
    c3 = ind(shuf, damage_fn(shuf, flood), sett, hubs)
    report.add("Metamorphic: edge-order invariance", c3.equals(cut))

    # 7. line-direction invariance (roads are undirected for connectivity)
    rev = edges.set_geometry(edges.geometry.apply(lambda g: LineString(list(g.coords)[::-1])))
    c4 = ind(rev, damage_fn(rev, flood), sett, hubs)
    report.add("Metamorphic: line-direction invariance", c4.equals(cut))


# ============================================================================
# 5. SECTION D — ACCURACY AGAINST A REFERENCE LAYER
# ============================================================================


def confusion_metrics(pred, truth) -> Dict[str, float]:
    """Strict (id-by-id) confusion matrix -> precision/recall/F1/IoU/accuracy."""
    p, t = np.asarray(pred, bool), np.asarray(truth, bool)
    tp, fp = int((p & t).sum()), int((p & ~t).sum())
    fn, tn = int((~p & t).sum()), int((~p & ~t).sum())
    div = lambda a, b: a / b if b else float("nan")
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": div(2 * prec * rec, prec + rec) if tp else (0.0 if (fp or fn) else float("nan")),
            "iou": div(tp, tp + fp + fn), "accuracy": div(tp + tn, tp + fp + fn + tn)}


def tolerant_precision_recall(pred_geoms: Sequence, ref_geoms: Sequence, tol_m: float) -> Dict:
    """
    TRICK 6 — two-sided spatial tolerance.
      precision = share of PREDICTED features within tol_m of any reference
      recall    = share of REFERENCE features within tol_m of any prediction
    Different denominators, so a 20 m satellite misregistration is not
    counted as both a false positive AND a false negative.
    Returns the per-feature boolean arrays too (used for bootstrap CIs).
    """
    pred_geoms, ref_geoms = list(pred_geoms), list(ref_geoms)
    if not pred_geoms or not ref_geoms:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
                "confirmed": np.zeros(len(pred_geoms), bool), "recalled": np.zeros(len(ref_geoms), bool)}
    ref_zone = prep(unary_union(ref_geoms).buffer(tol_m))
    pred_zone = prep(unary_union(pred_geoms).buffer(tol_m))
    confirmed = np.fromiter((ref_zone.intersects(g) for g in pred_geoms), bool, len(pred_geoms))
    recalled = np.fromiter((pred_zone.intersects(g) for g in ref_geoms), bool, len(ref_geoms))
    p, r = confirmed.mean(), recalled.mean()
    return {"precision": float(p), "recall": float(r),
            "f1": float(2 * p * r / (p + r)) if (p + r) else 0.0,
            "confirmed": confirmed, "recalled": recalled}


def bootstrap_rate_ci(flags, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """
    TRICK 7 — bootstrap CI for a yes/no rate without resampling rows.
    Resampling n Bernoulli(p_hat) items and averaging is *exactly*
    Binomial(n, p_hat) / n, so we draw that directly: O(n_boot) memory
    instead of O(n_boot * n).
    """
    flags = np.asarray(flags, bool)
    n = len(flags)
    if n == 0:
        return (float("nan"), float("nan"))
    draws = np.random.default_rng(seed).binomial(n, flags.mean(), size=n_boot) / n
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def evaluate_accuracy(report: Report, cfg: Config, edges: gpd.GeoDataFrame,
                      pred_flags: pd.Series, reference: Optional[gpd.GeoDataFrame]) -> Optional[pd.DataFrame]:
    """Compare predicted damaged edges to a reference layer; write accuracy CSV."""
    if reference is None or len(reference) == 0:
        report.add("Accuracy: reference layer available", False,
                   "skipped (no reference file) — add check-only reference labels to enable", severity="warn")
        return None
    reference = align_to(reference, edges.crs)
    res = tolerant_precision_recall(edges.geometry[pred_flags].values, reference.geometry.values, cfg.match_tol_m)
    rows = []
    for metric, arr in (("precision", res["confirmed"]), ("recall", res["recalled"])):
        lo, hi = bootstrap_rate_ci(arr, cfg.bootstrap_n, seed=cfg.seed)
        rows.append({"metric": metric, "value": res[metric], "ci95_low": lo, "ci95_high": hi, "n": len(arr)})
    rows.append({"metric": "f1", "value": res["f1"], "ci95_low": np.nan, "ci95_high": np.nan,
                 "n": int(pred_flags.sum())})
    table = pd.DataFrame(rows)
    table.to_csv(cfg.out_dir / "accuracy.csv", index=False)
    report.add("Accuracy: F1 >= 0.5 (tolerance %.0f m)" % cfg.match_tol_m, res["f1"] >= 0.5,
               f"precision={res['precision']:.2f}, recall={res['recall']:.2f}, f1={res['f1']:.2f}",
               severity="warn")
    return table


# ============================================================================
# 6. SECTION E — SENSITIVITY TO FLOOD-MASK UNCERTAINTY
# ============================================================================


def sensitivity_sweep(edges, flood, settlements, hubs, buffers_m: Sequence[float],
                      damage_fn: Callable = oracle_flooded_edges,
                      cutoff_fn: Callable = oracle_cutoff,
                      robust_share: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    SAR flood masks are uncertain by ~1-3 pixels.  Re-run the analysis with the
    flood polygon buffered by each value in `buffers_m` and see what changes.

    Returns (scenarios, stability):
      scenarios : one row per buffer — n flooded edges, n cut off, Jaccard
                  overlap of the cut-off set with the 0 m baseline.
      stability : one row per settlement — share of scenarios it is cut off
                  in, and TRICK 10's class: robust / fragile / safe.
    """
    flood = align_to(flood, edges.crs)
    union = unary_union(flood.geometry.values)
    cut_matrix, rows = {}, []
    base = None
    for b in sorted(buffers_m):
        g = flood.set_geometry([union.buffer(b)]) if b else flood
        f = damage_fn(edges, g)
        c = induced_cutoff(cutoff_fn, edges, f, settlements, hubs)
        cut_matrix[b] = c
        if base is None:
            base = set(c[c].index)
        cur = set(c[c].index)
        jac = len(base & cur) / len(base | cur) if (base | cur) else 1.0
        rows.append({"buffer_m": b, "edges_flooded": int(f.sum()), "settlements_cut_off": int(c.sum()),
                     "jaccard_vs_baseline": round(jac, 3)})
    mat = pd.DataFrame(cut_matrix)
    share = mat.mean(axis=1)
    cls = np.where(share >= robust_share, "robust", np.where(share > 0, "fragile", "safe"))
    stability = pd.DataFrame({"share_cut_off": share.round(3), "class": cls}, index=settlements.index)
    return pd.DataFrame(rows), stability


# ============================================================================
# 7. SECTION F — CASE STUDY (headline numbers, map, Markdown)
# ============================================================================


def compute_headline_numbers(edges, flooded: pd.Series, settlements, cut: pd.Series,
                             sid: str, pop_col: Optional[str], top_n: int = 10) -> Dict:
    """Summary numbers for the story: km flooded, share of network, who is cut off."""
    km = edges.geometry.length / 1000.0
    out = {"road_km_total": round(float(km.sum()), 1),
           "road_km_flooded": round(float(km[flooded].sum()), 1),
           "pct_network_flooded": round(100 * float(km[flooded].sum() / km.sum()), 2) if km.sum() else 0.0,
           "settlements_total": int(len(settlements)),
           "settlements_cut_off": int(cut.sum())}
    if pop_col:
        out["population_total"] = int(settlements[pop_col].fillna(0).sum())
        out["population_cut_off"] = int(settlements.loc[cut, pop_col].fillna(0).sum())
        top = settlements.loc[cut].nlargest(top_n, pop_col)[[sid, pop_col]]
    else:
        top = settlements.loc[cut].head(top_n)[[sid]]
    out["top_cut_off"] = top.to_dict("records")
    return out


def render_case_map(path: Path, edges, flooded, flood, settlements, cut, hubs=None) -> None:
    """Static PNG: grey roads, red flooded roads, blue flood, orange=cut off settlements."""
    import matplotlib
    matplotlib.use("Agg")                      # headless: no display needed
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    align_to(flood, edges.crs).plot(ax=ax, color="#9ecae1", alpha=0.5, zorder=1)
    edges[~flooded].plot(ax=ax, color="#888888", linewidth=0.6, zorder=2)
    edges[flooded].plot(ax=ax, color="#d62728", linewidth=1.4, zorder=3)
    align_to(settlements, edges.crs)[~cut].plot(ax=ax, color="#2ca02c", markersize=14, zorder=4)
    align_to(settlements, edges.crs)[cut].plot(ax=ax, color="#ff7f0e", markersize=40, edgecolor="k", zorder=5)
    if hubs is not None:
        align_to(hubs, edges.crs).plot(ax=ax, color="k", marker="*", markersize=160, zorder=6)
    ax.set_title("Flood damage and cut-off settlements")
    ax.set_axis_off()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_case_study_md(path: Path, cfg: Config, h: Dict, scenarios: Optional[pd.DataFrame],
                        stability: Optional[pd.DataFrame], accuracy: Optional[pd.DataFrame],
                        map_name: str) -> None:
    """Fill a Markdown template with the computed numbers (no manual copy-paste errors)."""
    L = [f"# Case study — {cfg.event_name}", "",
         f"*Prepared by {cfg.team}. Generated {dt.datetime.now():%Y-%m-%d %H:%M}.*", "",
         "## Headline", "",
         f"- Road network analysed: **{h['road_km_total']} km**; flooded: **{h['road_km_flooded']} km** "
         f"({h['pct_network_flooded']} %).",
         f"- Settlements cut off from the nearest hub: **{h['settlements_cut_off']} of {h['settlements_total']}**."]
    if "population_cut_off" in h:
        L.append(f"- Estimated people in cut-off settlements: **{h['population_cut_off']:,}** "
                 f"of {h['population_total']:,}.")
    L += ["", f"![map]({map_name})", "", "## Priority list (largest cut-off settlements)", ""]
    for r in h["top_cut_off"]:
        L.append("- " + ", ".join(f"{k}: {v}" for k, v in r.items()))
    if scenarios is not None:
        L += ["", "## How sensitive is this to the flood mask?", "",
              scenarios.to_markdown(index=False)]
    if stability is not None:
        vc = stability["class"].value_counts().to_dict()
        L += ["", f"Stability classes across buffers: {vc}. *Robust* = safe to prioritise; "
                  "*fragile* = verify on the ground first."]
    if accuracy is not None:
        L += ["", "## Accuracy vs reference (with 95 % bootstrap CI)", "", accuracy.round(3).to_markdown(index=False)]
    L += ["", "## Limitations", "",
          f"- Flood extent source: {cfg.flood_source}.",
          "- OSM road completeness varies; missing roads can create false 'cut-off' results "
          "(we exclude settlements already isolated before the flood).",
          "- Connectivity is binary (road exists / does not); partial damage, passability by foot "
          "or boat, and bridges' structural state are not modelled."]
    path.write_text("\n".join(L), encoding="utf-8")


# ============================================================================
# 8. SECTION G — HAND-OFF (hashes, manifest, GeoPackage, README, ZIP)
# ============================================================================


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Stream a file through SHA-256 in 1 MB chunks (constant memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_gdf_hash(gdf: gpd.GeoDataFrame) -> str:
    """
    TRICK 8 — hash the DATA, not the file.
    GPKG/GeoJSON bytes change with every write, so sha256_file can't tell
    "same data re-exported" from "data changed".  Here we sort by index, hash
    the attribute table, and hash geometries as WKB after snapping to a fixed
    grid (1e-7 deg or 1 cm) to kill float noise.
    """
    g = gdf.sort_index()
    h = hashlib.sha256()
    attrs = g.drop(columns=g.geometry.name)
    h.update(pd.util.hash_pandas_object(attrs, index=True).values.tobytes())
    grid = 1e-7 if (g.crs is not None and g.crs.is_geographic) else 0.01
    geoms = shapely.set_precision(g.geometry.to_numpy(), grid_size=grid)
    for wkb in shapely.to_wkb(geoms):
        if wkb is not None:
            h.update(wkb)
    return h.hexdigest()


def describe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Auto data-dictionary: column, dtype, non-null count, one example value."""
    rows = []
    for c in df.columns:
        s = df[c].dropna()
        ex = "" if s.empty else str(s.iloc[0])[:40]
        rows.append({"column": c, "dtype": str(df[c].dtype), "non_null": int(s.shape[0]), "example": ex})
    return pd.DataFrame(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "n/a"


def export_deliverables(cfg: Config, layers: Dict[str, gpd.GeoDataFrame]) -> List[Path]:
    """
    Write one consolidated GeoPackage (all layers, EPSG-native) plus
    web-friendly GeoJSON (EPSG:4326) per layer.  Returns the list of files.
    """
    gpkg = cfg.out_dir / "floodresq_deliverables.gpkg"
    if gpkg.exists():
        gpkg.unlink()                       # first layer must start a fresh file
    files = [gpkg]
    for name, gdf in layers.items():
        gdf.to_file(gpkg, layer=name, driver="GPKG")
        gj = cfg.out_dir / f"{name}.geojson"
        gdf.to_crs(4326).to_file(gj, driver="GeoJSON")
        files.append(gj)
    return files


def build_manifest(cfg: Config, files: Sequence[Path], layers: Dict[str, gpd.GeoDataFrame],
                   report: Report) -> Dict:
    """Machine-readable provenance: who/when/what versions/hashes of everything."""
    return {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "event": cfg.event_name, "team": cfg.team, "git_commit": _git_commit(),
        "python": platform.python_version(),
        "libs": {"geopandas": gpd.__version__, "shapely": shapely.__version__,
                 "networkx": nx.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "validation": {"summary": report.summary(), "ok": report.ok()},
        "layers": {n: {"rows": len(g), "crs": str(g.crs), "content_hash": canonical_gdf_hash(g)}
                   for n, g in layers.items()},
        "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in files if p.exists()},
    }


def write_readme(cfg: Config, layers: Dict[str, gpd.GeoDataFrame], path: Path, report: Report) -> None:
    """README for whoever receives the package: what is here, how to rerun, caveats."""
    L = [f"# FloodResQ deliverables — {cfg.event_name}", "",
         f"Prepared by: {cfg.team}", f"Validation: {report.summary()}", "",
         "## Files", "",
         "- `floodresq_deliverables.gpkg` — all layers (native CRS)",
         "- `*.geojson` — same layers in EPSG:4326 for web maps",
         "- `validation_report.csv` — every check that was run",
         "- `accuracy.csv`, `sensitivity_*.csv` — accuracy and robustness tables",
         "- `case_study.md`, `case_study_map.png` — the story",
         "- `manifest.json` — hashes + versions for reproducibility", "",
         "## Reproduce", "",
         "```", "python part05_validation_handoff.py selftest",
         "python part05_validation_handoff.py run --data-dir <dir> --out-dir <dir>", "```", "",
         "## Data dictionary", ""]
    for name, g in layers.items():
        L += [f"### {name}  ({len(g)} rows, {g.crs})", "", describe_columns(g.drop(columns=g.geometry.name)).to_markdown(index=False), ""]
    path.write_text("\n".join(L), encoding="utf-8")


def bundle_zip(zip_path: Path, files: Sequence[Path]) -> Path:
    """One ZIP to send to the team / judges."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if Path(f).exists():
                z.write(f, arcname=Path(f).name)
    return zip_path


# ============================================================================
# 9. ORCHESTRATION — run everything on the real data
# ============================================================================


def run_pipeline(cfg: Config) -> Report:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    report = Report()

    # ---- load -------------------------------------------------------------
    d = Path(cfg.data_dir)
    edges = load_layer(d / cfg.edges_file)
    sett = load_layer(d / cfg.settlements_file)
    flood = load_layer(d / cfg.flood_file)
    hubs = load_layer(d / cfg.hubs_file)
    damage = load_layer(d / cfg.damage_file)
    conn = load_layer(d / cfg.connectivity_file)
    ref = load_layer(d / cfg.reference_file)
    for name, layer in (("edges", edges), ("settlements", sett), ("flood", flood)):
        report.add(f"Input present: {name}", layer is not None, "required")
    if edges is None or sett is None or flood is None:
        return _finish(cfg, report, {})

    # ---- A. contracts -----------------------------------------------------
    edges = edges.reset_index(drop=True)
    sett = sett.reset_index(drop=True)
    edge_id = pick_column(edges, EDGE_ID_ALIASES, required=False)
    if edge_id is None:
        edges["edge_id"], edge_id = edges.index, "edge_id"
    sid = pick_column(sett, SETTLEMENT_ID_ALIASES, required=False)
    if sid is None:
        sett["settlement_id"], sid = sett.index, "settlement_id"
    pop_col = pick_column(sett, POP_ALIASES, required=False)

    metric = metric_crs_of(edges)
    check_layer_contract(report, "edges", edges, {"LineString"}, edge_id, cfg.sanity_bbox_wgs84)
    check_layer_contract(report, "settlements", sett, {"Point", "Polygon", "MultiPolygon"}, sid, cfg.sanity_bbox_wgs84)
    check_layer_contract(report, "flood", flood, {"Polygon", "MultiPolygon"}, None, cfg.sanity_bbox_wgs84)
    edges, sett, flood = align_to(edges, metric), align_to(sett, metric), align_to(flood, metric)
    hubs = align_to(hubs, metric) if hubs is not None else None

    # ---- B. oracle + cross-part invariants --------------------------------
    res = compare_to_oracle(report, cfg, edges, edge_id, sett, sid, flood, hubs, pop_col, damage, conn)
    flagged = pipeline_edge_flags(edges, edge_id, damage)
    flagged = flagged if flagged is not None else res["oracle_flooded"]
    cut = res.get("pipeline_cutoff", res["oracle_cutoff"])

    # ---- C. metamorphic (synthetic) ---------------------------------------
    run_metamorphic_suite(report)

    # ---- D. accuracy ------------------------------------------------------
    accuracy = evaluate_accuracy(report, cfg, edges, flagged, ref)

    # ---- E. sensitivity ---------------------------------------------------
    hubs_r = resolve_hubs(sett, hubs, pop_col)
    cutoff_fn = lambda e, f, s, h: oracle_cutoff(e, f, s, h, cfg.snap_tol_m)
    scenarios, stability = sensitivity_sweep(edges, flood, sett, hubs_r, cfg.flood_buffers_m,
                                             cutoff_fn=cutoff_fn, robust_share=cfg.robust_share)
    scenarios.to_csv(cfg.out_dir / "sensitivity_scenarios.csv", index=False)
    stability.assign(**{sid: sett[sid]}).to_csv(cfg.out_dir / "sensitivity_stability.csv", index=False)

    # ---- F. case study ----------------------------------------------------
    headline = compute_headline_numbers(edges, flagged, sett, cut, sid, pop_col)
    render_case_map(cfg.out_dir / "case_study_map.png", edges, flagged, flood, sett, cut, hubs_r)
    write_case_study_md(cfg.out_dir / "case_study.md", cfg, headline, scenarios, stability, accuracy,
                        "case_study_map.png")
    (cfg.out_dir / "headline.json").write_text(json.dumps(headline, indent=2, default=str))

    # ---- G. hand-off ------------------------------------------------------
    out_layers = {"road_edges": edges.assign(flooded=flagged.values),
                  "settlements": sett.assign(cut_off=cut.values, stability=stability["class"].values),
                  "flood_extent": flood}
    return _finish(cfg, report, out_layers)


def _finish(cfg: Config, report: Report, layers: Dict[str, gpd.GeoDataFrame]) -> Report:
    """Write report, deliverables, README, manifest and ZIP (even when checks failed)."""
    report.to_frame().to_csv(cfg.out_dir / "validation_report.csv", index=False)
    files: List[Path] = []
    if layers:
        files += export_deliverables(cfg, layers)
        write_readme(cfg, layers, cfg.out_dir / "README.md", report)
        files += [cfg.out_dir / n for n in ("README.md", "case_study.md", "case_study_map.png",
                                            "accuracy.csv", "sensitivity_scenarios.csv",
                                            "sensitivity_stability.csv", "headline.json")]
    files.append(cfg.out_dir / "validation_report.csv")
    manifest = build_manifest(cfg, files, layers, report)
    (cfg.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    bundle_zip(cfg.out_dir / "floodresq_handoff.zip", files + [cfg.out_dir / "manifest.json"])
    log.info("SUMMARY: %s", report.summary())
    return report


# ============================================================================
# 10. UNIT TESTS (run with: python part05_validation_handoff.py selftest)
# ============================================================================


class TestPart05(unittest.TestCase):
    def test_metamorphic_suite_passes_on_oracle(self):
        rep = Report()
        run_metamorphic_suite(rep)
        failed = [c.name for c in rep.checks if not c.passed]
        self.assertEqual(failed, [], failed)

    def test_confusion_metrics_known_values(self):
        m = confusion_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["f1"], 0.5)
        self.assertAlmostEqual(m["iou"], 1 / 3)

    def test_tolerance_forgives_small_shift(self):
        a = [LineString([(0, 0), (100, 0)])]
        b = [LineString([(0, 15), (100, 15)])]           # 15 m misregistration
        self.assertEqual(tolerant_precision_recall(a, b, 10)["f1"], 0.0)
        self.assertEqual(tolerant_precision_recall(a, b, 20)["f1"], 1.0)

    def test_bootstrap_ci_brackets_estimate(self):
        flags = np.array([True] * 70 + [False] * 30)
        lo, hi = bootstrap_rate_ci(flags, 4000)
        self.assertLess(lo, 0.7)
        self.assertGreater(hi, 0.7)
        self.assertLess(hi - lo, 0.25)

    def test_canonical_hash_ignores_row_order_but_sees_changes(self):
        edges, *_ = make_synthetic_scenario()
        h1 = canonical_gdf_hash(edges)
        self.assertEqual(h1, canonical_gdf_hash(edges.sample(frac=1, random_state=3)))
        moved = edges.copy()
        moved.loc[0, "geometry"] = LineString([(0, 0), (1, 1)])
        self.assertNotEqual(h1, canonical_gdf_hash(moved))

    def test_oracle_handles_pre_isolated_village(self):
        edges, sett, hubs, flood, _ = make_synthetic_scenario()
        far = gpd.GeoDataFrame({"settlement_id": ["Z"], "population": [1]},
                               geometry=[Point(520000, 3120000)], crs=edges.crs)
        both = pd.concat([sett, far], ignore_index=True)
        fl = oracle_flooded_edges(edges, flood)
        cut = induced_cutoff(oracle_cutoff, edges, fl, both, hubs)
        self.assertFalse(cut.iloc[-1])                    # TRICK 4: not blamed on flood

    def test_sensitivity_monotone_in_buffer(self):
        edges, sett, hubs, flood, _ = make_synthetic_scenario()
        sc, st = sensitivity_sweep(edges, flood, sett, hubs, (0, 50, 120))
        self.assertTrue(sc["settlements_cut_off"].is_monotonic_increasing)
        self.assertIn("robust", set(st["class"]))


def run_selftest() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPart05)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


# ============================================================================
# 11. COMMAND-LINE ENTRY POINT
# ============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="FloodResQ Part 05 — validation & hand-off")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="run unit + metamorphic tests on synthetic data")
    r = sub.add_parser("run", help="validate real outputs of Parts 01-04 and build the hand-off pack")
    r.add_argument("--data-dir", default="data/processed")
    r.add_argument("--out-dir", default="outputs/part05")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "selftest":
        return run_selftest()
    cfg = Config(data_dir=Path(args.data_dir), out_dir=Path(args.out_dir))
    report = run_pipeline(cfg)
    print("\n" + report.summary())
    return 0 if report.ok() else 2


if __name__ == "__main__":
    sys.exit(main())
