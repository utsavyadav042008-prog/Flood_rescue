"""
=====================================================================================
FloodResQ  |  PART 04  —  Connectivity & Cut-off Settlements  ("Who is cut off?")
=====================================================================================

WHAT THIS PART DOES (one paragraph)
-----------------------------------
Part 02 gave us a road graph. Part 03 told us which road segments / bridges were
hit by the flood. Part 04 answers the question rescue teams actually care about:

    "After the flood, which villages can STILL be reached from a relief hub,
     which can only be reached by a long detour, and which are completely
     CUT OFF?  And which single damaged road/bridge, if repaired, would
     reconnect the most people?"

INPUTS (adapt the Config block below to the exact filenames/columns from Parts 1-3)
-----------------------------------------------------------------------------------
  * road graph       : networkx MultiDiGraph (osmnx style: node attrs x,y ; edge attrs
                       length, travel_time (optional), highway, osmid, name, bridge)
                       with Part 03's damage label stored on each edge
                       (Config.damage_attr, e.g. "damage_class" = none/possible/likely/severe)
  * settlements      : GeoDataFrame of villages/towns (name, population optional)
  * hubs             : GeoDataFrame of relief origins (hospital, district HQ, shelter,
                       airstrip, army camp ...). "Reachable" always means "reachable
                       from at least one hub".

OUTPUTS  (in Config.out_dir/<scenario>/)
----------------------------------------
  settlements_status.geojson/.csv   every settlement + status + detour + blockers
  pockets.csv                       groups of mutually-connected, cut-off settlements
  damaged_edges.geojson/.csv        every damaged edge, CRITICAL vs BYPASSED, repair value
  summary.json                      headline numbers for slides / the demo
  map.png                           quick-look map
  + outputs/part04/scenario_comparison.csv  (optimistic / expected / pessimistic)

STATUS LABELS
-------------
  CONNECTED           reachable, travel time almost unchanged
  DEGRADED            reachable but only via a big detour (slow rescue)
  CUT_OFF             reachable BEFORE the flood, unreachable AFTER  <-- the headline
  NO_BASELINE_ACCESS  unreachable even before the flood (map/data gap, NOT flood effect)
  UNKNOWN_SNAP        no road node within Config.max_snap_m, so we can't judge

PIPELINE / FUNCTION INDEX  (read top to bottom — same order as the code)
------------------------------------------------------------------------
  1. Config                       all tunable numbers in one place
  2. load_graph / load_vector     read Part 02 / Part 01 outputs
  3. edge_time_s, _is_blocked     per-edge travel time + "is this edge impassable?"
  4. collapse_to_simple_graph     MultiDiGraph -> simple undirected Graph with TWO
                                  weights (t_base, t_flood) and a `blocked` flag
  5. snap_to_nodes                attach each settlement/hub to nearest road node
  6. analyse_scenario             the heart: reachability, status, pockets, blockers
  7. classify_damaged_edges       CRITICAL vs BYPASSED damaged edges + repair value
  8. write_outputs / plot_map     save results
  9. run_pipeline / main          glue + CLI  (python part04...py --demo  to test)

NON-OBVIOUS TRICKS USED  (each is also marked "TRICK #n" next to the code)
--------------------------------------------------------------------------
  TRICK #1  Snap on the PRE-FLOOD graph, never on the flooded one.
            If we snapped a village to the nearest *surviving* road, a flooded
            village would "teleport" onto some far-away working road and look
            connected. The village's anchor must not depend on the damage.

  TRICK #2  Multi-source Dijkstra instead of (settlements x hubs) shortest paths.
            One search that starts from ALL hubs at once gives every node its
            time-to-nearest-hub. Cost ~ one Dijkstra, not N*M of them.

  TRICK #3  Two weights on one graph + a filtered *view* (no copying).
            Each edge stores t_base (before flood) and t_flood (surviving roads).
            `nx.subgraph_view` hides blocked edges lazily, so "before vs after"
            runs on the same object with zero duplication.

  TRICK #4  Damaged edge is CRITICAL iff its two endpoints land in DIFFERENT
            connected components of the flooded graph.  No path search needed —
            one component labelling (O(V+E)) classifies every damaged edge.
            Damaged edges whose ends are still joined by another route are
            BYPASSED (damaged but harmless) and must NOT be the repair priority.

  TRICK #5  Compare against the baseline to separate "flood-caused" isolation
            from "map-data" isolation (NO_BASELINE_ACCESS). Otherwise every
            village on an unmapped track would be blamed on the flood.

  TRICK #6  Collapse one-way / parallel edges and block SYMMETRICALLY.
            Floodwater ignores one-way signs: if either direction of a physical
            segment is flagged, the whole segment is blocked. But two genuinely
            different parallel roads (different osmid) stay independent, so a
            surviving parallel road keeps the link alive.

  TRICK #7  Scenario sweep (optimistic / expected / pessimistic). Damage labels
            from satellite are uncertain, so we vary which classes count as
            "impassable" and report the range, not one fragile number.

Dependencies: networkx, geopandas, shapely, numpy, pandas, scipy, matplotlib
              (osmnx optional — only for reading GraphML with correct dtypes)
=====================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point


# =====================================================================================
# 1. CONFIG
# =====================================================================================
@dataclass
class Config:
    """Every knob lives here so you never hunt through the code to change one."""

    # --- file locations (EDIT to match your Part 01-03 outputs) --------------------
    graph_path: str = "data/processed/road_graph_damaged.graphml"
    settlements_path: str = "data/processed/settlements.gpkg"
    hubs_path: str = "data/processed/relief_hubs.gpkg"
    out_dir: str = "outputs/part04"

    # --- column names -----------------------------------------------------------------
    damage_attr: str = "damage_class"      # edge attribute written by Part 03
    name_col: str = "name"                 # settlement name column
    pop_col: str = "population"            # settlement population column (optional)

    # --- TRICK #7: which damage classes count as "road is impassable" ----------------
    # Lower-case strings. Adjust to whatever labels Part 03 produces.
    scenarios: dict = field(default_factory=lambda: {
        "optimistic":  {"severe"},
        "expected":    {"likely", "severe"},
        "pessimistic": {"possible", "likely", "severe"},
    })
    primary_scenario: str = "expected"     # the one used for the headline map

    # --- DEGRADED rule: BOTH must hold (avoids flagging 2 min -> 4 min trips) --------
    detour_ratio: float = 1.5              # flood_time / baseline_time >= this
    detour_extra_min: float = 30.0         # AND at least this many extra minutes

    # --- snapping ------------------------------------------------------------------
    max_snap_m: float = 2000.0             # farther than this from any road node => UNKNOWN_SNAP

    # --- fallback speeds (km/h) when the graph has no `travel_time` ------------------
    speeds_kph: dict = field(default_factory=lambda: {
        "motorway": 70, "trunk": 60, "primary": 50, "secondary": 40,
        "tertiary": 30, "unclassified": 25, "residential": 20,
        "track": 12, "path": 5, "service": 15, "default": 20,
    })


# =====================================================================================
# 2. LOADERS
# =====================================================================================
def load_graph(path: str) -> nx.MultiDiGraph:
    """
    Read the road graph produced in Part 02.

    Uses osmnx.load_graphml when available because it restores numeric types
    (plain networkx reads everything back as *strings*). If osmnx is missing we
    read with networkx and cast the fields we need by hand.
    """
    try:
        import osmnx as ox
        return ox.load_graphml(path)
    except ImportError:
        G = nx.read_graphml(path)
        for _, d in G.nodes(data=True):
            d["x"], d["y"] = float(d["x"]), float(d["y"])
        for _, _, _, d in G.edges(keys=True, data=True):
            for f in ("length", "travel_time"):
                if f in d:
                    d[f] = float(d[f])
        return nx.MultiDiGraph(G)


def load_vector(path: str) -> gpd.GeoDataFrame:
    """Read a GeoPackage/GeoJSON; assume WGS84 if the file has no CRS."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(4326)


# =====================================================================================
# 3. EDGE HELPERS
# =====================================================================================
def _first(value):
    """OSM attributes are sometimes lists (merged ways). Take the first item."""
    return value[0] if isinstance(value, (list, tuple)) and value else value


def edge_time_s(data: dict, cfg: Config) -> float:
    """
    Travel time of one edge in SECONDS.

    Prefers the `travel_time` attribute (osmnx add_edge_travel_times). If absent,
    falls back to length / speed-by-road-class so the code never crashes on a
    graph that Part 02 saved without times.
    """
    t = _first(data.get("travel_time"))
    if t is not None:
        try:
            return max(float(t), 0.1)
        except (TypeError, ValueError):
            pass
    length = float(_first(data.get("length", 0)) or 0)
    speed = cfg.speeds_kph.get(str(_first(data.get("highway", "default"))), cfg.speeds_kph["default"])
    return max(length / (speed / 3.6), 0.1)


def _is_blocked(data: dict, blocked_levels: set, cfg: Config) -> bool:
    """True if this edge's Part-03 damage label is in the scenario's 'impassable' set."""
    return str(_first(data.get(cfg.damage_attr, "none"))).strip().lower() in blocked_levels


def _is_bridge(data: dict) -> bool:
    b = str(_first(data.get("bridge", "no"))).strip().lower()
    return b not in ("no", "none", "nan", "false", "")


# =====================================================================================
# 4. COLLAPSE TO SIMPLE GRAPH  (TRICK #6 and #3)
# =====================================================================================
def collapse_to_simple_graph(G: nx.MultiDiGraph, blocked_levels: set, cfg: Config) -> nx.Graph:
    """
    Turn the osmnx MultiDiGraph into a SIMPLE UNDIRECTED graph H that carries both
    the "before" and "after" picture on every edge.

    Why undirected? In a flood emergency people walk, ride bikes, carry boats and
    ignore one-way signs. Direction only adds false "cut-offs" from sparse OSM
    one-way tagging in mountain areas.

    Edge attributes written on H[a][b]:
        t_base   : fastest time (s) using ALL roads between a and b (before flood)
        t_flood  : fastest time (s) using only NON-blocked roads (None if none)
        blocked  : True if EVERY physical road between a and b is blocked
        info     : list of dicts describing the blocked segments (name, bridge, level)

    TRICK #6 — how "segments" work:
        A physical road segment appears twice in a MultiDiGraph (a->b and b->a, same
        osmid). We group edges by (a, b, osmid) so both directions merge into ONE
        segment, and the segment is blocked if EITHER direction is flagged.
        Two different osmids between the same a and b are DIFFERENT roads, so if one
        is flooded and the other is not, the pair stays passable.
    """
    # ---- step 1: merge directions of the same physical road into segments ----------
    segs: dict = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        if u == v:                                   # ignore self loops
            continue
        a, b = sorted((u, v))
        key = (a, b, str(_first(d.get("osmid", f"k{k}"))))
        t = edge_time_s(d, cfg)
        blocked = _is_blocked(d, blocked_levels, cfg)
        seg = segs.get(key)
        if seg is None:
            segs[key] = {"t": t, "blocked": blocked, "bridge": _is_bridge(d),
                         "name": str(_first(d.get("name", ""))),
                         "level": str(_first(d.get(cfg.damage_attr, "none"))),
                         "osmid": key[2]}
        else:                                        # second direction of same road
            seg["t"] = min(seg["t"], t)
            if blocked and not seg["blocked"]:       # symmetric blocking
                seg["blocked"] = True
                seg["level"] = str(_first(d.get(cfg.damage_attr, "none")))
            seg["bridge"] = seg["bridge"] or _is_bridge(d)

    # ---- step 2: group segments per node pair and build H ---------------------------
    pairs: dict = {}
    for (a, b, _), seg in segs.items():
        pairs.setdefault((a, b), []).append(seg)

    H = nx.Graph()
    for n, d in G.nodes(data=True):                  # keep coordinates for snapping/plots
        H.add_node(n, x=float(d["x"]), y=float(d["y"]))
    for (a, b), group in pairs.items():
        ok = [s["t"] for s in group if not s["blocked"]]
        H.add_edge(
            a, b,
            t_base=min(s["t"] for s in group),
            t_flood=min(ok) if ok else None,
            blocked=not ok,
            info=[s for s in group if s["blocked"]],
        )
    return H


# =====================================================================================
# 5. SNAP SETTLEMENTS / HUBS TO ROAD NODES  (TRICK #1)
# =====================================================================================
def snap_to_nodes(G: nx.Graph, gdf: gpd.GeoDataFrame, max_snap_m: float) -> gpd.GeoDataFrame:
    """
    Attach every row of `gdf` to its nearest road node.

    Adds columns:  node (graph node id), snap_m (distance in metres), snap_ok (bool).

    TRICK #1: we snap against the *pre-flood* node set (all nodes of G, damaged or
    not). Nodes are never deleted from the graph, only edges are hidden, so a
    flooded village keeps its anchor and correctly shows up as isolated.

    Implementation detail: we project to a local UTM zone so distances are in real
    metres (degrees of longitude shrink with latitude), then use a cKDTree for
    O(log n) nearest-neighbour queries instead of looping.
    """
    ids = np.array(list(G.nodes))
    nodes = gpd.GeoDataFrame(
        {"node": ids},
        geometry=gpd.points_from_xy([G.nodes[n]["x"] for n in ids], [G.nodes[n]["y"] for n in ids]),
        crs=4326,
    )
    utm = nodes.estimate_utm_crs()
    node_xy = np.column_stack([nodes.to_crs(utm).geometry.x, nodes.to_crs(utm).geometry.y])

    pts = gdf.to_crs(utm).geometry.representative_point()   # polygons -> a point inside
    dist, idx = cKDTree(node_xy).query(np.column_stack([pts.x, pts.y]))

    out = gdf.copy()
    out["node"] = ids[idx]
    out["snap_m"] = dist
    out["snap_ok"] = dist <= max_snap_m
    return out


# =====================================================================================
# 6. THE HEART: ANALYSE ONE SCENARIO
# =====================================================================================
def analyse_scenario(H: nx.Graph, settlements: gpd.GeoDataFrame, hub_nodes: list, cfg: Config) -> dict:
    """
    Run the full "who is cut off?" analysis for ONE damage scenario.

    Returns dict with:
        settlements : GeoDataFrame with status, times, detour, pocket_id, blockers
        pockets     : DataFrame, one row per cut-off pocket
        edges       : GeoDataFrame of damaged edges (from classify_damaged_edges)
        summary     : dict of headline numbers
        H, comp_id  : graph and component labels (used by the plot)
    """
    # TRICK #3: a lazy *view* of H without blocked edges. No copy is made.
    G_flood = nx.subgraph_view(H, filter_edge=lambda u, v: not H[u][v]["blocked"])

    # TRICK #2: ONE multi-source Dijkstra from all hubs. Result: node -> seconds
    # to the nearest hub. `paths` lets us walk back the baseline route later.
    base_dist, base_paths = nx.multi_source_dijkstra(H, hub_nodes, weight="t_base")
    flood_dist = nx.multi_source_dijkstra_path_length(G_flood, hub_nodes, weight="t_flood")

    # Component labelling of the flooded network: node -> component number.
    comp_id = {}
    for i, comp in enumerate(nx.connected_components(G_flood)):
        for n in comp:
            comp_id[n] = i
    hub_comps = {comp_id[h] for h in hub_nodes}

    # ---- classify each settlement ----------------------------------------------------
    rows = []
    for _, r in settlements.iterrows():
        n = r["node"]
        tb, tf = base_dist.get(n), flood_dist.get(n)
        if not r["snap_ok"]:
            status = "UNKNOWN_SNAP"
        elif tb is None:                                       # TRICK #5
            status = "NO_BASELINE_ACCESS"
        elif tf is None:
            status = "CUT_OFF"
        else:
            extra = tf - tb
            ratio = tf / tb if tb > 0 else 1.0
            degraded = ratio >= cfg.detour_ratio and extra >= cfg.detour_extra_min * 60
            status = "DEGRADED" if degraded else "CONNECTED"

        # Which damaged edges sit on the ORIGINAL (pre-flood) route? Those are the culprits.
        blockers = []
        if status in ("CUT_OFF", "DEGRADED"):
            path = base_paths[n]
            for a, b in zip(path[:-1], path[1:]):
                if H[a][b]["blocked"]:
                    # For CUT_OFF only list CRITICAL edges (ends in different components);
                    # a bypassed edge on the old route is not why the village is cut off.
                    # For DEGRADED every blocked edge on the old route adds to the detour.
                    if status == "DEGRADED" or comp_id[a] != comp_id[b]:
                        blockers.append(_edge_label(a, b, H))

        rows.append({
            "baseline_min": None if tb is None else round(tb / 60, 1),
            "flood_min": None if tf is None else round(tf / 60, 1),
            "extra_min": None if (tb is None or tf is None) else round((tf - tb) / 60, 1),
            "status": status,
            "comp": comp_id.get(n),
            "blocked_on_route": ";".join(blockers),
        })

    res = pd.DataFrame(rows, index=settlements.index)
    S = settlements.join(res)
    S["pop"] = pd.to_numeric(S.get(cfg.pop_col, 0), errors="coerce").fillna(0)

    # ---- pockets: cut-off settlements that can still reach EACH OTHER ---------------
    # Settlements in the same flooded component form one pocket => one air-drop /
    # one boat run can serve the whole pocket.
    cut = S[S["status"] == "CUT_OFF"]
    pocket_rows, pocket_of_comp = [], {}
    for c, grp in sorted(cut.groupby("comp"), key=lambda kv: -kv[1]["pop"].sum()):
        pid = f"P{len(pocket_rows) + 1:02d}"
        pocket_of_comp[c] = pid
        pocket_rows.append({
            "pocket_id": pid,
            "n_settlements": len(grp),
            "population": int(grp["pop"].sum()),
            "settlements": "; ".join(grp[cfg.name_col].astype(str)) if cfg.name_col in grp else "",
            "centroid_lon": float(grp.geometry.x.mean()),
            "centroid_lat": float(grp.geometry.y.mean()),
        })
    S["pocket_id"] = S["comp"].map(pocket_of_comp)
    pockets = pd.DataFrame(pocket_rows)

    # ---- damaged edges: critical vs bypassed + repair value --------------------------
    edges = classify_damaged_edges(H, comp_id, hub_comps, cut, pocket_of_comp, cfg)

    # ---- summary -----------------------------------------------------------------------
    by_status = S.groupby("status").agg(settlements=("status", "size"), population=("pop", "sum"))
    summary = {
        "by_status": {k: {"settlements": int(v.settlements), "population": int(v.population)}
                      for k, v in by_status.iterrows()},
        "n_pockets": int(len(pockets)),
        "damaged_edges_total": int(len(edges)),
        "damaged_edges_critical": int((edges["kind"] == "CRITICAL").sum()) if len(edges) else 0,
        "damaged_edges_bypassed": int((edges["kind"] == "BYPASSED").sum()) if len(edges) else 0,
        "top_repair_targets": (
            edges[edges["kind"] == "CRITICAL"].head(5)[["edge", "name", "is_bridge", "reconnects_pop", "alternatives"]]
            .to_dict("records") if len(edges) else []
        ),
    }
    return {"settlements": S, "pockets": pockets, "edges": edges, "summary": summary,
            "H": H, "comp_id": comp_id}


def _edge_label(a, b, H: nx.Graph) -> str:
    """Readable id of a damaged edge, e.g. '123-456 (Trishuli Bridge)'."""
    a, b = sorted((a, b))
    names = {s["name"] for s in H[a][b]["info"] if s["name"] not in ("", "None", "nan")}
    return f"{a}-{b}" + (f" ({'/'.join(sorted(names))})" if names else "")


# =====================================================================================
# 7. CRITICAL vs BYPASSED DAMAGED EDGES  (TRICK #4)
# =====================================================================================
def classify_damaged_edges(H, comp_id, hub_comps, cut, pocket_of_comp, cfg) -> gpd.GeoDataFrame:
    """
    For every blocked edge decide whether it actually matters.

    TRICK #4:  edge (a,b) is CRITICAL  <=>  comp_id[a] != comp_id[b].
        If both ends are still in the same flooded component, some other route
        joins them, so the damage is BYPASSED (it hurts only by lengthening trips).
        This needs ONE component labelling, not a path search per edge.

    Repair value (`reconnects_pop`):
        If one side is a hub component and the other a cut-off pocket, repairing
        this edge alone re-links that pocket => value = pocket population.
        `alternatives` = how many critical edges join the SAME pair of components.
        Repairing ANY ONE of them works, so the engineer should pick the cheapest.
    """
    pocket_pop = cut.groupby("comp")["pop"].sum().to_dict()

    recs = []
    for a, b, d in H.edges(data=True):
        if not d["blocked"]:
            continue
        ca, cb = comp_id[a], comp_id[b]
        kind = "BYPASSED" if ca == cb else "CRITICAL"
        value = 0
        if kind == "CRITICAL":
            a_hub, b_hub = ca in hub_comps, cb in hub_comps
            if a_hub and not b_hub:
                value = pocket_pop.get(cb, 0)
            elif b_hub and not a_hub:
                value = pocket_pop.get(ca, 0)
        names = sorted({s["name"] for s in d["info"] if s["name"] not in ("", "None", "nan")})
        recs.append({
            "edge": f"{min(a, b)}-{max(a, b)}",
            "name": "/".join(names),
            "is_bridge": any(s["bridge"] for s in d["info"]),
            "damage_level": ",".join(sorted({s["level"] for s in d["info"]})),
            "kind": kind,
            "comp_pair": tuple(sorted((ca, cb))),
            "reconnects_pop": int(value),
            "pocket_id": pocket_of_comp.get(ca) or pocket_of_comp.get(cb),
            "geometry": LineString([(H.nodes[a]["x"], H.nodes[a]["y"]),
                                    (H.nodes[b]["x"], H.nodes[b]["y"])]),
        })
    if not recs:
        return gpd.GeoDataFrame(columns=["edge", "kind", "reconnects_pop", "geometry"], crs=4326)

    E = gpd.GeoDataFrame(recs, crs=4326)
    E["alternatives"] = E.groupby("comp_pair")["edge"].transform("size").where(E["kind"] == "CRITICAL", 0)
    E["comp_pair"] = E["comp_pair"].astype(str)              # tuples can't be written to GeoJSON
    return E.sort_values(["kind", "reconnects_pop"], ascending=[True, False]).reset_index(drop=True)


# =====================================================================================
# 8. OUTPUT
# =====================================================================================
STATUS_COLOURS = {"CONNECTED": "#2ca02c", "DEGRADED": "#ff7f0e", "CUT_OFF": "#d62728",
                  "NO_BASELINE_ACCESS": "#7f7f7f", "UNKNOWN_SNAP": "#bcbd22"}


def write_outputs(result: dict, hubs: gpd.GeoDataFrame, out: Path, cfg: Config) -> None:
    """Save the CSV/GeoJSON/JSON files for one scenario (+ map.png)."""
    out.mkdir(parents=True, exist_ok=True)
    S = result["settlements"].drop(columns=["comp"])
    S.to_file(out / "settlements_status.geojson", driver="GeoJSON")
    S.drop(columns="geometry").to_csv(out / "settlements_status.csv", index=False)
    result["pockets"].to_csv(out / "pockets.csv", index=False)
    E = result["edges"]
    if len(E):
        E.to_file(out / "damaged_edges.geojson", driver="GeoJSON")
        E.drop(columns="geometry").to_csv(out / "damaged_edges.csv", index=False)
    (out / "summary.json").write_text(json.dumps(result["summary"], indent=2, default=str))
    plot_map(result, hubs, out / "map.png")


def plot_map(result: dict, hubs: gpd.GeoDataFrame, path: Path) -> None:
    """Quick-look map: grey roads, red = critical damage, orange = bypassed damage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, S, E = result["H"], result["settlements"], result["edges"]
    fig, ax = plt.subplots(figsize=(9, 7))
    for a, b, d in H.edges(data=True):
        if not d["blocked"]:
            ax.plot([H.nodes[a]["x"], H.nodes[b]["x"]], [H.nodes[a]["y"], H.nodes[b]["y"]],
                    color="#cccccc", lw=0.8, zorder=1)
    if len(E):
        for kind, col in (("BYPASSED", "#ff7f0e"), ("CRITICAL", "#d62728")):
            sub = E[E["kind"] == kind]
            for g in sub.geometry:
                ax.plot(*g.xy, color=col, lw=2.5, ls="--" if kind == "BYPASSED" else "-", zorder=2)
    for status, col in STATUS_COLOURS.items():
        sub = S[S["status"] == status]
        if len(sub):
            ax.scatter(sub.geometry.x, sub.geometry.y, s=70, color=col, edgecolor="k",
                       label=f"{status} ({len(sub)})", zorder=3)
    ax.scatter(hubs.geometry.x, hubs.geometry.y, marker="*", s=300, color="blue", label="Hub", zorder=4)
    ax.set_title("FloodResQ Part 04 — who is cut off?  (red line = critical damage, dashed orange = bypassed)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =====================================================================================
# 9. GLUE
# =====================================================================================
def run_pipeline(G: nx.MultiDiGraph, settlements: gpd.GeoDataFrame, hubs: gpd.GeoDataFrame,
                 cfg: Config) -> dict:
    """
    Snap once, then run every scenario.  Returns {scenario_name: result_dict}.

    The snapping is done ONCE, outside the scenario loop (TRICK #1): every
    scenario sees identical anchors, so differences between scenarios are due to
    damage assumptions only.
    """
    settlements = snap_to_nodes(G, settlements, cfg.max_snap_m)
    hubs = snap_to_nodes(G, hubs, cfg.max_snap_m)
    hub_nodes = list(hubs.loc[hubs["snap_ok"], "node"].unique())
    if not hub_nodes:
        raise ValueError("No hub could be snapped to the road graph — check hub coordinates/CRS.")

    results = {}
    for name, levels in cfg.scenarios.items():              # TRICK #7
        H = collapse_to_simple_graph(G, {x.lower() for x in levels}, cfg)
        results[name] = analyse_scenario(H, settlements, hub_nodes, cfg)
        results[name]["hubs"] = hubs
    return results


def save_all(results: dict, cfg: Config) -> pd.DataFrame:
    """Write each scenario's files and a one-row-per-scenario comparison table."""
    root, comp = Path(cfg.out_dir), []
    for name, res in results.items():
        write_outputs(res, res["hubs"], root / name, cfg)
        st = res["summary"]["by_status"]
        comp.append({
            "scenario": name,
            "cut_off_settlements": st.get("CUT_OFF", {}).get("settlements", 0),
            "cut_off_population": st.get("CUT_OFF", {}).get("population", 0),
            "degraded_settlements": st.get("DEGRADED", {}).get("settlements", 0),
            "pockets": res["summary"]["n_pockets"],
            "critical_edges": res["summary"]["damaged_edges_critical"],
            "bypassed_edges": res["summary"]["damaged_edges_bypassed"],
        })
    table = pd.DataFrame(comp)
    root.mkdir(parents=True, exist_ok=True)
    table.to_csv(root / "scenario_comparison.csv", index=False)
    return table


# -------------------------------------------------------------------------------------
# Synthetic demo so you can test and study the logic with NO real data
# -------------------------------------------------------------------------------------
def build_demo():
    """
    7x5 grid of roads. A "river" runs between column 3 and column 4:
        row 0,1,3,4 crossings = severe     row 2 crossing = likely  (so scenarios differ)
    One extra damaged edge (row0, col1-col2) has a parallel route => BYPASSED.
    Hub is at the west edge (node 0). One village sits on an island road (never
    connected) to show NO_BASELINE_ACCESS, one is far from any road => UNKNOWN_SNAP.
    """
    cols, rows = 7, 5
    G = nx.MultiDiGraph()
    nid = lambda r, c: r * cols + c
    for r in range(rows):
        for c in range(cols):
            G.add_node(nid(r, c), x=85.0 + c * 0.02, y=28.0 + r * 0.02)

    def add_road(u, v, osmid, level="none", name="", bridge="no"):
        for a, b in ((u, v), (v, u)):                      # both directions, like OSM two-way
            G.add_edge(a, b, osmid=osmid, length=2000.0, highway="secondary",
                       **{"damage_class": level, "name": name, "bridge": bridge})

    oid = 1
    river = {0: "severe", 1: "severe", 2: "likely", 3: "severe", 4: "severe"}
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                lvl, nm, br = "none", "", "no"
                if c == 3:
                    lvl, nm, br = river[r], f"River Bridge {r}", "yes"
                if (r, c) == (0, 1):
                    lvl = "severe"                         # damaged but bypassable
                add_road(nid(r, c), nid(r, c + 1), oid, lvl, nm, br); oid += 1
            if r + 1 < rows:
                add_road(nid(r, c), nid(r + 1, c), oid); oid += 1

    # island: two nodes not connected to anything
    G.add_node(1000, x=85.30, y=28.30)
    G.add_node(1001, x=85.31, y=28.30)
    add_road(1000, 1001, oid)

    pts = {  # name: (node-ish location, population)
        "West_Village_A": (nid(1, 1), 800), "West_Village_B": (nid(3, 2), 450),
        "East_Village_A": (nid(0, 6), 1200), "East_Village_B": (nid(2, 5), 600),
        "East_Village_C": (nid(4, 6), 300), "Island_Hamlet": (1000, 90),
    }
    recs = [{"name": n, "population": p,
             "geometry": Point(G.nodes[k]["x"] + 0.002, G.nodes[k]["y"] + 0.002)}
            for n, (k, p) in pts.items()]
    recs.append({"name": "Remote_Hamlet", "population": 40, "geometry": Point(86.0, 29.0)})
    settlements = gpd.GeoDataFrame(recs, crs=4326)
    hubs = gpd.GeoDataFrame([{"name": "District_Hospital", "geometry": Point(85.0, 28.0)}], crs=4326)
    return G, settlements, hubs


def main():
    ap = argparse.ArgumentParser(description="FloodResQ Part 04 — connectivity & cut-off settlements")
    ap.add_argument("--demo", action="store_true", help="run on a built-in synthetic network")
    ap.add_argument("--out", default=None, help="output folder (default: Config.out_dir)")
    args = ap.parse_args()

    cfg = Config()
    if args.out:
        cfg.out_dir = args.out
    if args.demo:
        G, settlements, hubs = build_demo()
    else:
        G = load_graph(cfg.graph_path)
        settlements = load_vector(cfg.settlements_path)
        hubs = load_vector(cfg.hubs_path)

    results = run_pipeline(G, settlements, hubs, cfg)
    table = save_all(results, cfg)

    print("\n=== Scenario comparison ===")
    print(table.to_string(index=False))
    prim = results[cfg.primary_scenario]
    print(f"\n=== {cfg.primary_scenario}: settlements ===")
    cols = [c for c in (cfg.name_col, "status", "baseline_min", "flood_min", "extra_min",
                        "pocket_id", "blocked_on_route") if c in prim["settlements"].columns]
    print(prim["settlements"][cols].to_string(index=False))
    print(f"\n=== {cfg.primary_scenario}: top repair targets ===")
    print(json.dumps(prim["summary"]["top_repair_targets"], indent=2, default=str))
    print(f"\nFiles written to: {Path(cfg.out_dir).resolve()}")


if __name__ == "__main__":
    main()
