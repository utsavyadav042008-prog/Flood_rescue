"""Generate a contract-valid FAKE run so the dashboard/report can be built early.

Usage (Git Bash, repo root)::

    python scripts/make_mock_outputs.py                       # -> data/sample/mock_run
    python scripts/make_mock_outputs.py --dest /tmp/mock --no-rasters

The scenario mirrors the illustration in the challenge PDF: a river with a
wide flood corridor, a main road that dips into the corridor (road break), a
side road that crosses the river on a bridge, one hospital and four villages.

    A  stays connected          B, C  cut off by the broken main road
    D  cut off by the flooded bridge

Everything is marked ``is_mock = true`` (inside every file's provenance and in
run.json) so a mock can never be mistaken for a result; the final audit script
(``validate_run.py --forbid-mock``) refuses to pass a run that contains mock data.

How the geometry is built (the "trick")
---------------------------------------
Flood corridor, roads, buildings and the raster mask are ALL derived from the
same two analytic functions, ``river_y(x)`` and ``half_width(x)``.  So the
mask raster, the affected-road GeoJSON and the summary numbers agree with each
other by construction, exactly like real outputs should.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:               # works from any cwd, no install needed
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import fileio                                    # noqa: E402
from pipeline.config import load_config                        # noqa: E402
from pipeline.contracts import Artifact                        # noqa: E402
from pipeline.geo_math import lonlat_to_utm, utm_to_lonlat     # noqa: E402
from pipeline.run import RunPaths                              # noqa: E402
from pipeline.schemas import AOI, Provenance, Summary          # noqa: E402

MOCK_AOI = AOI(85.20, 28.05, 85.40, 28.20, name="mock-valley")
EVENT_DATE = "2026-08-26"
PIXEL_M = 50.0
STEP_M = 100.0                    # road densification step = segment length


class _Scenario:
    """Analytic valley in UTM metres plus helpers to sample it."""

    def __init__(self, aoi: AOI) -> None:
        self.epsg = aoi.utm_epsg()
        corners = [lonlat_to_utm(lon, lat, self.epsg)
                   for lon in (aoi.west, aoi.east) for lat in (aoi.south, aoi.north)]
        self.x_min = math.floor(min(c[0] for c in corners) / PIXEL_M) * PIXEL_M
        self.x_max = math.ceil(max(c[0] for c in corners) / PIXEL_M) * PIXEL_M
        self.y_min = math.floor(min(c[1] for c in corners) / PIXEL_M) * PIXEL_M
        self.y_max = math.ceil(max(c[1] for c in corners) / PIXEL_M) * PIXEL_M
        self.width_m = self.x_max - self.x_min
        self.y_mid = (self.y_min + self.y_max) / 2.0

    # river centre line and corridor half-width (works on floats AND numpy arrays)
    def river_y(self, x):
        import numpy as np
        return self.y_mid + 1500.0 * np.sin(2 * np.pi * (x - self.x_min) / 12000.0)

    def half_width(self, x):
        import numpy as np
        return 250.0 + 100.0 * np.sin(2 * np.pi * (x - self.x_min) / 7000.0 + 1.0)

    def inside(self, x: float, y: float) -> bool:
        return abs(y - float(self.river_y(x))) <= float(self.half_width(x))

    def fx(self, fraction: float) -> float:
        """x coordinate at a fraction of the AOI width."""
        return self.x_min + fraction * self.width_m

    def lonlat(self, x: float, y: float) -> list:
        lon, lat = utm_to_lonlat(x, y, self.epsg)
        return [round(lon, 6), round(lat, 6)]       # 1e-6 deg ~ 0.1 m


def _densify(points: list, step: float = STEP_M) -> list:
    """Insert points so consecutive vertices are at most ``step`` metres apart."""
    out = [points[0]]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        n = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / step))
        out += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(1, n + 1)]
    return out


def _fraction_inside(sc: _Scenario, a: tuple, b: tuple, samples: int = 5) -> float:
    pts = [(a[0] + (b[0] - a[0]) * k / (samples - 1), a[1] + (b[1] - a[1]) * k / (samples - 1))
           for k in range(samples)]
    return sum(sc.inside(x, y) for x, y in pts) / samples


def build_vectors(sc: _Scenario, min_fraction: float) -> dict:
    """Roads, bridge, buildings, settlements, hospital and the derived damage layers."""
    rng = random.Random(42)                          # deterministic mock

    # --- ways as lists of metric points -------------------------------------
    x_break = sc.fx(0.55)
    main = []
    x = sc.fx(0.03)
    while x <= sc.fx(0.97):
        dip = 860.0 * math.exp(-(((x - x_break) / 500.0) ** 2))   # road dips into corridor
        main.append((x, float(sc.river_y(x)) + 900.0 - dip))
        x += STEP_M
    # The side road to village D must start AT a vertex of the main road, exactly like two OSM
    # ways sharing a node; otherwise the road graph would not be connected there.
    xd, y_main_d = min(main, key=lambda p: abs(p[0] - sc.fx(0.40)))
    yr_d = float(sc.river_y(xd))
    ways = {
        "way/main": ("primary", False, "Mock Valley Road", main),
        "way/branch-north": ("track", False, "Village D track",
                             _densify([(xd, y_main_d), (xd, yr_d + 450.0)])),
        "way/bridge-1": ("track", True, "Mock Bridge",
                         _densify([(xd, yr_d + 450.0), (xd, yr_d - 450.0)])),
        "way/branch-south": ("track", False, "Village D track",
                             _densify([(xd, yr_d - 450.0), (xd, yr_d - 1400.0)])),
    }

    road_feats, aff_roads, aff_bridges, bridge_feats = [], [], [], []
    seg_ids: dict[str, list] = {k: [] for k in ways}
    affected_ids: dict[str, list] = {k: [] for k in ways}
    affected_len_m = 0.0
    for way_id, (highway, is_bridge, name, pts) in ways.items():
        for k, (a, b) in enumerate(zip(pts, pts[1:])):
            sid = f"{way_id}#{k}"
            seg_ids[way_id].append(sid)
            geom = {"type": "LineString", "coordinates": [sc.lonlat(*a), sc.lonlat(*b)]}
            road_feats.append(fileio.feature(geom, id=sid, highway_type=highway,
                                             bridge=is_bridge, name=name))
            frac = _fraction_inside(sc, a, b)
            if frac >= min_fraction:
                affected_ids[way_id].append(sid)
                affected_len_m += math.hypot(b[0] - a[0], b[1] - a[1]) * frac
                aff_roads.append(fileio.feature(
                    geom, object_id=sid, object_type="road", impact_status="affected",
                    impact_fraction=round(frac, 2), confidence=round(0.55 + 0.4 * frac, 2)))

    bridge_pts = ways["way/bridge-1"][3]
    bridge_geom = {"type": "LineString", "coordinates": [sc.lonlat(*p) for p in bridge_pts]}
    bridge_feats.append(fileio.feature(bridge_geom, id="way/bridge-1", name="Mock Bridge"))
    bfrac = sum(sc.inside(*p) for p in bridge_pts) / len(bridge_pts)
    if bfrac >= min_fraction:
        aff_bridges.append(fileio.feature(
            bridge_geom, object_id="way/bridge-1", object_type="bridge",
            impact_status="potentially_damaged", impact_fraction=round(bfrac, 2),
            confidence=round(0.55 + 0.4 * bfrac, 2)))

    # --- settlements + hospital --------------------------------------------------
    def on_main(fraction: float, offset: float = 0.0) -> tuple:
        x0 = sc.fx(fraction)
        return (x0, float(sc.river_y(x0)) + 900.0 + offset)

    places = {
        "node/H1": ("Mock District Hospital", on_main(0.08)),
        "node/A": ("Village A", on_main(0.25, 150)),
        "node/B": ("Village B", on_main(0.70, 150)),
        "node/C": ("Village C", on_main(0.88, 150)),
        "node/D": ("Village D", (xd, yr_d - 1400.0)),
    }
    names_ne = {"node/A": "गाउँ A", "node/B": "गाउँ B", "node/C": "गाउँ C", "node/D": "गाउँ D"}
    settlements = [fileio.feature({"type": "Point", "coordinates": sc.lonlat(*xy)},
                                  id=pid, name=name, name_ne=names_ne[pid])
                   for pid, (name, xy) in places.items() if pid != "node/H1"]
    hospitals = [fileio.feature({"type": "Point", "coordinates": sc.lonlat(*places["node/H1"][1])},
                                id="node/H1", name="Mock District Hospital",
                                destination_type="hospital")]

    # --- buildings ----------------------------------------------------------------
    centres = []
    for pid, (_, (px, py)) in places.items():
        if pid != "node/H1":
            centres += [(px + rng.uniform(-150, 150), py + rng.uniform(-100, 100)) for _ in range(8)]
    for _ in range(25):                                   # riverside houses (many get hit)
        x0 = rng.uniform(sc.fx(0.05), sc.fx(0.95))
        centres.append((x0, float(sc.river_y(x0)) + rng.uniform(-450, 450)))
    buildings, aff_buildings = [], []
    for i, (cx, cy) in enumerate(centres):
        h = 10.0
        ring = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h), (cx - h, cy - h)]
        geom = {"type": "Polygon", "coordinates": [[sc.lonlat(*p) for p in ring]]}
        bid = f"way/b{i}"
        buildings.append(fileio.feature(geom, id=bid, type="house"))
        frac = sum(sc.inside(*p) for p in ring[:4] + [(cx, cy)]) / 5.0
        if frac >= min_fraction:
            aff_buildings.append(fileio.feature(
                geom, object_id=bid, object_type="building", impact_status="potentially_damaged",
                impact_fraction=round(frac, 2), confidence=round(0.5 + 0.4 * frac, 2)))

    # --- cut-off result (hand-derived from the scenario, see module docstring) ---
    blocking_main = affected_ids["way/main"]
    cut = []
    for pid, why, blockers in (
        ("node/B", "Main road is flooded between this village and the hospital; "
                   "no alternative path in the pre-event network.", blocking_main),
        ("node/C", "Main road is flooded between this village and the hospital; "
                   "no alternative path in the pre-event network.", blocking_main),
        ("node/D", "The only bridge to this village overlaps the flood mask; "
                   "no alternative path in the pre-event network.", ["way/bridge-1"]),
    ):
        name, xy = places[pid]
        cut.append(fileio.feature({"type": "Point", "coordinates": sc.lonlat(*xy)},
                                  settlement_id=pid, name=name, reason=why,
                                  nearest_destination="Mock District Hospital",
                                  blocking_segment_ids=list(blockers)))

    return dict(roads=road_feats, bridges=bridge_feats, buildings=buildings,
                settlements=settlements, hospitals=hospitals,
                aff_roads=aff_roads, aff_bridges=aff_bridges, aff_buildings=aff_buildings,
                cut=cut, affected_len_km=affected_len_m / 1000.0,
                settlements_total=len(settlements))


def build_rasters(sc: _Scenario):
    """Synthetic SAR-like before/after, binary flood mask and probability map."""
    import numpy as np

    rng = np.random.default_rng(42)
    cols = int(round((sc.x_max - sc.x_min) / PIXEL_M))
    rows = int(round((sc.y_max - sc.y_min) / PIXEL_M))
    xs = sc.x_min + (np.arange(cols) + 0.5) * PIXEL_M
    ys = sc.y_max - (np.arange(rows) + 0.5) * PIXEL_M          # row 0 is the NORTH edge
    X, Y = np.meshgrid(xs, ys)
    dy = np.abs(Y - sc.river_y(X))
    corridor = dy <= sc.half_width(X)

    before = rng.normal(-8.0, 2.0, (rows, cols))                # rough terrain, dB
    before[dy <= 40.0] = rng.normal(-22.0, 1.0, (dy <= 40.0).sum())   # thin river = dark
    after = before.copy()
    debris = corridor & (rng.random((rows, cols)) < 0.7)
    ponds = corridor & ~debris
    after[debris] = rng.normal(-5.0, 1.5, debris.sum())         # wide rough debris corridor
    after[ponds] = rng.normal(-21.0, 1.0, ponds.sum())          # dark ponded water

    mask = corridor.astype("uint8")
    prob = np.where(corridor, rng.uniform(0.70, 0.98, (rows, cols)), rng.uniform(0.0, 0.15, (rows, cols)))
    return (before.astype("float32"), after.astype("float32"), mask, prob.astype("float32"),
            PIXEL_M * PIXEL_M / 1e6)


def build_mock_run(dest: Path, with_rasters: bool = True) -> RunPaths:
    """Write a complete mock run into ``dest`` and return its RunPaths."""
    cfg = load_config()
    run = RunPaths.create(MOCK_AOI, EVENT_DATE, cfg, exact_dir=dest, is_mock=True)
    sc = _Scenario(MOCK_AOI)
    epsg_txt = f"EPSG:{sc.epsg}"

    def prov(producer: str, source: str, crs: str = "EPSG:4326") -> Provenance:
        return Provenance(producer=producer, source=source, crs=crs, event_date=EVENT_DATE,
                          model_version=cfg["model"]["version"], is_mock=True)

    vec = build_vectors(sc, float(cfg["damage"]["min_impact_fraction"]))
    mock_src = "MOCK synthetic data (not real imagery)"

    # acquire ---------------------------------------------------------------
    fileio.write_json(run.path(Artifact.METADATA), {
        "source": mock_src, "before_date": "2026-08-20", "after_date": "2026-09-01",
        "relative_orbit_before": 99, "relative_orbit_after": 99,
        "orbit_direction": "ASCENDING", "crs": epsg_txt, "pixel_size_m": PIXEL_M,
        "polarisation": "VV"}, prov("mock.acquire", mock_src, epsg_txt))

    # osm -------------------------------------------------------------------
    osm_src = "MOCK synthetic OSM-like data"
    fileio.write_json(run.ensure_parent(Artifact.OSM_SNAPSHOT), {
        "snapshot_date": cfg["osm"]["snapshot_date"], "provider": cfg["osm"]["provider"]},
        prov("mock.osm", osm_src))
    for art, key in ((Artifact.OSM_ROADS, "roads"), (Artifact.OSM_BRIDGES, "bridges"),
                     (Artifact.OSM_BUILDINGS, "buildings"), (Artifact.OSM_SETTLEMENTS, "settlements"),
                     (Artifact.OSM_HOSPITALS, "hospitals")):
        fileio.write_geojson(run.path(art), vec[key], prov("mock.osm", osm_src))

    # damage / connectivity ---------------------------------------------------------
    for art, key in ((Artifact.AFFECTED_ROADS, "aff_roads"), (Artifact.AFFECTED_BRIDGES, "aff_bridges"),
                     (Artifact.AFFECTED_BUILDINGS, "aff_buildings"), (Artifact.CUTOFF_SETTLEMENTS, "cut")):
        fileio.write_geojson(run.path(art), vec[key], prov("mock.gis", "MOCK mask x MOCK OSM"))
    fileio.write_json(run.path(Artifact.DAMAGE_SUMMARY), {
        "affected_roads": len(vec["aff_roads"]),
        "affected_road_length_km": round(vec["affected_len_km"], 2),
        "affected_bridges": len(vec["aff_bridges"]),
        "affected_buildings": len(vec["aff_buildings"])}, prov("mock.gis", "MOCK"))
    cut_names = [f["properties"]["name"] for f in vec["cut"]]
    fileio.write_json(run.path(Artifact.CONNECTIVITY_SUMMARY), {
        "cutoff_settlements": len(cut_names), "cutoff_names": cut_names,
        "settlements_total": vec["settlements_total"],
        "destinations_used": ["Mock District Hospital"]}, prov("mock.gis", "MOCK"))

    # rasters (need rasterio) --------------------------------------------------------
    flood_area = None
    mean_conf = None
    if with_rasters:
        before, after, mask, probability, pixel_km2 = build_rasters(sc)
        tf = fileio.north_up_transform(sc.x_min, sc.y_max, PIXEL_M)
        for art, arr, p in ((Artifact.BEFORE, before, prov("mock.acquire", mock_src, epsg_txt)),
                            (Artifact.AFTER, after, prov("mock.acquire", mock_src, epsg_txt)),
                            (Artifact.FLOOD_MASK, mask, prov("mock.infer", mock_src, epsg_txt)),
                            (Artifact.FLOOD_CONFIDENCE, probability, prov("mock.infer", mock_src, epsg_txt))):
            fileio.write_geotiff(run.path(art), arr, transform=tf, epsg=sc.epsg, provenance=p)
        flood_area = round(float(mask.sum()) * pixel_km2, 2)
        mean_conf = round(float(probability[mask == 1].mean()), 3)

    # summary (Part 2 replaces this with the real aggregation) ---------------------
    summary = Summary(
        flood_area_km2=flood_area, affected_roads=len(vec["aff_roads"]),
        affected_road_length_km=round(vec["affected_len_km"], 2),
        affected_bridges=len(vec["aff_bridges"]), affected_buildings=len(vec["aff_buildings"]),
        cutoff_settlements=len(cut_names), cutoff_names=cut_names, mean_confidence=mean_conf,
        aoi_name=MOCK_AOI.name, event_date=EVENT_DATE, before_date="2026-08-20",
        after_date="2026-09-01", sentinel1_orbit="MOCK 99 ASCENDING", crs=epsg_txt,
        model_version=cfg["model"]["version"], osm_snapshot_date=cfg["osm"]["snapshot_date"],
        processing_date=run.manifest()["created_utc"], is_mock=True,
        warnings=["MOCK DATA - synthetic scenario, not a real result."]
        + ([] if with_rasters else ["Rasters were not generated (rasterio missing)."]))
    fileio.write_json(run.path(Artifact.SUMMARY), summary.to_dict(), prov("mock.summarize", "MOCK"))
    return run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", default=str(REPO_ROOT / "data" / "sample" / "mock_run"))
    ap.add_argument("--no-rasters", action="store_true", help="skip GeoTIFFs (no rasterio needed)")
    args = ap.parse_args(argv)

    with_rasters = not args.no_rasters
    if with_rasters and not fileio.rasterio_available():
        print("rasterio is not installed -> generating vectors/JSON only (use --no-rasters to silence).")
        with_rasters = False
    run = build_mock_run(Path(args.dest), with_rasters)
    print(f"Mock run written to {run.root}")
    print(f"Validate it with:  python scripts/validate_run.py {run.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
