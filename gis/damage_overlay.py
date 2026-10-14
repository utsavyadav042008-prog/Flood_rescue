"""
=====================================================================
FloodResQ - PART 03: DAMAGE OVERLAY  ("What was hit?")
Track B - Trishuli flood mapping from space (Teammate 3: GIS / Damage)
=====================================================================

WHAT THIS PART DOES
-------------------
Takes the FLOOD EXTENT (from the SAR/satellite teammate) and lays it on top of
the OSM data you cleaned in Part 01, then answers:

    * Which BUILDINGS are flooded?        (none / edge / partial / major)
    * Which ROAD segments are blocked?    (open / blocked / bridge_over_flood)
    * Which CRITICAL FACILITIES are hit?  (hospitals, schools, shelters ...)
    * How bad is it PER AREA?             (500 m grid for a heat-map)

It writes one GeoPackage (for Part 04 / QGIS) + GeoJSON files in EPSG:4326
(for the web map) + a JSON summary (for the dashboard / pitch numbers).

HOW IT CONNECTS TO THE OTHER PARTS
----------------------------------
  Part 01 (OSM acquisition)  -> GeoPackage with layers: buildings, roads, pois
  Part 02 (road graph)       -> optional: roads layer with u, v, key columns
  Part 03 (THIS FILE)        -> damage layers + `blocked_edges.csv`
  Part 04 (connectivity)     -> removes blocked edges from the Part 02 graph
                                and finds cut-off settlements

!! ASSUMPTIONS (edit the CONFIG block if your Part 01 differs) !!
  - Part 01 saved a GeoPackage with layers named "buildings", "roads", "pois".
  - Roads are LineStrings and may carry OSM tags: bridge, tunnel, highway,
    and (from Part 02) u / v / key edge identifiers.
  - POIs carry an `amenity` column (hospital, school, ...). Missing columns
    are tolerated - the code just skips what it cannot find.
  - Flood extent is either a binary GeoTIFF (1 = flood) or a vector file.

USAGE
-----
    python part03_damage_overlay.py \
        --osm data/osm_normalised.gpkg \
        --flood data/flood_extent.tif \
        --out outputs/part03

Dependencies: geopandas, shapely>=2, rasterio, numpy, pandas
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box, shape
from shapely.ops import unary_union

log = logging.getLogger("part03")


# =====================================================================
# 0. CONFIG  - every threshold lives here so you can tune in one place
# =====================================================================
@dataclass
class Config:
    # --- flood-extent cleaning ---
    min_flood_patch_m2: float = 1_000.0   # drop SAR speckle smaller than this
    # --- buildings ---
    edge_buffer_m: float = 10.0           # ~1 Sentinel-1 pixel of uncertainty
    partial_frac: float = 0.10            # >=10% of footprint wet  -> partial
    major_frac: float = 0.50              # >=50% of footprint wet  -> major
    # --- roads ---
    min_block_len_m: float = 20.0         # wet stretch >= 20 m (~2 px) blocks a road
    # --- critical facilities ---
    critical_amenities: tuple = ("hospital", "clinic", "school", "shelter",
                                 "police", "fire_station", "doctors")
    # --- aggregation ---
    grid_size_m: float = 500.0
    # --- layer names inside the Part 01 GeoPackage ---
    layer_buildings: str = "buildings"
    layer_roads: str = "roads"
    layer_pois: str = "pois"
    # columns we try to carry through untouched (if they exist)
    road_id_cols: tuple = ("u", "v", "key", "osmid")


# =====================================================================
# 1. LOADING + CRS HELPERS
# =====================================================================
def metric_crs_for(gdf: gpd.GeoDataFrame):
    """
    Return a UTM CRS appropriate for `gdf`.

    WHY: OSM is in lat/lon (EPSG:4326) where '1 unit' is degrees - areas and
    lengths computed there are meaningless. For the Trishuli valley
    (~85 E) this resolves to UTM zone 45N (EPSG:32645). Every area / length /
    buffer in this file is computed AFTER projecting to this CRS.
    """
    return gdf.estimate_utm_crs()


def safe_layer(path: Path, layer: str) -> gpd.GeoDataFrame | None:
    """Read one layer from a GeoPackage; return None (not a crash) if absent."""
    try:
        gdf = gpd.read_file(path, layer=layer)
    except Exception as exc:  # layer missing / driver issue
        log.warning("Layer '%s' not readable (%s) - skipping.", layer, exc)
        return None
    if gdf.empty:
        log.warning("Layer '%s' is empty - skipping.", layer)
        return None
    gdf["geometry"] = shapely.make_valid(gdf.geometry.values)  # see TRICK 6
    return gdf[~gdf.geometry.is_empty].copy()


def load_flood_extent(path: Path, target_crs, cfg: Config) -> shapely.Geometry:
    """
    Load the flood extent and return ONE cleaned (multi)polygon in `target_crs`.

    Accepts:
      * raster  (.tif/.tiff): pixel value > 0 is treated as flooded
      * vector  (.geojson/.gpkg/.shp): polygons used directly

    Steps: polygonise -> reproject -> drop speckle -> dissolve to one geometry.
    """
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        flood = _polygonise_raster(path)
    else:
        flood = gpd.read_file(path)

    if flood.crs is None:
        raise ValueError("Flood extent has no CRS - ask the SAR teammate to fix it.")
    flood = flood.to_crs(target_crs)
    flood["geometry"] = shapely.make_valid(flood.geometry.values)

    # Drop tiny specks: SAR speckle noise creates thousands of 1-2 pixel
    # 'floods' that would otherwise inflate the damage counts.
    flood = flood[flood.area >= cfg.min_flood_patch_m2]
    if flood.empty:
        raise ValueError("No flood polygons left after speckle filtering.")

    union = unary_union(flood.geometry.values)  # TRICK 2 explains why one geom
    log.info("Flood extent: %d patches, %.1f km2",
             len(flood), union.area / 1e6)
    return union


def _polygonise_raster(path: Path) -> gpd.GeoDataFrame:
    """
    Binary raster -> polygons using rasterio.features.shapes.

    `mask=` restricts polygonising to flooded pixels only, otherwise we would
    also get a giant polygon for every 'dry' region.
    """
    import rasterio
    from rasterio import features

    with rasterio.open(path) as src:
        band = src.read(1)
        flooded = (band > 0) & (band != src.nodata if src.nodata is not None else True)
        polys = [shape(geom) for geom, _ in
                 features.shapes(flooded.astype("uint8"),
                                 mask=flooded, transform=src.transform)]
        return gpd.GeoDataFrame(geometry=polys, crs=src.crs)


def subtract_permanent_water(flood, water_path: Path | None, target_crs):
    """
    OPTIONAL: remove permanent water (the normal river channel) so we only
    report NEW inundation. Pass a polygon file (e.g. JRC Global Surface Water
    or OSM natural=water). Returns the flood geometry unchanged if None.
    """
    if water_path is None:
        return flood
    water = gpd.read_file(water_path).to_crs(target_crs)
    return flood.difference(unary_union(shapely.make_valid(water.geometry.values)))


# =====================================================================
# 2. FAST SPATIAL PRIMITIVES
# =====================================================================
def make_fast_flood_index(flood: shapely.Geometry, cell_m: float = 1000.0):
    """
    Cut the (huge, complex) flood polygon into a grid of small pieces and
    return them as a GeoSeries with a spatial index.   >>> TRICK 1 <<<

    WHY: intersecting a building with ONE polygon that has 50 000 vertices
    costs O(vertices) every time. If we first chop that polygon into ~1 km
    tiles, each building only meets the few tiles under it, each tile having
    only a few hundred vertices. On a big flood this is 10-100x faster and
    gives exactly the same result (the union of the tiles IS the flood).
    """
    minx, miny, maxx, maxy = flood.bounds
    xs = np.arange(minx, maxx + cell_m, cell_m)
    ys = np.arange(miny, maxy + cell_m, cell_m)
    shapely.prepare(flood)
    tiles = []
    for x0 in xs[:-1]:
        for y0 in ys[:-1]:
            cell = box(x0, y0, x0 + cell_m, y0 + cell_m)
            if not shapely.intersects(flood, cell):
                continue                       # empty tile - skip entirely
            piece = shapely.intersection(flood, cell)
            if not piece.is_empty:
                tiles.append(piece)
    return gpd.GeoSeries(tiles)   # .sindex is built lazily on first query


def overlap_measure(geoms: gpd.GeoSeries, tiles: gpd.GeoSeries, kind: str) -> np.ndarray:
    """
    For each geometry, return how much of it lies inside the flood:
      kind="area"   -> m2  (for polygons)
      kind="length" -> m   (for lines)

    HOW: ask the tile spatial index which (geometry, tile) pairs touch, clip
    only those pairs, then SUM the clipped pieces per geometry with np.add.at.
    Summing across tiles is safe because the tiles do not overlap.
    """
    result = np.zeros(len(geoms))
    g_idx, t_idx = tiles.sindex.query(geoms.values, predicate="intersects")
    if len(g_idx) == 0:
        return result
    clipped = shapely.intersection(geoms.values[g_idx], tiles.values[t_idx])
    measure = shapely.area(clipped) if kind == "area" else shapely.length(clipped)
    np.add.at(result, g_idx, measure)   # TRICK 3: handles repeated indices
    return result


# =====================================================================
# 3. BUILDINGS
# =====================================================================
def classify_buildings(buildings: gpd.GeoDataFrame, tiles: gpd.GeoSeries,
                       flood: shapely.Geometry, cfg: Config) -> gpd.GeoDataFrame:
    """
    Add damage columns to each building footprint (input must be metric CRS):

      footprint_m2   - building area
      flooded_m2     - area inside the flood
      flooded_frac   - flooded_m2 / footprint_m2   (0..1)
      damage_class   - 'none' | 'edge' | 'partial' | 'major'

    'edge' = dry on paper but within `edge_buffer_m` of flood. SAR pixels are
    ~10 m and flood edges are fuzzy, so these are "probably wet" - shown as
    their own class instead of being silently dropped.   >>> TRICK 4 <<<
    """
    b = buildings.copy()
    b["footprint_m2"] = b.area
    b["flooded_m2"] = overlap_measure(b.geometry, tiles, "area")
    b["flooded_frac"] = (b["flooded_m2"] / b["footprint_m2"].replace(0, np.nan)).fillna(0).clip(0, 1)

    # Proximity test against the buffered flood (cheap: prepared geometry).
    near = shapely.buffer(flood, cfg.edge_buffer_m)
    shapely.prepare(near)
    is_near = shapely.intersects(near, b.geometry.values)

    cls = np.full(len(b), "none", dtype=object)
    cls[is_near] = "edge"
    cls[b["flooded_frac"] >= 1e-6] = "edge"           # touched but tiny
    cls[b["flooded_frac"] >= cfg.partial_frac] = "partial"
    cls[b["flooded_frac"] >= cfg.major_frac] = "major"
    b["damage_class"] = cls
    return b


# =====================================================================
# 4. ROADS
# =====================================================================
def _is_elevated(roads: gpd.GeoDataFrame) -> np.ndarray:
    """
    True for roads that pass OVER or UNDER water: bridge=*, tunnel=*,
    or layer > 0. Missing columns are treated as 'no'.
    """
    def tagged(col):
        if col not in roads:
            return np.zeros(len(roads), dtype=bool)
        # fillna("") FIRST: on newer pandas, astype(str) keeps NaN as a real
        # missing value (not the text "nan"), which would make every untagged
        # road look like a bridge.
        s = roads[col].fillna("").astype(str).str.strip().str.lower()
        return (~s.isin(["", "no", "nan", "none", "0", "false"])).to_numpy()

    elevated = tagged("bridge") | tagged("tunnel")
    if "layer" in roads:
        layer = pd.to_numeric(roads["layer"], errors="coerce").fillna(0)
        elevated |= (layer > 0).to_numpy()
    return elevated


def classify_roads(roads: gpd.GeoDataFrame, tiles: gpd.GeoSeries,
                   cfg: Config) -> gpd.GeoDataFrame:
    """
    Add columns to each road segment (metric CRS):

      length_m, flooded_len_m, flooded_frac
      status: 'open' | 'blocked' | 'bridge_over_flood'

    TRICK 5 - ROADS ARE NOT BUILDINGS: a 300 m road with a single 25 m flooded
    crossing is 8% wet but 100% IMPASSABLE. So status uses absolute wet length
    (>= min_block_len_m), NOT the fraction. The fraction is kept only for stats.

    TRICK 5b - BRIDGES: flood water flows UNDER a bridge. Flagging every bridge
    over the Trishuli as 'blocked' would falsely cut half the valley. Bridges
    in flood are labelled 'bridge_over_flood' (still passable, but worth an
    inspection - approaches may be washed out, which the neighbouring
    non-bridge segments will capture).
    """
    r = roads.copy()
    r["length_m"] = r.length
    r["flooded_len_m"] = overlap_measure(r.geometry, tiles, "length")
    r["flooded_frac"] = (r["flooded_len_m"] / r["length_m"].replace(0, np.nan)).fillna(0).clip(0, 1)

    wet = r["flooded_len_m"] >= cfg.min_block_len_m
    # Very short segments (< min_block_len) fully under water would never hit
    # the threshold - treat "mostly wet" short segments as blocked too.
    short_but_wet = (r["length_m"] < cfg.min_block_len_m) & (r["flooded_frac"] >= 0.5)
    wet |= short_but_wet

    elevated = _is_elevated(r)
    status = np.full(len(r), "open", dtype=object)
    status[wet & ~elevated] = "blocked"
    status[wet & elevated] = "bridge_over_flood"
    r["status"] = status
    return r


def blocked_edge_table(roads: gpd.GeoDataFrame, cfg: Config) -> pd.DataFrame:
    """
    The hand-off to PART 04: just the identifiers of blocked edges.
    Part 04 does:  G.remove_edges_from(zip(df.u, df.v, df.key))
    Falls back to the dataframe index if u/v/key are missing.
    """
    blocked = roads[roads["status"] == "blocked"]
    cols = [c for c in cfg.road_id_cols if c in blocked.columns]
    out = blocked[cols].copy() if cols else pd.DataFrame(index=blocked.index)
    out["flooded_len_m"] = blocked["flooded_len_m"].round(1)
    return out.reset_index(names="road_index")


# =====================================================================
# 5. CRITICAL FACILITIES
# =====================================================================
def classify_facilities(pois: gpd.GeoDataFrame, flood: shapely.Geometry,
                        cfg: Config) -> gpd.GeoDataFrame:
    """
    Keep only critical amenities and tag each as 'flooded' / 'at_risk' / 'dry'.
    at_risk = within edge_buffer_m of flood (same uncertainty logic as
    buildings). POIs given as polygons are converted to centroids first -
    OSM maps hospitals as both nodes and building outlines.
    """
    if "amenity" not in pois:
        log.warning("POIs have no 'amenity' column - facilities skipped.")
        return pois.iloc[0:0].copy()

    p = pois[pois["amenity"].isin(cfg.critical_amenities)].copy()
    if p.empty:
        return p
    p["geometry"] = p.geometry.representative_point()   # TRICK 7

    shapely.prepare(flood)
    inside = shapely.contains(flood, p.geometry.values)
    near = shapely.dwithin(p.geometry.values, flood, cfg.edge_buffer_m)
    p["flood_status"] = np.where(inside, "flooded", np.where(near, "at_risk", "dry"))
    return p


# =====================================================================
# 6. AGGREGATION  (heat-map layer + headline numbers)
# =====================================================================
def aggregate_to_grid(buildings: gpd.GeoDataFrame, roads: gpd.GeoDataFrame,
                      cfg: Config) -> gpd.GeoDataFrame:
    """
    Square grid (default 500 m) with per-cell counts:
      n_buildings, n_hit (partial+major), pct_hit, blocked_km

    Buildings are assigned to a cell by their CENTROID so each is counted
    exactly once (a footprint straddling two cells must not be double-counted).
    Cells with no buildings and no blocked road are dropped.
    """
    allgeom = pd.concat([buildings.geometry, roads.geometry])
    minx, miny, maxx, maxy = allgeom.total_bounds
    s = cfg.grid_size_m

    def cell_ids(x, y):
        return np.floor((x - minx) / s).astype(int), np.floor((y - miny) / s).astype(int)

    # buildings
    c = buildings.geometry.centroid
    bi, bj = cell_ids(c.x.values, c.y.values)
    bdf = pd.DataFrame({"i": bi, "j": bj,
                        "hit": buildings["damage_class"].isin(["partial", "major"]).values})
    bagg = (bdf.groupby(["i", "j"]).agg(n_buildings=("hit", "size"), n_hit=("hit", "sum"))
            .reset_index())

    # blocked road length, attributed to the segment midpoint's cell
    blk = roads[roads["status"] == "blocked"]
    if len(blk):
        mid = shapely.line_interpolate_point(blk.geometry.values, 0.5, normalized=True)
        ri, rj = cell_ids(shapely.get_x(mid), shapely.get_y(mid))
        rdf = pd.DataFrame({"i": ri, "j": rj, "m": blk["length_m"].values})
        ragg = rdf.groupby(["i", "j"], as_index=False).agg(blocked_m=("m", "sum"))
        ragg["blocked_km"] = ragg.pop("blocked_m") / 1000
    else:  # no blocked roads: keep the column so the schema never changes
        ragg = pd.DataFrame({"i": pd.Series(dtype=int), "j": pd.Series(dtype=int),
                             "blocked_km": pd.Series(dtype=float)})

    # Plain merge on the cell columns (more robust than index joins when one
    # side is empty).
    grid = bagg.merge(ragg, on=["i", "j"], how="outer").fillna(0)
    grid["pct_hit"] = np.where(grid["n_buildings"] > 0,
                               100 * grid["n_hit"] / grid["n_buildings"].replace(0, 1), 0).round(1)
    grid["geometry"] = [box(minx + i * s, miny + j * s, minx + (i + 1) * s, miny + (j + 1) * s)
                        for i, j in zip(grid["i"], grid["j"])]
    return gpd.GeoDataFrame(grid, geometry="geometry", crs=buildings.crs)


def build_summary(b, r, fac, flood) -> dict:
    """Headline numbers for the pitch / dashboard (plain JSON-serialisable)."""
    summary = {
        "flood_area_km2": round(flood.area / 1e6, 2),
        "buildings_total": int(len(b)),
        "buildings_by_class": b["damage_class"].value_counts().to_dict(),
        "roads_total_km": round(r["length_m"].sum() / 1000, 1),
        "roads_blocked_km": round(r.loc[r["status"] == "blocked", "length_m"].sum() / 1000, 2),
        "roads_by_status": r["status"].value_counts().to_dict(),
    }
    if fac is not None and len(fac):
        summary["facilities_by_status"] = (
            fac.groupby(["amenity", "flood_status"]).size().unstack(fill_value=0).to_dict("index"))
    return {k: (int_safe(v)) for k, v in summary.items()}


def int_safe(v):
    """Convert numpy ints (not JSON serialisable) recursively."""
    if isinstance(v, dict):
        return {str(k): int_safe(x) for k, x in v.items()}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# =====================================================================
# 7. EXPORT
# =====================================================================
def export_all(out_dir: Path, layers: dict[str, gpd.GeoDataFrame],
               blocked: pd.DataFrame, summary: dict):
    """
    * damage.gpkg      - all layers, metric CRS (Part 04 / QGIS)
    * <layer>.geojson  - EPSG:4326 copies (Leaflet / Mapbox want lon-lat)
    * blocked_edges.csv, summary.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / "damage.gpkg"
    if gpkg.exists():
        gpkg.unlink()           # avoid stale layers from a previous run
    for name, gdf in layers.items():
        if gdf is None or gdf.empty:
            continue
        gdf.to_file(gpkg, layer=name, driver="GPKG")
        # 'none'-class buildings are dropped from the web file: 100k+ dry
        # polygons make the browser map crawl and add no information.
        web = gdf
        if name == "buildings":
            web = gdf[gdf["damage_class"] != "none"]
        web.to_crs(4326).to_file(out_dir / f"{name}.geojson", driver="GeoJSON")
    blocked.to_csv(out_dir / "blocked_edges.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote outputs to %s", out_dir)


# =====================================================================
# 8. ORCHESTRATION
# =====================================================================
def run(osm_gpkg: Path, flood_path: Path, out_dir: Path,
        water_path: Path | None = None, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()

    buildings = safe_layer(osm_gpkg, cfg.layer_buildings)
    roads = safe_layer(osm_gpkg, cfg.layer_roads)
    pois = safe_layer(osm_gpkg, cfg.layer_pois)
    if buildings is None or roads is None:
        raise SystemExit("Need at least 'buildings' and 'roads' layers from Part 01.")

    crs = metric_crs_for(buildings)
    buildings, roads = buildings.to_crs(crs), roads.to_crs(crs)
    pois = pois.to_crs(crs) if pois is not None else None

    flood = load_flood_extent(flood_path, crs, cfg)
    flood = subtract_permanent_water(flood, water_path, crs)
    tiles = make_fast_flood_index(flood)

    b = classify_buildings(buildings, tiles, flood, cfg)
    r = classify_roads(roads, tiles, cfg)
    fac = classify_facilities(pois, flood, cfg) if pois is not None else None
    grid = aggregate_to_grid(b, r, cfg)
    blocked = blocked_edge_table(r, cfg)
    summary = build_summary(b, r, fac, flood)

    flood_gdf = gpd.GeoDataFrame(geometry=[flood], crs=crs)
    export_all(out_dir, {"buildings": b, "roads": r, "facilities": fac,
                         "grid": grid, "flood": flood_gdf}, blocked, summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osm", type=Path, required=True, help="Part 01 GeoPackage")
    ap.add_argument("--flood", type=Path, required=True, help="flood GeoTIFF or vector")
    ap.add_argument("--water", type=Path, default=None, help="permanent water polygons (optional)")
    ap.add_argument("--out", type=Path, default=Path("outputs/part03"))
    args = ap.parse_args()
    print(json.dumps(run(args.osm, args.flood, args.out, args.water), indent=2))


# =====================================================================
# APPENDIX - THE NON-OBVIOUS TRICKS, IN ONE PLACE
# =====================================================================
# TRICK 1  Tile the flood polygon (make_fast_flood_index)
#          One giant polygon makes every intersection slow. Chopping it into
#          1 km tiles + a spatial index gives the same answer much faster.
# TRICK 2  Dissolve first, then tile
#          unary_union merges overlapping patches, so no building/road is
#          counted twice. (Tiles then split that clean geometry.)
# TRICK 3  np.add.at instead of fancy-index +=
#          A building crossing 3 tiles appears 3 times in g_idx. `r[g_idx] += x`
#          would silently keep only ONE of them; np.add.at accumulates all.
# TRICK 4  Uncertainty band ('edge' class)
#          Satellite flood edges are +/- one pixel. A 10 m band flags
#          'probably wet' assets instead of pretending the edge is exact.
# TRICK 5  Road blocking uses absolute wet length, not percentage
#          A 25 m flooded crossing blocks a whole road even if it is 5% of it.
#          Bridges are separated: water flows under them.
# TRICK 6  shapely.make_valid on every input
#          OSM and polygonised rasters contain self-intersections that make
#          intersection() raise TopologyException mid-run.
# TRICK 7  representative_point(), not centroid
#          A centroid of an L-shaped hospital outline can fall outside it
#          (and outside the flood test). representative_point is guaranteed
#          to lie inside the polygon.
# TRICK 8  Grid assignment by centroid / segment midpoint
#          Each asset lands in exactly one cell -> counts add up to the totals.
# =====================================================================
