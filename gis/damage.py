"""Stage 'damage' - which roads, bridges and buildings overlap the flood mask.

Glue around T3's ``gis/damage_overlay.py`` (flood polygonising, 1 km tiling
index, building classes, road blocking by absolute wet length).

Reads:  flood_mask.tif, flood_confidence.tif (optional), osm/roads.geojson
        (segments), osm/bridges.geojson, osm/buildings.geojson
Writes: affected_roads.geojson, affected_bridges.geojson,
        affected_buildings.geojson, damage_summary.json

Wording (contract): ``impact_status`` is "impacted" (road segment with a wet
stretch long enough to block it, or building >= 50 % wet), "affected"
(building 10-50 % wet) or "potentially_damaged" (bridge over the flood: the
deck may survive, the approaches may not).  Never "destroyed": overlap with a
satellite mask is not proof of structural damage.
"""

from __future__ import annotations

import logging
from datetime import date

from pipeline.contracts import Artifact
from pipeline.errors import GISError
from pipeline.fileio import read_json, read_provenance, write_geojson, write_geojson_from_gdf, write_json
from pipeline.run import RunPaths
from pipeline.schemas import AOI, Provenance

log = logging.getLogger("floodresq.damage")


def _overlay_config(cfg: dict):
    from gis.damage_overlay import Config
    d = cfg.get("damage", {})
    return Config(min_flood_patch_m2=float(d.get("min_flood_patch_m2", 1000.0)),
                  edge_buffer_m=float(d.get("edge_buffer_m", 10.0)),
                  partial_frac=float(d.get("min_impact_fraction", 0.10)),
                  major_frac=float(d.get("major_impact_fraction", 0.50)),
                  min_block_len_m=float(d.get("road_min_block_m", 20.0)))


def _read_layer(path, crs):
    import geopandas as gpd
    import shapely
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    if gdf.empty:
        return gdf.to_crs(crs)
    gdf["geometry"] = shapely.make_valid(gdf.geometry.values)
    return gdf[~gdf.geometry.is_empty].to_crs(crs).reset_index(drop=True)


class _ConfidenceSampler:
    """Mean model probability over the WET part of an object (points every pixel)."""

    def __init__(self, path):
        import numpy as np
        import rasterio
        self.ok = path.exists()
        if self.ok:
            with rasterio.open(path) as src:
                self.arr = src.read(1).astype("float32")
                self.tf = src.transform
                self.res = abs(src.transform.a)
            self.arr[~np.isfinite(self.arr)] = 0.0

    def __call__(self, geom, flood) -> float:
        import numpy as np
        import shapely
        if not self.ok:
            return 0.5
        part = shapely.intersection(geom, flood)
        if part.is_empty:
            part = geom
        if part.geom_type in ("LineString", "MultiLineString"):
            n = max(2, int(part.length // self.res) + 1)
            pts = shapely.line_interpolate_point(part, np.linspace(0, 1, n), normalized=True) \
                if part.geom_type == "LineString" else \
                [p for g in part.geoms for p in shapely.line_interpolate_point(
                    g, np.linspace(0, 1, max(2, int(g.length // self.res) + 1)), normalized=True)]
        else:
            pts = [part.representative_point(), part.centroid]
        vals = []
        h, w = self.arr.shape
        for p in pts:
            col, row = (p.x - self.tf.c) / self.tf.a, (p.y - self.tf.f) / self.tf.e    # north-up grid
            r, c = int(row), int(col)
            if 0 <= r < h and 0 <= c < w:
                vals.append(float(self.arr[r, c]))
        return round(float(np.clip(np.mean(vals), 0, 1)), 3) if vals else 0.5


def _features(gdf, object_type: str, status_col: str, frac_col: str, sampler, flood, extra=()):
    """Rows -> contract properties (+ a few helpful extras for popups/tables)."""
    import geopandas as gpd
    rows = []
    for _, r in gdf.iterrows():
        props = {"object_id": str(r["id"]), "object_type": object_type,
                 "impact_status": r[status_col],
                 "impact_fraction": round(float(min(max(r[frac_col], 0.0), 1.0)), 3),
                 "confidence": sampler(r.geometry, flood)}
        for col in extra:
            if col in gdf.columns:
                v = r[col]
                props[col] = None if v is None or (isinstance(v, float) and v != v) else v
        rows.append({**props, "geometry": r.geometry})
    if not rows:
        return None
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


def run_damage(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None:
    import numpy as np
    from gis import damage_overlay as dov

    ocfg = _overlay_config(cfg)
    mask_path = run.path(Artifact.FLOOD_MASK)
    try:
        import rasterio
        with rasterio.open(mask_path) as src:
            epsg = src.crs.to_epsg() if src.crs else None
            crs = f"EPSG:{epsg}" if epsg else src.crs.to_wkt()
    except ImportError as exc:
        raise GISError("rasterio is needed to read the flood mask.", str(exc)) from exc

    roads = _read_layer(run.path(Artifact.OSM_ROADS), crs)
    bridges = _read_layer(run.path(Artifact.OSM_BRIDGES), crs)
    buildings = _read_layer(run.path(Artifact.OSM_BUILDINGS), crs)

    # ---- flood polygons (T3: polygonise -> drop speckle -> dissolve) --------------------------
    flood = None
    try:
        flood = dov.load_flood_extent(mask_path, crs, ocfg)
    except ValueError as exc:                     # no flood polygons left: computed, nothing found
        log.info("damage: %s", exc)

    mask_prov = read_provenance(mask_path) or {}
    prov = Provenance(producer="gis.damage", source="flood_mask.tif x pre-event OSM",
                      crs="EPSG:4326", event_date=event_date.isoformat(),
                      model_version=str(mask_prov.get("model_version", "")),
                      is_mock=bool(mask_prov.get("is_mock")))
    sampler = _ConfidenceSampler(run.path(Artifact.FLOOD_CONFIDENCE))

    aff = {"roads": None, "bridges": None, "buildings": None}
    stats = {"flood_polygon_area_km2": 0.0, "roads_total_km": 0.0, "affected_road_length_km": 0.0,
             "affected_segment_length_km": 0.0, "buildings_total": int(len(buildings)),
             "buildings_near_flood": 0, "buildings_by_class": {}, "bridges_total": int(len(bridges)),
             "affected_hospitals": []}
    if len(roads):
        stats["roads_total_km"] = round(float(roads.length.sum()) / 1000, 2)

    if flood is not None and not flood.is_empty:
        stats["flood_polygon_area_km2"] = round(flood.area / 1e6, 2)
        tiles = dov.make_fast_flood_index(flood)

        if len(roads):
            r = dov.classify_roads(roads, tiles, ocfg)
            hit = r[r["status"] == "blocked"].copy()
            hit["impact_status"] = "impacted"
            stats["affected_road_length_km"] = round(float(hit["flooded_len_m"].sum()) / 1000, 2)
            stats["affected_segment_length_km"] = round(float(hit["length_m"].sum()) / 1000, 2)
            hit["flooded_length_m"] = hit["flooded_len_m"].round(1)
            aff["roads"] = _features(hit, "road", "impact_status", "flooded_frac", sampler, flood,
                                     ("name", "highway_type", "way_id", "flooded_length_m"))

        if len(bridges):
            b = bridges.copy()
            b["length_m"] = b.length
            b["flooded_len_m"] = dov.overlap_measure(b.geometry, tiles, "length")
            b["flooded_frac"] = (b["flooded_len_m"] / b["length_m"].replace(0, np.nan)).fillna(0).clip(0, 1)
            wet = (b["flooded_len_m"] >= ocfg.min_block_len_m) | (b["flooded_frac"] >= ocfg.partial_frac)
            hit = b[wet].copy()
            hit["impact_status"] = "potentially_damaged"
            hit["flooded_length_m"] = hit["flooded_len_m"].round(1)
            aff["bridges"] = _features(hit, "bridge", "impact_status", "flooded_frac", sampler, flood,
                                       ("name", "highway_type", "flooded_length_m"))

        if len(buildings):
            bl = dov.classify_buildings(buildings, tiles, flood, ocfg)
            stats["buildings_by_class"] = {str(k): int(v) for k, v in bl["damage_class"].value_counts().items()}
            stats["buildings_near_flood"] = int((bl["damage_class"] == "edge").sum())
            hit = bl[bl["damage_class"].isin(["partial", "major"])].copy()
            hit["impact_status"] = np.where(hit["damage_class"] == "major", "impacted", "affected")
            aff["buildings"] = _features(hit, "building", "impact_status", "flooded_frac", sampler, flood,
                                         ("name", "type", "damage_class"))

        try:                                       # destinations standing in the flood (report line)
            dest = _read_layer(run.path(Artifact.OSM_HOSPITALS), crs)
            if len(dest):
                import shapely
                inside = shapely.intersects(flood, dest.geometry.values)
                stats["affected_hospitals"] = [str(n) for n in dest.loc[inside, "name"].tolist()]
        except Exception as exc:                   # optional extra, never fatal
            log.warning("damage: destination check skipped (%s)", exc)

    for key, artifact in (("roads", Artifact.AFFECTED_ROADS), ("bridges", Artifact.AFFECTED_BRIDGES),
                          ("buildings", Artifact.AFFECTED_BUILDINGS)):
        if aff[key] is not None and len(aff[key]):
            write_geojson_from_gdf(run.path(artifact), aff[key], prov)
        else:
            write_geojson(run.path(artifact), [], prov)

    summary = {
        "affected_roads": 0 if aff["roads"] is None else int(len(aff["roads"])),
        "affected_road_length_km": stats["affected_road_length_km"],
        "affected_bridges": 0 if aff["bridges"] is None else int(len(aff["bridges"])),
        "affected_buildings": 0 if aff["buildings"] is None else int(len(aff["buildings"])),
        **{k: v for k, v in stats.items() if k != "affected_road_length_km"},
        "thresholds": {"road_min_block_m": ocfg.min_block_len_m, "building_min_fraction": ocfg.partial_frac,
                       "building_major_fraction": ocfg.major_frac, "min_flood_patch_m2": ocfg.min_flood_patch_m2},
        "notes": ["'affected_road_length_km' is the length of road under the flood mask; "
                  "'affected_segment_length_km' is the full length of the road segments it blocks."],
    }
    write_json(run.path(Artifact.DAMAGE_SUMMARY), summary, prov)


def affected_ids(run: RunPaths) -> tuple:
    """(affected road segment ids, affected bridge way ids) - used by connectivity."""
    def ids(artifact):
        try:
            return [str(f["properties"]["object_id"]) for f in read_json(run.path(artifact))["features"]]
        except (OSError, ValueError, KeyError):
            return []
    return ids(Artifact.AFFECTED_ROADS), ids(Artifact.AFFECTED_BRIDGES)
