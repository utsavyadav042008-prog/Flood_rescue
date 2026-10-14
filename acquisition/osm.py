"""Stage 'osm' - pre-event OpenStreetMap layers in the data-contract format.

Glue between three teammates' code:

* T1 ``acquisition/ohsome.py``    downloads the map AS IT WAS before the event
                                  (ohsome ``time=<snapshot>``, cache, edit-date audit)
* T3 ``acquisition/overpass.py``  fallback downloader (Overpass *attic* query, also
                                  pre-event) and the tag normalisers (names, bridge
                                  flags, road classes)
* T3 ``gis/road_graph.py``        splits road ways into graph segments with stable
                                  ids ``<way>#<k>`` (shared by damage + connectivity)

Writes (into the run folder, all EPSG:4326):
    osm/snapshot.json       which snapshot/provider was used (data-rule evidence)
    osm/roads.geojson       road SEGMENTS: id, way_id, highway_type, bridge, name, ...
    osm/bridges.geojson     bridge WAYS: id, name, highway_type
    osm/buildings.geojson   footprints: id, type, name
    osm/settlements.geojson villages/hamlets inside the AOI: id, name, name_ne, place
    osm/hospitals.geojson   destinations: hospitals and towns, destination_type

Roads and destinations are fetched for the AOI grown by
``osm.network_buffer_km``: a village's route to the hospital often leaves the
box, and clipping the network at the box edge would invent cut-offs.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from pipeline.config import REPO_ROOT
from pipeline.contracts import Artifact
from pipeline.errors import DataRuleViolation, GISError
from pipeline.fileio import write_geojson, write_geojson_from_gdf, write_json
from pipeline.rules import assert_osm_snapshot_pre_event, effective_osm_snapshot
from pipeline.run import RunPaths
from pipeline.schemas import AOI, Provenance, parse_date

log = logging.getLogger("floodresq.osm")

#: ohsome filters (ohsome filter syntax).  Settlements exclude towns/cities:
#: those are destinations, and a town is trivially connected to itself.
LAYER_FILTERS = {
    "roads": "highway=* and type:way and not highway in (proposed, razed, abandoned, "
             "construction, platform, bus_stop, elevator, corridor)",
    "buildings": "building=* and geometry:polygon",
    "settlements": "place in (village, hamlet, isolated_dwelling) and type:node",
    "hospitals": "amenity=hospital or healthcare=hospital",
    "towns": "place in (city, town) and type:node",
}
NETWORK_LAYERS = ("roads", "hospitals", "towns")       # fetched on the buffered box
AOI_LAYERS = ("buildings", "settlements")              # fetched on the AOI only

OSM_ATTRIBUTION = "© OpenStreetMap contributors (ODbL)"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def snapshot_for(event: date, cfg: dict) -> date:
    """Latest allowed snapshot: the configured one, but always before the event."""
    snap = effective_osm_snapshot(cfg["osm"]["snapshot_date"], event)
    assert_osm_snapshot_pre_event(snap, event)
    return snap


def buffered_bbox(aoi: AOI, km: float) -> tuple:
    dlat = km / 110.574
    dlon = km / (111.320 * max(math.cos(math.radians(aoi.center[1])), 0.05))
    return (max(-180.0, aoi.west - dlon), max(-90.0, aoi.south - dlat),
            min(180.0, aoi.east + dlon), min(90.0, aoi.north + dlat))


def _raw_from_features(features: list, kind: str):
    """ohsome GeoJSON features -> T3's raw GeoDataFrame (uid, osm_type, osm_id, tags, geometry)."""
    import geopandas as gpd
    from shapely.geometry import shape

    rows = []
    for f in features:
        props = f.get("properties") or {}
        oid = str(props.get("@osmId") or f.get("id") or "")
        if not oid or f.get("geometry") is None:
            continue
        try:
            geom = shape(f["geometry"])
        except Exception:                                   # malformed geometry from the server
            continue
        if geom.is_empty:
            continue
        if kind == "point" and geom.geom_type != "Point":
            geom = geom.representative_point()
        osm_type, _, osm_id = oid.partition("/")
        tags = {k: v for k, v in props.items() if not str(k).startswith("@")}
        rows.append({"uid": oid, "osm_type": osm_type, "osm_id": osm_id, "tags": tags, "geometry": geom})
    if not rows:
        return gpd.GeoDataFrame({"uid": [], "osm_type": [], "osm_id": [], "tags": []},
                                geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _fetch_ohsome(bbox: tuple, event: date, snapshot: date, layers: dict, required: tuple, cache: Path):
    from acquisition.ohsome import OsmConfig, fetch_osm_snapshot
    from acquisition.sentinel import AcquisitionError

    ocfg = OsmConfig(layers=dict(layers), required_layers=required, cache_root=cache,
                     latest_allowed_snapshot=snapshot)
    try:
        snap = fetch_osm_snapshot(bbox, event, snapshot_date=snapshot, config=ocfg)
    except AcquisitionError as exc:
        if "not before" in str(exc) or "later than the snapshot" in str(exc):
            raise DataRuleViolation("OpenStreetMap data newer than the event was refused.", str(exc)) from exc
        raise
    from acquisition.ohsome import read_layer
    out = {}
    for name, lr in snap.layers.items():
        content = read_layer(Path(lr.path), None, None, None) or {"features": []}
        out[name] = content["features"]
    return out, list(snap.warnings), dict(snap.missing)


_OVERPASS_BODIES = {
    "roads": ('  way["highway"]["highway"!~"^(proposed|razed|abandoned|construction|platform|'
              'bus_stop|elevator|corridor)$"];', "line"),
    "buildings": ('  way["building"];\n  relation["building"];', "polygon"),
    "settlements": ('  node["place"~"^(village|hamlet|isolated_dwelling)$"];', "point"),
    "hospitals": ('  nwr["amenity"="hospital"];\n  nwr["healthcare"="hospital"];', "point"),
    "towns": ('  node["place"~"^(city|town)$"];', "point"),
}


def _fetch_overpass(bbox: tuple, snapshot: date, names: tuple, cache: Path) -> dict:
    """Fallback: Overpass attic query (map as it was on ``snapshot``) -> raw GeoDataFrames."""
    from acquisition.overpass import Config, OverpassClient, elements_to_raw_gdf

    client = OverpassClient(Config(bbox=bbox, out_dir=cache, cache_dir=cache / "overpass",
                                   date=snapshot.isoformat()))
    out = {}
    for name in names:
        body, kind = _OVERPASS_BODIES[name]
        raw = elements_to_raw_gdf(client.fetch(body), kind)
        if len(raw):
            raw["uid"] = raw["osm_type"].astype(str) + "/" + raw["osm_id"].astype(str)
        out[name] = raw
    return out


def fetch_raw_layers(aoi: AOI, event: date, snapshot: date, cfg: dict) -> tuple:
    """{layer: raw GeoDataFrame}, provider name, warnings.  ohsome first, then Overpass attic."""
    cache = REPO_ROOT / "data" / "raw" / "osm"
    big = buffered_bbox(aoi, float(cfg["osm"].get("network_buffer_km", 5.0)))
    kinds = {"roads": "line", "buildings": "polygon", "settlements": "point",
             "hospitals": "point", "towns": "point"}
    warnings: list = []
    provider = str(cfg["osm"].get("provider", "ohsome"))
    if provider == "ohsome":
        try:
            net, w1, miss1 = _fetch_ohsome(big, event, snapshot, {k: LAYER_FILTERS[k] for k in NETWORK_LAYERS},
                                           ("roads",), cache)
            loc, w2, miss2 = _fetch_ohsome(aoi.bbox, event, snapshot, {k: LAYER_FILTERS[k] for k in AOI_LAYERS},
                                           (), cache)
            warnings += w1 + w2 + [f"OSM layer '{k}' missing: {v}" for k, v in {**miss1, **miss2}.items()]
            feats = {**net, **loc}
            return ({n: _raw_from_features(feats.get(n, []), kinds[n]) for n in kinds},
                    "ohsome API (historical snapshot)", warnings)
        except DataRuleViolation:
            raise
        except Exception as exc:                            # network, server, parsing ...
            if not cfg["osm"].get("overpass_fallback", True):
                raise GISError("OpenStreetMap download failed (ohsome).", str(exc)) from exc
            log.warning("ohsome failed (%s); falling back to Overpass attic query", exc)
            warnings.append(f"ohsome API failed ({exc}); used the Overpass history (attic) database "
                            f"for the same pre-event snapshot date instead.")
    try:
        raw = _fetch_overpass(big, snapshot, NETWORK_LAYERS, cache)
        raw.update(_fetch_overpass(aoi.bbox, snapshot, AOI_LAYERS, cache))
    except Exception as exc:
        raise GISError("OpenStreetMap download failed (ohsome and Overpass).", str(exc)) from exc
    return raw, "Overpass API attic query (historical snapshot)", warnings


# --------------------------------------------------------------------------
# Normalisation (T3's normalisers) -> contract columns
# --------------------------------------------------------------------------
def _clip_to_box(gdf, bbox: tuple):
    from shapely.geometry import box
    if gdf.empty:
        return gdf
    return gdf[gdf.intersects(box(*bbox))].reset_index(drop=True)


def _name_ne(tags) -> Optional[str]:
    v = (tags or {}).get("name:ne")
    return str(v).strip() if v and str(v).strip() else None


def build_layers(raw: dict, aoi: AOI) -> dict:
    """Raw layers -> contract GeoDataFrames (EPSG:4326)."""
    import geopandas as gpd
    import pandas as pd
    from acquisition.overpass import best_name, normalise_buildings, normalise_places, normalise_roads
    from gis.road_graph import split_roads_to_segments

    utm = aoi.utm_epsg()
    out = {}

    # roads -> ways (normalised by T3) -> segments with ids "<way>#<k>"
    roads_raw = raw["roads"]
    if len(roads_raw):      # closed "area" highways come back as polygons: not part of the network
        roads_raw = roads_raw[roads_raw.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        roads_raw = roads_raw.reset_index(drop=True)
    if len(roads_raw):
        ways = normalise_roads(roads_raw, utm)
        ways = ways.rename(columns={"uid": "id", "highway": "highway_type", "is_bridge": "bridge",
                                    "is_tunnel": "tunnel", "is_ford": "ford"})
        keep = ["id", "highway_type", "bridge", "tunnel", "ford", "name", "road_class", "rank",
                "vehicle_ok", "surface", "geometry"]
        ways = ways[keep]
        segs = split_roads_to_segments(ways.rename(columns={"highway_type": "highway"}), id_col="id")
        segs = segs.rename(columns={"highway": "highway_type"})
        cols = ["id", "way_id", "highway_type", "bridge", "tunnel", "ford", "name", "road_class",
                "rank", "vehicle_ok", "surface", "length_m", "geometry"]
        segs = segs[[c for c in cols if c in segs.columns]]
        for c in ("bridge", "tunnel", "ford", "vehicle_ok"):
            segs[c] = segs[c].fillna(False).astype(bool)
        out["roads"] = segs
        bridges = ways[ways["bridge"].astype(bool)][["id", "name", "highway_type", "geometry"]]
        out["bridges"] = bridges.reset_index(drop=True)
    else:
        empty = gpd.GeoDataFrame({"id": [], "name": []}, geometry=[], crs="EPSG:4326")
        out["roads"], out["bridges"] = empty, empty.copy()

    # buildings (polygons only; make_valid can return collections)
    b_raw = raw["buildings"]
    if len(b_raw):
        b = normalise_buildings(b_raw, utm).rename(columns={"uid": "id", "building": "type"})
        b = b.explode(index_parts=False)
        b = b[b.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        b = b.dissolve(by="id", as_index=False, aggfunc="first")
        out["buildings"] = _clip_to_box(b[["id", "type", "name", "levels", "area_m2", "geometry"]], aoi.bbox)
    else:
        out["buildings"] = gpd.GeoDataFrame({"id": [], "type": []}, geometry=[], crs="EPSG:4326")

    # settlements (inside the AOI)
    s_raw = raw["settlements"]
    if len(s_raw):
        s = normalise_places(s_raw, utm).rename(columns={"uid": "id"})
        s["name_ne"] = [_name_ne(t) for t in s_raw["tags"]]
        s["name"] = [n.strip() if isinstance(n, str) and n.strip() else f"Unnamed {p if isinstance(p, str) else 'place'} ({i})"
                     for n, p, i in zip(s["name"], s["place"], s["id"])]
        out["settlements"] = _clip_to_box(s[["id", "name", "name_ne", "place", "population", "geometry"]], aoi.bbox)
    else:
        out["settlements"] = gpd.GeoDataFrame({"id": [], "name": []}, geometry=[], crs="EPSG:4326")

    # destinations: hospitals + towns
    parts = []
    for layer, dtype in (("hospitals", "hospital"), ("towns", "town")):
        r = raw[layer]
        if not len(r):
            continue
        d = gpd.GeoDataFrame({"id": r["uid"].astype(str),
                              "name": [best_name(t) or f"Unnamed {dtype} ({u})" for t, u in zip(r["tags"], r["uid"])],
                              "name_ne": [_name_ne(t) for t in r["tags"]],
                              "destination_type": dtype},
                             geometry=r.geometry.values, crs="EPSG:4326")
        parts.append(d)
    if parts:
        dest = pd.concat(parts, ignore_index=True).drop_duplicates("id")
        out["hospitals"] = gpd.GeoDataFrame(dest, geometry="geometry", crs="EPSG:4326")
    else:
        out["hospitals"] = gpd.GeoDataFrame({"id": [], "name": [], "destination_type": []},
                                            geometry=[], crs="EPSG:4326")
    return out


# --------------------------------------------------------------------------
# Stage entry point
# --------------------------------------------------------------------------
_TARGETS = {"roads": Artifact.OSM_ROADS, "bridges": Artifact.OSM_BRIDGES,
            "buildings": Artifact.OSM_BUILDINGS, "settlements": Artifact.OSM_SETTLEMENTS,
            "hospitals": Artifact.OSM_HOSPITALS}


def run_osm(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None:
    snapshot = snapshot_for(event_date, cfg)
    raw, provider, warnings = fetch_raw_layers(aoi, event_date, snapshot, cfg)
    layers = build_layers(raw, aoi)

    prov = Provenance(producer="acquisition.osm", source=f"OpenStreetMap {snapshot} via {provider}",
                      crs="EPSG:4326", event_date=event_date.isoformat())
    run.ensure_parent(Artifact.OSM_ROADS)
    for name, artifact in _TARGETS.items():
        gdf = layers[name]
        if len(gdf):
            write_geojson_from_gdf(run.path(artifact), gdf, prov)
        else:
            write_geojson(run.path(artifact), [], prov)

    counts = {name: int(len(layers[name])) for name in _TARGETS}
    if counts["hospitals"] == 0:
        warnings.append("No hospital or town is mapped within "
                        f"{cfg['osm'].get('network_buffer_km', 5)} km of the area: cut-off analysis "
                        "falls back to 'road exits' at the edge of the network.")
    if counts["settlements"] == 0:
        warnings.append("No village or hamlet is mapped in this area in the pre-event OSM snapshot.")
    write_json(run.path(Artifact.OSM_SNAPSHOT), {
        "snapshot_date": snapshot.isoformat(), "provider": provider,
        "event_date": event_date.isoformat(),
        "network_buffer_km": float(cfg["osm"].get("network_buffer_km", 5.0)),
        "counts": counts, "warnings": warnings, "attribution": OSM_ATTRIBUTION}, prov)
