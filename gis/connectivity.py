"""Stage 'connectivity' - which settlements lost their road link to a hospital/town.

Glue around T3's ``gis/cutoff.py`` (two-weight graph, multi-source Dijkstra
from all destinations, CUT_OFF vs NO_BASELINE_ACCESS vs DEGRADED, critical vs
bypassed damaged edges, repair priorities, pockets).

Reads:  osm/roads.geojson (segments "<way>#<k>"), osm/settlements.geojson,
        osm/hospitals.geojson, affected_roads.geojson, affected_bridges.geojson
Writes: cutoff_settlements.geojson, connectivity_summary.json

Graph: one edge per pair of consecutive vertices of every road segment, so the
result does not depend on how finely the segments were split; vertices are
merged on ~0.1 m rounding (OSM ways that share a node share the coordinate).
An affected road segment is removed; an affected bridge removes every segment
of that bridge way (``connectivity.block_flooded_bridges``; a debris flood
destroys bridges and their approaches far more often than plain water does).
"""

from __future__ import annotations

import logging
import math
from datetime import date

from pipeline.contracts import Artifact
from pipeline.errors import GISError
from pipeline.fileio import read_json, read_provenance, write_geojson, write_json
from pipeline.run import RunPaths
from pipeline.schemas import AOI, Provenance

log = logging.getLogger("floodresq.connectivity")

FOOT_ONLY = {"path", "footway", "steps", "bridleway", "cycleway", "pedestrian", "via_ferrata"}


def _key(x: float, y: float) -> tuple:
    return (round(x, 6), round(y, 6))


def _metres(a: tuple, b: tuple) -> float:
    lat = math.radians((a[1] + b[1]) / 2)
    return math.hypot((b[0] - a[0]) * 111_320 * math.cos(lat), (b[1] - a[1]) * 110_574)


def _lines(geom: dict):
    if geom.get("type") == "LineString":
        yield geom["coordinates"]
    elif geom.get("type") == "MultiLineString":
        yield from geom["coordinates"]


def blocked_segment_ids(roads_fc: dict, road_ids, bridge_ids) -> dict:
    """{segment id: damage class} - 'blocked' for flooded road segments, 'bridge'
    for every segment of a flooded bridge way.  Same convention as evaluation/oracle.py."""
    out = {str(r): "blocked" for r in road_ids}
    bridge_ids = {str(b) for b in bridge_ids}
    if bridge_ids:
        prefixes = tuple(f"{b}#" for b in bridge_ids)
        for f in roads_fc.get("features", []):
            p = f.get("properties") or {}
            rid = str(p.get("id", ""))
            way = str(p.get("way_id") or rid.split("#")[0])
            if rid in bridge_ids or way in bridge_ids or rid.startswith(prefixes):
                out.setdefault(rid, "bridge")
    return out


def build_graph(roads_fc: dict, damage: dict, include_footpaths: bool):
    """networkx MultiGraph in T3's Part-04 format (x/y = lon/lat, osmid = segment id)."""
    import networkx as nx
    G = nx.MultiGraph()
    ids: dict = {}                       # rounded (lon, lat) -> integer node id (T3's code needs ints)

    def node(k: tuple) -> int:
        if k not in ids:
            ids[k] = len(ids)
            G.add_node(ids[k], x=k[0], y=k[1])
        return ids[k]

    for f in roads_fc.get("features", []):
        p = f.get("properties") or {}
        hw = str(p.get("highway_type") or "")
        if not include_footpaths and hw in FOOT_ONLY:
            continue
        rid = str(p.get("id", ""))
        level = damage.get(rid, "none")
        for line in _lines(f.get("geometry") or {}):
            for a, b in zip(line, line[1:]):
                ka, kb = _key(a[0], a[1]), _key(b[0], b[1])
                if ka == kb:
                    continue
                G.add_edge(node(ka), node(kb), osmid=rid, length=_metres(ka, kb), highway=hw or "default",
                           name=str(p.get("name") or ""), bridge="yes" if p.get("bridge") else "no",
                           damage_class=level)
    return G


def _gdf(fc: dict):
    import geopandas as gpd
    feats = fc.get("features", [])
    if not feats:
        return gpd.GeoDataFrame({"id": [], "name": []}, geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")


def _exit_hubs(G, aoi_box: tuple):
    """Fallback destinations when no hospital/town is mapped: nodes where the network
    leaves the fetched box (the road continues towards the wider network)."""
    import geopandas as gpd
    from shapely.geometry import Point
    w, s, e, n = aoi_box
    tol_x, tol_y = (e - w) * 0.01, (n - s) * 0.01
    xy = [(G.nodes[k]["x"], G.nodes[k]["y"]) for k, deg in G.degree() if deg == 1]
    pts = [p for p in xy if p[0] - w < tol_x or e - p[0] < tol_x or p[1] - s < tol_y or n - p[1] < tol_y]
    rows = [{"name": "road exit to the wider network", "geometry": Point(p)} for p in pts]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326") if rows else None


def _reason(row, dest: str, names: list) -> str:
    road = f" (e.g. {', '.join(names[:3])})" if names else ""
    mins = row.get("baseline_min")
    base = f"{mins:.0f} min" if isinstance(mins, (int, float)) and mins == mins else "some time"
    return (f"Before the event this settlement could reach {dest} by road in about {base}. "
            f"The flood overlaps {len(row['blocking_ids'])} road segment(s) on that route{road}, "
            f"and no other mapped road route to any hospital or town remains.")


def run_connectivity(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None:
    from gis import cutoff
    from acquisition.osm import buffered_bbox

    ccfg = cfg.get("connectivity", {})
    block_bridges = bool(ccfg.get("block_flooded_bridges", True))
    include_foot = bool(ccfg.get("include_footpaths", False))

    roads_fc = read_json(run.path(Artifact.OSM_ROADS))
    settlements = _gdf(read_json(run.path(Artifact.OSM_SETTLEMENTS)))
    hubs = _gdf(read_json(run.path(Artifact.OSM_HOSPITALS)))
    if "population" not in settlements.columns:
        settlements["population"] = 0
    road_ids = [str(f["properties"]["object_id"]) for f in read_json(run.path(Artifact.AFFECTED_ROADS))["features"]]
    bridge_ids = [str(f["properties"]["object_id"]) for f in read_json(run.path(Artifact.AFFECTED_BRIDGES))["features"]]
    damage = blocked_segment_ids(roads_fc, road_ids, bridge_ids)

    G = build_graph(roads_fc, damage, include_foot)
    warnings: list = []
    if G.number_of_edges() == 0:
        raise GISError("The pre-event road network is empty here, so road access cannot be assessed.")

    hubs_used = hubs
    if hubs.empty:
        buffer_km = float(cfg.get("osm", {}).get("network_buffer_km", 5.0))
        hubs_used = _exit_hubs(G, buffered_bbox(aoi, buffer_km))
        if hubs_used is None:
            raise GISError("No hospital or town is mapped near this area and the road network has no "
                           "exit, so 'cut off' cannot be decided.")
        warnings.append("No hospital or town is mapped near this area: 'cut off' here means no road "
                        "route out of the network to the wider road system.")

    levels_primary = {"blocked", "bridge"} if block_bridges else {"blocked"}
    pcfg = cutoff.Config(
        scenarios={"roads_only": {"blocked"}, "roads_and_bridges": {"blocked", "bridge"}},
        primary_scenario="roads_and_bridges" if block_bridges else "roads_only",
        max_snap_m=float(ccfg.get("max_snap_m", 1500.0)),
        detour_ratio=float(ccfg.get("detour_ratio", 1.5)),
        detour_extra_min=float(ccfg.get("detour_extra_min", 30.0)),
        name_col="name", pop_col="population")
    if settlements.empty:
        results = None
    else:
        try:
            results = cutoff.run_pipeline(G, settlements, hubs_used, pcfg)
        except ValueError as exc:
            raise GISError("Hospitals/towns could not be linked to the road network.", str(exc)) from exc

    prov_mask = read_provenance(run.path(Artifact.AFFECTED_ROADS)) or {}
    prov = Provenance(producer="gis.connectivity", source="pre-event OSM road graph minus affected segments",
                      crs="EPSG:4326", event_date=event_date.isoformat(),
                      model_version=str(prov_mask.get("model_version", "")),
                      is_mock=bool(prov_mask.get("is_mock")))

    features, by_status, scen_table, pockets, repairs = [], {}, [], [], []
    if results is not None:
        res = results[pcfg.primary_scenario]
        S = res["settlements"]
        by_status = res["summary"]["by_status"]
        for _, row in S[S["status"] == "CUT_OFF"].iterrows():
            ids = [str(i) for i in row["blocking_ids"]]
            names = sorted({str(f["properties"].get("name") or "") for f in roads_fc["features"]
                            if str(f["properties"].get("id")) in set(ids)} - {"", "None"})
            dest = row["nearest_hub"] or "the nearest hospital or town"
            name = row["name"] if isinstance(row["name"], str) and row["name"].strip() else f"Unnamed ({row.get('id')})"
            props = {"settlement_id": str(row.get("id", name)), "name": name,
                     "name_ne": row.get("name_ne"), "reason": _reason(row, dest, names),
                     "nearest_destination": str(dest), "blocking_segment_ids": ids,
                     "baseline_minutes": row["baseline_min"], "pocket_id": row.get("pocket_id")}
            props = {k: (None if isinstance(v, float) and v != v else v) for k, v in props.items()}
            geom = row.geometry.__geo_interface__ if row.geometry is not None else None
            features.append({"type": "Feature", "geometry": geom, "properties": props})
        pockets = res["pockets"].to_dict("records") if len(res["pockets"]) else []
        repairs = res["summary"].get("top_repair_targets", [])
        for name, r in results.items():
            st = r["summary"]["by_status"]
            scen_table.append({"scenario": name,
                               "cutoff_settlements": st.get("CUT_OFF", {}).get("settlements", 0),
                               "degraded_settlements": st.get("DEGRADED", {}).get("settlements", 0)})

    write_geojson(run.path(Artifact.CUTOFF_SETTLEMENTS), features, prov)
    count = lambda s: int(by_status.get(s, {}).get("settlements", 0))     # noqa: E731
    write_json(run.path(Artifact.CONNECTIVITY_SUMMARY), {
        "cutoff_settlements": len(features),
        "cutoff_names": [f["properties"]["name"] for f in features],
        "settlements_total": int(len(settlements)),
        "destinations_used": sorted({str(n) for n in hubs_used["name"].tolist()}),
        "degraded_settlements": count("DEGRADED"),
        "connected_settlements": count("CONNECTED"),
        "no_baseline_access": count("NO_BASELINE_ACCESS"),
        "unknown_snap": count("UNKNOWN_SNAP"),
        "blocked_segments": sum(1 for v in damage.values() if v in levels_primary),
        "flooded_bridges_block_roads": block_bridges,
        "footpaths_included": include_foot,
        "pockets": pockets, "top_repair_targets": repairs, "scenarios": scen_table,
        "warnings": warnings,
        "notes": ["NO_BASELINE_ACCESS settlements had no mapped road route even before the event; "
                  "they are a map gap, not a flood effect, and are not counted as cut off."],
    }, prov)
