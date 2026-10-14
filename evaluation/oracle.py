"""A small, independent "oracle" for the cut-off analysis.

Purpose
-------
1. A BASELINE for T3: the same question answered with ~100 lines of networkx.
2. A TEST ORACLE: ``tests/test_stage_acceptance.py`` checks T3's real
   connectivity stage against this on the mock scenario.

Method: nodes are road vertices (rounded to ~1 m), edges are consecutive
vertices.  Settlements and destinations are snapped to the nearest node within
``snap_m`` metres.  A settlement is

* ``connected``           still reaches a destination after removing the affected edges
* ``cut_off``             reached one BEFORE, but no longer AFTER   <- the flood's effect
* ``unconnected_before``  never reached one (an OSM gap, NOT a flood effect)

Separating the last two is the honesty point: a village that was already
unreachable in the pre-event map must not be reported as "cut off by the flood".

Convention: a road feature's ``id`` is ``<way id>#<k>``; an affected bridge with
object id ``<way id>`` removes every road feature whose id starts with ``<way id>#``.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import networkx as nx
import numpy as np

CONNECTED, CUT_OFF, UNCONNECTED = "connected", "cut_off", "unconnected_before"


def _key(lon: float, lat: float) -> tuple:
    return (round(lon, 5), round(lat, 5))


def _metres(a: tuple, b: tuple) -> float:
    lat = math.radians((a[1] + b[1]) / 2)
    return math.hypot((b[0] - a[0]) * 111_320 * math.cos(lat), (b[1] - a[1]) * 110_574)


def _lines(geometry: dict):
    if geometry.get("type") == "LineString":
        yield geometry["coordinates"]
    elif geometry.get("type") == "MultiLineString":
        yield from geometry["coordinates"]


def _point(geometry: dict) -> Optional[tuple]:
    pts, stack = [], [geometry.get("coordinates")]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            if item and isinstance(item[0], (int, float)):
                pts.append(item)
            else:
                stack.extend(item)
    return None if not pts else (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def build_graph(roads_fc: dict, removed_ids: Iterable[str] = ()) -> nx.Graph:
    removed = set(removed_ids)
    graph = nx.Graph()
    for feature in roads_fc.get("features", []):
        props = feature.get("properties") or {}
        rid = str(props.get("id", ""))
        if rid in removed:
            continue
        for line in _lines(feature.get("geometry") or {}):
            for p, q in zip(line, line[1:]):
                a, b = _key(p[0], p[1]), _key(q[0], q[1])
                if a != b:
                    graph.add_edge(a, b, weight=_metres(a, b), id=rid)
    return graph


def expand_removed(affected_road_ids: Iterable[str], affected_bridge_ids: Iterable[str], roads_fc: dict) -> set:
    """Road ids to remove: the affected segments plus every segment of an affected bridge."""
    removed = set(affected_road_ids)
    prefixes = tuple(f"{b}#" for b in affected_bridge_ids)
    for feature in roads_fc.get("features", []):
        rid = str((feature.get("properties") or {}).get("id", ""))
        if rid in set(affected_bridge_ids) or (prefixes and rid.startswith(prefixes)):
            removed.add(rid)
    return removed


def _snap(point: tuple, nodes: list, max_m: float) -> Optional[tuple]:
    if not nodes or point is None:
        return None
    arr = np.array(nodes)
    lat = math.radians(point[1])
    d = np.hypot((arr[:, 0] - point[0]) * 111_320 * math.cos(lat), (arr[:, 1] - point[1]) * 110_574)
    i = int(d.argmin())
    return tuple(arr[i]) if d[i] <= max_m else None


def analyse(roads_fc: dict, settlements_fc: dict, destinations_fc: dict, removed_ids: Iterable[str] = (),
            snap_m: float = 300.0) -> dict:
    """``{settlement_id: {name, status, nearest_destination, blocking_segment_ids}}``."""
    removed = set(removed_ids)
    before, after = build_graph(roads_fc), build_graph(roads_fc, removed)
    nodes = list(before.nodes)
    dests = []
    for f in destinations_fc.get("features", []):
        node = _snap(_point(f.get("geometry") or {}), nodes, snap_m)
        if node is not None:
            dests.append((node, (f.get("properties") or {}).get("name", "")))

    def reach(graph):
        sources = [n for n, _ in dests if n in graph]
        return nx.multi_source_dijkstra(graph, sources) if sources else ({}, {})

    dist_before, paths_before = reach(before)
    dist_after, _ = reach(after)
    names = {n: name for n, name in dests}
    result = {}
    for f in settlements_fc.get("features", []):
        props = f.get("properties") or {}
        sid = str(props.get("id", props.get("name", "")))
        node = _snap(_point(f.get("geometry") or {}), nodes, snap_m)
        entry = {"name": props.get("name", sid), "status": UNCONNECTED, "nearest_destination": "",
                 "blocking_segment_ids": []}
        if node is not None and node in dist_before:
            path = paths_before[node]
            entry["nearest_destination"] = names.get(path[0], "")
            if node in dist_after:
                entry["status"] = CONNECTED
            else:
                entry["status"] = CUT_OFF
                entry["blocking_segment_ids"] = sorted(
                    {before[u][v]["id"] for u, v in zip(path, path[1:]) if before[u][v]["id"] in removed})
        result[sid] = entry
    return result
