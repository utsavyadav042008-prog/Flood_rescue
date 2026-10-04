"""
=============================================================================
FloodResQ  |  PART 02 — ROAD GRAPH CONSTRUCTION
=============================================================================
Owner : Teammate 3 (GIS / Damage / Connectivity)

WHAT THIS FILE DOES
-------------------
Turns the cleaned OSM road *lines* from Part 01 into a *graph*: a set of
nodes (junctions / dead-ends) and edges (road pieces between them).
A graph is what lets us later ask questions like:
    "If this bridge is flooded, which villages can no longer reach a hospital?"

INPUT  (from Part 01)
---------------------
A GeoDataFrame (or a GeoPackage / GeoJSON file) of road LineStrings in
EPSG:4326 with, ideally, these columns (all optional except geometry):
    highway   - OSM road class: 'primary', 'track', 'path', ...
    name      - road name
    maxspeed  - OSM maxspeed text, e.g. '40' or '30 mph'
    bridge    - 'yes' / NaN
    tunnel    - 'yes' / NaN
    oneway    - 'yes' / NaN   (stored, but NOT used for connectivity)
    osmid     - OSM way id

OUTPUT (consumed by Part 03 - flood overlay, Part 04 - connectivity)
--------------------------------------------------------------------
    nx.MultiGraph G  with
      node attrs : x, y (metres, projected CRS), lon, lat, component
      edge attrs : edge_id, length_m, highway, name, bridge, tunnel,
                   drivable, speed_kmh, travel_time_s, geometry (shapely)
      graph attrs: crs (projected CRS string)
    + edges.gpkg / nodes.gpkg / road_graph.pkl in the output folder

PIPELINE (each step = one function below)
-----------------------------------------
    load_roads -> project_and_explode -> snap_to_grid -> split_at_junctions
              -> attach_speeds -> build_graph -> label_components
              -> (optional) prune_small_components -> save_graph

HOW TO RUN
----------
    python part02_road_graph.py --roads roads.gpkg --out out_part02
    python part02_road_graph.py --demo              # self-test, no data needed

Requires: geopandas, shapely>=2.0, networkx, numpy, scipy
=============================================================================
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

# Coordinates are snapped to this grid (in metres) before we look for shared
# points. 0.1 m is far smaller than any real road feature but large enough to
# absorb floating-point noise from the CRS reprojection.
SNAP_GRID_M = 0.1

# Fallback speeds (km/h) when OSM has no usable `maxspeed`. Deliberately LOW:
# these are Himalayan roads (switchbacks, gravel, landslide debris), not
# highways. Tune these once you have field knowledge.
DEFAULT_SPEED_KMH = {
    "motorway": 60, "trunk": 40, "primary": 35, "secondary": 30,
    "tertiary": 25, "unclassified": 20, "residential": 20, "service": 10,
    "track": 12,       # unpaved farm / forest roads (4x4, tractor)
    "path": 4,         # footpaths: walking speed
    "footway": 4, "steps": 2, "bridleway": 4,
}
FALLBACK_SPEED_KMH = 15        # any highway class not listed above

# Road classes a normal rescue truck can use. Everything else stays in the
# graph (villages in Nepal are often reached ONLY by trail) but is flagged
# drivable=False so Part 04 can run "vehicle" and "on-foot" analyses.
DRIVABLE_CLASSES = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "service", "track",
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
    "tertiary_link",
}


# --------------------------------------------------------------------------
# STEP 1 — LOAD
# --------------------------------------------------------------------------
def load_roads(roads: str | Path | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Accept either a file path (gpkg / geojson / shp) or a GeoDataFrame.

    Guarantees on return:
      * has a geometry column with only LineString / MultiLineString rows
      * has a CRS (assumed EPSG:4326 if the file forgot to declare one)
      * has an integer `src_idx` column = original row number. This lets us
        trace any graph edge back to the exact Part-01 row it came from.
    """
    gdf = gpd.read_file(roads) if isinstance(roads, (str, Path)) else roads.copy()

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]

    if gdf.crs is None:
        print("[warn] roads have no CRS; assuming EPSG:4326")
        gdf = gdf.set_crs(4326)

    gdf = gdf.reset_index(drop=True)
    gdf["src_idx"] = gdf.index

    # Make sure every optional column exists so later code never KeyErrors.
    for col in ["highway", "name", "maxspeed", "bridge", "tunnel", "oneway", "osmid"]:
        if col not in gdf.columns:
            gdf[col] = None
    return gdf


# --------------------------------------------------------------------------
# STEP 2 — PROJECT + EXPLODE
# --------------------------------------------------------------------------
def project_and_explode(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject to a local metric CRS and split MultiLineStrings.

    WHY PROJECT: lengths/snapping in degrees are meaningless
    (1 degree of longitude != 1 degree of latitude in metres).
    `estimate_utm_crs()` picks the right UTM zone automatically
    (EPSG:32645 for the Trishuli valley, ~85 deg E).

    WHY EXPLODE: a MultiLineString is several disconnected pieces glued into
    one row; the graph needs each piece on its own row.
    """
    metric_crs = gdf.estimate_utm_crs()
    out = gdf.to_crs(metric_crs).explode(index_parts=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# STEP 3 — SNAP TO GRID
# --------------------------------------------------------------------------
def snap_to_grid(gdf: gpd.GeoDataFrame, grid: float = SNAP_GRID_M) -> gpd.GeoDataFrame:
    """Round every vertex onto a `grid`-metre lattice.

    >>> TRICK #1 (see 'CLEVER TRICKS' at the bottom) <<<
    After this, two vertices that are "the same place" are *bit-for-bit
    identical* numbers, so we can find junctions with exact equality instead
    of fuzzy distance searches.

    `shapely.set_precision` may collapse very short segments to nothing or
    to a single point; those rows are dropped afterwards.
    """
    snapped = shapely.set_precision(gdf.geometry.values, grid_size=grid)
    gdf = gdf.copy()
    gdf["geometry"] = snapped
    gdf = gdf[~gdf.geometry.is_empty & (gdf.geometry.length > 0)]
    gdf = gdf[gdf.geometry.geom_type == "LineString"].reset_index(drop=True)
    return gdf


# --------------------------------------------------------------------------
# STEP 4 — SPLIT AT JUNCTIONS
# --------------------------------------------------------------------------
def _find_junction_vertices(coords: np.ndarray, line_idx: np.ndarray,
                            n_lines: int, grid: float):
    """Return (vertex_id per coordinate, boolean 'is_node' per vertex_id).

    A vertex is a graph NODE if either:
      (a) it is the first or last point of some road line   (dead end / joint)
      (b) it is used by 2 or more DIFFERENT road lines      (T-junction or
          crossing where both ways share the OSM node)

    Everything is vectorised with numpy so 100k+ vertices take milliseconds.
    """
    # Integer lattice coordinates -> exact, hashable identity per location.
    ij = np.round(coords / grid).astype(np.int64)
    # np.unique on rows: `inverse` gives every coordinate a vertex id.
    uniq, vertex_id = np.unique(ij, axis=0, return_inverse=True)
    vertex_id = vertex_id.ravel()
    n_vert = len(uniq)

    # (b) count how many DISTINCT lines touch each vertex.
    # Deduplicate (vertex, line) pairs first so a line that visits the same
    # vertex twice (a loop) still counts as ONE line.
    pairs = np.unique(np.stack([vertex_id, line_idx], axis=1), axis=0)
    lines_per_vertex = np.bincount(pairs[:, 0], minlength=n_vert)
    is_node = lines_per_vertex >= 2

    # (a) endpoints of every line. A line's first/last coordinate is where
    # line_idx changes, so we can find them without a Python loop.
    first = np.r_[True, line_idx[1:] != line_idx[:-1]]
    last = np.r_[line_idx[1:] != line_idx[:-1], True]
    is_node[vertex_id[first | last]] = True

    return vertex_id, is_node


def split_at_junctions(gdf: gpd.GeoDataFrame, grid: float = SNAP_GRID_M) -> gpd.GeoDataFrame:
    """Cut every road line at each interior junction vertex.

    Example: road A runs west->east through 5 vertices. Road B starts at A's
    3rd vertex. After this function A becomes two rows
    (vertex 1-3 and vertex 3-5), so the junction is an endpoint of both and
    the graph builder can connect them.

    Each output row inherits all attributes of the line it came from and
    gets `seg_id` (unique) plus `src_idx` (original Part-01 row).

    >>> TRICK #2 <<<  We split ONLY at shared vertices, never by computing
    geometric line-line intersections. See bottom of file for why.
    """
    geoms = gdf.geometry.values
    coords, line_idx = shapely.get_coordinates(geoms, return_index=True)
    vertex_id, is_node = _find_junction_vertices(coords, line_idx, len(gdf), grid)
    node_flag = is_node[vertex_id]          # per-coordinate: "is this a cut point?"

    # Where each line's coordinates start in the flat arrays.
    starts = np.r_[0, np.flatnonzero(np.diff(line_idx)) + 1]
    ends = np.r_[starts[1:], len(coords)]

    attr_cols = [c for c in gdf.columns if c != "geometry"]
    attrs = gdf[attr_cols].to_dict("records")

    new_rows, new_geoms = [], []
    for li, (s, e) in enumerate(zip(starts, ends)):
        c = coords[s:e]
        flags = node_flag[s:e]
        cut_positions = np.flatnonzero(flags)       # includes first & last
        # Consecutive cut positions delimit one piece.
        for a, b in zip(cut_positions[:-1], cut_positions[1:]):
            piece = c[a:b + 1]
            if len(piece) >= 2:
                new_geoms.append(LineString(piece))
                new_rows.append(attrs[li])

    out = gpd.GeoDataFrame(new_rows, geometry=new_geoms, crs=gdf.crs)
    out["seg_id"] = np.arange(len(out))
    return out


# --------------------------------------------------------------------------
# STEP 5 — SPEEDS & TRAVEL TIME
# --------------------------------------------------------------------------
_SPEED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mph)?", re.IGNORECASE)


def parse_maxspeed(value) -> float | None:
    """Convert an OSM maxspeed tag to km/h, or None if unusable.

    Handles: 40, '40', '40 km/h', '25 mph', 'walk', 'none', lists/NaN.
    Only the first number is used ('30;40' -> 30).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    m = _SPEED_RE.search(str(value))
    if not m:
        return None
    kmh = float(m.group(1))
    if m.group(2):                      # 'mph'
        kmh *= 1.60934
    return kmh if kmh > 0 else None


def attach_speeds(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add drivable, speed_kmh, length_m, travel_time_s columns.

    Speed priority: explicit OSM maxspeed > default for the highway class
    > FALLBACK_SPEED_KMH.  OSM `highway` is sometimes a list-like string;
    we take the first token.
    """
    gdf = gdf.copy()
    hw = gdf["highway"].astype(str).str.strip("[]' ").str.split(",").str[0].str.strip("' ")
    hw = hw.replace({"None": "unknown", "nan": "unknown"})
    gdf["highway"] = hw

    explicit = gdf["maxspeed"].map(parse_maxspeed)
    default = hw.map(DEFAULT_SPEED_KMH).fillna(FALLBACK_SPEED_KMH)
    gdf["speed_kmh"] = explicit.fillna(default).astype(float)

    gdf["drivable"] = hw.isin(DRIVABLE_CLASSES)
    gdf["length_m"] = gdf.geometry.length
    gdf["travel_time_s"] = gdf["length_m"] / (gdf["speed_kmh"] / 3.6)

    for col in ("bridge", "tunnel"):
        gdf[col] = gdf[col].astype(str).str.lower().isin(["yes", "true", "1", "viaduct"])
    return gdf


# --------------------------------------------------------------------------
# STEP 6 — BUILD THE GRAPH
# --------------------------------------------------------------------------
def build_graph(segments: gpd.GeoDataFrame) -> nx.MultiGraph:
    """Create the networkx graph from the split segments.

    * Node identity = the segment's snapped endpoint coordinate. Because of
      the grid snap, plain dict lookup on (x, y) is exact.
    * We use a MultiGraph (undirected, parallel edges allowed): two roads can
      legitimately join the same pair of junctions (e.g. a bypass).
      Dropping one would silently lose a road we might later need to flag as
      flooded.
    * UNDIRECTED on purpose: for rescue by foot / 4x4 / boat-landing access
      one-way rules are not binding, and treating them as binding would
      create false "cut-off" results.
    """
    G = nx.MultiGraph(crs=str(segments.crs))

    node_ids: dict[tuple[float, float], int] = {}

    def node_for(xy):
        nid = node_ids.get(xy)
        if nid is None:
            nid = len(node_ids)
            node_ids[xy] = nid
            G.add_node(nid, x=xy[0], y=xy[1])
        return nid

    # Pull endpoints in one vectorised call instead of per-row .coords access.
    start_xy = shapely.get_coordinates(shapely.get_point(segments.geometry.values, 0))
    end_xy = shapely.get_coordinates(shapely.get_point(segments.geometry.values, -1))

    keep_cols = ["seg_id", "src_idx", "osmid", "name", "highway", "bridge", "tunnel",
                 "oneway", "drivable", "speed_kmh", "length_m", "travel_time_s"]

    for row, s, e in zip(segments.itertuples(index=False), start_xy, end_xy):
        u = node_for((float(s[0]), float(s[1])))
        v = node_for((float(e[0]), float(e[1])))
        data = {c: getattr(row, c) for c in keep_cols}
        data["edge_id"] = data["seg_id"]          # stable handle for Part 03
        data["geometry"] = row.geometry
        if u == v:
            # Closed loop (e.g. a roundabout) -> keep, but a self-loop never
            # helps connectivity; Part 04 can ignore it.
            data["is_loop"] = True
        G.add_edge(u, v, key=data["edge_id"], **data)

    # Geographic coordinates for every node (needed for web-map overlays).
    xs = np.array([G.nodes[n]["x"] for n in G.nodes])
    ys = np.array([G.nodes[n]["y"] for n in G.nodes])
    lonlat = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=segments.crs).to_crs(4326)
    for n, p in zip(G.nodes, lonlat):
        G.nodes[n]["lon"], G.nodes[n]["lat"] = p.x, p.y
    return G


# --------------------------------------------------------------------------
# STEP 7 — CONNECTED COMPONENTS
# --------------------------------------------------------------------------
def label_components(G: nx.MultiGraph) -> pd.DataFrame:
    """Tag each node with `component` (0 = biggest) and return a summary.

    A healthy road network is mostly ONE giant component. Many small
    components usually mean data problems (missing connecting way, bridge
    mapped without junction nodes), NOT real isolation. Always look at this
    table before trusting any "cut-off" result later.
    """
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    rows = []
    for cid, nodes in enumerate(comps):
        for n in nodes:
            G.nodes[n]["component"] = cid
        length = sum(d["length_m"] for _, _, d in G.subgraph(nodes).edges(data=True))
        rows.append({"component": cid, "n_nodes": len(nodes), "length_km": length / 1000})
    return pd.DataFrame(rows)


def prune_small_components(G: nx.MultiGraph, min_nodes: int = 5) -> nx.MultiGraph:
    """OPT-IN: delete components with fewer than `min_nodes` nodes.

    Not applied by default - a tiny component may be a real, genuinely
    isolated hamlet road. Use only after eyeballing the summary table.
    Returns a copy; the input graph is untouched.
    """
    H = G.copy()
    small = [n for n, d in H.nodes(data=True) if
             len(nx.node_connected_component(H, n)) < min_nodes]
    H.remove_nodes_from(small)
    return H


# --------------------------------------------------------------------------
# STEP 8 — HELPERS FOR LATER PARTS
# --------------------------------------------------------------------------
def graph_to_gdfs(G: nx.MultiGraph):
    """Return (nodes_gdf, edges_gdf) in the graph's projected CRS."""
    nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
    nodes.index.name = "node_id"
    nodes = gpd.GeoDataFrame(nodes.reset_index(),
                             geometry=gpd.points_from_xy(nodes.x, nodes.y),
                             crs=G.graph["crs"])

    recs = []
    for u, v, k, d in G.edges(keys=True, data=True):
        r = {kk: vv for kk, vv in d.items()}
        r["u"], r["v"] = u, v
        recs.append(r)
    edges = gpd.GeoDataFrame(recs, geometry="geometry", crs=G.graph["crs"])
    return nodes, edges


def nearest_nodes(G: nx.MultiGraph, points: gpd.GeoDataFrame, max_dist_m: float = 500):
    """Snap points (e.g. settlements) to their nearest graph node.

    Returns a DataFrame: index of `points`, node_id, snap_dist_m, snapped
    (False if farther than `max_dist_m` - such a village effectively has no
    road access at all, which is itself a finding).
    Uses a KD-tree, so 10k settlements x 50k nodes runs instantly.
    """
    ids = np.array(list(G.nodes))
    xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in ids])
    tree = cKDTree(xy)

    pts = points.to_crs(G.graph["crs"])
    px = np.column_stack([pts.geometry.x, pts.geometry.y])
    dist, idx = tree.query(px)
    return pd.DataFrame({"node_id": ids[idx], "snap_dist_m": dist,
                         "snapped": dist <= max_dist_m}, index=points.index)


def remove_edges(G: nx.MultiGraph, edge_ids) -> nx.MultiGraph:
    """Return a COPY of G without the given edge_ids (e.g. flooded roads).

    This is the hook Part 03/04 will use: flood overlay -> list of edge_ids
    -> remove_edges -> re-run label_components -> see who is cut off.
    Nodes are kept (so isolated nodes show up as their own components).
    """
    edge_ids = set(edge_ids)
    H = G.copy()
    doomed = [(u, v, k) for u, v, k, d in H.edges(keys=True, data=True)
              if d["edge_id"] in edge_ids]
    H.remove_edges_from(doomed)
    return H


# --------------------------------------------------------------------------
# STEP 9 — SAVE
# --------------------------------------------------------------------------
def save_graph(G: nx.MultiGraph, out_dir: str | Path) -> None:
    """Write edges.gpkg, nodes.gpkg and road_graph.pkl into `out_dir`.

    The pickle is the exact Python object (fast reload, keeps geometry);
    the GeoPackages open directly in QGIS for visual checking.
    GraphML is avoided on purpose: it cannot store shapely geometries.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes, edges = graph_to_gdfs(G)
    edges.to_file(out / "edges.gpkg", driver="GPKG")
    nodes.to_file(out / "nodes.gpkg", driver="GPKG")
    with open(out / "road_graph.pkl", "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path: str | Path) -> nx.MultiGraph:
    """Reload a graph saved by `save_graph` (for Parts 03 and 04)."""
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# ORCHESTRATOR
# --------------------------------------------------------------------------
def build_road_graph(roads, out_dir: str | Path | None = None,
                     grid: float = SNAP_GRID_M, verbose: bool = True) -> nx.MultiGraph:
    """Run the full Part 02 pipeline and (optionally) save the results."""
    gdf = load_roads(roads)
    n_in = len(gdf)
    gdf = project_and_explode(gdf)
    gdf = snap_to_grid(gdf, grid)
    segs = split_at_junctions(gdf, grid)
    segs = attach_speeds(segs)
    G = build_graph(segs)
    summary = label_components(G)

    if verbose:
        print(f"input rows      : {n_in}")
        print(f"after splitting : {len(segs)} segments")
        print(f"graph           : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"total length    : {segs.length_m.sum() / 1000:.1f} km "
              f"({segs.loc[segs.drivable, 'length_m'].sum() / 1000:.1f} km drivable)")
        print(f"components      : {len(summary)} "
              f"(largest holds {summary.n_nodes.iloc[0] / G.number_of_nodes():.0%} of nodes)")
    if out_dir:
        save_graph(G, out_dir)
        summary.to_csv(Path(out_dir) / "components.csv", index=False)
    return G


# --------------------------------------------------------------------------
# DEMO / SELF-TEST  (no real data required)
# --------------------------------------------------------------------------
def _demo_roads() -> gpd.GeoDataFrame:
    """Tiny synthetic network near Dhunche, Nepal (~85.3E, 28.1N).

        A: main road, west->east, 4 vertices
        B: side road starting at A's 3rd vertex (shared vertex = T-junction)
        C: bridge that crosses A geometrically WITHOUT sharing a vertex
           (must NOT become a junction)
        D: an isolated footpath (own component)
    """
    ox, oy = 85.30, 28.10
    d = 0.001
    mk = lambda pts: LineString([(ox + x * d, oy + y * d) for x, y in pts])
    rows = [
        dict(highway="primary", name="A", maxspeed="40", geometry=mk([(0, 0), (1, 0), (2, 0), (3, 0)])),
        dict(highway="track", name="B", geometry=mk([(2, 0), (2, 1), (2, 2)])),
        dict(highway="secondary", name="C", bridge="yes", geometry=mk([(1.5, -1), (1.5, 1)])),
        dict(highway="path", name="D", geometry=mk([(5, 5), (6, 5)])),
    ]
    return gpd.GeoDataFrame(rows, crs=4326)


def _run_demo():
    G = build_road_graph(_demo_roads(), out_dir="out_part02_demo")
    print("\nExpected: A split into 2 (at B's junction) -> 5 edges total")
    print("          bridge C NOT connected to A  -> D and C are separate components")
    nodes, edges = graph_to_gdfs(G)
    print(edges[["name", "highway", "length_m", "speed_kmh", "travel_time_s", "u", "v"]]
          .round(1).to_string(index=False))
    assert G.number_of_edges() == 5, "split logic broke"
    comps = {d["component"] for _, d in G.nodes(data=True)}
    assert len(comps) == 3, "bridge/crossing logic broke"
    print("\nDemo OK. Outputs in ./out_part02_demo")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FloodResQ Part 02 - road graph builder")
    ap.add_argument("--roads", help="Road lines file from Part 01 (gpkg/geojson/shp)")
    ap.add_argument("--out", default="out_part02", help="Output folder")
    ap.add_argument("--demo", action="store_true", help="Run self-test on synthetic data")
    args = ap.parse_args()
    if args.demo:
        _run_demo()
    elif args.roads:
        build_road_graph(args.roads, args.out)
    else:
        ap.error("give --roads FILE or --demo")


# ==========================================================================
# CLEVER TRICKS  (the non-obvious bits, explained)
# ==========================================================================
#
# TRICK #1 — "Snap first, then compare exactly"
#   The normal way to merge nearly-identical points is a distance search
#   ("is this point within 10 cm of that one?"), which is slow and
#   order-dependent. Instead `snap_to_grid` rounds every coordinate onto a
#   0.1 m lattice, then `_find_junction_vertices` converts coordinates to
#   INTEGER lattice indices (np.round(coords / grid).astype(int64)).
#   Integers compare exactly, so `np.unique(..., axis=0)` finds every
#   shared location in one vectorised pass, and the graph-node dictionary
#   can use plain (x, y) keys. No tolerance logic, no KD-tree needed to
#   build the graph.
#   Known limit: two points 1 cm apart that straddle a grid line round to
#   DIFFERENT cells. In OSM this does not occur because shared nodes have
#   identical coordinates to begin with; the snap only cleans up the
#   reprojection noise.
#
# TRICK #2 — Junctions come from SHARED VERTICES, not geometric crossings
#   Many tutorials call unary_union() to "node" all lines wherever they
#   cross. That is WRONG for road data: a bridge over a road, a tunnel under
#   a ridge road, or a flyover crosses geometrically but there is NO way to
#   get from one to the other. Noding them would invent junctions and make
#   a flooded valley road look connected to a dry bridge above it, hiding
#   cut-off villages. OSM only connects ways that share a node, so we only
#   cut at vertices used by >= 2 different lines (plus line endpoints).
#   Bonus: no attributes are lost (unary_union destroys them and forces an
#   error-prone re-join), and edge ids stay traceable to Part 01 rows.
#   If you ever import NON-OSM road data (e.g. satellite-derived roads)
#   where lines cross without shared vertices, you will need a separate
#   crossing-noding step - deliberately left out here.
#
# TRICK #3 — Finding line endpoints without a loop
#   shapely.get_coordinates(..., return_index=True) flattens all lines into
#   one big coordinate array plus a parallel array saying which line each
#   coordinate belongs to. A line's first coordinate is simply where that
#   index changes (`line_idx[1:] != line_idx[:-1]`), so endpoints,
#   "how many lines touch this vertex", and the cut positions are all
#   computed with array operations. Only the final slicing into new
#   LineStrings needs a Python loop.
#
# TRICK #4 — MultiGraph + edge_id as the key
#   Keeping parallel edges (instead of simplifying them away) and using the
#   segment id as the MultiGraph edge key means Part 03 can say "edge 1832
#   is flooded" and `remove_edges` deletes exactly that road piece, never
#   a neighbour that merely shares the same two junctions.
#
# TRICK #5 — Trails stay in the graph, flagged `drivable`
#   In the Trishuli valley many settlements are reachable only by foot trail.
#   Dropping `path`/`footway` would declare them "isolated" even in normal
#   conditions. We keep everything and let Part 04 run two analyses
#   (vehicle-only vs on-foot) by filtering on the `drivable` edge flag.
#
# TRICK #6 — Undirected on purpose
#   `oneway` is stored but ignored. Rescue teams are not bound by one-way
#   rules, and directed edges would produce false cut-off results.
# ==========================================================================
