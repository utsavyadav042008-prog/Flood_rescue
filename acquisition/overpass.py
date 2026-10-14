"""
=====================================================================
FloodResQ  |  PART 01  |  OSM Data Acquisition & Normalisation
=====================================================================

INTEGRATION NOTE (read first)
-----------------------------
In the integrated pipeline this module is the FALLBACK OSM source and the home
of the tag normalisers.  The primary source is ``acquisition/ohsome.py`` (the
ohsome API named in the brief).  Plain Overpass returns the CURRENT map, which
the data rules forbid for post-event dates, so ``Config.date`` was added: when
set, every query carries ``[date:"..."]`` and Overpass answers from its
history ("attic") database, i.e. the map AS IT WAS on that date.

WHAT THIS FILE DOES
-------------------
Downloads OpenStreetMap (OSM) data for the Trishuli flood area from the
Overpass API and turns the messy raw JSON into clean, consistently-named
GeoDataFrames that every later part of the pipeline can trust.

    Overpass API  ->  raw elements  ->  shapely geometry  ->  normalised
    layers  ->  one GeoPackage (+ manifest.json) on disk

OUTPUT LAYERS (all stored in EPSG:4326, metric values pre-computed in UTM)
-------------------------------------------------------------------------
    roads        LineString   highway=*  (+ bridge / ford / tunnel flags)
    bridges      LineString   subset of roads where is_bridge is True
    waterways    LineString   rivers, streams, canals
    water_bodies Polygon      natural=water
    buildings    Polygon      building=*
    places       Point        place=* (villages, hamlets ... = settlements)
    facilities   Point        hospitals, clinics, schools, police, shelters

HOW TO RUN
----------
    pip install geopandas shapely pandas requests pyogrio
    python part01_osm_acquisition.py                      # default bbox
    python part01_osm_acquisition.py --bbox 85.1 27.9 85.4 28.2
    python part01_osm_acquisition.py --layers roads places --refresh

    # from another part of the project:
    from part01_osm_acquisition import load_layers
    layers = load_layers("data/part01/osm_normalised.gpkg")
    roads = layers["roads"]

NOTE ON THE DEFAULT BBOX
------------------------
(west, south, east, north) = (85.05, 27.85, 85.45, 28.30) is a generous
box around the Trishuli valley (Rasuwa -> Nuwakot). It is only a starting
guess: replace it with the real AOI from your flood-extent file.

THE CLEVER TRICKS (each is also marked "TRICK #n" in the code)
--------------------------------------------------------------
 #1  Adaptive quadtree splitting: Overpass often answers HTTP 200 with
     a silent "remark" saying it ran out of time/memory and returns a
     TRUNCATED result. We detect that remark, split the tile into 4 and
     retry, so you never get a quietly incomplete dataset.
 #2  Tiles may overlap/duplicate freely: every element gets a globally
     unique id ("w123", "n456"), so duplicates from neighbouring tiles
     collapse with one drop_duplicates().
 #3  `out geom;` embeds coordinates inside each way, so we skip the
     classic (and slow) "download all nodes, then join" step.
 #4  The LAYER declares the geometry type, not the tags. Closed ways are
     ambiguous (a roundabout is a closed LINE, a building is a closed
     POLYGON), so each layer says what it wants.
 #5  Multipolygon relations are rebuilt with unary_union + polygonize,
     which stitches together ways that were split into pieces, without
     us having to order or orient the member ways by hand.
 #6  We FILTER with .intersects() instead of CLIPPING. Clipping would cut
     bridges/roads in half at the bbox edge and break connectivity later.
 #7  UTM zone is computed from the bbox centre, so lengths/areas are in
     real metres without hard-coding a CRS.
 #8  Disk cache keyed by a hash of the exact query text: re-runs are
     instant and a crashed run resumes where it stopped.
 #9  Mirror rotation + Retry-After handling: a rate-limited or dead
     server never kills the run.
 #10 Messy human tags ("3.5 m", "12'", "1,200") are parsed defensively
     and become real numbers or None, never crashes or strings.
=====================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

try:
    import geopandas as gpd  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing dependency: geopandas. Install with: "
        "pip install geopandas shapely pandas requests pyogrio"
    ) from exc

try:
    import pandas as pd  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing dependency: pandas. Install with: "
        "pip install geopandas shapely pandas requests pyogrio"
    ) from exc

try:
    import requests  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing dependency: requests. Install with: "
        "pip install geopandas shapely pandas requests pyogrio"
    ) from exc

try:
    from shapely import make_valid  # pyright: ignore[reportMissingModuleSource]
    from shapely.geometry import (  # pyright: ignore[reportMissingModuleSource]
        LineString,
        Point,
        Polygon,
        box,
    )
    from shapely.ops import polygonize, unary_union  # pyright: ignore[reportMissingModuleSource]
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing dependency: shapely. Install with: "
        "pip install geopandas shapely pandas requests pyogrio"
    ) from exc

log = logging.getLogger("part01")

# =====================================================================
# 1. CONFIGURATION
# =====================================================================


@dataclass
class Config:
    """All knobs in one place. Edit here, not deep inside functions."""

    # (west, south, east, north) in degrees, EPSG:4326
    bbox: tuple[float, float, float, float] = (85.05, 27.85, 85.45, 28.30)

    out_dir: Path = Path("data/part01")
    cache_dir: Path = Path("data/part01/cache")

    tile_deg: float = 0.15        # first-pass tile size (degrees)
    min_tile_deg: float = 0.02    # never split below this (~2 km)
    overpass_timeout: int = 120   # seconds the SERVER may spend per query
    max_retries: int = 4          # attempts per tile before giving up
    polite_pause: float = 1.0     # seconds between uncached requests
    refresh: bool = False         # True = ignore cache
    # Historical snapshot "YYYY-MM-DD" (attic query). None = current map, which
    # is NOT allowed as a pipeline input for a past event (data rules).
    date: Optional[str] = None

    # Public Overpass servers; we rotate through them on failure (TRICK #9)
    mirrors: tuple[str, ...] = (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    )
    user_agent: str = "FloodResQ-Hackathon/1.0 (OSM damage overlay; contact: team)"


# =====================================================================
# 2. GEOGRAPHY HELPERS
# =====================================================================


def make_tiles(bbox: tuple[float, float, float, float], step: float):
    """Cut a bbox into an even grid of tiles of roughly `step` degrees.

    We divide the box into n equal parts (instead of walking `step` and
    leaving a thin sliver at the edge), so tiles are all the same size.
    Returns a list of (west, south, east, north).
    """
    w, s, e, n = bbox
    nx = max(1, math.ceil((e - w) / step))
    ny = max(1, math.ceil((n - s) / step))
    dx, dy = (e - w) / nx, (n - s) / ny
    return [
        (w + i * dx, s + j * dy, w + (i + 1) * dx, s + (j + 1) * dy)
        for i in range(nx)
        for j in range(ny)
    ]


def split_bbox(b: tuple[float, float, float, float]):
    """Split one bbox into its 4 quadrants (used by TRICK #1)."""
    w, s, e, n = b
    mx, my = (w + e) / 2, (s + n) / 2
    return [(w, s, mx, my), (mx, s, e, my), (w, my, mx, n), (mx, my, e, n)]


def utm_epsg_for_bbox(bbox: tuple[float, float, float, float]) -> int:
    """TRICK #7: pick the right metric CRS automatically.

    UTM zones are 6 degrees wide: zone = floor((lon + 180) / 6) + 1.
    EPSG is 326xx for the northern hemisphere, 327xx for the southern.
    (Trishuli, ~85.2E, lands in zone 45N -> EPSG:32645.)
    """
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


# =====================================================================
# 3. OVERPASS CLIENT  (network, caching, retries, adaptive splitting)
# =====================================================================


class OverpassTooBig(Exception):
    """Server answered, but admitted the result is incomplete (see TRICK #1)."""


class OverpassClient:
    """Thin, robust wrapper around the Overpass interpreter endpoint."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mirror_idx = 0
        self.osm_timestamp: Optional[str] = None  # data freshness, for manifest

    # ---------- query construction ----------

    def build_query(self, body: str, bbox) -> str:
        """Wrap a layer's selector `body` into a complete Overpass QL query.

        TRICK #3: `out geom;` returns each way WITH its node coordinates
        inline. The older pattern `out body; >; out skel qt;` makes the
        server resolve and ship every node separately (bigger, slower, and
        we would have to join them back together ourselves).
        """
        w, s, e, n = bbox
        # [date:...] = attic query: the map as it was on that date (data rules).
        attic = f'[date:"{self.cfg.date}T00:00:00Z"]' if self.cfg.date else ""
        return (
            f"[out:json][timeout:{self.cfg.overpass_timeout}]{attic}"
            f"[bbox:{s:.6f},{w:.6f},{n:.6f},{e:.6f}];\n"
            f"(\n{body}\n);\n"
            f"out geom;"
        )

    # ---------- caching ----------

    def _cache_path(self, query: str) -> Path:
        """TRICK #8: the cache key is a hash of the EXACT query text.
        Change the bbox, the selector or the timeout and the key changes
        automatically, so a stale cache can never be served by mistake."""
        h = hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]
        return self.cfg.cache_dir / f"{h}.json"

    # ---------- one HTTP round-trip (with retries) ----------

    def _post(self, query: str) -> dict:
        """POST a query, rotating mirrors and backing off on failure.

        TRICK #9: HTTP 429 means "slow down" (we obey Retry-After),
        504/5xx means the server is overloaded (we hop to the next mirror).
        Only after `max_retries` failures do we give up.
        """
        last_err: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            url = self.cfg.mirrors[self._mirror_idx % len(self.cfg.mirrors)]
            try:
                r = requests.post(
                    url,
                    data={"data": query},
                    headers={"User-Agent": self.cfg.user_agent},
                    timeout=self.cfg.overpass_timeout + 30,
                )
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 15 * attempt))
                    log.warning("429 from %s, sleeping %ss", url, wait)
                    time.sleep(wait)
                elif r.status_code in (400,):
                    # Our query is malformed: retrying cannot help.
                    raise ValueError(f"Overpass rejected query:\n{r.text[:500]}")
                else:
                    log.warning("HTTP %s from %s", r.status_code, url)
                    self._mirror_idx += 1
                    time.sleep(2 ** attempt)
                    if r.status_code == 504:
                        raise OverpassTooBig("gateway timeout")
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_err = exc
                log.warning("network error on %s: %s", url, exc)
                self._mirror_idx += 1
                time.sleep(2 ** attempt)
            except ValueError as exc:
                # JSON decode errors land here too (HTML error page etc.)
                if "rejected" in str(exc):
                    raise
                last_err = exc
                self._mirror_idx += 1
        raise RuntimeError(f"Overpass failed after retries: {last_err}")

    # ---------- one tile, with TRICK #1 ----------

    def fetch_tile(self, body: str, bbox, depth: int = 0) -> list[dict]:
        """Fetch all elements for one bbox. Splits the bbox if it's too big.

        TRICK #1 (the important one): when a query is too heavy, Overpass
        frequently does NOT return an error. It returns HTTP 200 with a
        normal-looking JSON whose "remark" field says something like
        "runtime error: Query timed out". The elements list is then just
        whatever it managed to collect: silently incomplete data.
        We read `remark`; if present we split the bbox into four quadrants
        and recurse until the pieces are small enough to succeed.
        """
        query = self.build_query(body, bbox)
        cache = self._cache_path(query)

        if cache.exists() and not self.cfg.refresh:
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            try:
                data = self._post(query)
                remark = data.get("remark", "")
                if "runtime error" in remark or "out of memory" in remark:
                    raise OverpassTooBig(remark)
            except OverpassTooBig as exc:
                w, s, e, n = bbox
                if (e - w) / 2 < self.cfg.min_tile_deg:
                    raise RuntimeError(
                        f"Tile {bbox} still too big at minimum size: {exc}"
                    )
                log.info("splitting tile at depth %d (%s)", depth, exc)
                out: list[dict] = []
                for sub in split_bbox(bbox):
                    out += self.fetch_tile(body, sub, depth + 1)
                return out
            cache.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(self.cfg.polite_pause)

        ts = data.get("osm3s", {}).get("timestamp_osm_base")
        if ts:
            self.osm_timestamp = ts
        return data.get("elements", [])

    # ---------- the whole AOI ----------

    def fetch(self, body: str) -> list[dict]:
        """Fetch a layer for the whole AOI: tile it, fetch each tile, dedupe.

        TRICK #2: a road crossing a tile edge is returned by BOTH tiles.
        We do not care: every element has a unique key (type letter + id,
        e.g. "w8841", "n12", "r77"), and we keep one copy of each.
        Using the type letter matters because node 5 and way 5 are
        different objects that share the number 5.
        """
        elements: dict[str, dict] = {}
        tiles = make_tiles(self.cfg.bbox, self.cfg.tile_deg)
        for i, t in enumerate(tiles, 1):
            log.info("  tile %d/%d", i, len(tiles))
            for el in self.fetch_tile(body, t):
                elements[f"{el['type'][0]}{el['id']}"] = el
        return list(elements.values())


# =====================================================================
# 4. RAW ELEMENTS  ->  SHAPELY GEOMETRY
# =====================================================================


def _coords(geometry: list | None) -> list[tuple[float, float]]:
    """Convert Overpass [{'lat':..,'lon':..}, ...] to [(lon, lat), ...].

    Overpass can return `null` for nodes it could not resolve; we drop them.
    Note the order: shapely and GeoJSON are (x=lon, y=lat), Overpass talks
    lat-first. Mixing this up silently puts Nepal in Antarctica.
    """
    return [(p["lon"], p["lat"]) for p in (geometry or []) if p]


def _relation_to_polygon(el: dict):
    """TRICK #5: rebuild a multipolygon relation without ordering ways.

    Real-world multipolygons (a big lake, a courtyard building) are often
    split into several ways. Instead of chaining them end-to-end by hand,
    we throw all member lines into unary_union (which "nodes" them at their
    shared endpoints) and let polygonize() find every closed ring.
    Outer rings minus inner rings (holes) = final shape.
    """
    outers, inners = [], []
    for m in el.get("members", []):
        if m.get("type") != "way":
            continue
        c = _coords(m.get("geometry"))
        if len(c) < 2:
            continue
        (inners if m.get("role") == "inner" else outers).append(LineString(c))
    if not outers:
        return None
    outer_poly = unary_union(list(polygonize(unary_union(outers))))
    if outer_poly.is_empty:
        return None
    if inners:
        hole_poly = unary_union(list(polygonize(unary_union(inners))))
        outer_poly = outer_poly.difference(hole_poly)
    return outer_poly


def element_to_geometry(el: dict, kind: str):
    """Convert one Overpass element to a shapely geometry, or None.

    `kind` is what the LAYER wants ("line", "polygon", "point").

    TRICK #4: we never guess from tags whether a closed way is an area.
    A closed highway=* is a roundabout (a line!), a closed building=* is
    a footprint (a polygon!). The layer definition decides, which removes
    a whole class of bugs.
    """
    t = el["type"]

    if t == "node":
        return Point(el["lon"], el["lat"]) if kind == "point" else None

    if t == "way":
        c = _coords(el.get("geometry"))
        if kind == "line":
            return LineString(c) if len(c) >= 2 else None
        closed = len(c) >= 4 and c[0] == c[-1]
        if kind == "polygon":
            return make_valid(Polygon(c)) if closed else None
        if kind == "point":  # facilities mapped as areas -> a representative point
            if closed:
                return make_valid(Polygon(c)).representative_point()
            return LineString(c).centroid if len(c) >= 2 else None

    if t == "relation" and kind in ("polygon", "point"):
        g = _relation_to_polygon(el)
        if g is None:
            return None
        g = make_valid(g)
        return g.representative_point() if kind == "point" else g

    return None


def elements_to_raw_gdf(elements: list[dict], kind: str) -> gpd.GeoDataFrame:
    """Build the RAW GeoDataFrame: uid, osm_type, osm_id, tags(dict), geometry.

    `uid` ("w123") is our primary key for the whole project: it is unique
    across node/way/relation and can be used to cross-reference layers.
    """
    rows = []
    for el in elements:
        geom = element_to_geometry(el, kind)
        if geom is None or geom.is_empty:
            continue
        rows.append(
            {
                "uid": f"{el['type'][0]}{el['id']}",
                "osm_type": el["type"],
                "osm_id": el["id"],
                "tags": el.get("tags", {}),
                "geometry": geom,
            }
        )
    if not rows:  # a layer can legitimately be empty in a small AOI
        return gpd.GeoDataFrame(
            {"uid": [], "osm_type": [], "osm_id": [], "tags": []},
            geometry=[], crs="EPSG:4326",
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


# =====================================================================
# 5. TAG PARSING HELPERS  (the "normalisation" vocabulary)
# =====================================================================

_NUM = re.compile(r"[-+]?\d*\.?\d+")


def parse_number(value) -> Optional[float]:
    """TRICK #10: turn human-typed tag values into floats (or None).

    Handles: "3.5", "3.5 m", "3,5", "1,200", "12'" (feet), "4 ft".
    Anything unparseable returns None rather than raising: mappers type
    all sorts of things ("narrow", "yes", "2-3") and one bad value must
    never crash a pipeline that is processing 100,000 features.
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    s = s.replace(",", "") if re.search(r"\d,\d{3}\b", s) else s.replace(",", ".")
    m = _NUM.search(s)
    if not m:
        return None
    x = float(m.group())
    if "'" in s or "ft" in s or "feet" in s:
        x *= 0.3048
    return x


def parse_int(value) -> Optional[int]:
    x = parse_number(value)
    return int(round(x)) if x is not None else None


def best_name(tags: dict) -> Optional[str]:
    """Pick a usable display name: English first, then local, then Nepali.

    OSM in Nepal mixes `name` (often Latin transliteration or Devanagari),
    `name:en` and `name:ne`. Downstream maps want ONE string column.
    """
    for k in ("name:en", "name", "int_name", "name:ne"):
        v = tags.get(k)
        if v and str(v).strip():
            return str(v).strip()
    return None


def _tag_flag(tags: dict, key: str) -> bool:
    """True when `key` is present with any value other than 'no'."""
    v = tags.get(key)
    return bool(v) and str(v).lower() != "no"


# =====================================================================
# 6. LAYER NORMALISERS  (raw -> clean schema)
# =====================================================================

# Lower number = more important road. Used later to prefer main roads.
ROAD_RANK = {
    "motorway": 1, "trunk": 1, "primary": 2, "secondary": 3, "tertiary": 4,
    "unclassified": 5, "residential": 5, "living_street": 5, "service": 6,
    "track": 7, "road": 7,
    "path": 8, "footway": 8, "bridleway": 8, "steps": 8, "cycleway": 8,
    "pedestrian": 8, "construction": 9,
}
PAVED = {"paved", "asphalt", "concrete", "concrete:plates", "concrete:lanes",
         "paving_stones", "sett", "metal", "bricks"}
UNPAVED = {"unpaved", "gravel", "fine_gravel", "dirt", "earth", "ground",
           "mud", "sand", "grass", "compacted", "pebblestone", "rock"}


def _base_highway(h: str) -> str:
    """'primary_link' -> 'primary' so ramps share their parent's rank."""
    return h[:-5] if h.endswith("_link") else h


def normalise_roads(raw: gpd.GeoDataFrame, utm: int) -> gpd.GeoDataFrame:
    """Roads with flood-relevant flags.

    Key columns:
        road_class       cleaned highway value ('primary_link' -> 'primary')
        rank             1 (trunk) ... 9 (construction), 99 unknown
        vehicle_ok       False for footpaths/steps (rank >= 8)
        is_bridge        bridge=* present and not 'no'
        is_tunnel        tunnel=* present and not 'no' (culverts included)
        is_ford          ford=yes OR highway=ford  (flood-prone crossing!)
        paved            True / False / None (unknown)
        width_m, lanes   numeric or None
        length_m         true length in metres (computed in UTM)
    """
    recs = []
    for t in raw["tags"]:
        h = t.get("highway", "")
        base = _base_highway(h)
        rank = ROAD_RANK.get(base, 99)
        surf = (t.get("surface") or "").lower()
        recs.append(
            {
                "name": best_name(t),
                "highway": h,
                "road_class": base,
                "rank": rank,
                "vehicle_ok": rank < 8,
                "is_bridge": _tag_flag(t, "bridge"),
                "is_tunnel": _tag_flag(t, "tunnel"),
                "is_ford": t.get("ford") == "yes" or h == "ford",
                "surface": surf or None,
                "paved": True if surf in PAVED else (False if surf in UNPAVED else None),
                "width_m": parse_number(t.get("width")),
                "lanes": parse_int(t.get("lanes")),
                "oneway": t.get("oneway") in ("yes", "1", "true"),
            }
        )
    out = pd.concat([raw[["uid", "osm_id"]].reset_index(drop=True),
                     pd.DataFrame(recs)], axis=1)
    g = gpd.GeoDataFrame(out, geometry=raw.geometry.values, crs=raw.crs)
    g["length_m"] = g.to_crs(utm).length.round(1)
    return g


def normalise_waterways(raw: gpd.GeoDataFrame, utm: int) -> gpd.GeoDataFrame:
    recs = [
        {
            "name": best_name(t),
            "waterway": t.get("waterway"),
            "is_river": t.get("waterway") == "river",
            "width_m": parse_number(t.get("width")),
            "intermittent": t.get("intermittent") == "yes",
        }
        for t in raw["tags"]
    ]
    out = pd.concat([raw[["uid", "osm_id"]].reset_index(drop=True),
                     pd.DataFrame(recs)], axis=1)
    g = gpd.GeoDataFrame(out, geometry=raw.geometry.values, crs=raw.crs)
    g["length_m"] = g.to_crs(utm).length.round(1)
    return g


def normalise_water_bodies(raw: gpd.GeoDataFrame, utm: int) -> gpd.GeoDataFrame:
    recs = [{"name": best_name(t), "water": t.get("water")} for t in raw["tags"]]
    out = pd.concat([raw[["uid", "osm_id"]].reset_index(drop=True),
                     pd.DataFrame(recs)], axis=1)
    g = gpd.GeoDataFrame(out, geometry=raw.geometry.values, crs=raw.crs)
    g["area_m2"] = g.to_crs(utm).area.round(1)
    return g


def normalise_buildings(raw: gpd.GeoDataFrame, utm: int) -> gpd.GeoDataFrame:
    """Building footprints. `area_m2` is handy for estimating damage extent."""
    recs = [
        {
            "name": best_name(t),
            "building": t.get("building"),
            "levels": parse_int(t.get("building:levels")),
        }
        for t in raw["tags"]
    ]
    out = pd.concat([raw[["uid", "osm_id"]].reset_index(drop=True),
                     pd.DataFrame(recs)], axis=1)
    g = gpd.GeoDataFrame(out, geometry=raw.geometry.values, crs=raw.crs)
    g["area_m2"] = g.to_crs(utm).area.round(1)
    return g


PLACE_RANK = {"city": 1, "town": 2, "village": 3, "suburb": 4, "hamlet": 4,
              "neighbourhood": 4, "locality": 5, "isolated_dwelling": 5}


def normalise_places(raw: gpd.GeoDataFrame, utm: int) -> gpd.GeoDataFrame:
    """Named settlements (the nodes you later test for 'cut off')."""
    recs = [
        {
            "name": best_name(t),
            "place": t.get("place"),
            "rank": PLACE_RANK.get(t.get("place"), 99),
            "population": parse_int(t.get("population")),
        }
        for t in raw["tags"]
    ]
    out = pd.concat([raw[["uid", "osm_id"]].reset_index(drop=True),
                     pd.DataFrame(recs)], axis=1)
    return gpd.GeoDataFrame(out, geometry=raw.geometry.values, crs=raw.crs)


FACILITY_CATEGORY = {
    "hospital": "health", "clinic": "health", "doctors": "health",
    "pharmacy": "health", "health_post": "health",
    "school": "education", "college": "education",
    "university": "education", "kindergarten": "education",
    "police": "emergency", "fire_station": "emergency",
    "shelter": "shelter",
}


def normalise_facilities(raw: gpd.GeoDataFrame, utm: int) -> gpd.GeoDataFrame:
    """Critical facilities collapsed into 4 categories:
    health / education / emergency / shelter.

    `healthcare=*` is checked too, because many Nepali health posts are
    tagged only with healthcare=* and no amenity=*.
    """
    recs = []
    for t in raw["tags"]:
        a = t.get("amenity") or ""
        cat = FACILITY_CATEGORY.get(a) or (
            "health" if t.get("healthcare") else None
        )
        recs.append(
            {
                "name": best_name(t),
                "amenity": a or t.get("healthcare"),
                "category": cat,
                "beds": parse_int(t.get("beds")),
            }
        )
    out = pd.concat([raw[["uid", "osm_id"]].reset_index(drop=True),
                     pd.DataFrame(recs)], axis=1)
    g = gpd.GeoDataFrame(out, geometry=raw.geometry.values, crs=raw.crs)
    return g[g["category"].notna()].reset_index(drop=True)


# =====================================================================
# 7. LAYER REGISTRY  (what to download + how to clean it)
# =====================================================================


@dataclass(frozen=True)
class LayerSpec:
    name: str
    body: str   # Overpass QL selectors; the bbox is added by build_query
    kind: str   # "line" | "polygon" | "point"  (TRICK #4)
    normaliser: Callable[[gpd.GeoDataFrame, int], gpd.GeoDataFrame]


LAYERS: dict[str, LayerSpec] = {
    "roads": LayerSpec(
        "roads",
        '  way["highway"]["highway"!~"^(proposed|razed|abandoned|platform|'
        'bus_stop|elevator|corridor)$"];',
        "line", normalise_roads),
    "waterways": LayerSpec(
        "waterways",
        '  way["waterway"~"^(river|stream|canal|drain|ditch)$"];',
        "line", normalise_waterways),
    "water_bodies": LayerSpec(
        "water_bodies",
        '  way["natural"="water"];\n  relation["natural"="water"];',
        "polygon", normalise_water_bodies),
    "buildings": LayerSpec(
        "buildings",
        '  way["building"];\n  relation["building"];',
        "polygon", normalise_buildings),
    "places": LayerSpec(
        "places",
        '  node["place"~"^(city|town|village|hamlet|suburb|locality|'
        'isolated_dwelling|neighbourhood)$"];',
        "point", normalise_places),
    "facilities": LayerSpec(
        "facilities",
        '  nwr["amenity"~"^(hospital|clinic|doctors|pharmacy|school|college|'
        'university|kindergarten|police|fire_station|shelter)$"];\n'
        '  nwr["healthcare"];',
        "point", normalise_facilities),
}


# =====================================================================
# 8. POST-PROCESSING & QUALITY CONTROL
# =====================================================================


def filter_to_aoi(gdf: gpd.GeoDataFrame, bbox) -> gpd.GeoDataFrame:
    """TRICK #6: keep whole features that touch the AOI; do NOT clip.

    Clipping would slice a bridge or road at the bbox edge, producing
    dangling half-features that break connectivity analysis. With
    intersects() a road that merely enters the box is kept intact.
    """
    if gdf.empty:
        return gdf
    return gdf[gdf.intersects(box(*bbox))].reset_index(drop=True)


def derive_bridges(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Bridges are the single most flood-critical asset, so they get their
    own layer (same schema as roads, just filtered)."""
    return roads[roads["is_bridge"]].reset_index(drop=True)


def quality_report(layers: dict[str, gpd.GeoDataFrame]) -> dict:
    """Cheap sanity checks printed and saved to the manifest."""
    rep = {}
    for name, g in layers.items():
        info = {"count": int(len(g))}
        if len(g):
            info["invalid_geometries"] = int((~g.geometry.is_valid).sum())
            info["duplicate_uids"] = int(g["uid"].duplicated().sum())
            info["unnamed_pct"] = (
                round(100 * float(g["name"].isna().mean()), 1)
                if "name" in g else None
            )
        rep[name] = info
    if "roads" in layers and len(layers["roads"]):
        r = layers["roads"]
        rep["roads"].update(
            bridges=int(r["is_bridge"].sum()),
            fords=int(r["is_ford"].sum()),
            total_km=round(float(r["length_m"].sum()) / 1000, 1),
        )
    return rep


# =====================================================================
# 9. SAVE / LOAD
# =====================================================================


def save_geopackage(layers: dict[str, gpd.GeoDataFrame], path: Path):
    """Write every non-empty layer into ONE GeoPackage file (one layer per
    table). A single file is easier to hand to teammates than 7 shapefiles,
    and unlike shapefiles it keeps full column names and types."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()  # GPKG appends layers; start clean to avoid stale layers
    for name, g in layers.items():
        if g.empty:
            log.warning("layer %s is empty, not written", name)
            continue
        g.to_file(path, layer=name, driver="GPKG")


def load_layers(gpkg: str | Path) -> dict[str, gpd.GeoDataFrame]:
    """Convenience loader for downstream parts: {layer_name: GeoDataFrame}."""
    gpkg = Path(gpkg)
    names = gpd.list_layers(gpkg)["name"].tolist()
    return {n: gpd.read_file(gpkg, layer=n) for n in names}


# =====================================================================
# 10. ORCHESTRATION
# =====================================================================


def acquire_and_normalise(cfg: Config, layer_names: list[str] | None = None):
    """The whole of Part 01 in one function.

    For each requested layer:
        download (tiled, cached) -> geometry -> normalise -> AOI filter
    then derive bridges, run QC, save GeoPackage + manifest.json.
    Returns (layers_dict, manifest_dict).
    """
    names = layer_names or list(LAYERS)
    utm = utm_epsg_for_bbox(cfg.bbox)
    client = OverpassClient(cfg)
    log.info("AOI %s | metric CRS EPSG:%d", cfg.bbox, utm)

    layers: dict[str, gpd.GeoDataFrame] = {}
    for name in names:
        spec = LAYERS[name]
        log.info("Layer '%s'", name)
        elements = client.fetch(spec.body)
        raw = elements_to_raw_gdf(elements, spec.kind)
        if raw.empty:
            layers[name] = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            continue
        clean = spec.normaliser(raw, utm)
        layers[name] = filter_to_aoi(clean, cfg.bbox)

    if "roads" in layers and len(layers["roads"]):
        layers["bridges"] = derive_bridges(layers["roads"])

    gpkg = cfg.out_dir / "osm_normalised.gpkg"
    save_geopackage(layers, gpkg)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "osm_data_timestamp": client.osm_timestamp,
        "bbox_wsen": cfg.bbox,
        "storage_crs": "EPSG:4326",
        "metric_crs": f"EPSG:{utm}",
        "geopackage": str(gpkg),
        "quality": quality_report(layers),
    }
    (cfg.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Done. Manifest:\n%s", json.dumps(manifest["quality"], indent=2))
    return layers, manifest


def main():
    p = argparse.ArgumentParser(description="FloodResQ Part 01: OSM acquisition")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    p.add_argument("--layers", nargs="+", choices=list(LAYERS))
    p.add_argument("--out", type=Path, default=Path("data/part01"))
    p.add_argument("--refresh", action="store_true", help="ignore the cache")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = Config(out_dir=a.out, cache_dir=a.out / "cache", refresh=a.refresh)
    if a.bbox:
        cfg.bbox = tuple(a.bbox)
    acquire_and_normalise(cfg, a.layers)


if __name__ == "__main__":
    main()
