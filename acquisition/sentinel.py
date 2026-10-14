"""
acquisition/sentinel.py  —  PART 1: Acquisition & Scene-Pair Selection
=======================================================================

WHAT THIS FILE DOES (one paragraph)
-----------------------------------
Given an AOI (bounding box) and an event date, find on the Copernicus Data
Space (CDSE) the Sentinel-1 radar scenes that best describe "before" and
"after" the event, make sure both come from the SAME orbit track, download
them, and write ``scene_pair.json`` saying exactly what was chosen and why.
It does NO image processing (that is Part 2).

PUBLIC ENTRY POINT
------------------
    pair = get_satellite_data((85.1, 27.9, 85.5, 28.3), "2026-08-26")
    # (that bbox is purely illustrative, not an official AOI)

    pair.before.zip_paths   ->  list of zip files for the "before" image
    pair.after.zip_paths    ->  list of zip files for the "after" image
    pair.warnings           ->  human-readable caveats (goes into the report)

DEPENDENCIES: ``requests`` and ``numpy`` only (no shapely/pyproj; see TRICK 3).
Credentials: environment variables CDSE_USERNAME and CDSE_PASSWORD (needed
only to DOWNLOAD; searching the catalogue needs no account).

MAP OF THE FILE (read top to bottom, each block is independent)
---------------------------------------------------------------
  1. Errors ................ clear exception types the dashboard can show
  2. Config ................ every tunable number in one place
  3. AOI + CoverageGrid .... "how much of my area does this scene cover?"
  4. Scene / Acquisition ... data objects (one zip file / one satellite pass)
  5. Catalogue parsing ..... raw CDSE JSON  ->  Scene objects
  6. Pair selection ........ the decision logic (pure functions, no network)
  7. CDSEClient ............ search + download (all the network code)
  8. get_satellite_data .... glues everything together
  9. CLI ................... ``python -m acquisition.sentinel --help``

NON-OBVIOUS TRICKS USED HERE (search for "TRICK n" in the code)
---------------------------------------------------------------
  TRICK 1  Group slices into passes ("datatakes") using the file NAME, so an
           AOI straddling two slices of one pass still counts as one image.
  TRICK 2  One wide search, then "relax" in memory (a ladder), instead of
           several HTTP round-trips.
  TRICK 3  Coverage is measured by sampling a grid of points inside the AOI,
           so unions/differences of scene footprints become simple boolean
           array operations (no polygon-clipping library needed).
  TRICK 4  Greedy minimal slice cover: never download a 1-2 GB slice that only
           adds a sliver of the AOI.
  TRICK 5  Lexicographic ranking with coverage BUCKETED, so 97 % vs 99 %
           coverage does not override more important preferences.
  TRICK 6  Keep the best K candidates per side, not just the nearest scene:
           the nearest scene may only cover a corner of the AOI.
  TRICK 7  Event-day ambiguity: a scene on the event date is flagged, because
           a date alone cannot say if it was before or after the flood.
  TRICK 8  Catalogue search needs no login; only downloads do. So the choice
           can be previewed instantly (``download=False``).
  TRICK 9  Redirects handled by hand so the Authorization header survives the
           hop to the download server (requests drops it on cross-host hops).
  TRICK 10 Download to ``.part`` + HTTP Range resume + token refresh on 401 +
           MD5 check + atomic rename: a half-file can never look complete.
  TRICK 11 Search results cached on disk; if the network dies during a demo,
           the stale cache is used instead of failing.
  TRICK 12 Selection code is pure (no network) so it is unit-tested offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import requests

log = logging.getLogger("acquisition.sentinel")


# ============================================================================
# 1. ERRORS
# ============================================================================
# Distinct types so Teammate 4's dashboard can show a friendly message per case
# instead of a stack trace.

class AcquisitionError(Exception):
    """Base class for everything this module raises on purpose."""


class InvalidInputError(AcquisitionError, ValueError):
    """Bad bbox / date / AOI too big."""


class AuthError(AcquisitionError):
    """Missing or wrong CDSE credentials."""


class CatalogueError(AcquisitionError):
    """Catalogue search failed (network, server error)."""


class DownloadError(AcquisitionError):
    """Download failed or file is corrupt."""


class NoValidPairError(AcquisitionError):
    """No same-track before/after pair exists. ``reasons`` explains why,
    track by track, so the user learns what to change (date, area...)."""

    def __init__(self, message: str, reasons: Sequence[str] = ()):
        super().__init__(message)
        self.message = message
        self.reasons = list(reasons)

    def __str__(self) -> str:
        if not self.reasons:
            return self.message
        return self.message + "\n" + "\n".join(f"  - {r}" for r in self.reasons)


class _Transient(Exception):
    """Internal: 'try again' (HTTP 429/5xx, expired token, incomplete file)."""


# ============================================================================
# 2. CONFIG
# ============================================================================

@dataclass(frozen=True)
class Relaxation:
    """One rung of the 'relaxation ladder' (see TRICK 2).

    window_days   how far from the event date we accept scenes
    min_coverage  fraction of the AOI that BOTH images must cover
    label         text added to the warnings when this rung is the one used
    """
    window_days: int
    min_coverage: float
    label: str


@dataclass
class AcquisitionConfig:
    # --- CDSE endpoints (verify against the current CDSE docs if they change)
    catalogue_url: str = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    token_url: str = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
                      "protocol/openid-connect/token")
    download_url: str = "https://download.dataspace.copernicus.eu/odata/v1"
    collection: str = "SENTINEL-1"
    # IW = Interferometric Wide swath, GRDH = ground-range detected, high res
    # (~10 m pixels): the standard land product for flood mapping.
    product_type: str = "IW_GRDH_1S"

    # --- selection behaviour. Tried in order; first rung that works wins.
    ladder: tuple = (
        Relaxation(40, 0.95, "standard"),
        Relaxation(80, 0.95, "wider time window"),
        Relaxation(80, 0.60, "partial AOI coverage"),
    )
    top_k: int = 3                 # candidates kept per side per track (TRICK 6)
    min_slice_gain: float = 0.01   # a slice must add >=1 % of the AOI (TRICK 4)
    coverage_bucket: float = 0.05  # coverage rounded to 5 % steps (TRICK 5)
    long_gap_warn_days: int = 24   # warn if before->after gap exceeds this
    max_aoi_km2: float = 2500.0    # ~50 x 50 km. Bigger => ask user to tile
    grid_size: int = 256           # coverage sampling grid is N x N points

    # --- search
    search_page_size: int = 100
    search_max_pages: int = 20
    search_cache_dir: Optional[Path] = Path("data/cache/search")
    search_cache_ttl_h: float = 24.0

    # --- network behaviour
    max_retries: int = 5
    retry_base_delay_s: float = 2.0
    timeout_s: float = 60.0
    chunk_bytes: int = 1 << 20     # 1 MiB streaming chunks
    verify_md5: bool = True


# ============================================================================
# 3. AOI + COVERAGE GRID
# ============================================================================

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class AOI:
    """Area of interest as a bbox in EPSG:4326: (min_lon, min_lat, max_lon, max_lat).

    Order reminder: GeoJSON/Shapely use (lon, lat); Folium uses [lat, lon].
    This class always uses (lon, lat).
    """
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def __post_init__(self):
        if not all(map(math.isfinite, self.as_tuple())):
            raise InvalidInputError("bbox contains NaN/inf")
        if not (-180 <= self.min_lon < self.max_lon <= 180):
            raise InvalidInputError(
                f"Invalid longitudes {self.min_lon}..{self.max_lon}. Expected "
                "(min_lon, min_lat, max_lon, max_lat) with min < max, in degrees.")
        if not (-90 <= self.min_lat < self.max_lat <= 90):
            raise InvalidInputError(
                f"Invalid latitudes {self.min_lat}..{self.max_lat}. Did you "
                "swap lat and lon? Expected (min_lon, min_lat, max_lon, max_lat).")

    @classmethod
    def from_bbox(cls, bbox: Sequence[float]) -> "AOI":
        if len(bbox) != 4:
            raise InvalidInputError("bbox must have 4 numbers")
        return cls(*map(float, bbox))

    def as_tuple(self) -> tuple:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    def area_km2(self) -> float:
        """Exact area of a lon/lat box on a sphere:
        R^2 * dLon(rad) * (sin(lat2) - sin(lat1)).  Accurate to ~0.3 % vs the
        WGS84 ellipsoid: plenty for a size cap, and needs no pyproj."""
        d_lon = math.radians(self.max_lon - self.min_lon)
        d_sin = math.sin(math.radians(self.max_lat)) - math.sin(math.radians(self.min_lat))
        return EARTH_RADIUS_KM ** 2 * d_lon * d_sin

    def odata_polygon(self) -> str:
        """WKT ring for the OData 'Intersects' filter (lon lat order, closed)."""
        a, b, c, d = self.min_lon, self.min_lat, self.max_lon, self.max_lat
        return f"POLYGON(({a:.5f} {b:.5f},{c:.5f} {b:.5f},{c:.5f} {d:.5f},{a:.5f} {d:.5f},{a:.5f} {b:.5f}))"

    def key(self) -> str:
        """Short stable id for this AOI (used in cache/file names)."""
        return hashlib.sha1(",".join(f"{v:.5f}" for v in self.as_tuple()).encode()).hexdigest()[:10]


def _points_in_ring(px: np.ndarray, py: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon test.

    Casts a ray to the right from every point and flips an 'inside' flag each
    time the ray crosses a polygon edge. ``px``/``py`` can be whole 2-D grids,
    so one call tests 65 000 points at once.
    """
    x, y = ring[:, 0], ring[:, 1]
    inside = np.zeros(px.shape, dtype=bool)
    j = len(ring) - 1
    with np.errstate(all="ignore"):          # horizontal edges divide by ~0; harmless
        for i in range(len(ring)):
            xi, yi, xj, yj = x[i], y[i], x[j], y[j]
            crosses = ((yi > py) != (yj > py)) & (
                px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi)
            inside ^= crosses
            j = i
    return inside


class CoverageGrid:
    """TRICK 3 — measure coverage by sampling instead of polygon clipping.

    We lay an N x N grid of points over the AOI. A scene "covers" the points
    that fall inside its footprint, stored as a boolean array. Then:
        union of slices      ->  a | b
        what is still missing ->  ~covered
        coverage fraction     ->  mask.mean()
    No shapely, no invalid-geometry errors, trivially fast, and accurate to
    about 1/N (~0.4 % at N=256), far below our 95 % thresholds.

    Distortion note: the grid is uniform in degrees, so areas are weighted by
    1 instead of cos(lat). Over a 40 km AOI cos(lat) changes < 0.5 %, so the
    ratios are effectively exact.
    """

    def __init__(self, aoi: AOI, n: int = 256):
        self.aoi = aoi
        # cell CENTRES, not corners, so edges are not over-counted
        lons = aoi.min_lon + (np.arange(n) + 0.5) / n * (aoi.max_lon - aoi.min_lon)
        lats = aoi.min_lat + (np.arange(n) + 0.5) / n * (aoi.max_lat - aoi.min_lat)
        self.px, self.py = np.meshgrid(lons, lats)
        self._cache: dict = {}

    def mask(self, scene: "Scene") -> np.ndarray:
        """Boolean n x n array: which AOI sample points does this scene cover?
        Cached per scene (a scene is tested against the same grid many times)."""
        m = self._cache.get(scene.product_id)
        if m is None:
            m = np.zeros(self.px.shape, dtype=bool)
            for ring in scene.rings:
                m |= _points_in_ring(self.px, self.py, np.asarray(ring, dtype=float))
            self._cache[scene.product_id] = m
        return m

    def union(self, scenes: Iterable["Scene"]) -> np.ndarray:
        out = np.zeros(self.px.shape, dtype=bool)
        for s in scenes:
            out |= self.mask(s)
        return out

    @staticmethod
    def fraction(mask: np.ndarray) -> float:
        return float(mask.mean())


# ============================================================================
# 4. DATA OBJECTS
# ============================================================================

@dataclass
class Scene:
    """ONE Sentinel-1 product = one zip file = one 'slice' of a satellite pass.

    A pass (continuous recording) is chopped by ESA into ~25 s slices. An AOI
    near a slice boundary needs two slices; that is why ``Acquisition`` exists.
    """
    product_id: str                 # CDSE UUID (used for downloading)
    name: str                       # e.g. S1A_IW_GRDH_1SDV_20260826T...SAFE
    platform: str                   # S1A / S1C ...
    start: datetime                 # sensing start, UTC, timezone-aware
    end: datetime
    rel_orbit: int                  # relative orbit = the "track" number
    direction: str                  # ASCENDING / DESCENDING
    rings: list                     # footprint outer rings [[ [lon,lat],... ], ...]
    abs_orbit: Optional[int] = None
    datatake_id: Optional[str] = None
    size_bytes: Optional[int] = None
    md5: Optional[str] = None
    online: bool = True             # False = long-term archive, may need staging
    local_path: Optional[str] = None

    @property
    def track_key(self) -> tuple:
        """Scenes are comparable pixel-by-pixel ONLY if this key is equal.
        Different relative orbit or direction = different viewing angle."""
        return (self.rel_orbit, self.direction)

    @property
    def datatake_key(self) -> tuple:
        """Identifies one continuous pass (TRICK 1). Slices of the same pass
        share platform + absolute orbit + datatake id. If the name could not be
        parsed we fall back to platform + calendar day (safe because we only
        ever group within one track)."""
        if self.abs_orbit is not None and self.datatake_id:
            return (self.platform, self.abs_orbit, self.datatake_id)
        return (self.platform, self.start.date().isoformat())

    @property
    def stem(self) -> str:
        return self.name[:-5] if self.name.endswith(".SAFE") else self.name

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id, "name": self.name,
            "platform": self.platform,
            "start_utc": self.start.isoformat(), "end_utc": self.end.isoformat(),
            "relative_orbit": self.rel_orbit, "direction": self.direction,
            "absolute_orbit": self.abs_orbit, "datatake_id": self.datatake_id,
            "size_bytes": self.size_bytes, "md5": self.md5, "online": self.online,
            "local_path": self.local_path, "footprint_rings": self.rings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        """Inverse of ``to_dict`` (used by Part 2 to reload scene_pair.json)."""
        return cls(
            product_id=d["product_id"], name=d["name"], platform=d["platform"],
            start=_parse_dt(d["start_utc"]), end=_parse_dt(d["end_utc"]),
            rel_orbit=int(d["relative_orbit"]), direction=d["direction"],
            rings=d["footprint_rings"], abs_orbit=d.get("absolute_orbit"),
            datatake_id=d.get("datatake_id"), size_bytes=d.get("size_bytes"),
            md5=d.get("md5"), online=d.get("online", True),
            local_path=d.get("local_path"))


@dataclass
class Acquisition:
    """One satellite PASS over the AOI = one or more slices of the same datatake.

    This is what 'the before image' really is. Usually 1 slice; 2 when the AOI
    straddles a slice boundary (Part 2 then has to stitch them, e.g. with
    SNAP's slice-assembly step).
    """
    slices: list

    @property
    def start(self) -> datetime:
        return min(s.start for s in self.slices)

    @property
    def date(self) -> date:
        return self.start.date()          # UTC calendar day

    @property
    def platform(self) -> str:
        return self.slices[0].platform

    @property
    def track_key(self) -> tuple:
        return self.slices[0].track_key

    @property
    def zip_paths(self) -> list:
        return [s.local_path for s in self.slices if s.local_path]

    def cover_mask(self, grid: CoverageGrid) -> np.ndarray:
        return grid.union(self.slices)

    def trimmed(self, grid: CoverageGrid, min_gain: float) -> "Acquisition":
        """TRICK 4 — greedy minimal cover.

        The search returns every slice that merely TOUCHES the AOI, including
        ones overlapping only a sliver. Each is a 1-2 GB download. So: keep
        picking the slice that adds the most still-uncovered AOI, and stop
        when the best remaining slice adds less than ``min_gain`` (default 1 %).
        This is the classic greedy set-cover heuristic; with <= 3 slices it is
        effectively optimal.
        """
        pool = list(self.slices)
        covered = np.zeros(grid.px.shape, dtype=bool)
        chosen: list = []
        while pool:
            gains = [(float((grid.mask(s) & ~covered).mean()), s) for s in pool]
            gain, best = max(gains, key=lambda g: g[0])
            if gain <= 0:
                break
            if chosen and gain < min_gain:     # always keep the first one
                break
            chosen.append(best)
            pool.remove(best)
            covered |= grid.mask(best)
        return Acquisition(sorted(chosen or self.slices[:1], key=lambda s: s.start))

    def to_dict(self) -> dict:
        return {"date_utc": self.date.isoformat(), "platform": self.platform,
                "n_slices": len(self.slices),
                "slices": [s.to_dict() for s in self.slices]}

    @classmethod
    def from_dict(cls, d: dict) -> "Acquisition":
        return cls([Scene.from_dict(x) for x in d["slices"]])


@dataclass
class ScenePair:
    """The final answer of Part 1 (also serialised to scene_pair.json)."""
    aoi: AOI
    event_date: date
    before: Acquisition
    after: Acquisition
    coverage: float                         # AOI fraction covered by BOTH
    relaxation: str                         # which ladder rung produced this
    before2: Optional[Acquisition] = None   # optional: previous cycle, same track
    warnings: list = field(default_factory=list)

    @property
    def track(self) -> tuple:
        return self.before.track_key

    @property
    def gap_days(self) -> int:
        return (self.after.date - self.before.date).days

    def all_scenes(self) -> list:
        out = list(self.before.slices) + list(self.after.slices)
        if self.before2:
            out += list(self.before2.slices)
        return out

    def to_dict(self) -> dict:
        d = {
            "event_date": self.event_date.isoformat(),
            "aoi_bbox_4326": list(self.aoi.as_tuple()),
            "track": {"relative_orbit": self.track[0], "direction": self.track[1]},
            "gap_days": self.gap_days,
            "aoi_coverage": round(self.coverage, 4),
            "relaxation": self.relaxation,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "before2": self.before2.to_dict() if self.before2 else None,
            "warnings": self.warnings,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return d

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "ScenePair":
        return cls(
            aoi=AOI.from_bbox(d["aoi_bbox_4326"]),
            event_date=date.fromisoformat(d["event_date"]),
            before=Acquisition.from_dict(d["before"]),
            after=Acquisition.from_dict(d["after"]),
            coverage=float(d["aoi_coverage"]), relaxation=d["relaxation"],
            before2=Acquisition.from_dict(d["before2"]) if d.get("before2") else None,
            warnings=list(d.get("warnings", [])))

    @classmethod
    def load_json(cls, path) -> "ScenePair":
        """Reload what ``save_json`` wrote, so Part 2 can run without Part 1's network step."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    def summary(self) -> str:
        lines = [
            f"Track {self.track[0]} {self.track[1]}   (event {self.event_date})",
            f"  before : {self.before.date}  {self.before.platform}  "
            f"{len(self.before.slices)} slice(s)",
            f"  after  : {self.after.date}  {self.after.platform}  "
            f"{len(self.after.slices)} slice(s)",
            f"  gap    : {self.gap_days} days     AOI coverage: {self.coverage:.1%}"
            f"     rung: {self.relaxation}",
        ]
        if self.before2:
            lines.append(f"  before2: {self.before2.date}  {self.before2.platform}")
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


# ============================================================================
# 5. CATALOGUE PARSING   (raw CDSE JSON -> Scene)
# ============================================================================

# Sentinel-1 file names are packed with metadata:
#   S1A_IW_GRDH_1SDV_20260826T120000_20260826T120025_055000_06B000_ABCD.SAFE
#   |   |  |    ||||  start           stop            abs.orbit datatake uid
#   |   |  |    |||+- V : first polarisation V  (with D => VV+VH)
#   |   |  |    ||+-- D : Dual polarisation (S = single)
#   |   |  |    |+--- S : Standard product
#   |   |  |    +---- level 1
#   |   |  +--------- product type GRDH
#   |   +------------ mode IW
#   +---------------- platform
_NAME_RE = re.compile(
    r"^(?P<platform>S1[A-D])_(?P<mode>\w{2})_(?P<ptype>\w{4})_1S(?P<cls>[SD])(?P<pol>[VH])_"
    r"(?P<t0>\d{8}T\d{6})_(?P<t1>\d{8}T\d{6})_(?P<abs>\d{6})_(?P<dt>[0-9A-F]{6})_")


def _parse_dt(text: str) -> datetime:
    """ISO text -> timezone-aware UTC datetime. Always UTC: mixing naive and
    aware datetimes crashes, and local time would shift the event day."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _rings_from_wkt(text: str) -> list:
    """Fallback footprint parser for strings like
    "geography'SRID=4326;MULTIPOLYGON (((x y, x y, ...)))'". Every innermost
    parenthesised group with commas is a ring. (Holes would be misread as
    extra outer rings; Sentinel-1 footprints have none.)"""
    rings = []
    for grp in re.findall(r"\(([^()]+)\)", text):
        if "," not in grp:
            continue
        pts = [list(map(float, p.split()[:2])) for p in grp.split(",") if p.strip()]
        if len(pts) >= 3:
            rings.append(pts)
    return rings


def _rings_from_item(item: dict) -> list:
    geo = item.get("GeoFootprint")
    if isinstance(geo, dict):
        if geo.get("type") == "Polygon":
            return [geo["coordinates"][0]]
        if geo.get("type") == "MultiPolygon":
            return [poly[0] for poly in geo["coordinates"]]
    fp = item.get("Footprint")
    return _rings_from_wkt(fp) if isinstance(fp, str) else []


def parse_product(item: dict) -> Optional[Scene]:
    """Convert one OData product dict into a ``Scene``.

    Returns None (and the scene is silently skipped) when it is not exactly
    what we need: IW GRDH, dual-pol VV+VH, with a usable footprint and track.
    """
    name = item.get("Name", "")
    attrs = {a.get("Name"): a.get("Value") for a in item.get("Attributes", [])}

    m = _NAME_RE.match(name)
    if m:
        # TRICK (small): the file name tells us the polarisation, so we do not
        # depend on a catalogue attribute whose spelling could change.
        if not (m["mode"] == "IW" and m["ptype"] == "GRDH"
                and m["cls"] == "D" and m["pol"] == "V"):
            return None
        platform, abs_orbit, datatake = m["platform"], int(m["abs"]), m["dt"]
    else:
        if attrs.get("polarisationChannels") not in ("VV&VH", "VV+VH"):
            return None
        serial = attrs.get("platformSerialIdentifier")
        platform = f"S1{serial}" if serial else "S1?"
        abs_orbit, datatake = None, None

    try:
        rel_orbit = int(attrs["relativeOrbitNumber"])
        direction = str(attrs["orbitDirection"]).upper()
        start = _parse_dt(item["ContentDate"]["Start"])
        end = _parse_dt(item["ContentDate"]["End"])
    except (KeyError, ValueError, TypeError):
        return None

    rings = _rings_from_item(item)
    if not rings:
        return None

    md5 = None
    for c in item.get("Checksum") or []:
        if str(c.get("Algorithm", "")).upper() == "MD5":
            md5 = str(c.get("Value", "")).lower() or None

    return Scene(
        product_id=item["Id"], name=name, platform=platform, start=start, end=end,
        rel_orbit=rel_orbit, direction=direction, rings=rings,
        abs_orbit=abs_orbit, datatake_id=datatake,
        size_bytes=int(item["ContentLength"]) if item.get("ContentLength") else None,
        md5=md5, online=bool(item.get("Online", True)))


def build_odata_filter(aoi: AOI, start: date, end: date, cfg: AcquisitionConfig) -> str:
    """The OData $filter string. Polarisation is checked client-side (see
    parse_product) to keep this query simple and robust."""
    return " and ".join([
        f"Collection/Name eq '{cfg.collection}'",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi.odata_polygon()}')",
        f"ContentDate/Start gt {start.isoformat()}T00:00:00.000Z",
        f"ContentDate/Start lt {end.isoformat()}T23:59:59.999Z",
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"and att/OData.CSC.StringAttribute/Value eq '{cfg.product_type}')",
    ])


# ============================================================================
# 6. PAIR SELECTION   (pure logic: no network, fully unit-tested)  — TRICK 12
# ============================================================================

def group_acquisitions(scenes: Iterable[Scene]) -> list:
    """TRICK 1 — merge slices of the same pass into one ``Acquisition``.

    Why not group by date? Two different passes can share a date, and slices of
    one pass can straddle midnight. The datatake key (platform + absolute orbit
    + datatake id, all read from the file name) identifies a pass exactly.
    """
    buckets: dict = {}
    for s in scenes:
        buckets.setdefault((s.track_key, s.datatake_key), []).append(s)
    acqs = [Acquisition(sorted(v, key=lambda s: s.start)) for v in buckets.values()]
    return sorted(acqs, key=lambda a: a.start)


def group_by_track(acqs: Iterable[Acquisition]) -> dict:
    """{(relative_orbit, direction): [Acquisition, ...] sorted by time}.
    THE safety rule of this whole module: we only ever pair acquisitions that
    live inside the same bucket here."""
    out: dict = {}
    for a in acqs:
        out.setdefault(a.track_key, []).append(a)
    return {k: sorted(v, key=lambda a: a.start) for k, v in out.items()}


@dataclass
class _Candidate:
    before: Acquisition
    after: Acquisition
    coverage: float
    event_date: date
    bucket: float

    @property
    def gap_days(self) -> int:
        return (self.after.date - self.before.date).days

    def rank(self) -> tuple:
        """TRICK 5 — lexicographic ranking, smaller is better.

        1. coverage bucket  (higher is better; BUCKETED so 97 % ~ 99 %)
        2. platform mismatch (same satellite is the cleaner comparison)
        3. gap before->after (shorter = less seasonal/soil-moisture change)
        4. delay of 'after' relative to the event (sooner is more useful)

        Because it is a tuple, a later criterion only breaks ties of earlier
        ones. Without bucketing, 99.1 % vs 97.9 % coverage would always beat
        'same platform', which is not what we want.
        """
        return (-round(self.coverage / self.bucket),
                int(self.before.platform != self.after.platform),
                self.gap_days,
                (self.after.date - self.event_date).days)


def _best_candidate(by_track: dict, grid: CoverageGrid, event_date: date,
                    relax: Relaxation, cfg: AcquisitionConfig) -> Optional[_Candidate]:
    """Best pair on ANY track for one ladder rung, or None."""
    lo = event_date - timedelta(days=relax.window_days)
    hi = event_date + timedelta(days=relax.window_days)
    best: Optional[_Candidate] = None

    for _key, acqs in by_track.items():
        # TRICK 7: 'before' is strictly earlier than the event DAY; 'after' is
        # the event day or later (and flagged later if it IS the event day).
        befores = [a for a in acqs if lo <= a.date < event_date]
        afters = [a for a in acqs if event_date <= a.date <= hi]
        # TRICK 6: newest K befores and oldest K afters, not only the nearest.
        befores = sorted(befores, key=lambda a: a.start, reverse=True)[:cfg.top_k]
        afters = sorted(afters, key=lambda a: a.start)[:cfg.top_k]

        for b, a in product(befores, afters):
            bt = b.trimmed(grid, cfg.min_slice_gain)
            at = a.trimmed(grid, cfg.min_slice_gain)
            cov = grid.fraction(bt.cover_mask(grid) & at.cover_mask(grid))
            if cov < relax.min_coverage:
                continue
            cand = _Candidate(bt, at, cov, event_date, cfg.coverage_bucket)
            if best is None or cand.rank() < best.rank():
                best = cand
    return best


def _find_before2(by_track: dict, grid: CoverageGrid, chosen: _Candidate,
                  relax: Relaxation, cfg: AcquisitionConfig) -> Optional[Acquisition]:
    """Optional second pre-event image: the latest earlier pass on the SAME
    track (some flood datasets train on 2 pre-event + 1 post-event images)."""
    options = [a for a in by_track.get(chosen.before.track_key, [])
               if a.date < chosen.before.date]
    options.sort(key=lambda a: a.start, reverse=True)
    for a in options[:cfg.top_k]:
        t = a.trimmed(grid, cfg.min_slice_gain)
        if grid.fraction(t.cover_mask(grid)) >= relax.min_coverage:
            return t
    return None


def _diagnose(by_track: dict, grid: CoverageGrid, event_date: date,
              cfg: AcquisitionConfig) -> list:
    """Explain a failure track by track. A message like 'track 41 has an
    after-scene but no before-scene' tells the user what to change."""
    max_w = max(r.window_days for r in cfg.ladder)
    if not by_track:
        return [f"No Sentinel-1 IW GRD dual-pol (VV+VH) scene intersects the AOI within "
                f"+/-{max_w} days of {event_date}. Check the AOI location and that "
                "Sentinel-1 data exists for that period."]
    lines = []
    for (orbit, direction), acqs in sorted(by_track.items()):
        tag = f"track {orbit} {direction}"
        bs = [a for a in acqs if a.date < event_date]
        as_ = [a for a in acqs if a.date >= event_date]
        if not bs:
            lines.append(f"{tag}: no scene before {event_date}")
        elif not as_:
            lines.append(f"{tag}: no scene on/after {event_date} (latest is {acqs[-1].date}); "
                         "post-event data may not be published yet")
        else:
            best = max(grid.fraction(b.cover_mask(grid) & a.cover_mask(grid))
                       for b in bs[-cfg.top_k:] for a in as_[:cfg.top_k])
            lines.append(f"{tag}: has before and after, but best AOI coverage by both "
                         f"is only {best:.0%}")
    return lines


def select_pair(scenes: Iterable[Scene], aoi: AOI, event_date: date,
                cfg: Optional[AcquisitionConfig] = None,
                want_before2: bool = False) -> ScenePair:
    """THE decision function: scenes in, ``ScenePair`` out (or NoValidPairError).

    TRICK 2 — relaxation ladder: ``scenes`` came from ONE wide search. We now
    walk the ladder (strict -> looser) filtering in memory. The first rung with
    a valid pair wins and, if it is not the strictest, a warning says so.
    """
    cfg = cfg or AcquisitionConfig()
    grid = CoverageGrid(aoi, cfg.grid_size)
    by_track = group_by_track(group_acquisitions(scenes))

    for i, relax in enumerate(cfg.ladder):
        cand = _best_candidate(by_track, grid, event_date, relax, cfg)
        if cand is None:
            continue
        pair = ScenePair(aoi=aoi, event_date=event_date, before=cand.before,
                         after=cand.after, coverage=cand.coverage,
                         relaxation=relax.label)
        if want_before2:
            pair.before2 = _find_before2(by_track, grid, cand, relax, cfg)
            if pair.before2 is None:
                pair.warnings.append("A second pre-event image on the same track was "
                                     "requested but none was found.")
        _add_warnings(pair, i, relax, cfg)
        return pair

    raise NoValidPairError(
        "No same-track before/after Sentinel-1 pair found for this AOI and date.",
        _diagnose(by_track, grid, event_date, cfg))


def _add_warnings(pair: ScenePair, rung_index: int, relax: Relaxation,
                  cfg: AcquisitionConfig) -> None:
    """Turn the situation into sentences for the report's limitations text."""
    w = pair.warnings
    if rung_index > 0:
        w.append(f"Search was relaxed ({relax.label}: window +/-{relax.window_days} d, "
                 f"min coverage {relax.min_coverage:.0%}).")
    if pair.after.date == pair.event_date:                       # TRICK 7
        w.append("The 'after' scene was acquired on the event date itself; "
                 "its time of day relative to the event is unknown.")
    if pair.gap_days > cfg.long_gap_warn_days:
        w.append(f"Long gap of {pair.gap_days} days between before and after: "
                 "seasonal change, soil moisture or vegetation may cause false changes.")
    if pair.before.platform != pair.after.platform:
        w.append(f"Before ({pair.before.platform}) and after ({pair.after.platform}) "
                 "come from different satellites on the same track.")
    if pair.coverage < 0.999:
        w.append(f"The pair covers {pair.coverage:.1%} of the AOI; the rest has no data.")
    if any(not s.online for s in pair.all_scenes()):
        w.append("Some selected scenes are in the long-term archive (offline) and "
                 "may take longer to retrieve.")


# ============================================================================
# 7. CDSE CLIENT   (all network code lives here)
# ============================================================================

class CDSEClient:
    """Search the catalogue and download products.

    ``session`` can be injected (tests pass a fake). Credentials come from the
    arguments or from the CDSE_USERNAME / CDSE_PASSWORD environment variables.
    """

    def __init__(self, cfg: Optional[AcquisitionConfig] = None,
                 username: Optional[str] = None, password: Optional[str] = None,
                 session: Optional[requests.Session] = None):
        self.cfg = cfg or AcquisitionConfig()
        self.username = username or os.environ.get("CDSE_USERNAME")
        self.password = password or os.environ.get("CDSE_PASSWORD")
        self.session = session or requests.Session()
        self._tok: Optional[str] = None
        self._tok_expiry = 0.0

    # ------------------------------------------------------------- plumbing
    def _sleep_before_retry(self, attempt: int) -> None:
        """Exponential backoff with jitter: 2 s, 4 s, 8 s... (+0-25 %), max 60 s.
        Jitter stops many clients retrying in lockstep."""
        delay = min(60.0, self.cfg.retry_base_delay_s * 2 ** (attempt - 1))
        time.sleep(delay * (1 + 0.25 * random.random()))

    def _with_retries(self, fn, what: str, error_cls=CatalogueError):
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                return fn()
            except (_Transient, requests.ConnectionError, requests.Timeout) as exc:
                if attempt == self.cfg.max_retries:
                    raise error_cls(f"{what} failed after {attempt} attempts: {exc}") from exc
                log.warning("%s: %s (attempt %d/%d), retrying", what, exc, attempt,
                            self.cfg.max_retries)
                self._sleep_before_retry(attempt)

    # ----------------------------------------------------------------- auth
    def _token(self) -> str:
        """Bearer token, refreshed 30 s before expiry. CDSE tokens live only
        ~10 minutes, shorter than a 2 GB download on a slow link (see TRICK 10)."""
        if self._tok and time.time() < self._tok_expiry - 30:
            return self._tok
        if not (self.username and self.password):
            raise AuthError("Downloading needs a free CDSE account. Set the environment "
                            "variables CDSE_USERNAME and CDSE_PASSWORD.")
        resp = self.session.post(
            self.cfg.token_url, timeout=self.cfg.timeout_s,
            data={"grant_type": "password", "client_id": "cdse-public",
                  "username": self.username, "password": self.password})
        if resp.status_code != 200:
            raise AuthError(f"CDSE login failed (HTTP {resp.status_code}). "
                            "Check username/password.")
        body = resp.json()
        self._tok = body["access_token"]
        self._tok_expiry = time.time() + float(body.get("expires_in", 600))
        return self._tok

    def _invalidate_token(self) -> None:
        self._tok, self._tok_expiry = None, 0.0

    # --------------------------------------------------------------- search
    def search(self, aoi: AOI, start: date, end: date) -> list:
        """Raw OData product dicts, with a disk cache (TRICK 11).

        Fresh cache -> returned without touching the network. Network failure
        -> stale cache used with a warning. No cache + failure -> CatalogueError.
        TRICK 8: no credentials needed for any of this.
        """
        cache = self._cache_path(aoi, start, end)
        fresh = self._read_cache(cache, self.cfg.search_cache_ttl_h)
        if fresh is not None:
            log.info("search: using cached results (%d products)", len(fresh))
            return fresh
        try:
            raw = self._search_network(aoi, start, end)
        except CatalogueError:
            stale = self._read_cache(cache, None)
            if stale is not None:
                log.warning("Catalogue unreachable; using STALE cached search results.")
                return stale
            raise
        self._write_cache(cache, raw)
        return raw

    def _search_network(self, aoi: AOI, start: date, end: date) -> list:
        params = {"$filter": build_odata_filter(aoi, start, end, self.cfg),
                  "$expand": "Attributes", "$orderby": "ContentDate/Start asc",
                  "$top": str(self.cfg.search_page_size)}
        url, items = self.cfg.catalogue_url, []
        for _page in range(self.cfg.search_max_pages):
            data = self._with_retries(lambda: self._get_json(url, params), "catalogue search")
            items.extend(data.get("value", []))
            url, params = data.get("@odata.nextLink"), None   # nextLink is a full URL
            if not url:
                break
        else:
            log.warning("Search hit the page limit (%d); results may be truncated. "
                        "Use a smaller AOI.", self.cfg.search_max_pages)
        return items

    def _get_json(self, url: str, params: Optional[dict]) -> dict:
        resp = self.session.get(url, params=params, timeout=self.cfg.timeout_s)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise _Transient(f"HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise CatalogueError(f"Catalogue returned HTTP {resp.status_code}: "
                                 f"{getattr(resp, 'text', '')[:200]}")
        return resp.json()

    def _cache_path(self, aoi: AOI, start: date, end: date) -> Optional[Path]:
        if not self.cfg.search_cache_dir:
            return None
        key = hashlib.sha1(f"{aoi.key()}|{start}|{end}|{self.cfg.collection}|"
                           f"{self.cfg.product_type}".encode()).hexdigest()[:16]
        return Path(self.cfg.search_cache_dir) / f"search_{key}.json"

    @staticmethod
    def _read_cache(path: Optional[Path], max_age_h: Optional[float]):
        if not path or not path.exists():
            return None
        if max_age_h is not None and time.time() - path.stat().st_mtime > max_age_h * 3600:
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_cache(path: Optional[Path], raw: list) -> None:
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw))
        except OSError as exc:                     # cache is best-effort
            log.debug("could not write search cache: %s", exc)

    # ------------------------------------------------------------- download
    def _get_following_redirects(self, url: str, headers: dict, max_hops: int = 6):
        """TRICK 9 — follow redirects manually.

        The CDSE download endpoint answers with a redirect to another host.
        ``requests`` deliberately drops the Authorization header when a redirect
        leaves the original host, which yields a confusing 401/403. So we turn
        automatic redirects off and re-send our headers at every hop.
        """
        for _ in range(max_hops):
            resp = self.session.get(url, headers=headers, stream=True,
                                    allow_redirects=False, timeout=self.cfg.timeout_s)
            if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
                url = resp.headers["Location"]
                resp.close()
                continue
            return resp
        raise DownloadError("Too many redirects while downloading")

    def _stream_to_part(self, scene: Scene, part: Path) -> None:
        """One download attempt that APPENDS to ``part`` (resume support)."""
        offset = part.stat().st_size if part.exists() else 0
        if scene.size_bytes is not None:
            if offset > scene.size_bytes:            # corrupt leftover: start over
                part.unlink()
                offset = 0
            elif offset == scene.size_bytes:         # nothing left to fetch
                return
        headers = {"Authorization": f"Bearer {self._token()}"}
        if offset:
            headers["Range"] = f"bytes={offset}-"    # "send me the rest"
        url = f"{self.cfg.download_url}/Products({scene.product_id})/$value"

        resp = self._get_following_redirects(url, headers)
        try:
            sc = resp.status_code
            if sc == 401:                            # token expired mid-download
                self._invalidate_token()
                raise _Transient("token expired (401); will re-login and resume")
            if sc in (202, 429) or sc >= 500:        # 202 = product being staged
                raise _Transient(f"HTTP {sc}")
            if sc in (403, 404):
                raise DownloadError(f"HTTP {sc} for {scene.name}: not allowed or not found")
            if sc not in (200, 206):
                raise DownloadError(f"Unexpected HTTP {sc} for {scene.name}")
            mode = "ab"
            if sc == 200 and offset:                 # server ignored Range: restart
                mode = "wb"
            with open(part, mode) as fh:
                for chunk in resp.iter_content(self.cfg.chunk_bytes):
                    if chunk:
                        fh.write(chunk)
        finally:
            resp.close()

    def download(self, scene: Scene, dest_dir: Path) -> Path:
        """Download one product to ``dest_dir/<name>.zip`` and return the path.

        TRICK 10 — the file is written as ``.zip.part`` and renamed (an atomic
        operation) only after size and MD5 check out. Therefore:
          * if ``<name>.zip`` exists it is complete (cache hit, instant);
          * a crash or Ctrl-C leaves only a ``.part`` that the next run resumes
            using an HTTP Range request, not from zero.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        final = dest_dir / f"{scene.stem}.zip"
        part = dest_dir / f"{scene.stem}.zip.part"

        if final.exists() and (scene.size_bytes is None
                               or final.stat().st_size == scene.size_bytes):
            log.info("cache hit: %s", final.name)
            return final

        def attempt():
            self._stream_to_part(scene, part)
            size = part.stat().st_size if part.exists() else 0
            if scene.size_bytes is not None and size != scene.size_bytes:
                raise _Transient(f"incomplete file {size}/{scene.size_bytes} bytes")

        self._with_retries(attempt, f"download of {scene.name}", error_cls=DownloadError)

        if self.cfg.verify_md5 and scene.md5:
            h = hashlib.md5()
            with open(part, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 22), b""):
                    h.update(block)
            if h.hexdigest() != scene.md5:
                part.unlink(missing_ok=True)         # corrupt: do not resume garbage
                raise DownloadError(f"MD5 mismatch for {scene.name}; file deleted, retry.")
        os.replace(part, final)                      # atomic on the same filesystem
        log.info("downloaded %s (%.2f GB)", final.name, final.stat().st_size / 1e9)
        return final


# ============================================================================
# 8. GLUE:  get_satellite_data
# ============================================================================

def _coerce_date(value) -> date:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise InvalidInputError(f"Event date must be YYYY-MM-DD, got {value!r}") from exc


def run_key(aoi: AOI, event_date: date) -> str:
    """Folder name shared with later parts: <aoi hash>_<event date>."""
    return f"{aoi.key()}_{event_date.isoformat()}"


def search_scenes(client: CDSEClient, aoi: AOI, event_date: date,
                  cfg: AcquisitionConfig) -> list:
    """One wide catalogue search covering the widest rung of the ladder
    (TRICK 2), parsed into Scene objects, de-duplicated."""
    max_w = max(r.window_days for r in cfg.ladder)
    today = datetime.now(timezone.utc).date()
    raw = client.search(aoi, event_date - timedelta(days=max_w),
                        min(event_date + timedelta(days=max_w), today))
    scenes, seen, skipped = [], set(), 0
    for item in raw:
        s = parse_product(item)
        if s is None:
            skipped += 1
        elif s.product_id not in seen:
            seen.add(s.product_id)
            scenes.append(s)
    log.info("search: %d usable scenes (%d skipped: wrong mode/polarisation or "
             "missing metadata)", len(scenes), skipped)
    if raw and not scenes:
        log.warning("All %d results were skipped. If this is unexpected, the CDSE "
                    "attribute names may have changed; check parse_product().", len(raw))
    return scenes


def get_satellite_data(bbox: Sequence[float], event_date, *,
                       download: bool = True,
                       want_before2: bool = False,
                       out_dir: str | Path = "data/raw/s1",
                       processed_dir: str | Path = "data/processed",
                       config: Optional[AcquisitionConfig] = None,
                       client: Optional[CDSEClient] = None,
                       write_json: bool = True) -> ScenePair:
    """Part 1 entry point:  AOI + date  ->  ScenePair (+ files on disk).

    bbox          (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
    event_date    'YYYY-MM-DD' string or ``date``
    download      False = only choose and report (no login needed, TRICK 8)
    want_before2  also look for a second, earlier pre-event pass
    Raises: InvalidInputError, NoValidPairError, CatalogueError, AuthError,
            DownloadError (all subclasses of AcquisitionError).
    """
    cfg = config or AcquisitionConfig()
    event_date = _coerce_date(event_date)
    aoi = AOI.from_bbox(bbox)

    # ---- validate before spending any time on the network
    today = datetime.now(timezone.utc).date()
    if event_date > today:
        raise InvalidInputError(f"Event date {event_date} is in the future.")
    if aoi.area_km2() > cfg.max_aoi_km2:
        raise InvalidInputError(
            f"AOI is {aoi.area_km2():.0f} km2, above the {cfg.max_aoi_km2:.0f} km2 cap. "
            "Use a smaller area or process it in tiles.")

    client = client or CDSEClient(cfg)
    scenes = search_scenes(client, aoi, event_date, cfg)
    pair = select_pair(scenes, aoi, event_date, cfg, want_before2=want_before2)
    log.info("selected pair:\n%s", pair.summary())

    if download:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [s for s in pair.all_scenes()
                if not (out_dir / f"{s.stem}.zip").exists()]
        need = sum(s.size_bytes or 0 for s in todo)
        free = shutil.disk_usage(out_dir).free
        if need and free < need * 1.1:               # fail early, not at 90 %
            raise DownloadError(f"Need ~{need / 1e9:.1f} GB but only "
                                f"{free / 1e9:.1f} GB free in {out_dir}.")
        for s in pair.all_scenes():
            s.local_path = str(client.download(s, out_dir))

    if write_json:
        path = Path(processed_dir) / run_key(aoi, event_date) / "scene_pair.json"
        pair.save_json(path)
        log.info("wrote %s", path)
    return pair


# ============================================================================
# 9. CLI
# ============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Pick (and optionally download) a same-track Sentinel-1 before/after pair.")
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    p.add_argument("--date", required=True, help="event date YYYY-MM-DD")
    p.add_argument("--no-download", action="store_true",
                   help="only choose scenes and print them (no login needed)")
    p.add_argument("--before2", action="store_true", help="also pick a second pre-event pass")
    p.add_argument("--out-dir", default="data/raw/s1")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    try:
        pair = get_satellite_data(args.bbox, args.date, download=not args.no_download,
                                  want_before2=args.before2, out_dir=args.out_dir)
    except AcquisitionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(pair.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
