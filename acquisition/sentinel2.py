"""
acquisition/sentinel2.py  —  PART 4 (1/2): Sentinel-2 search, selection, band access
===================================================================================

WHAT THIS FILE DOES (one paragraph)
-----------------------------------
Sentinel-2 is the OPTIONAL half of the system: a colour camera that shows muddy
flood water, stripped vegetation and landslide scars, but only when the sky is
clear. This file finds Sentinel-2 L2A scenes around the event, decides WHICH ONE
before and WHICH ONE after is actually usable over the AOI (clear sky), and
hands out the file paths of the bands we need. All pixel work is in
``preprocessing/sentinel2.py``.

NOTHING HERE MAY BREAK THE PIPELINE. The brief says "Sentinel-2 when the sky is
clear": in monsoon season it often is not. "No usable optical image" is a normal
result, reported as status "unavailable", never an exception that escapes.

MAP OF THE FILE
---------------
  1. Config ............. OpticalConfig (selection AND pixel settings, one place)
  2. S2Product / S2Pass . one tile file / one satellite pass (several tiles)
  3. Catalogue .......... parse CDSE JSON, search, rank candidates
  4. Selection .......... lazy clear-sky choice (pure logic, no network)
  5. Band sources ....... where a band's file comes from (zip / single files)

NON-OBVIOUS TRICKS USED HERE (search for "TRICK n")
---------------------------------------------------
  TRICK 1  The catalogue's cloudCover is for the WHOLE 100 km tile. A tile that
           is 80 % cloudy can be perfectly clear over our 30 km AOI, and the
           reverse. So it is only used to ORDER candidates; the real test is the
           clear fraction over the AOI, measured on the scene classification
           layer (SCL), which is a small file.
  TRICK 2  LAZY evaluation with early exit: candidates are tested one by one in
           rank order and testing stops at the first "good enough" scene, so we
           usually fetch 1-2 SCL files instead of 10.
  TRICK 3  The REQUIRED side (after the event) is evaluated first. If no clear
           post-event image exists the before side is never evaluated: no
           downloads are wasted on a result we could not use.
  TRICK 4  Tiles are grouped into passes by the sensing time in the file NAME.
           An AOI crossing a tile border still counts as one image.
  TRICK 5  Processing baseline from the name (N0511 = 05.11). Since baseline
           04.00 reflectance = (DN - 1000) / 10000; ignoring the offset biases
           every index.
  TRICK 6  Two band sources chained: try fetching single band files (small), and
           fall back to the whole product zip. The zip downloader is Part 1's
           resumable, checksummed one, reused through duck typing (S2Product
           has the same attributes as a Part 1 Scene).
  TRICK 7  Coverage by footprints uses Part 1's sampling grid (CoverageGrid), so
           it needs no geometry library.
"""
from __future__ import annotations

import logging
import os
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from acquisition.sentinel import (AcquisitionConfig, AcquisitionError, AOI, CDSEClient,
                                  CoverageGrid, _rings_from_item)

log = logging.getLogger("acquisition.sentinel2")


class S2FetchError(AcquisitionError):
    """A Sentinel-2 band file could not be obtained."""


# ============================================================================
# 1. CONFIG
# ============================================================================

@dataclass
class OpticalConfig:
    # ---- search / selection
    window_days: int = 20              # +/- days around the event
    min_coverage: float = 0.90         # footprint must cover this share of the AOI
    max_tile_cloud: float = 95.0       # tile-level cloudCover above this is skipped (TRICK 1)
    max_scl_checks: int = 4            # at most this many SCL tests per side
    min_clear: float = 0.65            # clear share of the AOI needed to use a scene
    good_enough_clear: float = 0.90    # stop testing as soon as a scene reaches this (TRICK 2)
    clear_bucket: float = 0.10         # clear share compared in 10 % steps
    cloud_rank_bucket: float = 20.0    # tile cloud compared in 20 % steps when ordering
    # ---- pixel processing (these affect results => hashed)
    clear_classes: tuple = (4, 5, 6)   # SCL: vegetation, not-vegetated, water
    cloud_classes: tuple = (3, 8, 9, 10)   # shadow, cloud medium/high, cirrus
    cloud_buffer_px: int = 6           # grow cloud/shadow by 60 m: edges and missed shadows
    bright_blue: Optional[float] = 0.30    # B02 reflectance above this = probably cloud/haze
    boa_offset_override: Optional[float] = None
    water_abs: float = 0.05            # water index (max of NDWI, MNDWI) above this = water
    water_delta: float = 0.15          # ...and it must have risen this much since before
    veg_before_min: float = 0.30       # NDVI before must show real vegetation
    veg_drop: float = 0.25             # NDVI drop that counts as vegetation loss
    min_blob_px: int = 15              # remove isolated specks from evidence layers
    # ---- not hashed
    preview_max_px: int = 1600
    cache_ttl_h: float = 24.0          # re-check "unavailable"/"partial" after this long
    scl_margin_px: int = 3

    def effective_params(self) -> dict:
        skip = {"preview_max_px", "cache_ttl_h"}
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items() if k not in skip}


def s2_client_config(cfg: Optional[OpticalConfig] = None) -> AcquisitionConfig:
    """Part 1's client configured for Sentinel-2 L2A instead of Sentinel-1 GRD."""
    return AcquisitionConfig(collection="SENTINEL-2", product_type="S2MSI2A")


# ============================================================================
# 2. S2Product / S2Pass
# ============================================================================

# S2B_MSIL2A_20260820T050701_N0511_R019_T45RUL_20260820T091245.SAFE
#  |    |      |               |     |    |
#  |    |      sensing start   |     |    tile id
#  |    product level          |     relative orbit
#  platform                    processing baseline (05.11)
_S2_NAME = re.compile(
    r"^(?P<platform>S2[A-D])_MSIL2A_(?P<t>\d{8}T\d{6})_N(?P<nb>\d{4})_R(?P<orbit>\d{3})_"
    r"T(?P<tile>[0-9A-Z]{5})_")


@dataclass
class S2Product:
    """ONE tile of a Sentinel-2 L2A pass = one zip file.

    Has the same attributes Part 1's downloader reads from a Scene (product_id,
    name, stem, size_bytes, md5), so ``CDSEClient.download`` works on it (TRICK 6).
    """
    product_id: str
    name: str
    platform: str
    sensing: datetime               # UTC; identical for all tiles of one pass
    rel_orbit: int
    tile: str
    baseline: float                 # 5.11 for N0511
    rings: list                     # footprint outer rings
    cloud_cover: Optional[float] = None
    size_bytes: Optional[int] = None
    md5: Optional[str] = None
    online: bool = True

    @property
    def stem(self) -> str:
        return self.name[:-5] if self.name.endswith(".SAFE") else self.name

    @property
    def date(self) -> date:
        return self.sensing.date()

    def to_dict(self) -> dict:
        return {"product_id": self.product_id, "name": self.name, "platform": self.platform,
                "sensing_utc": self.sensing.isoformat(), "relative_orbit": self.rel_orbit,
                "tile": self.tile, "baseline": self.baseline, "cloud_cover": self.cloud_cover,
                "size_bytes": self.size_bytes}


@dataclass
class S2Pass:
    """One satellite pass over the AOI: usually 1 tile, sometimes 2 (TRICK 4)."""
    tiles: list

    @property
    def sensing(self) -> datetime:
        return self.tiles[0].sensing

    @property
    def date(self) -> date:
        return self.tiles[0].date

    @property
    def platform(self) -> str:
        return self.tiles[0].platform

    @property
    def key(self) -> str:
        return f"{self.date.isoformat()}_{self.platform}"

    @property
    def cloud_cover(self) -> Optional[float]:
        vals = [t.cloud_cover for t in self.tiles if t.cloud_cover is not None]
        return float(np.mean(vals)) if vals else None

    def coverage(self, grid: CoverageGrid) -> float:
        """Share of the AOI inside the union of this pass's tile footprints (TRICK 7)."""
        return grid.fraction(grid.union(_FootprintOnly(t) for t in self.tiles))

    def to_dict(self) -> dict:
        return {"date_utc": self.date.isoformat(), "platform": self.platform,
                "cloud_cover_tile_pct": self.cloud_cover,
                "tiles": [t.to_dict() for t in self.tiles]}


class _FootprintOnly:
    """Adapter so Part 1's CoverageGrid (which expects .product_id and .rings)
    can measure an S2 tile footprint."""

    def __init__(self, t: S2Product):
        self.product_id, self.rings = t.product_id, t.rings


def boa_offset(baseline: float, override: Optional[float] = None) -> float:
    """TRICK 5 — the DN offset to ADD before dividing by 10000.

    Processing baseline 04.00 and later (since January 2022) stores
    DN = reflectance * 10000 + 1000, so the offset is -1000. Older baselines have
    none. Without this correction every reflectance is 0.1 too high and the
    water indices drift."""
    if override is not None:
        return float(override)
    return -1000.0 if baseline >= 4.0 else 0.0


# ============================================================================
# 3. CATALOGUE
# ============================================================================

def parse_s2_product(item: dict) -> Optional[S2Product]:
    """One CDSE OData product dict -> S2Product, or None if it is not an L2A tile
    we can use (wrong level, no footprint, unparseable name)."""
    name = item.get("Name", "")
    m = _S2_NAME.match(name)
    if not m:
        return None
    rings = _rings_from_item(item)
    if not rings:
        return None
    try:
        sensing = datetime.strptime(m["t"], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    attrs = {a.get("Name"): a.get("Value") for a in item.get("Attributes", [])}
    try:
        cloud = float(attrs["cloudCover"]) if attrs.get("cloudCover") is not None else None
    except (TypeError, ValueError):
        cloud = None
    md5 = None
    for c in item.get("Checksum") or []:
        if str(c.get("Algorithm", "")).upper() == "MD5":
            md5 = str(c.get("Value", "")).lower() or None
    return S2Product(
        product_id=item["Id"], name=name, platform=m["platform"], sensing=sensing,
        rel_orbit=int(m["orbit"]), tile=m["tile"], baseline=int(m["nb"]) / 100.0,
        rings=rings, cloud_cover=cloud,
        size_bytes=int(item["ContentLength"]) if item.get("ContentLength") else None,
        md5=md5, online=bool(item.get("Online", True)))


def search_s2(client: CDSEClient, aoi: AOI, event_date: date, cfg: OpticalConfig) -> list:
    """Search L2A tiles intersecting the AOI within +/- window_days of the event.
    Uses Part 1's client, including its on-disk search cache (so a dead network
    during a demo falls back to the last search)."""
    today = datetime.now(timezone.utc).date()
    raw = client.search(aoi, event_date - timedelta(days=cfg.window_days),
                        min(event_date + timedelta(days=cfg.window_days), today))
    seen, products = set(), []
    for item in raw:
        p = parse_s2_product(item)
        if p is not None and p.product_id not in seen:
            seen.add(p.product_id)
            products.append(p)
    log.info("sentinel-2 search: %d L2A tiles (%d results skipped)", len(products),
             len(raw) - len(products))
    return products


def group_passes(products: Sequence[S2Product]) -> list:
    """TRICK 4 — tiles of one pass share platform + sensing time (in the name)."""
    buckets: dict = {}
    for p in products:
        buckets.setdefault((p.platform, p.sensing), []).append(p)
    return sorted((S2Pass(sorted(v, key=lambda t: t.tile)) for v in buckets.values()),
                  key=lambda ps: ps.sensing)


def rank_candidates(passes: Sequence[S2Pass], event_date: date, side: str,
                    cfg: OpticalConfig) -> list:
    """Candidates for one side, best first, at most ``max_scl_checks``.

    'before' = strictly before the event day; 'after' = the event day or later.
    Order: tile cloud cover in 20 % steps (TRICK 1: only a hint), then closeness
    to the event. Scenes with absurd tile cloud (> max_tile_cloud) are dropped.
    """
    if side == "before":
        pool = [p for p in passes if p.date < event_date]
    else:
        pool = [p for p in passes if p.date >= event_date]
    pool = [p for p in pool if p.cloud_cover is None or p.cloud_cover <= cfg.max_tile_cloud]

    def key(p: S2Pass):
        cloud = 2 if p.cloud_cover is None else round(p.cloud_cover / cfg.cloud_rank_bucket)
        return (cloud, abs((p.date - event_date).days), p.sensing)

    return sorted(pool, key=key)[:cfg.max_scl_checks]


# ============================================================================
# 4. SELECTION  (pure: the caller supplies the function that measures clearness)
# ============================================================================

@dataclass
class S2Choice:
    before: Optional[S2Pass]
    after: Optional[S2Pass]
    status: str                       # "used" | "partial" | "unavailable"
    reason_code: str                  # machine-readable: ok, no_candidates, no_clear_after, ...
    reason: str                       # human sentence for the report/dashboard
    evaluated: dict = field(default_factory=dict)     # pass key -> clear fraction measured


def pick_side(candidates: Sequence[S2Pass], clear_fn: Callable, event_date: date,
              cfg: OpticalConfig) -> tuple:
    """TRICK 2 — test candidates in rank order, with early exit.

    ``clear_fn(pass)`` measures the clear share of the AOI (it fetches the SCL
    file, so each call is expensive). Scenes below ``min_clear`` are rejected.
    Among acceptable ones the best is: higher clear share (in 10 % steps), then
    closer to the event. We stop at the first scene >= ``good_enough_clear``.
    Returns (best pass or None, {pass key: clear fraction}).
    """
    evaluated: dict = {}
    best, best_key = None, None
    for p in candidates:
        c = float(clear_fn(p))
        evaluated[p.key] = round(c, 3)
        if c < cfg.min_clear:
            continue
        k = (-round(c / cfg.clear_bucket), abs((p.date - event_date).days))
        if best is None or k < best_key:
            best, best_key = p, k
        if c >= cfg.good_enough_clear:
            break
    return best, evaluated


def choose_pair(before_candidates: Sequence[S2Pass], after_candidates: Sequence[S2Pass],
                clear_fn: Callable, event_date: date, cfg: OpticalConfig) -> S2Choice:
    """Choose the optical before/after passes.

    Status rules (the after image is REQUIRED: a before-only result says nothing
    about the flood):
        both found      -> "used"
        only after      -> "partial"  (water evidence from the after image alone)
        no after        -> "unavailable"
    TRICK 3: the after side is evaluated FIRST; if it fails the before side is
    never touched, so no downloads are wasted.
    """
    if not after_candidates:
        return S2Choice(None, None, "unavailable", "no_candidates_after",
                        "No Sentinel-2 scene exists on or after the event date in the "
                        "search window (or all are filtered as too cloudy).")
    after, ev_after = pick_side(after_candidates, clear_fn, event_date, cfg)
    if after is None:
        return S2Choice(None, None, "unavailable", "no_clear_after",
                        f"No post-event Sentinel-2 scene is clear enough over the AOI "
                        f"(needed {cfg.min_clear:.0%}; best tested {max(ev_after.values()):.0%}). "
                        "Radar only.", ev_after)
    if not before_candidates:
        return S2Choice(None, after, "partial", "no_candidates_before",
                        "No pre-event Sentinel-2 scene found: optical evidence uses the "
                        "after image only.", ev_after)
    before, ev_before = pick_side(before_candidates, clear_fn, event_date, cfg)
    evaluated = {**ev_before, **ev_after}
    if before is None:
        return S2Choice(None, after, "partial", "no_clear_before",
                        "No clear pre-event Sentinel-2 scene: optical evidence uses the "
                        "after image only.", evaluated)
    return S2Choice(before, after, "used", "ok", "Clear before and after scenes found.", evaluated)


# ============================================================================
# 5. BAND SOURCES  (path_for(product, band) -> something the raster reader can open)
# ============================================================================

# Preferred resolutions per band (metres). B11 and SCL only exist at 20 m and 60 m.
BAND_RES = {"B02": (10, 20, 60), "B03": (10, 20, 60), "B04": (10, 20, 60),
            "B08": (10, 20, 60), "B11": (20, 60), "SCL": (20, 60)}


def find_member(names: Sequence[str], band: str) -> Optional[str]:
    """Pick the archive member holding ``band``, trying resolutions in order.
    ('.../IMG_DATA/R10m/T45RUL_20260820T050701_B02_10m.jp2')"""
    for res in BAND_RES[band]:
        pat = re.compile(rf"IMG_DATA/R{res}m/[^/]*_{band}_{res}m\.jp2$")
        for n in names:
            if pat.search(n):
                return n
    return None


class ZipBandSource:
    """Downloads the whole product zip with Part 1's robust downloader (resume,
    MD5, token refresh) and points GDAL INSIDE it ("/vsizip/..."): no unzipping."""

    def __init__(self, client: CDSEClient, cache_dir="data/raw/s2"):
        self.client, self.cache_dir = client, Path(cache_dir)
        self._names: dict = {}

    def path_for(self, product: S2Product, band: str) -> str:
        zpath = Path(self.client.download(product, self.cache_dir))
        if zpath not in self._names:
            with zipfile.ZipFile(zpath) as zf:
                self._names[zpath] = zf.namelist()
        member = find_member(self._names[zpath], band)
        if member is None:
            raise S2FetchError(f"{band} not found inside {zpath.name}")
        return f"/vsizip/{zpath.as_posix()}/{member}"


class NodeBandSource:
    """Downloads ONLY the needed band file through CDSE's OData 'Nodes' API
    (a few MB instead of ~1 GB). UNVERIFIED: written from the CDSE documentation
    as I remember it; on any problem it raises S2FetchError and the chain falls
    back to ZipBandSource."""

    def __init__(self, client: CDSEClient, cache_dir="data/raw/s2"):
        self.client, self.cache_dir = client, Path(cache_dir)

    def _list(self, url: str) -> list:
        headers = {"Authorization": f"Bearer {self.client._token()}"}
        resp = self.client._get_following_redirects(f"{url}/Nodes", headers)
        try:
            if resp.status_code != 200:
                raise S2FetchError(f"node listing HTTP {resp.status_code}")
            data = resp.json()
        finally:
            resp.close()
        return data.get("result") or data.get("value") or []

    @staticmethod
    def _child(children: list, wanted: str) -> str:
        for c in children:
            if c.get("Name") == wanted:
                return c["Name"]
        raise S2FetchError(f"node {wanted!r} not found")

    def path_for(self, product: S2Product, band: str) -> str:
        dest = self.cache_dir / product.stem / f"{band}.jp2"
        if dest.exists() and dest.stat().st_size > 0:
            return str(dest)
        base = f"{self.client.cfg.download_url}/Products({product.product_id})/Nodes({product.name})"
        granules = self._list(f"{base}/Nodes(GRANULE)")
        if not granules:
            raise S2FetchError("no GRANULE folder")
        gbase = f"{base}/Nodes(GRANULE)/Nodes({granules[0]['Name']})/Nodes(IMG_DATA)"
        for res in BAND_RES[band]:
            rbase = f"{gbase}/Nodes(R{res}m)"
            try:
                files = self._list(rbase)
            except S2FetchError:
                continue
            for f in files:
                if re.search(rf"_{band}_{res}m\.jp2$", f.get("Name", "")):
                    return self._download(f"{rbase}/Nodes({f['Name']})/$value", dest,
                                          f.get("ContentLength"))
        raise S2FetchError(f"{band} not found for {product.stem}")

    def _download(self, url: str, dest: Path, size: Optional[int]) -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".jp2.part")
        resp = self.client._get_following_redirects(
            url, {"Authorization": f"Bearer {self.client._token()}"})
        try:
            if resp.status_code != 200:
                raise S2FetchError(f"band download HTTP {resp.status_code}")
            with open(part, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    if chunk:
                        fh.write(chunk)
        finally:
            resp.close()
        if size is not None and part.stat().st_size != int(size):
            part.unlink(missing_ok=True)
            raise S2FetchError("band download incomplete")
        os.replace(part, dest)                       # atomic, as in Part 1
        return str(dest)


class FallbackBandSource:
    """TRICK 6 — try sources in order; any failure moves on to the next."""

    def __init__(self, sources: Sequence):
        self.sources = list(sources)

    def path_for(self, product: S2Product, band: str) -> str:
        errors = []
        for s in self.sources:
            try:
                return s.path_for(product, band)
            except AcquisitionError as exc:
                errors.append(f"{type(s).__name__}: {exc}")
                log.warning("band source %s failed for %s %s: %s",
                            type(s).__name__, product.stem, band, exc)
        raise S2FetchError(f"could not get {band} for {product.stem}: " + " | ".join(errors))


def default_band_source(client: CDSEClient, cache_dir="data/raw/s2") -> FallbackBandSource:
    return FallbackBandSource([NodeBandSource(client, cache_dir), ZipBandSource(client, cache_dir)])
