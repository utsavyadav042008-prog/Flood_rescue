"""
preprocessing/sentinel2.py  —  PART 4 (2/2): Sentinel-2 optical layers
======================================================================

WHAT THIS FILE DOES (one paragraph)
-----------------------------------
Takes the shared analysis grid from Part 3 and, if a clear Sentinel-2 image pair
exists, adds OPTICAL evidence on exactly that grid: reflectance bands, water and
vegetation indices, a clear-sky mask, a small "optical says new water / stripped
vegetation" evidence layer, and before/after RGB pictures for the dashboard. If no
usable image exists it says so ("unavailable") and everything else keeps working.

PUBLIC ENTRY POINT
------------------
    res = build_optical_layers(scene_pair, grid, package_dir)
    res.status        "used" | "partial" | "unavailable"
    res.reason        a sentence for the report / dashboard
    res.files         dict of written files (may be empty)

OUTPUT FILES (all on the Part 3 grid; written next to Part 3's files)
---------------------------------------------------------------------
    s2_before.tif, s2_after.tif           5 bands reflectance: B02 B03 B04 B08 B11
    s2_indices_before.tif / _after.tif    3 bands: NDWI, MNDWI, NDVI
    s2_valid_mask.tif                     uint8, 2 bands: clear_before, clear_after
    s2_evidence.tif                       uint8, 2 bands: new_water, vegetation_loss
    quicklook_before.png / _after.png     RGB pictures (transparent where no data)
    optical.json                          status + selection + thresholds + bounds
    (and an "optical" entry merged into Part 3's metadata.json)
  If only the after image is usable (status "partial") the *_before files are not
  written and the evidence uses the after image alone.

MAP OF THE FILE
---------------
  1. Reading ............ windowed reads of band files (rasterio only at the edge)
  2. Reflectance + masks  DN -> reflectance, SCL -> clear mask
  3. Indices + evidence .. NDWI / MNDWI / NDVI and change evidence
  4. Quicklooks .......... dependency-free PNG writer, shared colour stretch
  5. Orchestration ....... never raises, caching, metadata merge, CLI

NON-OBVIOUS TRICKS USED HERE (search for "TRICK n")
---------------------------------------------------
  TRICK 1  Read only the AOI window of each band file (a JP2 tile is 10980 x 10980
           px). The window is found by projecting the grid corners into the TILE's
           own UTM zone, which may differ from ours near 84 E.
  TRICK 2  Categorical data (SCL classes) is resampled with NEAREST neighbour.
           Bilinear would invent class "5.5" between 4 and 7.
  TRICK 3  DN == 0 is no-data and becomes NaN BEFORE interpolation, so edges do
           not blend with fake zeros.
  TRICK 4  Reflectance offset from the processing baseline (see sentinel2 part 1).
  TRICK 5  The cloud mask is GROWN by 60 m (clouds have fuzzy edges and shadows
           the classifier misses), plus a "too bright in blue" test for thin
           cloud and haze that SCL sometimes calls vegetation.
  TRICK 6  Water evidence is a CHANGE: a high water index now, a low one before,
           and a big rise. Absolute thresholds alone fail on muddy flood water,
           so the change criterion carries more weight than the level.
  TRICK 7  Debris evidence from vegetation loss: if NDVI collapses where it was
           green, mud/rock replaced it. Radar struggles with debris; optics does
           not.
  TRICK 8  Evidence never fires on unknown pixels: cloud means "don't know", not
           "no flood". The valid mask tells the user which is which.
  TRICK 9  After-only mode: with no clear before image, permanent water from
           Part 3 is subtracted so existing rivers are not called new flood.
  TRICK 10 The PNG writer is 15 lines of zlib + struct: no Pillow needed. Before
           and after share ONE colour stretch, so the pair is visually comparable.
  TRICK 11 build_optical_layers catches every exception and returns
           "unavailable"; optional means optional. Real errors are not cached.
  TRICK 12 The required (after) side is tested first and tests stop early, so a
           failed attempt costs few downloads.

HONEST LIMITS (for the report)
------------------------------
  * Optical needs clear sky; in monsoon season often none exists.
  * Water/vegetation thresholds are starting values; tune on the Trishuli case.
  * Optical and radar images are NOT simultaneous; dates are recorded.
  * Sentinel-2 evidence supports the radar result. It is not ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import struct
import sys
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

from acquisition.sentinel import AOI, CDSEClient, CoverageGrid, ScenePair, run_key
from acquisition.sentinel2 import (OpticalConfig, S2Choice, S2FetchError, S2Pass, S2Product,
                                   boa_offset, choose_pair, default_band_source, group_passes,
                                   rank_candidates, s2_client_config, search_s2)
from preprocessing.alignment import GridSpec, remove_small_blobs, resample_to_grid
from preprocessing.sentinel1 import GeoRaster, RasterioIO
from preprocessing.utm import lonlat_to_utm, parse_utm_crs, utm_to_lonlat

log = logging.getLogger("preprocessing.sentinel2")

BANDS = ("B02", "B03", "B04", "B08", "B11")     # reflectance bands we keep (blue..SWIR)
INDEX_NAMES = ("NDWI", "MNDWI", "NDVI")
EVIDENCE_NAMES = ("new_water", "vegetation_loss")


# ============================================================================
# 1. READING
# ============================================================================

@dataclass
class TileInfo:
    """Geometry of one band file: CRS, pixel size, top-left corner, size."""
    crs: str
    res: float
    x0: float
    y0: float
    width: int
    height: int


class RasterioTileReader:
    """Reads band files (JPEG2000 inside zips or on disk) with rasterio.
    The only place rasterio is used in Part 4. Injectable (TRICK 12 of part 3)."""

    @staticmethod
    def _rio():
        try:
            import rasterio
            return rasterio
        except ImportError as exc:                           # pragma: no cover
            raise S2FetchError("rasterio (with JPEG2000 support) is required") from exc

    def info(self, path) -> TileInfo:
        with self._rio().open(path) as src:
            t = src.transform
            if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
                raise S2FetchError(f"{path}: rotated raster not supported")
            return TileInfo(src.crs.to_string(), abs(t.a), t.c, t.f, src.width, src.height)

    def read(self, path, col0: int, row0: int, ncols: int, nrows: int) -> np.ndarray:
        from rasterio.windows import Window
        with self._rio().open(path) as src:
            return src.read(1, window=Window(col0, row0, ncols, nrows)).astype("float32")


def grid_window_in_tile(grid: GridSpec, info: TileInfo, margin_px: int = 3) -> Optional[tuple]:
    """TRICK 1 — which pixels of the band file do we need?

    Project 200 points along the grid's outline into the TILE's CRS (via lon/lat
    when the tile lies in another UTM zone), take their bounding box, add a small
    margin for interpolation, and clip to the file. Returns
    (col0, row0, ncols, nrows) or None when the file does not touch the grid.
    """
    t = np.linspace(0.0, 1.0, 50, endpoint=False)
    x1, y1 = grid.x0 + grid.width * grid.res, grid.y0 - grid.height * grid.res
    e = np.concatenate([grid.x0 + t * (x1 - grid.x0), np.full(50, x1),
                        x1 - t * (x1 - grid.x0), np.full(50, grid.x0)])
    n = np.concatenate([np.full(50, grid.y0), grid.y0 - t * (grid.y0 - y1),
                        np.full(50, y1), y1 + t * (grid.y0 - y1)])
    if info.crs != grid.crs:
        zg, ng = parse_utm_crs(grid.crs)
        zt, nt = parse_utm_crs(info.crs)
        e, n = lonlat_to_utm(*utm_to_lonlat(e, n, zg, ng), zt, nt)
    col, row = (e - info.x0) / info.res, (info.y0 - n) / info.res
    c0 = max(0, int(math.floor(col.min())) - margin_px)
    r0 = max(0, int(math.floor(row.min())) - margin_px)
    c1 = min(info.width, int(math.ceil(col.max())) + margin_px)
    r1 = min(info.height, int(math.ceil(row.max())) + margin_px)
    if c1 <= c0 or r1 <= r0:
        return None
    return c0, r0, c1 - c0, r1 - r0


def read_band_on_grid(reader, path, grid: GridSpec, order: int,
                      zero_is_nodata: bool, margin_px: int = 3) -> Optional[np.ndarray]:
    """One band file -> (H, W) float32 array on ``grid``; None if no overlap.

    order=1 (bilinear) for reflectance, order=0 (nearest) for SCL classes
    (TRICK 2). ``zero_is_nodata`` turns DN 0 into NaN before interpolating
    (TRICK 3). Resampling is Part 3's ``resample_to_grid``, so a 10 m band on the
    same lattice is an exact copy and a 20 m band is interpolated.
    """
    info = reader.info(path)
    win = grid_window_in_tile(grid, info, margin_px)
    if win is None:
        return None
    raw = reader.read(path, *win)
    if zero_is_nodata:
        raw = np.where(raw == 0, np.nan, raw)
    src = GeoRaster(raw[None].astype(np.float32), info.x0 + win[0] * info.res,
                    info.y0 - win[1] * info.res, info.res, info.crs)
    return resample_to_grid(src, grid, order=order).data[0]


def combine_first(arrays: Sequence[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    """First non-NaN value wins (used to merge tiles of one pass)."""
    out = None
    for a in arrays:
        if a is None:
            continue
        out = a.copy() if out is None else np.where(np.isnan(out), a, out)
    return out


# ============================================================================
# 2. REFLECTANCE + MASKS
# ============================================================================

def to_reflectance(dn: np.ndarray, offset: float) -> np.ndarray:
    """DN -> surface reflectance: (DN + offset) / 10000, never negative (TRICK 4).
    NaN (no data) stays NaN."""
    return np.maximum((dn + offset) / 10000.0, 0.0).astype(np.float32)


def load_pass_scl(p: S2Pass, source, reader, grid: GridSpec, cfg: OpticalConfig) -> np.ndarray:
    """SCL classes of a pass on the grid as uint8 (0 = no data / outside the tile)."""
    arrays = [read_band_on_grid(reader, source.path_for(t, "SCL"), grid, order=0,
                                zero_is_nodata=False, margin_px=cfg.scl_margin_px)
              for t in p.tiles]
    scl = combine_first(arrays)
    if scl is None:
        return np.zeros((grid.height, grid.width), dtype=np.uint8)
    return np.nan_to_num(scl, nan=0.0).astype(np.uint8)


def load_pass_bands(p: S2Pass, source, reader, grid: GridSpec, cfg: OpticalConfig) -> dict:
    """Reflectance of BANDS on the grid: {band: (H, W) float32, NaN = no data}.
    Offsets are applied per tile (each tile has its own baseline), then tiles merge."""
    out = {}
    for band in BANDS:
        arrays = []
        for t in p.tiles:
            dn = read_band_on_grid(reader, source.path_for(t, band), grid, order=1,
                                   zero_is_nodata=True, margin_px=cfg.scl_margin_px)
            if dn is not None:
                arrays.append(to_reflectance(dn, boa_offset(t.baseline, cfg.boa_offset_override)))
        merged = combine_first(arrays)
        out[band] = merged if merged is not None else np.full(
            (grid.height, grid.width), np.nan, dtype=np.float32)
    return out


def clear_from_scl(scl: np.ndarray, cfg: OpticalConfig) -> np.ndarray:
    """True where the SCL says the surface is visible: vegetation, bare ground or
    water. Cloud shadow / cloud / cirrus are GROWN by ``cloud_buffer_px`` (TRICK 5)
    because their edges are fuzzy and shadows are often missed. Dark-area pixels
    (class 2), snow (11) and unclassified (7) are not clear: in mountains they are
    mostly topographic shadow or snow, which fool water indices."""
    clear = np.isin(scl, cfg.clear_classes)
    cloudish = np.isin(scl, cfg.cloud_classes)
    if cfg.cloud_buffer_px > 0 and cloudish.any():
        cloudish = ndimage.binary_dilation(cloudish, structure=np.ones((3, 3), dtype=bool),
                                           iterations=cfg.cloud_buffer_px)
    return clear & ~cloudish


def clear_mask(scl: np.ndarray, refl: dict, cfg: OpticalConfig) -> np.ndarray:
    """SCL clear mask, plus: all bands finite, and not too bright in blue
    (TRICK 5: thin cloud and haze that SCL sometimes labels as vegetation)."""
    clear = clear_from_scl(scl, cfg)
    for b in BANDS:
        clear &= np.isfinite(refl[b])
    if cfg.bright_blue is not None:
        clear &= ~(np.nan_to_num(refl["B02"], nan=0.0) > cfg.bright_blue)
    return clear


# ============================================================================
# 3. INDICES + EVIDENCE
# ============================================================================

def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN where the denominator is ~0 (dark pixels)."""
    s = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 1e-4, (a - b) / s, np.nan).astype(np.float32)


def compute_indices(refl: dict) -> np.ndarray:
    """(3, H, W): NDWI (green vs NIR), MNDWI (green vs SWIR), NDVI (NIR vs red).

    NDWI/MNDWI are high over water. MNDWI uses SWIR, which water absorbs even when
    muddy, so it is the better one for sediment-laden flood water; snow and
    shadows fool both, which is why those are masked. NDVI is high over healthy
    vegetation."""
    return np.stack([normalized_difference(refl["B03"], refl["B08"]),
                     normalized_difference(refl["B03"], refl["B11"]),
                     normalized_difference(refl["B08"], refl["B04"])])


def change_evidence(idx_before: Optional[np.ndarray], idx_after: np.ndarray,
                    clear_before: Optional[np.ndarray], clear_after: np.ndarray,
                    permanent_water: Optional[np.ndarray], cfg: OpticalConfig) -> tuple:
    """What the optical pair says changed. Returns (evidence (2,H,W) uint8, mode).

    CHANGE mode (both images): new_water = water index above ``water_abs`` now,
    not before, and risen by ``water_delta`` (TRICK 6). vegetation_loss = NDVI was
    green and dropped by ``veg_drop`` (TRICK 7; debris, mud, landslide scars).
    AFTER-ONLY mode: new_water = water now and not in Part 3's permanent water
    mask (TRICK 9); no vegetation evidence.
    Both only fire on clear pixels (TRICK 8); small specks are removed.
    """
    wi_a = np.fmax(idx_after[0], idx_after[1])
    if idx_before is not None and clear_before is not None:
        wi_b = np.fmax(idx_before[0], idx_before[1])
        valid = clear_after & clear_before
        water = valid & (wi_a > cfg.water_abs) & ~(wi_b > cfg.water_abs) \
            & ((wi_a - wi_b) > cfg.water_delta)
        veg = valid & (idx_before[2] > cfg.veg_before_min) \
            & ((idx_before[2] - idx_after[2]) > cfg.veg_drop) & ~water
        mode = "change"
    else:
        pw = permanent_water.astype(bool) if permanent_water is not None else False
        water = clear_after & (wi_a > cfg.water_abs) & ~pw
        veg = np.zeros_like(water)
        mode = "after_only"
    water = remove_small_blobs(water, cfg.min_blob_px)
    veg = remove_small_blobs(veg, cfg.min_blob_px)
    return np.stack([water, veg]).astype(np.uint8), mode


# ============================================================================
# 4. QUICKLOOKS
# ============================================================================

def write_png_rgba(path, rgba: np.ndarray) -> None:
    """Write an (H, W, 4) uint8 array as a PNG using only zlib + struct (TRICK 10).

    A PNG is: signature, an IHDR chunk (size, 8-bit RGBA), an IDAT chunk (rows,
    each prefixed with filter byte 0, deflate-compressed) and IEND. Each chunk is
    length + type + data + CRC32."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(png)


def shared_stretch(refl_list: Sequence[dict], clear_list: Sequence[np.ndarray]) -> tuple:
    """TRICK 10 — ONE (low, high) brightness stretch for ALL images of the pair:
    2nd and 98th percentile of the clear RGB pixels of every image together.
    Stretching each image on its own would make before and after look equally
    bright even if one is much darker (e.g. after a flood)."""
    vals = []
    for refl, clear in zip(refl_list, clear_list):
        for b in ("B04", "B03", "B02"):
            v = refl[b][clear]
            if v.size:
                vals.append(v[::max(1, v.size // 200000)])
    if not vals:
        return 0.0, 0.3
    v = np.concatenate(vals)
    lo, hi = np.percentile(v, [2, 98])
    return float(lo), float(max(hi, lo + 1e-3))


def _downscale(a: np.ndarray, max_px: int) -> np.ndarray:
    """Block-average an (H, W, C) array so that its longer side <= max_px."""
    f = int(math.ceil(max(a.shape[:2]) / max_px))
    if f <= 1:
        return a
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:h, :w].reshape(h // f, f, w // f, f, -1).mean(axis=(1, 3))


def make_quicklook(refl: dict, stretch: tuple, max_px: int) -> np.ndarray:
    """True-colour RGBA picture: red=B04, green=B03, blue=B02, same stretch for
    all images; alpha 0 where there is no data. Clouds stay visible on purpose: a
    human looking at the picture should see why the pixels are not clear."""
    rgb = np.stack([refl["B04"], refl["B03"], refl["B02"]], axis=-1)
    lo, hi = stretch
    alpha = np.isfinite(rgb).all(axis=-1).astype(np.float32)
    rgb = np.clip((np.nan_to_num(rgb, nan=0.0) - lo) / (hi - lo), 0, 1)
    small = _downscale(np.concatenate([rgb, alpha[..., None]], axis=-1), max_px)
    out = np.zeros(small.shape, dtype=np.uint8)
    out[..., :3] = np.round(small[..., :3] * 255)
    out[..., 3] = np.where(small[..., 3] > 0.5, 255, 0)
    return out


def grid_bounds_lonlat(grid: GridSpec) -> list:
    """[[south, west], [north, east]] of the grid, the format folium's ImageOverlay
    wants. The picture is a UTM rectangle drawn as a lon/lat rectangle: the
    distortion over a 30 km AOI is a few metres, invisible on a map."""
    zone, north = parse_utm_crs(grid.crs)
    e = [grid.x0, grid.x0 + grid.width * grid.res]
    n = [grid.y0 - grid.height * grid.res, grid.y0]
    lon, lat = utm_to_lonlat(np.array([e[0], e[1], e[0], e[1]]), np.array([n[0], n[0], n[1], n[1]]),
                             zone, north)
    return [[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]]


# ============================================================================
# 5. ORCHESTRATION
# ============================================================================

@dataclass
class OpticalResult:
    status: str                         # "used" | "partial" | "unavailable"
    reason_code: str = ""
    reason: str = ""
    files: dict = field(default_factory=dict)
    selection: dict = field(default_factory=dict)
    evidence_mode: str = ""
    qa: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    quicklook_bounds: Optional[list] = None
    from_cache: bool = False

    def to_dict(self) -> dict:
        return {"status": self.status, "reason_code": self.reason_code, "reason": self.reason,
                "files": self.files, "selection": self.selection,
                "evidence_mode": self.evidence_mode, "qa": self.qa, "warnings": self.warnings,
                "quicklook_bounds_south_west_north_east": self.quicklook_bounds}

    @classmethod
    def from_dict(cls, d: dict) -> "OpticalResult":
        return cls(d["status"], d.get("reason_code", ""), d.get("reason", ""), d.get("files", {}),
                   d.get("selection", {}), d.get("evidence_mode", ""), d.get("qa", {}),
                   d.get("warnings", []), d.get("quicklook_bounds_south_west_north_east"), True)

    def summary(self) -> str:
        lines = [f"Optical (Sentinel-2): {self.status.upper()}"
                 f"{'  (cache)' if self.from_cache else ''}  -  {self.reason}"]
        lines += [f"  {k:<22} {v}" for k, v in self.files.items()]
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def _optical_hash(cfg: OpticalConfig, grid: GridSpec, pair: ScenePair) -> str:
    blob = json.dumps({"cfg": cfg.effective_params(), "grid": grid.to_dict(),
                       "event": pair.event_date.isoformat(), "aoi": pair.aoi.as_tuple()},
                      sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _load_cache(out_dir: Path, ohash: str, cfg: OpticalConfig) -> Optional[OpticalResult]:
    """Reuse a previous result if the settings match, its files still exist, it
    was not an error, and (unless it is a final "used") it is not too old: new
    scenes may have been published since."""
    path = out_dir / "optical.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if d.get("params_hash") != ohash or d.get("reason_code") == "error":
        return None
    if not all(Path(f).exists() for f in d.get("files", {}).values()):
        return None
    if d.get("status") != "used":
        age_h = (datetime.now(timezone.utc)
                 - datetime.fromisoformat(d["created_utc"])).total_seconds() / 3600
        if age_h > cfg.cache_ttl_h:
            return None
    return OpticalResult.from_dict(d)


def _merge_metadata(meta_path: Path, optical: dict) -> None:
    """Add/replace the "optical" entry of Part 3's metadata.json (the contract)."""
    if not meta_path.exists():
        log.warning("metadata.json not found; optical status only in optical.json")
        return
    try:
        meta = json.loads(meta_path.read_text())
        meta["optical"] = optical
        meta_path.write_text(json.dumps(meta, indent=2))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not merge optical status into metadata.json: %s", exc)


def _as_selection(choice: S2Choice, event_date, pair: ScenePair) -> dict:
    sel = {"evaluated_clear_fraction": choice.evaluated}
    for side, p in (("before", choice.before), ("after", choice.after)):
        if p is not None:
            sel[side] = {**p.to_dict(), "days_from_event": (p.date - event_date).days}
    if choice.after is not None:
        sel["days_after_vs_s1_after"] = (choice.after.date - pair.after.date).days
    if choice.before is not None:
        sel["days_before_vs_s1_before"] = (choice.before.date - pair.before.date).days
    return sel


def _write_u8(io, path: Path, grid: GridSpec, arr: np.ndarray, names: Sequence[str]) -> None:
    io.write(path, grid.empty(arr.shape[0], 0, np.uint8).with_data(arr.astype(np.uint8)),
             list(names), {}, dtype="uint8", nodata=None)


def _build(pair: ScenePair, grid: GridSpec, out_dir: Path, cfg: OpticalConfig, client, source,
           reader, io, permanent_water) -> OpticalResult:
    event = pair.event_date
    products = search_s2(client, pair.aoi, event, cfg)
    if not products:
        return OpticalResult("unavailable", "no_candidates",
                             "No Sentinel-2 L2A scene covers the AOI in the search window.")
    cov = CoverageGrid(pair.aoi, 128)
    passes = [p for p in group_passes(products) if p.coverage(cov) >= cfg.min_coverage]
    if not passes:
        return OpticalResult("unavailable", "no_coverage",
                             f"No single Sentinel-2 pass covers {cfg.min_coverage:.0%} of the AOI.")

    scl_cache: dict = {}

    def clear_fn(p: S2Pass) -> float:
        """Measure the clear share of the AOI for a pass (fetches only the SCL)."""
        if p.key not in scl_cache:
            scl_cache[p.key] = load_pass_scl(p, source, reader, grid, cfg)
        return float(clear_from_scl(scl_cache[p.key], cfg).mean())

    choice = choose_pair(rank_candidates(passes, event, "before", cfg),
                         rank_candidates(passes, event, "after", cfg), clear_fn, event, cfg)
    selection = _as_selection(choice, event, pair)
    if choice.after is None:
        return OpticalResult("unavailable", choice.reason_code, choice.reason, selection=selection)

    warnings: list = []
    loaded: dict = {}
    for side, p in (("before", choice.before), ("after", choice.after)):
        if p is None:
            continue
        refl = load_pass_bands(p, source, reader, grid, cfg)
        if p.key not in scl_cache:          # (explicit check: a dict.get default would be
            scl_cache[p.key] = load_pass_scl(p, source, reader, grid, cfg)   # evaluated EAGERLY)
        clear = clear_mask(scl_cache[p.key], refl, cfg)
        loaded[side] = (refl, clear, compute_indices(refl))
        if clear.mean() < 0.9:
            warnings.append(f"The {side} optical image is only {clear.mean():.0%} clear over the "
                            "AOI; cloudy parts carry no optical information.")
    if choice.after.date == event:
        warnings.append("The optical 'after' image was acquired on the event date itself; its "
                        "time relative to the flood is unknown.")
    for k, label in (("days_after_vs_s1_after", "after"), ("days_before_vs_s1_before", "before")):
        if abs(selection.get(k, 0)) > 3:
            warnings.append(f"The optical {label} image is {abs(selection[k])} days away from the "
                            f"radar {label} image: they do not show the same moment.")

    refl_a, clear_a, idx_a = loaded["after"]
    before = loaded.get("before")
    evidence, mode = change_evidence(before[2] if before else None, idx_a,
                                     before[1] if before else None, clear_a,
                                     permanent_water, cfg)
    if mode == "after_only" and permanent_water is None:
        warnings.append("No pre-event optical image and no permanent-water mask supplied: "
                        "existing rivers and lakes may appear as new water.")
    valid_cnt = float(clear_a.sum()) or 1.0
    qa = {"clear_fraction_after": round(float(clear_a.mean()), 4),
          "clear_fraction_before": round(float(before[1].mean()), 4) if before else None,
          "new_water_fraction_of_clear": round(float(evidence[0][clear_a].sum()) / valid_cnt, 4),
          "vegetation_loss_fraction_of_clear": round(float(evidence[1][clear_a].sum()) / valid_cnt, 4)}
    if qa["new_water_fraction_of_clear"] > 0.4:
        warnings.append("Over 40 % of the clear area is flagged as new water: thresholds are "
                        "probably too loose for this scene. Inspect the quicklook.")

    # ---- write
    files: dict = {}

    def write_f(name: str, arr: np.ndarray, desc: Sequence[str], tag: str) -> None:
        path = out_dir / f"{name}.tif"
        io.write(path, grid.empty(arr.shape[0]).with_data(arr.astype(np.float32)), list(desc),
                 {"units": tag})
        files[name] = str(path)

    for side, tpl in (("before", before), ("after", loaded["after"])):
        if tpl is None:
            continue
        refl, _, idx = tpl
        write_f(f"s2_{side}", np.stack([refl[b] for b in BANDS]), [f"{b}_reflectance" for b in BANDS],
                "reflectance")
        write_f(f"s2_indices_{side}", idx, INDEX_NAMES, "index")
    valid_stack = np.stack([before[1] if before else np.zeros_like(clear_a), clear_a])
    _write_u8(io, out_dir / "s2_valid_mask.tif", grid, valid_stack, ("clear_before", "clear_after"))
    files["s2_valid_mask"] = str(out_dir / "s2_valid_mask.tif")
    _write_u8(io, out_dir / "s2_evidence.tif", grid, evidence, EVIDENCE_NAMES)
    files["s2_evidence"] = str(out_dir / "s2_evidence.tif")

    stretch = shared_stretch([x[0] for x in loaded.values()], [x[1] for x in loaded.values()])
    for side, tpl in loaded.items():
        qp = out_dir / f"quicklook_{side}.png"
        write_png_rgba(qp, make_quicklook(tpl[0], stretch, cfg.preview_max_px))
        files[f"quicklook_{side}"] = str(qp)

    return OpticalResult(choice.status, choice.reason_code, choice.reason, files, selection, mode,
                         qa, warnings, grid_bounds_lonlat(grid))


def build_optical_layers(pair: ScenePair, grid: GridSpec, out_dir, cfg: Optional[OpticalConfig] = None,
                         client: Optional[CDSEClient] = None, source=None, reader=None, io=None,
                         permanent_water: Optional[np.ndarray] = None,
                         raise_errors: bool = False) -> OpticalResult:
    """Part 4 entry point. NEVER raises unless ``raise_errors`` (TRICK 11): any
    failure becomes status "unavailable" with the error in ``reason``, because the
    optical layer is optional and must not take the pipeline down.

    out_dir           the Part 3 package folder (data/processed/<key>/)
    permanent_water   Part 3's permanent_water mask as a bool (H, W) array, used
                      only in after-only mode (TRICK 9)
    """
    cfg = cfg or OpticalConfig()
    io = io or RasterioIO()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ohash = _optical_hash(cfg, grid, pair)

    cached = _load_cache(out_dir, ohash, cfg)
    if cached is not None:
        log.info("optical layers: cache hit (%s)", cached.status)
        return cached

    try:
        client = client or CDSEClient(s2_client_config(cfg))
        source = source or default_band_source(client)
        reader = reader or RasterioTileReader()
        result = _build(pair, grid, out_dir, cfg, client, source, reader, io, permanent_water)
    except Exception as exc:                                   # noqa: BLE001 (deliberate)
        if raise_errors:
            raise
        log.exception("optical layers failed; continuing with radar only")
        result = OpticalResult("unavailable", "error", f"{type(exc).__name__}: {exc}")

    payload = {**result.to_dict(), "params_hash": ohash,
               "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "config": cfg.effective_params()}
    (out_dir / "optical.json").write_text(json.dumps(payload, indent=2))
    _merge_metadata(out_dir / "metadata.json", {k: payload[k] for k in
                    ("status", "reason_code", "reason", "files", "selection", "evidence_mode", "qa",
                     "warnings", "quicklook_bounds_south_west_north_east", "created_utc")})
    return result


# ============================================================================
# CLI
# ============================================================================

def grid_from_metadata(meta: dict) -> GridSpec:
    g = meta["grid"]
    return GridSpec(g["crs"], float(g["res_m"]), float(g["x0"]), float(g["y0"]),
                    int(g["width"]), int(g["height"]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Optional Sentinel-2 optical layers (Part 4).")
    p.add_argument("--scene-pair", required=True, help="scene_pair.json from Part 1")
    p.add_argument("--processed-root", default="data/processed")
    p.add_argument("--s2-cache", default="data/raw/s2")
    p.add_argument("--search-only", action="store_true",
                   help="only search and rank candidates (no band downloads)")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    pair = ScenePair.load_json(a.scene_pair)
    out_dir = Path(a.processed_root) / run_key(pair.aoi, pair.event_date)
    cfg = OpticalConfig()
    if a.search_only:
        client = CDSEClient(s2_client_config(cfg))
        passes = group_passes(search_s2(client, pair.aoi, pair.event_date, cfg))
        for side in ("before", "after"):
            print(f"-- {side} candidates (best first)")
            for ps in rank_candidates(passes, pair.event_date, side, cfg):
                print(f"   {ps.date} {ps.platform} tile cloud {ps.cloud_cover}% "
                      f"tiles {[t.tile for t in ps.tiles]}")
        return 0
    meta_path = out_dir / "metadata.json"
    if not meta_path.exists():
        print("ERROR: run Part 3 first (metadata.json with the grid is missing).", file=sys.stderr)
        return 1
    grid = grid_from_metadata(json.loads(meta_path.read_text()))
    io = RasterioIO()
    pw_path = out_dir / "permanent_water.tif"
    pw = io.read(pw_path).data[0] > 0.5 if pw_path.exists() else None
    client = CDSEClient(s2_client_config(cfg))
    res = build_optical_layers(pair, grid, out_dir, cfg, client,
                               default_band_source(client, a.s2_cache), io=io, permanent_water=pw)
    print(res.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
