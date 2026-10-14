"""Turn the run's rasters into small PNG images for the map and the tabs.

Why not just ``ImageOverlay(raster)``?
--------------------------------------
Our rasters are in UTM metres.  A web map is a lon/lat grid.  UTM north is
rotated against true north (the "grid convergence", about 0.8 deg at the edge
of the Trishuli zone), so drawing the raster's bounding box straight onto the
map shears it by hundreds of metres across a 35 km area.  Instead we do an
INVERSE MAPPING: for every pixel of the output lon/lat grid, compute where it
falls in the UTM raster and copy that pixel (nearest neighbour).  No GDAL
warp needed, and the output size is capped at ``max_px`` so the browser never
receives a 20-megapixel image.

Important principle: this is a PICTURE.  Every number in the dashboard comes
from ``summary.json``, never from these thumbnails (nearest-neighbour
resampling can drop pixels of thin features).

Only ``numpy`` and the standard library are needed here, except
:func:`render_overlay_file` / :func:`render_gray_file`, which read rasters with
``rasterio`` (imported lazily).
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline.contracts import Artifact
from pipeline.geo_math import lonlat_to_utm_array, utm_to_lonlat
from pipeline.run import RunPaths

Transform6 = tuple  # (a, b, c, d, e, f) as in rasterio/affine: x = a*col + b*row + c


# --------------------------------------------------------------------------
# PNG encoder (standard library only)
# --------------------------------------------------------------------------
def encode_png_rgba(rgba: np.ndarray) -> bytes:
    """Encode an ``(h, w, 4)`` uint8 array as a PNG file.

    Written by hand with ``zlib`` + ``struct`` so the dashboard does not need
    Pillow or matplotlib just to draw a translucent overlay.  Each row is
    prefixed with filter byte 0 ("no filter"), which every decoder supports.
    """
    arr = np.ascontiguousarray(rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("rgba must have shape (height, width, 4)")
    height, width = arr.shape[:2]

    def chunk(tag: bytes, payload: bytes) -> bytes:
        # length, tag, payload, CRC32 over tag+payload
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    raw = b"".join(b"\x00" + arr[row].tobytes() for row in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))   # 8-bit RGBA
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


# --------------------------------------------------------------------------
# Geometry: UTM raster -> lon/lat grid
# --------------------------------------------------------------------------
def footprint_lonlat(transform: Transform6, shape: tuple, epsg: int) -> tuple:
    """``(west, south, east, north)`` of the raster's footprint in degrees."""
    a, b, c, d, e, f = transform[:6]
    rows, cols = shape[:2]
    corners = [(c, f), (a * cols + c, d * cols + f),
               (b * rows + c, e * rows + f), (a * cols + b * rows + c, d * cols + e * rows + f)]
    lonlat = [utm_to_lonlat(x, y, epsg) for x, y in corners]
    lons, lats = [p[0] for p in lonlat], [p[1] for p in lonlat]
    return (min(lons), min(lats), max(lons), max(lats))


def warp_to_lonlat(array: np.ndarray, transform: Transform6, epsg: int,
                   bounds: Optional[tuple] = None, max_px: int = 1200):
    """Resample a north-up UTM raster onto a lon/lat grid (nearest neighbour).

    Returns ``(out, inside, bounds)``: ``out`` has the raster's dtype, ``inside``
    is a boolean array that is False for output pixels outside the raster, and
    ``bounds`` is ``(west, south, east, north)``.  Rotated transforms (b or d
    not zero) are not supported (Sentinel grids are north-up).
    """
    a, b, c, d, e, f = transform[:6]
    if abs(b) > 1e-9 or abs(d) > 1e-9:
        raise ValueError("rotated rasters are not supported")
    rows, cols = array.shape[:2]
    west, south, east, north = bounds or footprint_lonlat(transform, array.shape, epsg)

    # Output size: keep the aspect ratio of the ground area, longest side = max_px.
    lat_mid = np.radians((south + north) / 2.0)
    width_m = (east - west) * 111_320.0 * np.cos(lat_mid)
    height_m = (north - south) * 110_574.0
    scale = max_px / max(width_m, height_m)
    out_w = max(1, int(round(width_m * scale)))
    out_h = max(1, int(round(height_m * scale)))

    lons = west + (np.arange(out_w) + 0.5) * (east - west) / out_w
    lats = north - (np.arange(out_h) + 0.5) * (north - south) / out_h      # row 0 = north
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    easting, northing = lonlat_to_utm_array(lon_grid, lat_grid, epsg)

    col = np.floor((easting - c) / a).astype(np.int64)
    row = np.floor((northing - f) / e).astype(np.int64)        # e < 0 for north-up grids
    inside = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
    out = np.zeros((out_h, out_w), dtype=array.dtype)
    out[inside] = array[row[inside], col[inside]]
    return out, inside, (west, south, east, north)


# --------------------------------------------------------------------------
# Colouring
# --------------------------------------------------------------------------
def mask_rgba(values: np.ndarray, inside: np.ndarray, rgb=(31, 120, 180), alpha: int = 170) -> np.ndarray:
    """Binary mask -> translucent colour where the mask is 1, fully transparent elsewhere."""
    rgba = np.zeros(values.shape + (4,), dtype=np.uint8)
    on = inside & (values > 0)
    rgba[on, :3] = rgb
    rgba[on, 3] = alpha
    return rgba


def probability_rgba(values: np.ndarray, inside: np.ndarray, rgb=(31, 120, 180), max_alpha: int = 200) -> np.ndarray:
    """Probability 0..1 -> same colour with opacity proportional to the probability."""
    p = np.clip(np.nan_to_num(values.astype(float)), 0.0, 1.0)
    rgba = np.zeros(values.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(inside & (p >= 0.05), (p * max_alpha).astype(np.uint8), 0)
    return rgba


def gray_rgba(values: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Backscatter (dB) -> opaque grey image with a 2-98 percentile stretch.

    Percentile stretching ignores a few extreme pixels, so the whole valley
    stays readable instead of being washed out by one bright rock face.
    """
    v = values.astype(float)
    finite = np.isfinite(v)
    if not finite.any():
        return np.zeros(v.shape + (4,), dtype=np.uint8)
    lo, hi = np.percentile(v[finite], [low_pct, high_pct])
    scaled = np.clip((np.nan_to_num(v, nan=lo) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    grey = (scaled * 255).astype(np.uint8)
    rgba = np.empty(v.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = grey
    rgba[..., 3] = 255
    return rgba


def shared_stretch(arrays, low_pct: float = 2.0, high_pct: float = 98.0) -> tuple:
    """ONE (lo, hi) for several images, so before and after use the same grey scale.

    Stretching each image on its own (what a viewer does by default) makes a
    flooded "after" image look as bright as the "before" one and hides the change.
    """
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if a is not None])
    if vals.size == 0:
        return (-25.0, 0.0)
    if vals.size > 2_000_000:
        vals = vals[:: vals.size // 2_000_000]
    lo, hi = np.percentile(vals, [low_pct, high_pct])
    return float(lo), float(max(hi, lo + 1e-6))


def unit_gray(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """dB -> 0..1 with a fixed stretch; NaN stays NaN."""
    return np.clip((values.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def gray_from_unit(unit: np.ndarray) -> np.ndarray:
    """0..1 grey (NaN = transparent) -> RGBA."""
    rgba = np.zeros(unit.shape + (4,), dtype=np.uint8)
    finite = np.isfinite(unit)
    g = (np.nan_to_num(unit) * 255).astype(np.uint8)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = g
    rgba[..., 3] = np.where(finite, 255, 0)
    return rgba


def change_rgba(diff_db: np.ndarray, limit_db: float = 8.0) -> np.ndarray:
    """After - before (dB) -> diverging colours on a light background.

    Blue = darker after (smooth surface: new water); orange = brighter after
    (rough surface: debris, rubble).  Small changes stay near-white, so only real
    change stands out.  NaN = transparent.
    """
    t = np.clip(np.nan_to_num(diff_db) / limit_db, -1.0, 1.0)
    base = np.array([248, 250, 252], dtype=np.float32)
    blue = np.array([29, 78, 216], dtype=np.float32)
    orange = np.array([194, 65, 12], dtype=np.float32)
    mag = np.abs(t)[..., None]
    target = np.where((t < 0)[..., None], blue, orange)
    rgb = base * (1 - mag) + target * mag
    rgba = np.zeros(diff_db.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = rgb.astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(diff_db), 255, 0)
    return rgba


def nan_smooth(a: np.ndarray, k: int = 2) -> np.ndarray:
    """Mean over a (2k+1)^2 window ignoring NaN (radar speckle hides change pixel by pixel)."""
    try:
        from scipy import ndimage
    except ImportError:
        return a
    ok = np.isfinite(a)
    size = 2 * k + 1
    s = ndimage.uniform_filter(np.where(ok, a, 0.0).astype(np.float64), size=size, mode="nearest")
    n = ndimage.uniform_filter(ok.astype(np.float64), size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 1e-9, s / np.maximum(n, 1e-9), np.nan)
    return np.where(ok, out, np.nan).astype(np.float32)


def downsample(array: np.ndarray, max_px: int) -> np.ndarray:
    """Keep every k-th pixel so the longest side is at most ``max_px``."""
    step = max(1, int(np.ceil(max(array.shape[:2]) / max_px)))
    return array[::step, ::step]


# --------------------------------------------------------------------------
# Pure orchestration (testable without rasterio)
# --------------------------------------------------------------------------
def overlay_png(kind: str, array: np.ndarray, transform: Transform6, epsg: int, max_px: int = 1200):
    """Return ``(png_bytes, (west, south, east, north))`` for ``kind`` in {'mask', 'probability'}."""
    values, inside, bounds = warp_to_lonlat(array, transform, epsg, max_px=max_px)
    rgba = mask_rgba(values, inside) if kind == "mask" else probability_rgba(values, inside)
    return encode_png_rgba(rgba), bounds


# --------------------------------------------------------------------------
# Disk helpers (need rasterio) with a modification-time cache
# --------------------------------------------------------------------------
def _read_raster(path: Path):
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is required to draw raster layers (pip install rasterio)") from exc
    with rasterio.open(path) as src:
        t = src.transform
        epsg = src.crs.to_epsg() if src.crs else None
        if epsg is None:
            raise RuntimeError(f"{path.name} has no EPSG code")
        return src.read(1), (t.a, t.b, t.c, t.d, t.e, t.f), epsg


def _is_fresh(target: Path, source: Path) -> bool:
    return target.exists() and target.stat().st_mtime >= source.stat().st_mtime


def render_overlay_file(run: RunPaths, artifact: Artifact, kind: str, max_px: int = 1200):
    """Write ``figures/_overlay_<name>.png`` (+ ``.json`` with bounds); return ``(path, bounds)``.

    Regenerated only when the raster is newer than the cached picture
    (make-style, like the pipeline cache).  The leading underscore marks it
    as a dashboard cache file, not a contract artifact.
    """
    source = run.path(artifact)
    stem = Path(artifact.value).stem
    png = run.root / "figures" / f"_overlay_{stem}.png"
    meta = png.with_suffix(".json")
    if _is_fresh(png, source) and meta.exists():
        try:
            return png, tuple(json.loads(meta.read_text(encoding="utf-8"))["bounds"])
        except (OSError, ValueError, KeyError):
            pass
    array, transform, epsg = _read_raster(source)
    data, bounds = overlay_png(kind, array, transform, epsg, max_px)
    png.parent.mkdir(parents=True, exist_ok=True)
    png.write_bytes(data)
    meta.write_text(json.dumps({"bounds": list(bounds)}), encoding="utf-8")
    return png, bounds


def load_pair(run: RunPaths, max_px: int = 900) -> dict:
    """Before/after VV (dB, downsampled) + shared stretch + smoothed change, for the comparison tab.

    Returns {"before": unit 0..1, "after": unit 0..1, "change": dB, "stretch": (lo, hi)}.
    Raises RuntimeError if rasterio is missing or a file cannot be read.
    """
    before, _, _ = _read_raster(run.path(Artifact.BEFORE))
    after, _, _ = _read_raster(run.path(Artifact.AFTER))
    before = downsample(before.astype(np.float32), max_px)
    after = downsample(after.astype(np.float32), max_px)
    if before.shape != after.shape:
        raise RuntimeError("before and after images have different sizes")
    lo, hi = shared_stretch([before, after])
    k = 2 if max(before.shape) > 200 else 1
    change = nan_smooth(after, k) - nan_smooth(before, k)
    return {"before": unit_gray(before, lo, hi), "after": unit_gray(after, lo, hi),
            "change": change, "stretch": (lo, hi)}


def png_bytes(kind: str, pair: dict, mix: float = 0.0) -> bytes:
    """PNG for ``kind`` in {'before', 'after', 'change', 'blend'} (blend uses ``mix`` 0..1)."""
    if kind == "change":
        return encode_png_rgba(change_rgba(pair["change"]))
    if kind == "blend":
        return encode_png_rgba(gray_from_unit((1.0 - mix) * pair["before"] + mix * pair["after"]))
    return encode_png_rgba(gray_from_unit(pair[kind]))


def render_gray_file(run: RunPaths, artifact: Artifact, max_px: int = 900) -> Path:
    """Grey preview of a backscatter raster for the Before/After tab (cached)."""
    source = run.path(artifact)
    png = run.root / "figures" / f"_preview_{Path(artifact.value).stem}.png"
    if _is_fresh(png, source):
        return png
    array, _, _ = _read_raster(source)
    png.parent.mkdir(parents=True, exist_ok=True)
    png.write_bytes(encode_png_rgba(gray_rgba(downsample(array, max_px))))
    return png
