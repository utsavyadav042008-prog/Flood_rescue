"""
geo_io.py  --  Part 4: reading and writing rasters (GeoTIFF) behind a tiny interface
====================================================================================

Repo location: models/geo_io.py

WHY THIS FILE EXISTS
--------------------
Teammate 3 can only overlay our mask on the map if the output GeoTIFF has the
EXACT same georeference (coordinate system + position + pixel size) as the input
radar images.  So reading and writing rasters correctly is a contract, not a
detail.  This file keeps that logic in one small place.

TRICK T9 (swappable raster backend):
    The real format is GeoTIFF (via `rasterio`).  But `rasterio` is awkward to
    install and needs real data, which makes tests painful.  So all code in
    Part 4 talks to a tiny interface - `Raster`, `read_raster`, `write_raster` -
    and the backend is chosen by FILE EXTENSION:
        .tif / .tiff  -> rasterio (production)
        .npz          -> a plain numpy stand-in (tests and the demo)
    The inference code is identical for both, so the same tests that pass on
    .npz exercise every line of the pipeline except the two thin rasterio
    functions at the bottom of section 3.

CONTENTS
    1. RasterMeta / Raster        what we need to know about a raster
    2. check_same_grid            before/after must sit on the same pixel grid
    3. read_raster / write_raster (backend chosen by extension)
    4. validate_outputs           the checker Teammate 3 can run on our files
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# 1. DATA CONTAINERS
# =============================================================================
@dataclass
class RasterMeta:
    """Georeference of a raster.

    width, height : pixels
    crs           : coordinate system as a string, e.g. "EPSG:32645" (None = unknown)
    transform     : the 6 numbers (a, b, c, d, e, f) of the affine transform that
                    maps pixel (col,row) -> map (x,y):  x = a*col + b*row + c ,
                    y = d*col + e*row + f   (this is rasterio/GDAL's convention)
    pixel_size_m  : pixel edge length in METRES when the CRS is projected in
                    metres, else None.  Used only for area numbers: in a
                    degrees-based CRS a pixel has no fixed area, so we refuse to
                    report km^2 instead of reporting a wrong value.
    """
    width: int
    height: int
    crs: Optional[str]
    transform: Tuple[float, float, float, float, float, float]
    pixel_size_m: Optional[float] = None


@dataclass
class Raster:
    data: np.ndarray        # float32 [C, H, W]; NaN where the file marked no-data
    meta: RasterMeta


# =============================================================================
# 2. GRID CHECK
# =============================================================================
class GridMismatchError(ValueError):
    """before.tif and after.tif are not on the same pixel grid."""


def check_same_grid(a: RasterMeta, b: RasterMeta, tol: float = 1e-6) -> None:
    """Raise unless both rasters have the same size, CRS and transform.

    Comparing change pixel-by-pixel only makes sense on an identical grid.
    Teammate 1 must resample both images onto one grid; if they did not, we stop
    with a clear message instead of producing a silently shifted flood map.
    """
    problems: List[str] = []
    if (a.width, a.height) != (b.width, b.height):
        problems.append(f"size {a.width}x{a.height} vs {b.width}x{b.height}")
    if a.crs != b.crs:
        problems.append(f"CRS {a.crs} vs {b.crs}")
    if not np.allclose(a.transform, b.transform, atol=tol, rtol=0):
        problems.append(f"transform {tuple(round(v, 4) for v in a.transform)} vs "
                        f"{tuple(round(v, 4) for v in b.transform)}")
    if problems:
        raise GridMismatchError("before/after rasters are not on the same grid: " + "; ".join(problems)
                                + ". Ask Teammate 1 to resample both to one grid.")


# =============================================================================
# 3. READ / WRITE  (backend chosen by file extension)
# =============================================================================
def _is_tif(path) -> bool:
    return Path(path).suffix.lower() in (".tif", ".tiff")


def read_raster(path) -> Raster:
    """Read a raster as float32 [C,H,W]; no-data pixels become NaN."""
    return _read_geotiff(path) if _is_tif(path) else _read_npz(path)


def write_raster(path, data: np.ndarray, ref: RasterMeta, dtype, nodata=None,
                 tags: Optional[Dict[str, object]] = None) -> None:
    """Write a SINGLE-band raster `data` [H,W] using the georeference of `ref`.

    `tags` are free-text key/values stored inside the file (model version,
    threshold, ...).  TRICK T10: provenance travels INSIDE the GeoTIFF, so even
    if someone renames or e-mails the file, `gdalinfo flood_mask.tif` still says
    which model and threshold produced it.
    """
    if data.shape != (ref.height, ref.width):
        raise ValueError(f"data {data.shape} does not match grid {(ref.height, ref.width)}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tags = {k: str(v) for k, v in (tags or {}).items()}
    if _is_tif(path):
        _write_geotiff(path, data, ref, dtype, nodata, tags)
    else:
        _write_npz(path, data, ref, dtype, nodata, tags)


# ---- numpy stand-in backend (tests / demo) -----------------------------------
def _read_npz(path) -> Raster:
    with np.load(path, allow_pickle=False) as z:
        data = z["data"].astype(np.float32)
        meta = json.loads(str(z["meta_json"]))
    meta["transform"] = tuple(meta["transform"])
    if data.ndim == 2:
        data = data[None]
    return Raster(data, RasterMeta(**meta))


def _write_npz(path, data, ref, dtype, nodata, tags) -> None:
    np.savez(path, data=np.asarray(data).astype(dtype), meta_json=json.dumps(asdict(ref)),
             nodata=np.array(-1 if nodata is None else nodata), tags_json=json.dumps(tags))


def make_npz_raster(path, data: np.ndarray, meta: RasterMeta) -> None:
    """Create a stand-in 'GeoTIFF' (.npz) from a [C,H,W] array (tests / demo)."""
    np.savez(path, data=data.astype(np.float32), meta_json=json.dumps(asdict(meta)))


# ---- rasterio backend (production; the ONLY code that needs rasterio) ---------
def _read_geotiff(path) -> Raster:
    import rasterio
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        if src.nodata is not None and np.isfinite(src.nodata):
            data[data == src.nodata] = np.nan
        t = src.transform
        pixel_m = None
        if src.crs is not None and src.crs.is_projected and \
                str(src.crs.linear_units).lower() in ("metre", "meter", "m"):
            pixel_m = float(abs(t.a))
        meta = RasterMeta(src.width, src.height, src.crs.to_string() if src.crs else None,
                          (t.a, t.b, t.c, t.d, t.e, t.f), pixel_m)
    return Raster(data, meta)


def _write_geotiff(path, data, ref, dtype, nodata, tags) -> None:
    import rasterio
    from affine import Affine
    big = min(ref.width, ref.height) >= 256
    with rasterio.open(path, "w", driver="GTiff", height=ref.height, width=ref.width, count=1,
                       dtype=np.dtype(dtype).name, crs=ref.crs, transform=Affine(*ref.transform),
                       nodata=nodata, compress="lzw", tiled=big,
                       **({"blockxsize": 256, "blockysize": 256} if big else {})) as dst:
        dst.write(np.asarray(data).astype(dtype), 1)
        dst.update_tags(**tags)


# =============================================================================
# 4. OUTPUT CHECKER
# =============================================================================
def validate_outputs(mask_path, conf_path, ref_path) -> List[str]:
    """Return a list of problems (empty list = the files honour the contract).

    Contract (agreed with Teammates 3 and 4):
      * flood_mask:       one band, values only in {0, 1, 255}
      * flood_confidence: one band, values in [0, 1]
      * both on exactly the SAME grid (size, CRS, transform) as the input
    Teammate 3 can run this on any file we hand over.
    """
    problems: List[str] = []
    ref = read_raster(ref_path)
    for name, p in (("mask", mask_path), ("confidence", conf_path)):
        r = read_raster(p)
        try:
            check_same_grid(ref.meta, r.meta)
        except GridMismatchError as e:
            problems.append(f"{name}: {e}")
        if r.data.shape[0] != 1:
            problems.append(f"{name}: expected 1 band, found {r.data.shape[0]}")
        if name == "mask":
            vals = set(np.unique(r.data[np.isfinite(r.data)]).astype(int).tolist())
            if not vals <= {0, 1, 255}:
                problems.append(f"mask has unexpected values {sorted(vals - {0, 1, 255})}")
        else:
            v = r.data[np.isfinite(r.data)]
            if v.size and (v.min() < -1e-6 or v.max() > 1 + 1e-6):
                problems.append(f"confidence outside [0,1]: {v.min():.3f}..{v.max():.3f}")
    return problems
