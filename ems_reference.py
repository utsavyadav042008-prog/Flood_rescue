"""
ems_reference.py  --  Part 5: loading the EMSR927 reference AND protecting the data rule
========================================================================================

Repo location: models/ems_reference.py

THE RULE (challenge text): Copernicus EMS, UNOSAT or any published damage map are
"NOT allowed as inputs ... These may only be used to check your results. Breaking
this rule leads to DISQUALIFICATION."

This file does two jobs:
    A. turn the reference (vector or raster) into a boolean mask on OUR grid
    B. make accidental rule-breaking visible and hard:
         audit_no_reference_leak()   inputs/training lists must not be EMS data
         lock_prediction()/verify_lock()   freeze our prediction BEFORE opening the reference

CONTENTS
    1. GeoJSON rasterizer (pure PIL; no rasterio needed)
    2. load_reference / load_footprint
    3. leak audit
    4. prediction lock
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "part4_code"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from geo_io import RasterMeta, read_raster  # noqa: E402
from pack_model import sha256_file  # noqa: E402


class DataRuleViolation(RuntimeError):
    """A reference (EMS/UNOSAT) file was found where only allowed inputs may be."""


class LockError(RuntimeError):
    """The prediction changed after it was locked."""


# =============================================================================
# 1. GEOJSON -> MASK
# =============================================================================
def _pixel_coords(ring: Sequence[Sequence[float]], meta: RasterMeta, transformer=None) -> np.ndarray:
    """Map coordinates -> (col, row) pixel coordinates (float), through the inverse
    of the affine transform; optionally reprojecting first (needs pyproj)."""
    pts = np.asarray(ring, np.float64)[:, :2]
    if transformer is not None:
        xs, ys = transformer.transform(pts[:, 0], pts[:, 1])
        pts = np.stack([xs, ys], axis=1)
    a, b, c, d, e, f = meta.transform
    det = a * e - b * d
    x, y = pts[:, 0] - c, pts[:, 1] - f
    col = (e * x - b * y) / det
    row = (-d * x + a * y) / det
    return np.stack([col, row], axis=1)


def _fill_polygon(rings_px: List[np.ndarray], shape, factor: int) -> np.ndarray:
    """Rasterize ONE polygon (exterior + holes) into a bool [H,W] array.

    TRICK T5 (sub-pixel accuracy with only PIL):
        PIL fills whole pixels, which biases area by up to a pixel along every
        edge.  We draw at `factor` x resolution inside the polygon's bounding
        box only, then average factor x factor blocks and keep pixels at least
        50% covered.  That approximates "pixel centre inside the polygon" to
        well under a pixel, and working in the bounding box keeps memory small
        and lets polygons that overlap other polygons stay correct (holes only
        erase their OWN polygon).
    """
    from PIL import Image, ImageDraw
    h, w = shape
    allp = np.concatenate(rings_px)
    x0, y0 = int(np.floor(allp[:, 0].min())) - 1, int(np.floor(allp[:, 1].min())) - 1
    x1, y1 = int(np.ceil(allp[:, 0].max())) + 1, int(np.ceil(allp[:, 1].max())) + 1
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    out = np.zeros(shape, bool)
    if x1c <= x0c or y1c <= y0c:
        return out
    bw, bh = x1c - x0c, y1c - y0c
    img = Image.new("L", (bw * factor, bh * factor), 0)
    draw = ImageDraw.Draw(img)
    for i, ring in enumerate(rings_px):
        pts = [((x - x0c) * factor, (y - y0c) * factor) for x, y in ring]
        draw.polygon(pts, fill=255 if i == 0 else 0)             # exterior, then holes
    arr = np.asarray(img, np.float32) / 255.0
    cov = arr.reshape(bh, factor, bw, factor).mean(axis=(1, 3))
    out[y0c:y1c, x0c:x1c] = cov >= 0.5
    return out


def _geometries(geojson: Dict) -> List[Dict]:
    t = geojson.get("type")
    if t == "FeatureCollection":
        return [g for f in geojson["features"] for g in _geometries(f)]
    if t == "Feature":
        return _geometries(geojson["geometry"]) if geojson.get("geometry") else []
    if t == "GeometryCollection":
        return [g for sub in geojson["geometries"] for g in _geometries(sub)]
    return [geojson] if t in ("Polygon", "MultiPolygon") else []


def rasterize_geojson(geojson: Dict, meta: RasterMeta, src_crs: Optional[str] = None) -> np.ndarray:
    """GeoJSON (Polygon / MultiPolygon, any nesting) -> bool mask [H,W] on `meta`'s grid.

    src_crs: CRS of the coordinates (GeoJSON is normally EPSG:4326).  If it
    differs from the grid CRS, pyproj is needed (pip install pyproj).
    Points and lines are ignored (they have no area).
    """
    transformer = None
    if src_crs and meta.crs and src_crs != meta.crs:
        try:
            from pyproj import Transformer
        except ImportError:
            raise ImportError(f"reference CRS {src_crs} differs from the grid CRS {meta.crs}; "
                              f"install pyproj to reproject (pip install pyproj)")
        transformer = Transformer.from_crs(src_crs, meta.crs, always_xy=True)
    shape = (meta.height, meta.width)
    factor = 4 if meta.height * meta.width <= 4_000_000 else (2 if meta.height * meta.width <= 16_000_000 else 1)
    mask = np.zeros(shape, bool)
    for g in _geometries(geojson):
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            rings = [_pixel_coords(r, meta, transformer) for r in poly]
            mask |= _fill_polygon(rings, shape, factor)
    return mask


# =============================================================================
# 2. LOADING
# =============================================================================
def load_reference(paths: Sequence, meta: RasterMeta, src_crs: Optional[str] = "EPSG:4326") -> np.ndarray:
    """Union of one or more reference files -> bool mask on OUR grid.

    Accepts GeoJSON (.geojson/.json), raster masks on the same grid (.tif/.npz,
    non-zero = flood), and (optionally, needs geopandas) .shp/.gpkg/.kml.
    EMS flood products usually come as several layers (flood extent, debris,
    ...): pass every layer you consider "flooded or debris-covered".
    """
    out = np.zeros((meta.height, meta.width), bool)
    for p in paths:
        suffix = Path(p).suffix.lower()
        if suffix in (".geojson", ".json"):
            out |= rasterize_geojson(json.loads(Path(p).read_text()), meta, src_crs)
        elif suffix in (".tif", ".tiff", ".npz"):
            r = read_raster(p)
            if (r.meta.width, r.meta.height) != (meta.width, meta.height):
                raise ValueError(f"reference raster {Path(p).name} is not on our grid; resample it first")
            out |= np.nan_to_num(r.data[0]) > 0
        elif suffix in (".shp", ".gpkg", ".kml"):
            try:
                import geopandas as gpd
                from shapely.geometry import mapping
            except ImportError:
                raise ImportError("reading shapefiles needs geopandas: pip install geopandas")
            gdf = gpd.read_file(p).to_crs(meta.crs)
            gj = {"type": "GeometryCollection", "geometries": [mapping(g) for g in gdf.geometry if g is not None]}
            out |= rasterize_geojson(gj, meta, src_crs=meta.crs)
        else:
            raise ValueError(f"unsupported reference format: {p}")
    return out


def load_footprint(path, meta: RasterMeta, src_crs: Optional[str] = "EPSG:4326") -> np.ndarray:
    """Reference area of interest (where the reference actually mapped) -> bool mask."""
    return load_reference([path], meta, src_crs)


# =============================================================================
# 3. LEAK AUDIT
# =============================================================================
FORBIDDEN_NAME_PARTS = ("emsr", "unosat", "mapping.emergency", "copernicus_ems", "copernicus-ems")


def audit_no_reference_leak(input_files: Iterable, reference_files: Iterable = (),
                            extra_listed_paths: Iterable = ()) -> List[str]:
    """Return violations (empty list = clean).

    TRICK T6 (two independent tripwires, because file names lie):
      (a) NAME check: any input/training path containing emsr / unosat / ...
      (b) CONTENT check: SHA-256 of every input compared with SHA-256 of every
          reference file.  A reference file renamed to 'before.tif' still
          matches, which the name check alone would miss.
    `extra_listed_paths` is for text lists such as the training file list from
    Part 1's splits.json.  It is cheap, so run it before EVERY case-study run.
    """
    violations: List[str] = []
    ref_hash = {sha256_file(r): str(r) for r in reference_files if Path(r).exists()}
    for p in list(input_files) + list(extra_listed_paths):
        low = str(p).lower()
        for bad in FORBIDDEN_NAME_PARTS:
            if bad in low:
                violations.append(f"path '{p}' contains '{bad}' (EMS/UNOSAT data may not be an input)")
                break
    for p in input_files:
        if Path(p).exists():
            h = sha256_file(p)
            if h in ref_hash:
                violations.append(f"input '{p}' is byte-identical to reference file '{ref_hash[h]}'")
    return violations


# =============================================================================
# 4. PREDICTION LOCK
# =============================================================================
def lock_prediction(paths: Dict[str, str], lock_path, threshold: float, threshold_source: str) -> Dict:
    """Record SHA-256 + time of our prediction files BEFORE the reference is read.

    TRICK T7 (pre-registration, like a sealed envelope):
        Scientists "pre-register" a hypothesis so they cannot quietly tweak it
        after seeing the answer.  We do the same: the comparison step refuses to
        run unless the prediction files still match the hashes written here, and
        the report quotes the lock time and threshold.  Any re-tuning after
        seeing EMSR927 would require re-running prediction, which creates a NEW
        lock that is visible in the file history.
        HONEST LIMIT: this is a process control that makes accidents visible and
        gives judges something checkable.  It cannot stop someone determined to
        cheat, which is why the system must also run live on the judges' data.
    """
    lock = {"locked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "threshold": threshold,
            "threshold_source": threshold_source,
            "files": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in paths.items()}}
    Path(lock_path).write_text(json.dumps(lock, indent=2))
    return lock


def verify_lock(lock_path) -> Dict:
    lock = json.loads(Path(lock_path).read_text())
    for k, info in lock["files"].items():
        if not Path(info["path"]).exists() or sha256_file(info["path"]) != info["sha256"]:
            raise LockError(f"prediction file '{k}' changed after the lock ({lock['locked_at']}); "
                            f"re-run the prediction and accept the new lock")
    return lock
