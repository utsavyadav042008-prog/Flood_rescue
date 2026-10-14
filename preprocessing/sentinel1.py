"""
preprocessing/sentinel1.py  —  PART 2: Sentinel-1 Preprocessing
================================================================

WHAT THIS FILE DOES (one paragraph)
-----------------------------------
Part 1 gave us downloaded Sentinel-1 GRD zip files (a "before" pass and an
"after" pass on the same track). A raw GRD file cannot be compared directly:
it has border noise, uneven brightness, grainy speckle and mountain
distortion. This part turns each pass into ONE clean, georeferenced GeoTIFF with
two bands, VV and VH backscatter in dB, on a 10 m UTM grid, cropped to the AOI.

PUBLIC ENTRY POINT
------------------
    pair   = ScenePair.load_json("data/processed/<key>/scene_pair.json")  # Part 1
    result = preprocess_pair(pair)
    result.before.tif_path   ->  s1_before_<hash>.tif   (band 1 = VV dB, band 2 = VH dB)
    result.after.tif_path    ->  s1_after_<hash>.tif
    result.warnings          ->  caveats for the report

THE HEAVY LIFTING IS DONE BY ESA SNAP (``gpt`` command line). Python's job is:
build the SNAP graph, run it, stitch slices, convert to dB, clean up edges, run
sanity checks, cache, and write metadata. Install SNAP separately (free), and set
the environment variable SNAP_GPT to the gpt executable if it is not on PATH.

MAP OF THE FILE (read top to bottom)
------------------------------------
  1. Errors ................ SnapError carries a human hint, not just a stack trace
  2. Config ................ every tunable in one place + the params hash
  3. Small helpers ......... UTM zone, dB, normalisation, erosion
  4. GeoRaster + IO ........ tiny raster container; rasterio only at the edges
  5. Footprint-edge maths .. decides if it is safe to crop before processing
  6. SNAP graph ............ build the XML, run gpt, diagnose failures
  7. Per-slice processing .. one zip -> linear VV / VH rasters
  8. Mosaic + clean + QA ... stitch slices, linear -> dB, sanity checks
  9. Orchestration ......... preprocess_acquisition / preprocess_pair, caching
 10. CLI ................... ``python -m preprocessing.sentinel1 --help``

NON-OBVIOUS TRICKS USED HERE (search for "TRICK n" in the code)
---------------------------------------------------------------
  TRICK 1  ONE config object processes before AND after, and a params hash is
           stamped on both outputs. "Same settings" is guaranteed, not hoped for.
  TRICK 2  SNAP writes the final CRS (UTM) directly and snaps its output to a
           global 10 m lattice (alignToStandardGrid). Before, after and every
           slice then share the same pixel lattice, so stitching is an integer
           paste and Part 3 barely has to resample (resampling blurs).
  TRICK 3  SNAP reads the .zip directly (no 4 GB unzip) and the graph XML is
           generated from code and saved next to the outputs as a receipt.
  TRICK 4  Crop BEFORE processing only when the AOI is far from the scene edge
           (distance to the footprint, from Part 1). Edge noise removal needs
           the real scene edge, so near the edge we process the full scene.
  TRICK 5  Speckle filtering and stitching happen in LINEAR units; dB is the very
           last step. Averaging or filtering in dB is mathematically wrong.
  TRICK 6  VV and VH are written as separate single-band files, so we never rely
           on SNAP's band order. A VV-minus-VH plausibility check catches a swap.
  TRICK 7  No-data is 0 in SNAP output. log10(0) = -inf would poison everything,
           so <= 0 becomes NaN, never -inf and never a fake very low dB value.
  TRICK 8  Edge guard: erode the valid mask a few pixels, but with
           border_value=1 so the edge of the AOI crop is NOT eaten.
  TRICK 9  Apply-Orbit-File continues on failure: precise orbit files for very
           recent scenes may not exist yet; we fall back and record it.
  TRICK 10 nodataValueAtSea=false: otherwise terrain correction blanks pixels
           at DEM height 0, which would erase rivers and flat wet ground.
  TRICK 11 SNAP errors are matched against known log patterns and turned into a
           plain-English fix ("raise --mem-gb", "needs internet for orbit file").
  TRICK 12 SNAP runner and raster I/O are injected, so the logic is unit-tested
           offline with fakes, and ``--dry-run`` shows graphs without SNAP.
  TRICK 13 Cache key = product ids + params hash; changing the speckle filter
           re-processes, re-running the same thing returns instantly.

IMPORTANT: SNAP parameter names below follow my knowledge of SNAP 9/10. They
could not be run here. Verify each operator once with ``gpt <Operator> -h`` and
use ``--dry-run`` to inspect the generated graphs before the first real run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

from acquisition.sentinel import (AOI, Acquisition, Scene, ScenePair,
                                  _points_in_ring, run_key)

log = logging.getLogger("preprocessing.sentinel1")

# Bump when the generated graph changes in a way that changes pixel values, so
# old cached outputs are not reused by mistake (TRICK 13).
GRAPH_VERSION = "1"


# ============================================================================
# 1. ERRORS
# ============================================================================

class PreprocessingError(Exception):
    """Base class for everything this module raises on purpose."""


class SnapNotFoundError(PreprocessingError):
    """The SNAP ``gpt`` executable could not be found."""


class SnapError(PreprocessingError):
    """gpt failed. ``hint`` is a plain-English suggestion, ``log_path`` the full log."""

    def __init__(self, message: str, hint: str = "", log_path: Optional[Path] = None):
        super().__init__(message)
        self.message, self.hint, self.log_path = message, hint, log_path

    def __str__(self) -> str:
        out = self.message
        if self.hint:
            out += f"\n  Hint: {self.hint}"
        if self.log_path:
            out += f"\n  Full log: {self.log_path}"
        return out


class NoUsableDataError(PreprocessingError):
    """Processing worked but (almost) no valid pixels remain."""


# ============================================================================
# 2. CONFIG
# ============================================================================

@dataclass
class PreprocessConfig:
    # ---- output grid
    res_m: float = 10.0
    crs_epsg: Optional[int] = None        # None = UTM zone of the AOI centre
    crop_margin_km: float = 0.5           # final crop slightly larger than AOI (Part 3 resamples)

    # ---- SNAP processing choices (these change pixel values => part of the hash)
    speckle_filter: str = "Refined Lee"
    speckle_size: int = 5                 # ignored by filters with a fixed window
    dem_name: str = "Copernicus 30m Global DEM"   # SNAP downloads tiles (needs internet)
    dem_path: Optional[str] = None        # or a local DEM GeoTIFF (offline, reproducible)
    dem_apply_egm: bool = True            # Copernicus DEM heights are above the geoid
    orbit_type: str = "Sentinel Precise (Auto Download)"
    orbit_continue_on_fail: bool = True   # TRICK 9
    align_to_standard_grid: bool = True   # TRICK 2
    save_incidence_angle: bool = False    # also output local incidence angle (see Part 3 notes)

    # ---- cleaning (also part of the hash)
    edge_guard_px: int = 3
    clip_db: tuple = (-50.0, 20.0)

    # ---- speed/safety only (NOT part of the hash)
    subset_early: str = "auto"            # "auto" | "always" | "never"   (TRICK 4)
    margin_km: float = 3.0                # context kept around AOI when cropping early
    edge_safe_km: float = 12.0            # AOI must be this far inside the scene to crop early

    # ---- QA thresholds
    min_valid_fraction: float = 0.5       # below this: warn
    fatal_valid_fraction: float = 0.02    # below this: raise NoUsableDataError
    vv_vh_gap_db: tuple = (1.0, 20.0)     # plausible median(VV) - median(VH) over land

    # ---- SNAP runtime
    gpt_path: Optional[str] = None
    snap_mem_gb: int = 8
    snap_threads: Optional[int] = None
    timeout_s: int = 7200

    def effective_params(self, aoi: AOI, crs: str) -> dict:
        """Everything that can change a pixel value. Hashed (TRICK 1 / 13)."""
        return {
            "graph_version": GRAPH_VERSION, "res_m": self.res_m, "crs": crs,
            "aoi": [round(v, 5) for v in aoi.as_tuple()],
            "crop_margin_km": self.crop_margin_km,
            "speckle_filter": self.speckle_filter, "speckle_size": self.speckle_size,
            "dem": self.dem_path or self.dem_name, "dem_apply_egm": self.dem_apply_egm,
            "orbit_type": self.orbit_type, "align_to_standard_grid": self.align_to_standard_grid,
            "save_incidence_angle": self.save_incidence_angle,
            "edge_guard_px": self.edge_guard_px, "clip_db": list(self.clip_db),
        }

    def params_hash(self, aoi: AOI, crs: str) -> str:
        blob = json.dumps(self.effective_params(aoi, crs), sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]


# ============================================================================
# 3. SMALL HELPERS
# ============================================================================

def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat).
    zone = floor((lon + 180) / 6) + 1 ;  north: 32600 + zone, south: 32700 + zone.
    Nepal east of 84 E -> zone 45 -> EPSG:32645; west of 84 E -> zone 44.
    Part 3 should import this function so both parts always agree."""
    zone = min(60, max(1, int((lon + 180.0) // 6) + 1))
    return (32600 if lat >= 0 else 32700) + zone


def to_db(linear: np.ndarray) -> np.ndarray:
    """10*log10(x), with every non-positive or non-finite input -> NaN (TRICK 7).

    SNAP stores 'no data' as 0. Naive log10(0) gives -inf, which breaks
    statistics, normalisation and the neural network. We also deliberately do
    NOT replace zeros with a tiny epsilon: that would invent very dark pixels
    that look exactly like calm water.
    """
    lin = np.asarray(linear, dtype=np.float32)
    out = np.full(lin.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(lin) & (lin > 0)
    out[ok] = 10.0 * np.log10(lin[ok])
    return out


def normalize_db(x: np.ndarray, lo: float = -30.0, hi: float = 0.0) -> np.ndarray:
    """Map dB values from [lo, hi] to [0, 1] (clipping outside). NaN stays NaN.

    Teammate 2 imports this so TRAINING and INFERENCE scale identically.
    Whatever range the training data used must be used here (see Part 2 notes).
    """
    x = np.asarray(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def erode_valid(valid: np.ndarray, px: int) -> np.ndarray:
    """TRICK 8 — shrink the valid area by ``px`` pixels around internal no-data.

    Swath edges and cut seams leave a ragged rim of unreliable pixels. Erosion
    removes it. ``border_value=1`` treats the outside of the array as VALID;
    without it, erosion would also eat ``px`` pixels all around the AOI crop,
    which is real data we want to keep.
    """
    if px <= 0:
        return valid
    return ndimage.binary_erosion(valid, structure=np.ones((3, 3), dtype=bool),
                                  iterations=px, border_value=1)


# ============================================================================
# 4. GeoRaster + IO   (rasterio is imported lazily, only inside RasterioIO)
# ============================================================================

@dataclass
class GeoRaster:
    """Minimal north-up raster with square pixels in a projected CRS.

    data  (bands, rows, cols) float32, NaN = no data
    x0,y0 coordinates of the TOP-LEFT corner (west/north edge of pixel 0,0)
    res   pixel size in metres
    crs   "EPSG:32645"
    This is all SNAP's UTM output needs; keeping it tiny makes the logic
    testable without any GIS library.
    """
    data: np.ndarray
    x0: float
    y0: float
    res: float
    crs: str

    @property
    def bands(self) -> int:
        return self.data.shape[0]

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def width(self) -> int:
        return self.data.shape[2]

    def with_data(self, data: np.ndarray) -> "GeoRaster":
        return GeoRaster(data, self.x0, self.y0, self.res, self.crs)


def grid_offset_px(r: GeoRaster) -> tuple:
    """Distance (in pixels, 0..0.5) between this raster's origin and the global
    lattice of multiples of ``res``. ~0 means SNAP honoured alignToStandardGrid."""
    def frac(v):
        f = (v / r.res) % 1.0
        return min(f, 1.0 - f)
    return frac(r.x0), frac(r.y0)


class RasterioIO:
    """Reads/writes GeoTIFFs with rasterio. The ONLY place rasterio is used."""

    @staticmethod
    def _rio():
        try:
            import rasterio
            return rasterio
        except ImportError as exc:                       # pragma: no cover
            raise PreprocessingError("rasterio is required: pip install rasterio") from exc

    def read(self, path) -> GeoRaster:
        rio = self._rio()
        with rio.open(path) as src:
            t = src.transform
            if abs(t.b) > 1e-9 or abs(t.d) > 1e-9 or abs(abs(t.a) - abs(t.e)) > 1e-6:
                raise PreprocessingError(
                    f"{path}: rotated or non-square pixels are not supported "
                    "(expected SNAP's north-up UTM output)")
            data = src.read().astype("float32")
            nodata = src.nodata
            if nodata is not None and not np.isnan(nodata):
                data[data == nodata] = np.nan
            return GeoRaster(data, t.c, t.f, abs(t.a), src.crs.to_string())

    def write(self, path, r: GeoRaster, descriptions: Optional[Sequence[str]] = None,
              tags: Optional[dict] = None, dtype: str = "float32",
              nodata: Optional[float] = float("nan")) -> None:
        """Write a GeoTIFF. Defaults suit dB/float data (float32, NaN no-data).
        Masks use ``dtype="uint8", nodata=None`` (Part 3): 1 byte per pixel."""
        rio = self._rio()
        from rasterio.transform import from_origin
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with rio.open(path, "w", driver="GTiff", height=r.height, width=r.width,
                      count=r.bands, dtype=dtype, crs=r.crs,
                      transform=from_origin(r.x0, r.y0, r.res, r.res), nodata=nodata,
                      tiled=True, compress="deflate",
                      predictor=3 if dtype.startswith("float") else 2) as dst:
            dst.write(r.data.astype(dtype))
            for i, d in enumerate(descriptions or [], start=1):
                dst.set_band_description(i, d)
            if tags:
                dst.update_tags(**{k: str(v) for k, v in tags.items()})


# ============================================================================
# 5. FOOTPRINT-EDGE MATHS  (decides TRICK 4)
# ============================================================================

def perimeter_points(aoi: AOI, n: int = 40) -> np.ndarray:
    """(4n, 2) array of (lon, lat) points along the AOI rectangle's outline."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    a, b, c, d = aoi.min_lon, aoi.min_lat, aoi.max_lon, aoi.max_lat
    return np.vstack([
        np.column_stack([a + t * (c - a), np.full(n, b)]),
        np.column_stack([np.full(n, c), b + t * (d - b)]),
        np.column_stack([c - t * (c - a), np.full(n, d)]),
        np.column_stack([np.full(n, a), d - t * (d - b)]),
    ])


def distance_to_footprint_edge_km(points: np.ndarray, rings: list) -> float:
    """SIGNED distance (km) from the nearest of ``points`` to the footprint
    boundary: positive = all points inside, negative = at least one outside.

    Uses a local flat-earth conversion (1 deg lat = 110.57 km, 1 deg lon =
    111.32*cos(lat) km): accurate to well under 1 % over a few hundred km.
    """
    pts = np.asarray(points, dtype=float)
    lon0, lat0 = pts[:, 0].mean(), pts[:, 1].mean()
    kx, ky = 111.32 * math.cos(math.radians(lat0)), 110.57

    def km(a: np.ndarray) -> np.ndarray:
        return np.column_stack([(a[:, 0] - lon0) * kx, (a[:, 1] - lat0) * ky])

    p_km = km(pts)
    inside = np.zeros(len(pts), dtype=bool)
    dmin = np.full(len(pts), np.inf)
    for ring in rings:
        r = np.asarray(ring, dtype=float)[:, :2]
        if not np.allclose(r[0], r[-1]):
            r = np.vstack([r, r[0]])                        # close the ring
        inside |= _points_in_ring(pts[:, 0], pts[:, 1], r)
        rk = km(r)
        a, b = rk[:-1], rk[1:]
        ab = b - a
        l2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
        ap = p_km[:, None, :] - a[None, :, :]
        t = np.clip((ap * ab[None]).sum(axis=2) / l2[None], 0.0, 1.0)
        proj = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(p_km[:, None, :] - proj, axis=2).min(axis=1)
        dmin = np.minimum(dmin, d)
    return float(np.where(inside, dmin, -dmin).min())


def buffered_aoi(aoi: AOI, margin_km: float) -> AOI:
    """AOI grown by ``margin_km`` on every side (clamped to valid lon/lat)."""
    lat_c = (aoi.min_lat + aoi.max_lat) / 2
    dlat = margin_km / 110.57
    dlon = margin_km / (111.32 * max(math.cos(math.radians(lat_c)), 0.05))
    return AOI(max(-180.0, aoi.min_lon - dlon), max(-90.0, aoi.min_lat - dlat),
               min(180.0, aoi.max_lon + dlon), min(90.0, aoi.max_lat + dlat))


def choose_subset_mode(aoi: AOI, scene: Scene, cfg: PreprocessConfig) -> tuple:
    """TRICK 4 — decide whether we may crop BEFORE the heavy processing.

    Cropping first is a huge saving (a full GRD scene is ~25 000 x 17 000 px).
    But border-noise removal works on the true scene edge. If the AOI is close
    to that edge (or partly outside this slice, e.g. near a slice boundary) we
    process the whole slice, slower but correct.

    Returns (crop_early: bool, distance_km: float)  (distance is signed).
    """
    area = buffered_aoi(aoi, cfg.margin_km)
    dist = distance_to_footprint_edge_km(perimeter_points(area), scene.rings)
    if cfg.subset_early == "always":
        return True, dist
    if cfg.subset_early == "never":
        return False, dist
    return dist >= cfg.edge_safe_km, dist


# ============================================================================
# 6. SNAP GRAPH + RUNNER
# ============================================================================

def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _add_node(graph: ET.Element, node_id: str, operator: str,
              sources: Sequence[str], params: dict) -> None:
    node = ET.SubElement(graph, "node", id=node_id)
    ET.SubElement(node, "operator").text = operator
    src = ET.SubElement(node, "sources")
    for i, ref in enumerate(sources):
        # gpt expects <sourceProduct refid="..."/> (first) then <sourceProduct.1 .../>
        tag = "sourceProduct" if i == 0 else f"sourceProduct.{i}"
        ET.SubElement(src, tag, refid=ref)
    p = ET.SubElement(node, "parameters", {"class": "com.bc.ceres.binding.dom.XppDomElement"})
    for k, v in params.items():
        if v is not None:
            ET.SubElement(p, k).text = _fmt(v)


def build_graph_xml(zip_path: Path, outputs: dict, aoi: AOI, crs: str,
                    cfg: PreprocessConfig, crop_early: bool) -> str:
    """Generate the SNAP processing graph as XML text (TRICK 3).

    outputs: {"VV": Path, "VH": Path, optional "LIA": Path}  (GeoTIFF files)

    Chain (order follows ESA's standard Sentinel-1 GRD workflow):

      Read(zip) -> [Subset early] -> Apply-Orbit-File -> Remove-GRD-Border-Noise
        -> ThermalNoiseRemoval -> Calibration(sigma0, linear) -> Speckle-Filter
        -> Terrain-Correction(UTM, 10 m, standard grid) -> Subset(AOI) per band -> Write

    The chain ends in SEPARATE single-band files (TRICK 6). Output stays LINEAR
    (TRICK 5): we convert to dB later in Python.
    """
    g = ET.Element("graph", id="s1_preprocess")
    ET.SubElement(g, "version").text = "1.0"

    _add_node(g, "Read", "Read", [], {"file": str(zip_path), "formatName": "SENTINEL-1"})
    last = "Read"

    if crop_early:
        _add_node(g, "Subset-Early", "Subset", [last], {
            "geoRegion": buffered_aoi(aoi, cfg.margin_km).odata_polygon(),
            "subSamplingX": 1, "subSamplingY": 1, "fullSwath": False, "copyMetadata": True})
        last = "Subset-Early"

    _add_node(g, "Orbit", "Apply-Orbit-File", [last], {
        "orbitType": cfg.orbit_type, "polyDegree": 3,
        "continueOnFail": cfg.orbit_continue_on_fail})                       # TRICK 9
    _add_node(g, "Border", "Remove-GRD-Border-Noise", ["Orbit"], {
        "selectedPolarisations": "VV,VH", "borderLimit": 1000, "trimThreshold": 0.5})
    _add_node(g, "Thermal", "ThermalNoiseRemoval", ["Border"], {
        "selectedPolarisations": "VV,VH", "removeThermalNoise": True})
    _add_node(g, "Calib", "Calibration", ["Thermal"], {
        "selectedPolarisations": "VV,VH", "outputSigmaBand": True,
        "outputImageScaleInDb": False})
    _add_node(g, "Speckle", "Speckle-Filter", ["Calib"], {
        "filter": cfg.speckle_filter, "filterSizeX": cfg.speckle_size,
        "filterSizeY": cfg.speckle_size})

    tc = {
        "sourceBands": "Sigma0_VV,Sigma0_VH",
        "demResamplingMethod": "BILINEAR_INTERPOLATION",
        "imgResamplingMethod": "BILINEAR_INTERPOLATION",
        "pixelSpacingInMeter": float(cfg.res_m),
        "mapProjection": crs,                                  # final CRS directly (TRICK 2)
        "alignToStandardGrid": cfg.align_to_standard_grid,
        "standardGridOriginX": 0.0, "standardGridOriginY": 0.0,
        "nodataValueAtSea": False,                             # TRICK 10
        "saveSelectedSourceBand": True,
        "saveLocalIncidenceAngle": cfg.save_incidence_angle,
    }
    if cfg.dem_path:
        tc.update({"demName": "External DEM", "externalDEMFile": str(cfg.dem_path),
                   "externalDEMNoDataValue": 0.0, "externalDEMApplyEGM": cfg.dem_apply_egm})
    else:
        tc["demName"] = cfg.dem_name
    _add_node(g, "TC", "Terrain-Correction", ["Speckle"], tc)

    crop_wkt = buffered_aoi(aoi, cfg.crop_margin_km).odata_polygon()
    bands = {"VV": "Sigma0_VV", "VH": "Sigma0_VH"}
    if cfg.save_incidence_angle:
        bands["LIA"] = "localIncidenceAngle"
    for key, band in bands.items():
        _add_node(g, f"Crop-{key}", "Subset", ["TC"], {
            "sourceBands": band, "geoRegion": crop_wkt, "copyMetadata": False})
        _add_node(g, f"Write-{key}", "Write", [f"Crop-{key}"], {
            "file": str(outputs[key]), "formatName": "GeoTIFF"})

    ET.indent(g)
    return ET.tostring(g, encoding="unicode")


# Known failure signatures in SNAP logs -> what the human should do (TRICK 11).
_SNAP_HINTS = [
    (r"OutOfMemoryError|Java heap space|GC overhead",
     "SNAP ran out of memory. Raise --mem-gb (e.g. 12), shrink the AOI, or keep early cropping on."),
    (r"[Uu]nknown operator|operator.*not found|No operator",
     "This SNAP version lacks an operator used here. Update SNAP (9+) and check `gpt -h`."),
    (r"orbit|Orbit.*(not found|unavailable|download)",
     "Orbit file could not be fetched. SNAP needs internet for 'Auto Download', or place "
     "orbit files in ~/.snap/auxdata/Orbits. (continueOnFail normally avoids a hard stop.)"),
    (r"Copernicus.*DEM|DEM.*(download|not found|unreachable)|externalDEM",
     "DEM problem. Needs internet for the built-in DEM, or set a local DEM with --dem."),
    (r"[Nn]o intersection|outside.*product|empty.*region|region.*empty",
     "The AOI does not intersect this scene. Re-run Part 1 or check the AOI coordinates."),
    (r"Cannot read|not a valid|Unexpected EOF|zip.*(corrupt|invalid)|ZipException",
     "The input zip looks corrupt or incomplete. Delete it and download it again (Part 1)."),
    (r"parameter.*(not|unknown)|Unknown parameter|illegal parameter",
     "An operator parameter name differs in your SNAP version. Run `gpt <Operator> -h` and "
     "adjust build_graph_xml()."),
]


def diagnose_snap_log(text: str) -> str:
    """Plain-English suggestion for a failed gpt run, or '' if nothing matches."""
    for pattern, hint in _SNAP_HINTS:
        if re.search(pattern, text):
            return hint
    return ""


def find_gpt(explicit: Optional[str] = None) -> str:
    """Locate SNAP's gpt: explicit arg, $SNAP_GPT, PATH, then common install dirs."""
    candidates = [explicit, os.environ.get("SNAP_GPT"), shutil.which("gpt"),
                  str(Path.home() / "esa-snap" / "bin" / "gpt"),
                  str(Path.home() / "snap" / "bin" / "gpt"),
                  "/opt/snap/bin/gpt", "/usr/local/snap/bin/gpt",
                  r"C:\Program Files\esa-snap\bin\gpt.exe",
                  r"C:\Program Files\snap\bin\gpt.exe"]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    raise SnapNotFoundError(
        "SNAP 'gpt' not found. Install ESA SNAP (free), then either add its bin folder "
        "to PATH or set SNAP_GPT=/path/to/gpt (or pass --gpt).")


class SnapRunner:
    """Runs ``gpt graph.xml``; the only code that spawns SNAP (injectable, TRICK 12)."""

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self._gpt: Optional[str] = None

    @property
    def gpt(self) -> str:
        if self._gpt is None:
            self._gpt = find_gpt(self.cfg.gpt_path)
        return self._gpt

    def command(self, graph_path: Path) -> list:
        mem = self.cfg.snap_mem_gb
        cmd = [self.gpt, str(graph_path), f"-J-Xmx{mem}G", "-c", f"{max(1, int(mem * 0.6))}G"]
        if self.cfg.snap_threads:
            cmd += ["-q", str(self.cfg.snap_threads)]
        return cmd

    def run(self, graph_path: Path, log_path: Path) -> None:
        cmd = self.command(graph_path)
        log.info("SNAP: %s", " ".join(cmd))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "w") as fh:
                proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                      timeout=self.cfg.timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise SnapError(f"SNAP exceeded the {self.cfg.timeout_s}s time limit.",
                            "Use a smaller AOI or raise timeout_s.", log_path) from exc
        text = log_path.read_text(errors="replace")
        # gpt sometimes prints an error but exits 0, so check the text as well.
        if proc.returncode != 0 or re.search(r"^Error:|\bERROR\b.*Exception", text, re.M):
            tail = "\n".join(text.strip().splitlines()[-12:])
            raise SnapError(f"gpt failed (exit {proc.returncode}). Last log lines:\n{tail}",
                            diagnose_snap_log(text), log_path)


# ============================================================================
# 7. PER-SLICE PROCESSING
# ============================================================================

@dataclass
class SliceOutput:
    vv: Optional[GeoRaster]
    vh: Optional[GeoRaster]
    lia: Optional[GeoRaster]
    info: dict


def process_slice(scene: Scene, aoi: AOI, role: str, k: int, work_dir: Path, crs: str,
                  cfg: PreprocessConfig, io, runner, dry_run: bool = False) -> SliceOutput:
    """One downloaded zip -> linear sigma0 VV / VH rasters (UTM, 10 m, cropped).

    Writes ``graph.xml`` (the receipt) and ``snap.log`` into the slice folder.
    With ``dry_run`` only the graph and the command are produced.
    """
    zip_path = Path(scene.local_path) if scene.local_path else \
        Path("data/raw/s1") / f"{scene.stem}.zip"
    if not dry_run and not zip_path.exists():
        raise PreprocessingError(f"{zip_path} does not exist. Run Part 1 with download=True.")

    crop_early, dist_km = choose_subset_mode(aoi, scene, cfg)
    sd = Path(work_dir) / f"slice{k}"
    sd.mkdir(parents=True, exist_ok=True)
    outs = {"VV": sd / "sigma0_vv_lin.tif", "VH": sd / "sigma0_vh_lin.tif"}
    if cfg.save_incidence_angle:
        outs["LIA"] = sd / "lia.tif"

    graph_path = sd / "graph.xml"
    graph_path.write_text(build_graph_xml(zip_path, outs, aoi, crs, cfg, crop_early))
    info = {"scene": scene.name, "crop_early": crop_early,
            "distance_to_scene_edge_km": round(dist_km, 2), "graph": str(graph_path)}
    if dry_run:
        info["command"] = " ".join(runner.command(graph_path)) if runner else ""
        return SliceOutput(None, None, None, info)

    runner.run(graph_path, sd / "snap.log")
    for key, path in outs.items():
        if not Path(path).exists():
            raise SnapError(f"gpt finished but did not write {path.name}.",
                            "Check snap.log; a Write node may have been skipped.", sd / "snap.log")
    return SliceOutput(io.read(outs["VV"]), io.read(outs["VH"]),
                       io.read(outs["LIA"]) if "LIA" in outs else None, info)


# ============================================================================
# 8. MOSAIC + CLEAN + QA   (pure numpy: fully testable)
# ============================================================================

def mosaic(rasters: Sequence[GeoRaster]) -> tuple:
    """Stitch rasters that share a CRS and pixel size into one.

    Returns (raster, max_fractional_offset_px).

    Because of TRICK 2 all inputs sit on the same global lattice, so each raster
    is pasted at an INTEGER pixel offset (no resampling, no blur). If a SNAP
    version ignored alignToStandardGrid, offsets are rounded to the nearest pixel
    (error <= 0.5 px = 5 m) and the worst error is reported so a warning can say so.
    Overlaps: the earlier raster wins (slices of one pass have equal radiometry,
    so seams are invisible).
    """
    if not rasters:
        raise PreprocessingError("nothing to mosaic")
    ref = rasters[0]
    for r in rasters[1:]:
        if r.crs != ref.crs or abs(r.res - ref.res) > 1e-6 or r.bands != ref.bands:
            raise PreprocessingError("cannot mosaic rasters with different CRS / pixel size / bands")
    if len(rasters) == 1:
        return ref, 0.0

    offs, max_frac = [], 0.0
    for r in rasters:
        cx, cy = (r.x0 - ref.x0) / ref.res, (ref.y0 - r.y0) / ref.res   # y grows downward
        ix, iy = int(round(cx)), int(round(cy))
        max_frac = max(max_frac, abs(cx - ix), abs(cy - iy))
        offs.append((ix, iy))
    min_x = min(o[0] for o in offs)
    min_y = min(o[1] for o in offs)
    width = max(o[0] + r.width for o, r in zip(offs, rasters)) - min_x
    height = max(o[1] + r.height for o, r in zip(offs, rasters)) - min_y

    canvas = np.full((ref.bands, height, width), np.nan, dtype=np.float32)
    for (ix, iy), r in zip(offs, rasters):
        view = canvas[:, iy - min_y: iy - min_y + r.height, ix - min_x: ix - min_x + r.width]
        empty = np.isnan(view)
        view[empty] = r.data[empty]                       # fill only still-empty pixels
    out = GeoRaster(canvas, ref.x0 + min_x * ref.res, ref.y0 - min_y * ref.res, ref.res, ref.crs)
    return out, max_frac


def clean_linear_to_db(vv_lin: GeoRaster, vh_lin: GeoRaster, cfg: PreprocessConfig) -> GeoRaster:
    """Linear sigma0 (VV, VH) -> one 2-band dB raster, edge-guarded and clipped.

    Steps: valid = both polarisations finite and > 0  ->  erode (TRICK 8)  ->
    convert to dB (TRICK 5, 7)  ->  clip to cfg.clip_db (keeps outliers finite).
    """
    if vv_lin.data.shape != vh_lin.data.shape:
        raise PreprocessingError("VV and VH rasters have different shapes")
    vv, vh = vv_lin.data[0], vh_lin.data[0]
    valid = np.isfinite(vv) & (vv > 0) & np.isfinite(vh) & (vh > 0)
    valid = erode_valid(valid, cfg.edge_guard_px)
    lo, hi = cfg.clip_db
    out = np.stack([np.clip(to_db(np.where(valid, vv, np.nan)), lo, hi),
                    np.clip(to_db(np.where(valid, vh, np.nan)), lo, hi)]).astype(np.float32)
    return vv_lin.with_data(out)


def qa_report(db: GeoRaster, cfg: PreprocessConfig) -> tuple:
    """Sanity statistics + warnings for a 2-band dB raster -> (stats, warnings).

    Catches the silent failures: nearly empty output, mostly-dark output, and
    swapped polarisations (TRICK 6). Over land VV is normally a few dB brighter
    than VH; a median gap outside cfg.vv_vh_gap_db means channels are swapped or
    something else is wrong.
    """
    vv, vh = db.data[0], db.data[1]
    valid = np.isfinite(vv) & np.isfinite(vh)
    frac = float(valid.mean()) if valid.size else 0.0
    stats: dict = {"valid_fraction": round(frac, 4), "shape": [db.height, db.width]}
    warns: list = []
    if frac < cfg.fatal_valid_fraction:
        raise NoUsableDataError(
            f"Only {frac:.1%} of the output has valid data. The AOI may lie outside the "
            "swath, or processing failed. Check snap.log and the AOI.")
    if frac < cfg.min_valid_fraction:
        warns.append(f"Only {frac:.0%} of the AOI has valid radar data.")
    v, h = vv[valid], vh[valid]
    stats.update({
        "vv_db_p1_p50_p99": [round(float(x), 2) for x in np.percentile(v, [1, 50, 99])],
        "vh_db_p1_p50_p99": [round(float(x), 2) for x in np.percentile(h, [1, 50, 99])],
        "vv_dark_fraction_below_-20dB": round(float((v < -20).mean()), 4)})
    gap = float(np.median(v) - np.median(h))
    stats["median_vv_minus_vh_db"] = round(gap, 2)
    lo, hi = cfg.vv_vh_gap_db
    if not (lo <= gap <= hi):
        warns.append(f"Median VV-VH gap is {gap:.1f} dB (expected {lo:g}..{hi:g}). "
                     "The polarisations may be swapped or the data is unusual.")
    if stats["vv_dark_fraction_below_-20dB"] > 0.5:
        warns.append("More than half the valid pixels are very dark (< -20 dB): the scene may "
                     "be dominated by water or radar shadow, or calibration is off.")
    return stats, warns


# ============================================================================
# 9. ORCHESTRATION
# ============================================================================

@dataclass
class AcquisitionResult:
    role: str                       # "before" | "after" | "before2"
    params_hash: str
    crs: str
    tif_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    lia_path: Optional[Path] = None
    qa: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    from_cache: bool = False
    slices: list = field(default_factory=list)      # per-slice info (dry-run shows commands)


@dataclass
class PreprocessedPair:
    before: AcquisitionResult
    after: AcquisitionResult
    before2: Optional[AcquisitionResult]
    out_dir: Path
    params_hash: str
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Preprocessed (params {self.params_hash}) in {self.out_dir}"]
        for r in (self.before, self.after, self.before2):
            if r:
                lines.append(f"  {r.role:<7}: {r.tif_path}  valid="
                             f"{r.qa.get('valid_fraction', '?')}{'  (cache)' if r.from_cache else ''}")
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def preprocess_acquisition(acq: Acquisition, aoi: AOI, role: str, out_dir: Path,
                           cfg: PreprocessConfig, io=None, runner=None,
                           dry_run: bool = False) -> AcquisitionResult:
    """Preprocess ONE pass (1+ slices) into ``s1_<role>_<hash>.tif``.

    Cache (TRICK 13): if the output and its metadata exist and were made from the
    same product ids with the same params hash, return immediately.
    """
    io = io or RasterioIO()
    runner = runner or SnapRunner(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c_lon, c_lat = (aoi.min_lon + aoi.max_lon) / 2, (aoi.min_lat + aoi.max_lat) / 2
    crs = f"EPSG:{cfg.crs_epsg or utm_epsg(c_lon, c_lat)}"
    phash = cfg.params_hash(aoi, crs)
    product_ids = sorted(s.product_id for s in acq.slices)

    tif = out_dir / f"s1_{role}_{phash}.tif"
    meta_path = tif.with_suffix(".json")
    lia = out_dir / f"s1_{role}_lia_{phash}.tif" if cfg.save_incidence_angle else None

    if not dry_run and tif.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("product_ids") == product_ids and meta.get("params_hash") == phash:
                log.info("%s: cache hit (%s)", role, tif.name)
                return AcquisitionResult(role, phash, crs, tif, meta_path, lia,
                                         meta.get("qa", {}), meta.get("warnings", []), True)
        except (OSError, json.JSONDecodeError):
            pass                                          # unreadable cache: just redo

    work = out_dir / f"work_{role}_{phash}"
    outputs = [process_slice(s, aoi, role, k, work, crs, cfg, io, runner, dry_run)
               for k, s in enumerate(acq.slices)]
    slice_info = [o.info for o in outputs]
    if dry_run:
        return AcquisitionResult(role, phash, crs, None, None, None, {}, [], False, slice_info)

    warnings: list = []
    vv, off1 = mosaic([o.vv for o in outputs])
    vh, off2 = mosaic([o.vh for o in outputs])
    fx, fy = grid_offset_px(vv)
    if max(fx, fy, off1, off2) > 0.05:
        warnings.append(
            f"Output grid is not on the {cfg.res_m:g} m standard lattice (offset up to "
            f"{max(fx, fy, off1, off2):.2f} px). SNAP may have ignored alignToStandardGrid; "
            "Part 3 will resample, which slightly blurs.")
    db = clean_linear_to_db(vv, vh, cfg)
    qa, qa_warn = qa_report(db, cfg)
    warnings += qa_warn

    tags = {"units": "dB", "bands": "VV,VH", "role": role, "params_hash": phash,
            "speckle_filter": cfg.speckle_filter, "dem": cfg.dem_path or cfg.dem_name,
            "products": ";".join(s.name for s in acq.slices),
            "acquired_utc": acq.start.isoformat()}
    io.write(tif, db, descriptions=["VV_dB", "VH_dB"], tags=tags)
    if lia is not None and outputs[0].lia is not None:
        lia_r, _ = mosaic([o.lia for o in outputs])
        io.write(lia, lia_r, descriptions=["local_incidence_angle_deg"], tags={"units": "deg"})

    meta = {
        "role": role, "params_hash": phash, "product_ids": product_ids,
        "product_names": [s.name for s in acq.slices],
        "acquired_utc": acq.start.isoformat(), "platform": acq.platform,
        "track": {"relative_orbit": acq.track_key[0], "direction": acq.track_key[1]},
        "crs": crs, "res_m": cfg.res_m, "units": "dB", "bands": ["VV", "VH"],
        "origin_xy": [db.x0, db.y0], "shape": [db.height, db.width],
        "params": cfg.effective_params(aoi, crs), "slices": slice_info,
        "qa": qa, "warnings": warnings,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    meta_path.write_text(json.dumps(meta, indent=2))
    # The work folder is deliberately kept: graph.xml + snap.log are the receipts of
    # what ran, and the per-slice linear GeoTIFFs are small (AOI-sized).
    return AcquisitionResult(role, phash, crs, tif, meta_path, lia, qa, warnings, False, slice_info)


def preprocess_pair(pair: ScenePair, cfg: Optional[PreprocessConfig] = None,
                    out_root: str | Path = "data/interim/s1", io=None, runner=None,
                    dry_run: bool = False) -> PreprocessedPair:
    """Part 2 entry point: ScenePair (from Part 1) -> cleaned dB GeoTIFFs.

    TRICK 1: before, after (and before2) are processed with the SAME cfg object
    and the same AOI, and we assert the resulting params hashes match. If someone
    later edits this function to treat the two passes differently, the tripwire fires.
    """
    cfg = cfg or PreprocessConfig()
    io = io or RasterioIO()
    runner = runner or SnapRunner(cfg)
    out_dir = Path(out_root) / run_key(pair.aoi, pair.event_date)

    roles = [("before", pair.before), ("after", pair.after)]
    if pair.before2:
        roles.append(("before2", pair.before2))
    results = {role: preprocess_acquisition(acq, pair.aoi, role, out_dir, cfg, io, runner, dry_run)
               for role, acq in roles}

    hashes = {r.params_hash for r in results.values()}
    crs_set = {r.crs for r in results.values()}
    assert len(hashes) == 1 and len(crs_set) == 1, \
        "before/after were processed with different settings: this must never happen"

    warnings = list(pair.warnings)                        # carry Part 1's caveats forward
    for r in results.values():
        warnings += [f"[{r.role}] {w}" for w in r.warnings]
    pre = PreprocessedPair(results["before"], results["after"], results.get("before2"),
                           out_dir, hashes.pop(), warnings)
    if not dry_run:
        (out_dir / "preprocess_summary.json").write_text(json.dumps({
            "params_hash": pre.params_hash, "crs": results["before"].crs,
            "before": str(pre.before.tif_path), "after": str(pre.after.tif_path),
            "before2": str(pre.before2.tif_path) if pre.before2 else None,
            "warnings": warnings}, indent=2))
    return pre


# ============================================================================
# 10. CLI
# ============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Sentinel-1 preprocessing (Part 2).")
    p.add_argument("--scene-pair", required=True, help="scene_pair.json written by Part 1")
    p.add_argument("--out-root", default="data/interim/s1")
    p.add_argument("--dry-run", action="store_true",
                   help="write the SNAP graphs and print commands; do not run SNAP")
    p.add_argument("--speckle", default=None, help='e.g. "Refined Lee", "Lee", "Boxcar"')
    p.add_argument("--dem", default=None, help="local DEM GeoTIFF (otherwise SNAP downloads one)")
    p.add_argument("--gpt", default=None, help="path to SNAP gpt")
    p.add_argument("--mem-gb", type=int, default=None)
    p.add_argument("--crs-epsg", type=int, default=None)
    p.add_argument("--save-lia", action="store_true", help="also write local incidence angle")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    cfg = PreprocessConfig()
    if a.speckle:
        cfg.speckle_filter = a.speckle
    if a.dem:
        cfg.dem_path = a.dem
    if a.gpt:
        cfg.gpt_path = a.gpt
    if a.mem_gb:
        cfg.snap_mem_gb = a.mem_gb
    if a.crs_epsg:
        cfg.crs_epsg = a.crs_epsg
    cfg.save_incidence_angle = a.save_lia

    try:
        pair = ScenePair.load_json(a.scene_pair)
        runner = None
        if a.dry_run:
            class _DryRunner:                              # no SNAP needed to print commands
                def command(self, graph):
                    return ["gpt", str(graph), f"-J-Xmx{cfg.snap_mem_gb}G"]
            runner = _DryRunner()
        res = preprocess_pair(pair, cfg, a.out_root, runner=runner, dry_run=a.dry_run)
    except PreprocessingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if a.dry_run:
        for r in (res.before, res.after, res.before2):
            for s in (r.slices if r else []):
                print(f"[{r.role}] crop_early={s['crop_early']} "
                      f"(edge distance {s['distance_to_scene_edge_km']} km)\n  {s['command']}")
    else:
        print(res.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
