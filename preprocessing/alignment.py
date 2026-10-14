"""
preprocessing/alignment.py  —  PART 3: Alignment, Common Grid, DEM & Helper Masks
================================================================================

WHAT THIS FILE DOES (one paragraph)
-----------------------------------
Part 2 produced cleaned before/after radar images. This part is the HAND-OFF to
the rest of the team: it puts everything on ONE identical pixel grid (same CRS,
same 10 m lattice, same shape), adds the elevation model and its slope, and
computes three helper masks (valid data, unreliable terrain, permanent water).
Teammates 2 and 3 can then load any two of these files and stack/overlay them
with no reprojection, no guessing and no silent shifts.

PUBLIC ENTRY POINT
------------------
    pkg = build_analysis_package(scene_pair, PreInputs.from_summary_json(path))
    pkg.files["before"]   -> .../before.tif            2 bands (VV, VH) dB, float32
    pkg.files["after"]    -> .../after.tif
    pkg.files["dem"]      -> .../dem.tif               metres
    pkg.files["slope"]    -> .../slope.tif             degrees
    pkg.files["valid_mask"], ["terrain_mask"], ["permanent_water"]   uint8
    pkg.metadata_path     -> .../metadata.json         the contract

FILE CONTRACT (fixed names so nobody has to ask)
------------------------------------------------
    all rasters: same CRS (UTM of the AOI), 10 m pixels, same origin and shape
    float rasters: NaN = no data;  masks: uint8, 1 = true
    terrain_mask is a BIT FLAGS band (see TERRAIN_* constants below)

MAP OF THE FILE
---------------
  1. Config + errors
  2. GridSpec ................ the one shared grid (+ build_grid)
  3. Resampling .............. onto the grid (exact copy when aligned)
  4. DEM onto the grid ....... inverse mapping with coarse coordinate nodes
  5. Slope + radar geometry .. slope, look direction, local effects, cast shadow
  6. Masks ................... terrain bit flags, permanent water, valid
  7. Self-check .............. assert_aligned
  8. Package builder ......... orchestration, metadata, caching
  9. CLI

NON-OBVIOUS TRICKS USED HERE (search for "TRICK n")
---------------------------------------------------
  TRICK 1  Grid origin is snapped to multiples of the pixel size. Part 2 asked
           SNAP for the same lattice, so "resampling" Part 2's rasters onto this
           grid is an exact array copy: no interpolation, no blur.
  TRICK 2  Own UTM maths (preprocessing/utm.py): no pyproj, and everything here
           can be unit-tested offline.
  TRICK 3  DEM is sampled by INVERSE MAPPING: for each grid pixel find its lon/lat
           and read the DEM there. The exact lon/lat is only computed every 32nd
           pixel and interpolated in between (error < 1 mm, ~1000x fewer
           trigonometric evaluations).
  TRICK 4  Interpolation is NaN-aware: any NaN neighbour makes the result NaN,
           and outside the source raster is NaN, never a fake 0.
  TRICK 5  terrain_mask packs five different causes into ONE uint8 band using bit
           flags, so teammates can use ``mask > 0`` or ask "which cause?".
  TRICK 6  Radar look direction comes from the orbit direction (ASC/DESC) and the
           Sentinel-1 orbit inclination, not from metadata we do not have.
  TRICK 7  Cast shadow (a ridge hiding the valley behind it) is found by a
           horizon scan along the look direction, on a 3x coarser DEM. np.fmax
           ignores NaN so holes in the DEM do not poison the scan.
  TRICK 8  Permanent water = dark in the before image(s) AND flat AND not in radar
           shadow, because shadow is also dark. Otsu threshold with a sanity
           guard and a fixed fallback, since Otsu invents a split when there is
           no water at all. Two pre-event images are AND-ed when available.
  TRICK 9  If the DEM cannot be obtained the package is still built (terrain
           masks off) with a loud warning, unless require_dem=True.
  TRICK 10 Every build self-checks with assert_aligned and stamps a params hash
           that includes Part 2's hash (chain of custody); same inputs = cache hit.
  TRICK 11 DEM source and raster I/O are injected, so tests run offline.

HONEST LIMITS (also for the report)
-----------------------------------
  * Slope is computed from a 30 m DEM resampled to 10 m: effective resolution 30 m.
  * Cast shadow assumes a typical IW incidence angle (config), +/- 8 degrees.
  * Terrain masks are FLAGS for "radar result unreliable here", not proof of water.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

from acquisition.dem import (DEM_ATTRIBUTION, CopernicusDemDownloader, DemError,
                             LatLonRaster, LocalDemSource)
from acquisition.sentinel import AOI, ScenePair, run_key
from preprocessing.sentinel1 import (GeoRaster, RasterioIO, buffered_aoi,   # noqa: F401
                                     normalize_db, perimeter_points, utm_epsg)
from preprocessing.utm import lonlat_to_utm, parse_utm_crs, utm_to_lonlat

log = logging.getLogger("preprocessing.alignment")

# Bit flags stored in terrain_mask.tif (TRICK 5)
TERRAIN_STEEP = 1          # slope above max_slope_deg
TERRAIN_LAYOVER = 2        # slope faces the radar more steeply than the beam angle
TERRAIN_SHADOW = 4         # slope faces away so steeply the beam barely grazes it
TERRAIN_CAST_SHADOW = 8    # a ridge between the pixel and the radar hides it
TERRAIN_FORESHORTEN = 16   # strongly compressed slope facing the radar
TERRAIN_BITS = {"steep_slope": TERRAIN_STEEP, "layover": TERRAIN_LAYOVER,
                "local_shadow": TERRAIN_SHADOW, "cast_shadow": TERRAIN_CAST_SHADOW,
                "foreshortening": TERRAIN_FORESHORTEN}
_SHADOW_BITS = TERRAIN_LAYOVER | TERRAIN_SHADOW | TERRAIN_CAST_SHADOW

S1_ORBIT_INCLINATION_DEG = 98.18


# ============================================================================
# 1. CONFIG + ERRORS
# ============================================================================

class AlignmentError(Exception):
    """Rasters do not line up, or an input is unusable."""


@dataclass
class AlignmentConfig:
    res_m: float = 10.0
    # ---- terrain flags
    max_slope_deg: float = 30.0          # steep slopes: radar result unreliable
    incidence_deg: float = 39.0          # IW swath centre (29-46 across the swath)
    grazing_deg: float = 80.0            # local incidence at/above this ~ shadow
    foreshortening_deg: float = 10.0     # local incidence below this ~ foreshortened
    cast_shadow: bool = True
    cast_max_dist_m: float = 6000.0
    cast_step_m: float = 60.0
    cast_coarse_factor: int = 3          # run the horizon scan on a 3x coarser DEM
    # ---- permanent water
    water_max_slope_deg: float = 10.0
    water_default_db: float = -18.0      # fallback threshold (VV)
    water_otsu_range: tuple = (-26.0, -14.0)
    water_min_class_gap_db: float = 6.0
    water_dark_fraction_range: tuple = (0.001, 0.4)
    water_min_blob_px: int = 20
    # ---- DEM
    dem_margin_km: float = 3.0
    dem_node_step: int = 32
    require_dem: bool = False
    # ---- speed / memory only (not hashed)
    block_rows: int = 256

    def effective_params(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "block_rows"}
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


# ============================================================================
# 2. GRID
# ============================================================================

@dataclass(frozen=True)
class GridSpec:
    """THE shared grid. x0, y0 = top-left corner (west/north edge of pixel 0,0)."""
    crs: str
    res: float
    x0: float
    y0: float
    width: int
    height: int

    def empty(self, bands: int = 1, fill: float = np.nan, dtype=np.float32) -> GeoRaster:
        return GeoRaster(np.full((bands, self.height, self.width), fill, dtype=dtype),
                         self.x0, self.y0, self.res, self.crs)

    def matches(self, r: GeoRaster) -> bool:
        return (r.crs == self.crs and abs(r.res - self.res) < 1e-9
                and abs(r.x0 - self.x0) < 1e-6 and abs(r.y0 - self.y0) < 1e-6
                and r.width == self.width and r.height == self.height)

    def to_dict(self) -> dict:
        return {"crs": self.crs, "res_m": self.res, "x0": self.x0, "y0": self.y0,
                "width": self.width, "height": self.height,
                "transform": [self.res, 0.0, self.x0, 0.0, -self.res, self.y0]}


def build_grid(aoi: AOI, crs: str, res: float = 10.0) -> GridSpec:
    """The smallest grid on the lattice of multiples of ``res`` that covers the AOI.

    TRICK 1: x0 and y0 are rounded OUTWARD to multiples of ``res`` (never to the
    AOI corner itself). Two runs, two machines, two AOIs on the same area all land
    on the same lattice, and it is the lattice SNAP was told to use in Part 2.

    The AOI is a rectangle in lon/lat, which is a slightly bent quadrilateral in
    UTM, so we project 200 points along its outline and take their bounding box.
    """
    zone, north = parse_utm_crs(crs)
    pts = perimeter_points(aoi, 200)
    e, n = lonlat_to_utm(pts[:, 0], pts[:, 1], zone, north)
    x0 = math.floor(e.min() / res) * res
    x1 = math.ceil(e.max() / res) * res
    y0 = math.ceil(n.max() / res) * res
    y1 = math.floor(n.min() / res) * res
    return GridSpec(crs, float(res), float(x0), float(y0),
                    max(1, int(round((x1 - x0) / res))), max(1, int(round((y0 - y1) / res))))


# ============================================================================
# 3. RESAMPLING ONTO THE GRID
# ============================================================================

def _aligned_offset(src: GeoRaster, grid: GridSpec) -> Optional[tuple]:
    """(dx, dy) if ``src`` lies on exactly the same lattice as ``grid`` (same CRS,
    same pixel size, whole-pixel offset); otherwise None."""
    if src.crs != grid.crs or abs(src.res - grid.res) > 1e-9:
        return None
    dx, dy = (grid.x0 - src.x0) / src.res, (src.y0 - grid.y0) / src.res
    if abs(dx - round(dx)) > 1e-3 or abs(dy - round(dy)) > 1e-3:
        return None
    return int(round(dx)), int(round(dy))


def resample_to_grid(src: GeoRaster, grid: GridSpec, order: int = 1,
                     block_rows: int = 256) -> GeoRaster:
    """Return ``src`` on ``grid``. Pixels outside ``src`` become NaN.

    Two paths:
      FAST (TRICK 1): src is on the same lattice -> an integer window copy.
                      Exact, instant, no interpolation at all.
      GENERAL:        inverse mapping. For every grid pixel centre compute where
                      it falls in src, then interpolate (order 1 = bilinear,
                      0 = nearest). Handles other pixel sizes, sub-pixel shifts
                      and a different UTM zone (via lon/lat).

    TRICK 4: interpolation uses mode="constant", cval=NaN. A NaN anywhere in the
    2x2 neighbourhood yields NaN, so no-data never leaks into plausible values.
    """
    off = _aligned_offset(src, grid)
    if off is not None:
        dx, dy = off
        out = np.full((src.bands, grid.height, grid.width), np.nan, dtype=np.float32)
        r0, r1 = max(0, -dy), min(grid.height, src.height - dy)
        c0, c1 = max(0, -dx), min(grid.width, src.width - dx)
        if r1 > r0 and c1 > c0:
            out[:, r0:r1, c0:c1] = src.data[:, r0 + dy:r1 + dy, c0 + dx:c1 + dx]
        return grid.empty(src.bands).with_data(out)

    same_crs = src.crs == grid.crs
    if not same_crs:
        zt, nt = parse_utm_crs(grid.crs)
        zs, ns = parse_utm_crs(src.crs)
    out = np.full((src.bands, grid.height, grid.width), np.nan, dtype=np.float32)
    cols = np.arange(grid.width)
    for r0 in range(0, grid.height, block_rows):
        r1 = min(grid.height, r0 + block_rows)
        e_t = grid.x0 + (cols + 0.5) * grid.res                     # (W,)
        n_t = grid.y0 - (np.arange(r0, r1) + 0.5) * grid.res        # (nb,)
        if same_crs:                                                # separable: cheap
            col_s = ((e_t - src.x0) / src.res - 0.5)[None, :]
            row_s = ((src.y0 - n_t) / src.res - 0.5)[:, None]
            row_s, col_s = np.broadcast_arrays(row_s, col_s)
        else:                                                       # via lon/lat
            ee, nn = np.meshgrid(e_t, n_t)
            lon, lat = utm_to_lonlat(ee, nn, zt, nt)
            e_s, n_s = lonlat_to_utm(lon, lat, zs, ns)
            col_s, row_s = (e_s - src.x0) / src.res - 0.5, (src.y0 - n_s) / src.res - 0.5
        for b in range(src.bands):
            out[b, r0:r1] = ndimage.map_coordinates(
                src.data[b], [row_s, col_s], order=order, mode="constant",
                cval=np.nan, prefilter=False)
    return grid.empty(src.bands).with_data(out)


# ============================================================================
# 4. DEM ONTO THE GRID
# ============================================================================

def dem_to_grid(dem: LatLonRaster, grid: GridSpec, node_step: int = 32,
                block_rows: int = 256) -> np.ndarray:
    """Sample a lon/lat DEM onto the UTM grid -> (H, W) float32 metres, NaN outside.

    TRICK 3 — exact lon/lat of every pixel would need ~25 million inverse
    projections for a 50 km AOI. Instead we compute it exactly on a coarse
    lattice (every ``node_step`` pixels = 320 m) and bilinearly interpolate the
    lon and lat FIELDS between nodes. The mapping is smooth (curvature ~1e-7 per
    metre), so the interpolation error is millimetres, far below DEM accuracy.
    Rows are processed in blocks to bound memory.
    """
    zone, north = parse_utm_crs(grid.crs)
    kr = int(math.ceil((grid.height - 1) / node_step)) + 1
    kc = int(math.ceil((grid.width - 1) / node_step)) + 1
    e_nodes = grid.x0 + (np.arange(kc) * node_step + 0.5) * grid.res
    n_nodes = grid.y0 - (np.arange(kr) * node_step + 0.5) * grid.res
    lon_n, lat_n = utm_to_lonlat(*np.meshgrid(e_nodes, n_nodes), zone, north)

    out = np.empty((grid.height, grid.width), dtype=np.float32)
    c_idx = (np.arange(grid.width) / node_step)[None, :]
    for r0 in range(0, grid.height, block_rows):
        r1 = min(grid.height, r0 + block_rows)
        rr, cc = np.broadcast_arrays((np.arange(r0, r1) / node_step)[:, None], c_idx)
        lon = ndimage.map_coordinates(lon_n, [rr, cc], order=1, mode="nearest")
        lat = ndimage.map_coordinates(lat_n, [rr, cc], order=1, mode="nearest")
        col = (lon - dem.lon0) / dem.dlon - 0.5
        row = (dem.lat0 - lat) / dem.dlat - 0.5
        out[r0:r1] = ndimage.map_coordinates(dem.data, [row, col], order=1,
                                             mode="constant", cval=np.nan, prefilter=False)
    return out


# ============================================================================
# 5. SLOPE + RADAR GEOMETRY
# ============================================================================

def gradients(dem: np.ndarray, res: float) -> tuple:
    """Horn's 3x3 method. Returns (p, q): rise per metre toward EAST and NORTH.

    Computed in metres on the projected grid. (Doing this in degrees gives
    nonsense: one degree of longitude is not one degree of latitude.) Image rows
    run SOUTH, so the north gradient is minus the row gradient. Edges are padded
    by replication; NaN propagates.
    """
    z = np.pad(dem.astype(np.float32), 1, mode="edge")
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    p = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * res)
    q = -((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * res)
    return p, q


def slope_degrees(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan(np.hypot(p, q))).astype(np.float32)


def look_azimuth_deg(direction: str, lat: float,
                     inclination_deg: float = S1_ORBIT_INCLINATION_DEG) -> float:
    """Compass direction (clockwise from north) the radar LOOKS toward.

    TRICK 6: Sentinel-1 flies a near-polar orbit (inclination 98.18 deg) and looks
    to its RIGHT. The ground-track heading at latitude phi follows
        sin(heading) = cos(inclination) / cos(phi)      (ascending)
    and descending is the mirror image. Look = heading + 90. At 28 N: ascending
    looks ~81 deg (east-ish), descending ~279 deg (west-ish). Ignores Earth
    rotation (~1 degree), irrelevant for shadow geometry at 30 m.
    """
    ratio = max(-1.0, min(1.0, math.cos(math.radians(inclination_deg))
                          / max(math.cos(math.radians(lat)), 1e-6)))
    az_asc = math.degrees(math.asin(ratio))
    heading = az_asc if str(direction).upper().startswith("A") else 180.0 - az_asc
    return (heading + 90.0) % 360.0


def range_slope_deg(p: np.ndarray, q: np.ndarray, look_az_deg: float) -> np.ndarray:
    """Slope ALONG the direction toward the sensor, in degrees.
    Positive = the ground rises toward the radar (faces it); negative = faces away.
    The direction from ground to sensor is (-sin a, -cos a) in (east, north)."""
    a = math.radians(look_az_deg)
    return np.degrees(np.arctan(-p * math.sin(a) - q * math.cos(a)))


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """f x f block average ignoring NaN (the ragged remainder is NaN-padded)."""
    h, w = a.shape
    ap = np.pad(a, ((0, (-h) % f), (0, (-w) % f)), constant_values=np.nan)
    blocks = ap.reshape(ap.shape[0] // f, f, ap.shape[1] // f, f)
    fin = np.isfinite(blocks)
    n = fin.sum(axis=(1, 3))
    s = np.where(fin, blocks, 0.0).sum(axis=(1, 3))
    return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)


def cast_shadow(dem: np.ndarray, res: float, look_az_deg: float, incidence_deg: float,
                max_dist_m: float = 6000.0, step_m: float = 60.0, coarse: int = 3) -> np.ndarray:
    """Boolean mask of pixels hidden from the radar by terrain between them and it.

    Idea (TRICK 7): stand on a pixel and look back toward the satellite. The
    radar ray climbs at an angle of (90 - incidence) above the horizon, i.e. its
    height above the pixel after d metres is d / tan(incidence). If any terrain
    along that line is higher than the ray, the pixel is in shadow:
        z(d) - d/tan(i)  >  z(0)
    We scan d = step, 2*step ... max_dist and keep the running maximum of the
    left-hand side. np.fmax (not np.maximum) ignores NaN, so DEM holes do not
    wipe out the whole scan. The scan runs on a ``coarse``x coarser DEM
    (30 m instead of 10 m): 9x fewer pixels, and shadow edges are smooth anyway.
    Beyond the array edge the nearest value is repeated (mode="nearest").
    """
    h, w = dem.shape
    coarse = max(1, int(coarse))
    dem_c = _block_mean(dem, coarse) if coarse > 1 else dem.astype(np.float32)
    res_c = res * coarse
    a = math.radians(look_az_deg)
    tan_i = math.tan(math.radians(incidence_deg))
    rr, cc = np.indices(dem_c.shape, dtype=np.float32)
    best = np.full(dem_c.shape, -np.inf, dtype=np.float32)
    for k in range(1, int(max_dist_m // step_m) + 1):
        d = k * step_m
        row = rr + np.float32(d * math.cos(a) / res_c)       # rows run south
        col = cc - np.float32(d * math.sin(a) / res_c)
        z = ndimage.map_coordinates(dem_c, [row, col], order=1, mode="nearest", prefilter=False)
        np.fmax(best, z - np.float32(d / tan_i), out=best)
    shadow_c = best > dem_c                                  # NaN compares False
    full = np.repeat(np.repeat(shadow_c, coarse, axis=0), coarse, axis=1)[:h, :w]
    return full


# ============================================================================
# 6. MASKS
# ============================================================================

def build_terrain_mask(dem: np.ndarray, res: float, look_az_deg: float, cfg: AlignmentConfig,
                       lia: Optional[np.ndarray] = None) -> tuple:
    """Bit-flag mask of pixels where a radar backscatter change is NOT trustworthy.

    Returns (mask uint8, slope float32, info dict).  See the TERRAIN_* constants.

    Local effects use the DEM and the nominal incidence angle: with sigma the
    slope toward the radar, the local incidence angle is  theta - sigma.
        < 0                       -> layover     (ground leans over the radar beam)
        0 .. foreshortening_deg   -> foreshortening
        >= grazing_deg            -> local shadow (beam barely grazes the slope)
    If SNAP's measured local incidence angle (``lia``, degrees, from Part 2's
    optional output) is available it replaces this estimate.
    Pixels where the DEM is NaN get no flags: unknown is not the same as bad.
    """
    p, q = gradients(dem, res)
    slope = slope_degrees(p, q)
    known = np.isfinite(dem)
    mask = np.zeros(dem.shape, dtype=np.uint8)
    mask[known & (slope > cfg.max_slope_deg)] |= TERRAIN_STEEP

    if lia is not None:
        theta_loc = lia
        mask[np.isfinite(lia) & (lia <= 0)] |= TERRAIN_LAYOVER
        mask[np.isfinite(lia) & (lia > 0) & (lia <= cfg.foreshortening_deg)] |= TERRAIN_FORESHORTEN
        mask[np.isfinite(lia) & (lia >= cfg.grazing_deg)] |= TERRAIN_SHADOW
        source = "snap_local_incidence_angle"
    else:
        sigma = range_slope_deg(p, q, look_az_deg)
        theta_loc = cfg.incidence_deg - sigma
        mask[known & (theta_loc < 0)] |= TERRAIN_LAYOVER
        mask[known & (theta_loc >= 0) & (theta_loc < cfg.foreshortening_deg)] |= TERRAIN_FORESHORTEN
        mask[known & (theta_loc >= cfg.grazing_deg)] |= TERRAIN_SHADOW
        source = "dem_slope_and_nominal_incidence"

    if cfg.cast_shadow:
        cs = cast_shadow(dem, res, look_az_deg, cfg.incidence_deg, cfg.cast_max_dist_m,
                         cfg.cast_step_m, cfg.cast_coarse_factor)
        mask[known & cs] |= TERRAIN_CAST_SHADOW

    info = {"local_geometry_source": source, "look_azimuth_deg": round(look_az_deg, 1),
            "incidence_deg": cfg.incidence_deg,
            "fractions": {name: round(float(((mask & bit) > 0).mean()), 4)
                          for name, bit in TERRAIN_BITS.items()}}
    return mask, slope, info


def otsu_threshold(values: np.ndarray, lo: float = -35.0, hi: float = 5.0, bins: int = 256) -> tuple:
    """Otsu's histogram threshold. Returns (threshold, gap_between_class_means,
    fraction_of_values_below_threshold). Maximises the between-class variance."""
    v = np.clip(values[np.isfinite(values)], lo, hi)
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean0 = m0 / w0
        mean1 = (mt - m0) / w1
        between = w0 * w1 * (mean0 - mean1) ** 2
    between = np.nan_to_num(between)
    # With clean two-level data every threshold inside the gap scores the same.
    # argmax would pick the FIRST one, i.e. right on top of the dark values (and
    # then ``value < threshold`` would miss them). Take the MIDDLE of the plateau.
    plateau = np.flatnonzero(between >= between.max() * (1.0 - 1e-9))
    k = int(plateau[len(plateau) // 2])
    return float(centres[k]), float(mean1[k] - mean0[k]), float(w0[k] / w0[-1])


def choose_water_threshold(vv_db: np.ndarray, eligible: np.ndarray, cfg: AlignmentConfig) -> tuple:
    """TRICK 8 — Otsu with a safety net. Returns (threshold_db, method_string).

    Otsu ALWAYS returns a threshold, even for a histogram with one hump (no
    water in the AOI), where it would slice the land in half and call half the
    area "permanent water". So we accept it only if all of these hold:
      * enough pixels, * threshold in a plausible range for water,
      * the two classes are well separated, * dark class is a plausible share.
    Otherwise we fall back to the fixed default.
    """
    vals = vv_db[eligible & np.isfinite(vv_db)]
    if vals.size < 500:
        return cfg.water_default_db, "default (too few pixels)"
    t, gap, dark = otsu_threshold(vals)
    lo, hi = cfg.water_otsu_range
    flo, fhi = cfg.water_dark_fraction_range
    if not (lo <= t <= hi):
        return cfg.water_default_db, f"default (otsu {t:.1f} dB outside {lo:g}..{hi:g})"
    if gap < cfg.water_min_class_gap_db:
        return cfg.water_default_db, f"default (classes only {gap:.1f} dB apart)"
    if not (flo <= dark <= fhi):
        return cfg.water_default_db, f"default (dark share {dark:.1%} implausible)"
    return t, "otsu"


def remove_small_blobs(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Delete connected groups of True pixels smaller than ``min_px`` (speckle)."""
    if min_px <= 1 or not mask.any():
        return mask
    labels, n = ndimage.label(mask)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = sizes >= min_px
    keep[0] = False
    return keep[labels]


def permanent_water_mask(pre_vv_db: Sequence[np.ndarray], slope: np.ndarray,
                         terrain: np.ndarray, valid: np.ndarray, cfg: AlignmentConfig) -> tuple:
    """Pixels that were already water BEFORE the event (so not 'new flood').

    A pixel counts if it is dark in VV in EVERY pre-event image (two images, when
    available, filter out one-off wet soil) AND flat AND not in radar shadow
    (shadow is dark too: TRICK 8) AND valid. Tiny specks are removed.
    Returns (mask bool, info dict).
    """
    flat = ~(slope >= cfg.water_max_slope_deg)           # NaN slope (no DEM) counts as allowed
    not_shadow = (terrain & _SHADOW_BITS) == 0
    eligible = valid & flat & not_shadow
    result = eligible.copy()
    info = {"thresholds_db": [], "methods": []}
    for vv in pre_vv_db:
        t, how = choose_water_threshold(vv, eligible, cfg)
        info["thresholds_db"].append(round(t, 2))
        info["methods"].append(how)
        result &= np.isfinite(vv) & (vv < t)
    result = remove_small_blobs(result, cfg.water_min_blob_px)
    info["fraction"] = round(float(result.mean()), 4)
    return result, info


def valid_mask(*arrays: np.ndarray) -> np.ndarray:
    """True where EVERY band of every given (bands, H, W) array is finite."""
    ok = np.ones(arrays[0].shape[1:], dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a).all(axis=0)
    return ok


# ============================================================================
# 7. SELF-CHECK
# ============================================================================

def assert_aligned(rasters: dict, grid: Optional[GridSpec] = None) -> None:
    """Raise AlignmentError, listing every offender, unless all rasters share the
    same CRS, pixel size, origin and shape (and equal ``grid`` if given).
    Run on every build and by Teammates 2/3 on their inputs (TRICK 10)."""
    items = list(rasters.items())
    if not items:
        return
    ref_name, ref = items[0]
    problems = []
    if grid is not None and not grid.matches(ref):
        problems.append(f"{ref_name}: does not match the grid spec")
    for name, r in items[1:]:
        diffs = []
        if r.crs != ref.crs:
            diffs.append(f"crs {r.crs} != {ref.crs}")
        if abs(r.res - ref.res) > 1e-9:
            diffs.append(f"res {r.res} != {ref.res}")
        if abs(r.x0 - ref.x0) > 1e-6 or abs(r.y0 - ref.y0) > 1e-6:
            diffs.append(f"origin ({r.x0}, {r.y0}) != ({ref.x0}, {ref.y0})")
        if (r.height, r.width) != (ref.height, ref.width):
            diffs.append(f"shape {(r.height, r.width)} != {(ref.height, ref.width)}")
        if diffs:
            problems.append(f"{name}: " + "; ".join(diffs))
    if problems:
        raise AlignmentError("Rasters are not aligned:\n  - " + "\n  - ".join(problems))


# ============================================================================
# 8. PACKAGE BUILDER
# ============================================================================

@dataclass
class PreInputs:
    """What Part 3 needs from Part 2 (decouples the two parts' classes)."""
    before_tif: Path
    after_tif: Path
    before2_tif: Optional[Path] = None
    lia_tif: Optional[Path] = None
    crs: str = ""
    params_hash: str = ""
    warnings: list = field(default_factory=list)

    @classmethod
    def from_preprocessed(cls, pre) -> "PreInputs":
        """From Part 2's ``PreprocessedPair`` object."""
        return cls(pre.before.tif_path, pre.after.tif_path,
                   pre.before2.tif_path if pre.before2 else None,
                   pre.before.lia_path, pre.before.crs, pre.params_hash, list(pre.warnings))

    @classmethod
    def from_summary_json(cls, path) -> "PreInputs":
        """From Part 2's preprocess_summary.json (also picks up the optional
        local-incidence-angle file written next to the before image)."""
        d = json.loads(Path(path).read_text())
        before = Path(d["before"])
        lia = before.with_name(before.name.replace("s1_before_", "s1_before_lia_"))
        return cls(before, Path(d["after"]), Path(d["before2"]) if d.get("before2") else None,
                   lia if lia.exists() else None, d.get("crs", ""), d.get("params_hash", ""),
                   list(d.get("warnings", [])))


@dataclass
class AnalysisPackage:
    out_dir: Path
    grid: GridSpec
    files: dict
    metadata_path: Path
    qa: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    from_cache: bool = False

    def summary(self) -> str:
        lines = [f"Analysis package in {self.out_dir}{'  (cache)' if self.from_cache else ''}",
                 f"  grid: {self.grid.crs}, {self.grid.res:g} m, {self.grid.width} x "
                 f"{self.grid.height} px"]
        lines += [f"  {k:<16} {v}" for k, v in self.files.items()]
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def _analysis_hash(cfg: AlignmentConfig, pre: PreInputs, grid: GridSpec, dem_desc: str) -> str:
    blob = json.dumps({"cfg": cfg.effective_params(), "pre": pre.params_hash,
                       "grid": grid.to_dict(), "dem": dem_desc}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def build_analysis_package(pair: ScenePair, pre: PreInputs, cfg: Optional[AlignmentConfig] = None,
                           dem_source=None, io=None,
                           processed_root: str | Path = "data/processed") -> AnalysisPackage:
    """Part 3 entry point. Builds (or reuses) the files described at the top.

    Steps: grid -> resample before/after(/before2) -> DEM -> slope -> terrain mask
    -> valid mask -> permanent water -> write -> metadata -> self-check.
    """
    cfg = cfg or AlignmentConfig()
    io = io or RasterioIO()
    aoi = pair.aoi
    crs = pre.crs or f"EPSG:{utm_epsg((aoi.min_lon + aoi.max_lon) / 2, (aoi.min_lat + aoi.max_lat) / 2)}"
    grid = build_grid(aoi, crs, cfg.res_m)
    dem_source = dem_source or CopernicusDemDownloader()
    dem_desc = dem_source.describe() if hasattr(dem_source, "describe") else type(dem_source).__name__
    out_dir = Path(processed_root) / run_key(aoi, pair.event_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"
    ahash = _analysis_hash(cfg, pre, grid, dem_desc)

    names = ["before", "after", "dem", "slope", "valid_mask", "terrain_mask", "permanent_water"]
    if pre.before2_tif:
        names.append("before2")
    files = {n: out_dir / f"{n}.tif" for n in names}

    # ---- cache (TRICK 10)
    if meta_path.exists() and all(f.exists() for f in files.values()):
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("analysis_hash") == ahash:
                log.info("analysis package: cache hit")
                return AnalysisPackage(out_dir, grid, files, meta_path, meta.get("qa", {}),
                                       meta.get("warnings", []), True)
        except (OSError, json.JSONDecodeError):
            pass

    warnings: list = list(pre.warnings)

    # ---- radar images onto the grid
    def load(path) -> GeoRaster:
        r = resample_to_grid(io.read(path), grid, order=1, block_rows=cfg.block_rows)
        if not np.isfinite(r.data).any():
            raise AlignmentError(f"{path} has no data inside the AOI grid. Does the AOI "
                                 "match the area that was preprocessed?")
        return r

    before, after = load(pre.before_tif), load(pre.after_tif)
    before2 = load(pre.before2_tif) if pre.before2_tif else None
    lia = None
    if pre.lia_tif and Path(pre.lia_tif).exists():
        lia = resample_to_grid(io.read(pre.lia_tif), grid, order=1).data[0]

    # ---- DEM (TRICK 9: fail soft)
    dem_arr = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    dem_ok = True
    try:
        dem_ll = dem_source.load(buffered_aoi(aoi, cfg.dem_margin_km).as_tuple())
        dem_arr = dem_to_grid(dem_ll, grid, cfg.dem_node_step, cfg.block_rows)
        if not np.isfinite(dem_arr).any():
            raise DemError("DEM does not cover the AOI")
    except Exception as exc:                                  # noqa: BLE001 (deliberate)
        if cfg.require_dem:
            raise
        dem_ok = False
        warnings.append(f"DEM unavailable ({exc}). Slope, terrain mask and DEM-based "
                        "filtering are DISABLED; radar shadow may be mistaken for water.")

    # ---- masks
    look_az = look_azimuth_deg(pair.before.track_key[1], (aoi.min_lat + aoi.max_lat) / 2)
    valid = valid_mask(before.data, after.data)
    if dem_ok:
        terrain, slope, terrain_info = build_terrain_mask(dem_arr, grid.res, look_az, cfg, lia)
    else:
        terrain = np.zeros(dem_arr.shape, dtype=np.uint8)
        slope = np.full(dem_arr.shape, np.nan, dtype=np.float32)
        terrain_info = {"local_geometry_source": "none (no DEM)"}
    pre_vv = [before.data[0]] + ([before2.data[0]] if before2 is not None else [])
    water, water_info = permanent_water_mask(pre_vv, slope, terrain, valid, cfg)

    qa = {"valid_fraction": round(float(valid.mean()), 4),
          "terrain_flagged_fraction": round(float((terrain > 0).mean()), 4),
          "permanent_water_fraction": water_info["fraction"],
          "dem_valid_fraction": round(float(np.isfinite(dem_arr).mean()), 4)}
    if qa["valid_fraction"] < 0.5:
        warnings.append(f"Only {qa['valid_fraction']:.0%} of the grid has valid before+after data.")
    if qa["terrain_flagged_fraction"] > 0.5:
        warnings.append(f"{qa['terrain_flagged_fraction']:.0%} of the AOI is flagged as "
                        "unreliable terrain: results there are weak (steep valley). Consider "
                        "Sentinel-2 or checking thresholds.")
    if any(m != "otsu" for m in water_info["methods"]):
        warnings.append("Permanent-water threshold fell back to the default: "
                        + "; ".join(water_info["methods"]))

    # ---- write (contract)
    tags = {"units": "dB", "bands": "VV,VH", "params_hash": ahash}
    io.write(files["before"], before, ["VV_dB", "VH_dB"], tags)
    io.write(files["after"], after, ["VV_dB", "VH_dB"], tags)
    if before2 is not None:
        io.write(files["before2"], before2, ["VV_dB", "VH_dB"], tags)
    io.write(files["dem"], grid.empty(1).with_data(dem_arr[None]), ["elevation_m"], {"units": "m"})
    io.write(files["slope"], grid.empty(1).with_data(slope[None]), ["slope_deg"], {"units": "deg"})
    u8 = dict(dtype="uint8", nodata=None)
    io.write(files["valid_mask"], grid.empty(1, 0, np.uint8).with_data(valid[None].astype(np.uint8)),
             ["valid"], {}, **u8)
    io.write(files["terrain_mask"], grid.empty(1, 0, np.uint8).with_data(terrain[None]),
             ["terrain_flags"], {"bits": json.dumps(TERRAIN_BITS)}, **u8)
    io.write(files["permanent_water"], grid.empty(1, 0, np.uint8).with_data(water[None].astype(np.uint8)),
             ["permanent_water"], {}, **u8)

    # ---- self-check: everything we produced must be on the grid
    check = {"before": before, "after": after, "dem": grid.empty(1).with_data(dem_arr[None])}
    if before2 is not None:
        check["before2"] = before2
    assert_aligned(check, grid)

    meta = {
        "analysis_hash": ahash, "preprocess_params_hash": pre.params_hash,
        "event_date": pair.event_date.isoformat(), "aoi_bbox_4326": list(aoi.as_tuple()),
        "grid": grid.to_dict(), "units": {"before/after": "dB", "dem": "m", "slope": "deg"},
        "bands": ["VV", "VH"], "nodata": "NaN for float rasters; masks are uint8 (1 = true)",
        "before": {"date": pair.before.date.isoformat(), "platform": pair.before.platform},
        "after": {"date": pair.after.date.isoformat(), "platform": pair.after.platform},
        "gap_days": pair.gap_days,
        "track": {"relative_orbit": pair.track[0], "direction": pair.track[1]},
        "files": {k: str(v) for k, v in files.items()},
        "terrain_bits": TERRAIN_BITS, "terrain": terrain_info,
        "permanent_water": water_info, "dem_source": dem_desc if dem_ok else None,
        "config": cfg.effective_params(), "qa": qa, "warnings": warnings,
        "attributions": ["Contains modified Copernicus Sentinel data 2026.", DEM_ATTRIBUTION],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return AnalysisPackage(out_dir, grid, files, meta_path, qa, warnings, False)


# ============================================================================
# 9. CLI
# ============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build the shared analysis grid package (Part 3).")
    p.add_argument("--scene-pair", required=True, help="scene_pair.json from Part 1")
    p.add_argument("--preprocess-summary", default=None,
                   help="preprocess_summary.json from Part 2 (default: data/interim/s1/<key>/)")
    p.add_argument("--dem", default=None, help="DEM GeoTIFF or folder of tiles (else downloads)")
    p.add_argument("--dem-cache", default="data/raw/dem")
    p.add_argument("--processed-root", default="data/processed")
    p.add_argument("--interim-root", default="data/interim/s1")
    p.add_argument("--no-cast-shadow", action="store_true")
    p.add_argument("--require-dem", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    try:
        pair = ScenePair.load_json(a.scene_pair)
        summary = a.preprocess_summary or (
            Path(a.interim_root) / run_key(pair.aoi, pair.event_date) / "preprocess_summary.json")
        pre = PreInputs.from_summary_json(summary)
        cfg = AlignmentConfig(cast_shadow=not a.no_cast_shadow, require_dem=a.require_dem)
        src = LocalDemSource(a.dem) if a.dem else CopernicusDemDownloader(a.dem_cache)
        pkg = build_analysis_package(pair, pre, cfg, src, processed_root=a.processed_root)
    except (AlignmentError, DemError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(pkg.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
