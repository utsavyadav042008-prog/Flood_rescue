"""
preprocessing/utm.py  —  UTM projection in pure numpy (helper for PART 3)
=========================================================================

WHY THIS FILE EXISTS
--------------------
Part 3 must convert between longitude/latitude (degrees) and UTM (metres) for
every pixel of the DEM and for the grid corners. The usual tool is ``pyproj``,
but:
  * it is one more dependency that must be installed on every teammate's machine;
  * it cannot be imported in the offline test environment, so any code that
    uses it could not be unit-tested here.

So this is a small, self-contained implementation of the Transverse Mercator
projection using Kruger's series (Karney 2011). It is accurate to far below a
millimetre inside a UTM zone, works on whole numpy arrays, and is verified in
``tests/test_alignment.py`` against independent calculations (the meridian arc
length integrated numerically, the east-west scale, round trips, and pyproj
itself when it happens to be installed).

NON-OBVIOUS DETAILS
-------------------
TRICK  The same series gives BOTH directions, so "pixel -> lon/lat" (inverse
       mapping, used to sample the DEM) is as exact as "lon/lat -> pixel".
TRICK  Everything is vectorised: pass arrays, get arrays. No Python loops over
       pixels anywhere.

WGS84 ellipsoid, UTM scale factor 0.9996, false easting 500 000 m, false
northing 0 (north) or 10 000 000 m (south).
"""
from __future__ import annotations

import re
from typing import Tuple

import numpy as np

# ---- WGS84 + UTM constants ---------------------------------------------------
_A = 6378137.0                         # semi-major axis (m)
_F = 1.0 / 298.257223563               # flattening
_K0 = 0.9996                           # UTM central-meridian scale factor
_E0 = 500000.0                         # false easting (m)
_N0_SOUTH = 10000000.0                 # false northing in the southern hemisphere

_N = _F / (2.0 - _F)                   # third flattening "n"
# Rectifying radius: the radius of a sphere with the same meridian length.
_AH = _A / (1.0 + _N) * (1.0 + _N**2 / 4.0 + _N**4 / 64.0 + _N**6 / 256.0)

# Series coefficients (Kruger, 4 terms: error < 1e-9 m inside a UTM zone).
_ALPHA = (
    _N / 2 - 2 * _N**2 / 3 + 5 * _N**3 / 16 + 41 * _N**4 / 180,
    13 * _N**2 / 48 - 3 * _N**3 / 5 + 557 * _N**4 / 1440,
    61 * _N**3 / 240 - 103 * _N**4 / 140,
    49561 * _N**4 / 161280,
)
_BETA = (
    _N / 2 - 2 * _N**2 / 3 + 37 * _N**3 / 96 - _N**4 / 360,
    _N**2 / 48 + _N**3 / 15 - 437 * _N**4 / 1440,
    17 * _N**3 / 480 - 37 * _N**4 / 840,
    4397 * _N**4 / 161280,
)
_DELTA = (
    2 * _N - 2 * _N**2 / 3 - 2 * _N**3 + 116 * _N**4 / 45,
    7 * _N**2 / 3 - 8 * _N**3 / 5 - 227 * _N**4 / 45,
    56 * _N**3 / 15 - 136 * _N**4 / 35,
    4279 * _N**4 / 630,
)


def central_meridian(zone: int) -> float:
    """Longitude (degrees) of the middle of a UTM zone. Zone 45 -> 87 E."""
    return (zone - 1) * 6.0 - 180.0 + 3.0


def parse_utm_crs(crs: str) -> Tuple[int, bool]:
    """'EPSG:32645' -> (45, True) ; 'EPSG:32719' -> (19, False).

    Raises ValueError for anything that is not a WGS84 UTM code, so a wrong CRS
    fails loudly here instead of silently producing shifted maps."""
    m = re.fullmatch(r"(?:EPSG:)?(32[67])(\d{2})", str(crs).strip().upper())
    if not m or not (1 <= int(m.group(2)) <= 60):
        raise ValueError(f"{crs!r} is not a WGS84 UTM CRS (expected EPSG:326xx or EPSG:327xx)")
    return int(m.group(2)), m.group(1) == "326"


def lonlat_to_utm(lon, lat, zone: int, north: bool = True):
    """Longitude/latitude in degrees -> (easting, northing) in metres.

    ``lon`` and ``lat`` can be scalars or numpy arrays of the same shape.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lam = np.radians(lon - central_meridian(zone))
    sphi = np.sin(np.radians(lat))
    c = 2.0 * np.sqrt(_N) / (1.0 + _N)
    t = np.sinh(np.arctanh(sphi) - c * np.arctanh(c * sphi))
    xi = np.arctan2(t, np.cos(lam))
    eta = np.arctanh(np.sin(lam) / np.sqrt(1.0 + t * t))
    e_sum, n_sum = eta.copy(), xi.copy()
    for j, a in enumerate(_ALPHA, start=1):
        e_sum = e_sum + a * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        n_sum = n_sum + a * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    easting = _E0 + _K0 * _AH * e_sum
    northing = (0.0 if north else _N0_SOUTH) + _K0 * _AH * n_sum
    return easting, northing


def utm_to_lonlat(easting, northing, zone: int, north: bool = True):
    """(easting, northing) in metres -> (lon, lat) in degrees. Exact inverse of
    ``lonlat_to_utm`` (round-trip error < 1e-9 degrees)."""
    e = np.asarray(easting, dtype=np.float64)
    n = np.asarray(northing, dtype=np.float64)
    xi = (n - (0.0 if north else _N0_SOUTH)) / (_K0 * _AH)
    eta = (e - _E0) / (_K0 * _AH)
    xi_p, eta_p = xi.copy(), eta.copy()
    for j, b in enumerate(_BETA, start=1):
        xi_p = xi_p - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p = eta_p - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    chi = np.arcsin(np.sin(xi_p) / np.cosh(eta_p))
    phi = chi.copy()
    for j, d in enumerate(_DELTA, start=1):
        phi = phi + d * np.sin(2 * j * chi)
    lam = np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    return central_meridian(zone) + np.degrees(lam), np.degrees(phi)
