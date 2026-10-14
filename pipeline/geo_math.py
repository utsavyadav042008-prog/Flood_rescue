"""Dependency-free WGS84 <-> UTM conversion (pure Python, ``math`` only).

Why does this exist when ``pyproj`` does the same thing?
--------------------------------------------------------
On Windows, ``rasterio`` / ``pyproj`` / ``geopandas`` are the packages most
likely to fail to install on Day 1.  The mock-data generator, the AOI helpers
and the contract tests must still run in that situation, so the small amount
of map projection they need lives here.  Real pipeline stages (T1-T3) should
keep using pyproj/rasterio; this module is for *tooling and tests*.

Method: Krueger n-series (3rd order) for the Transverse Mercator projection.
Accuracy is far better than a millimetre inside a UTM zone, which the unit
test checks against an independent numerical meridian-arc integral.
"""

from __future__ import annotations

import math

# WGS84 ellipsoid -----------------------------------------------------------
_A = 6378137.0                       # semi-major axis (m)
_F = 1.0 / 298.257223563             # flattening
_N = _F / (2.0 - _F)                 # third flattening "n"
_K0 = 0.9996                         # UTM scale factor on the central meridian
_E0 = 500_000.0                      # false easting
_N0_SOUTH = 10_000_000.0             # false northing in the southern hemisphere
_E = 2.0 * math.sqrt(_N) / (1.0 + _N)  # first eccentricity (identity in n)
_A_HAT = _A / (1.0 + _N) * (1.0 + _N**2 / 4.0 + _N**4 / 64.0)

_ALPHA = (
    _N / 2 - 2 * _N**2 / 3 + 5 * _N**3 / 16,
    13 * _N**2 / 48 - 3 * _N**3 / 5,
    61 * _N**3 / 240,
)
_BETA = (
    _N / 2 - 2 * _N**2 / 3 + 37 * _N**3 / 96,
    _N**2 / 48 + _N**3 / 15,
    17 * _N**3 / 480,
)
_DELTA = (
    2 * _N - 2 * _N**2 / 3 - 2 * _N**3,
    7 * _N**2 / 3 - 8 * _N**3 / 5,
    56 * _N**3 / 15,
)


def utm_zone_from_epsg(epsg: int) -> tuple[int, bool]:
    """Return ``(zone, is_north)`` for EPSG:326zz (north) or EPSG:327zz (south)."""
    if 32601 <= epsg <= 32660:
        return epsg - 32600, True
    if 32701 <= epsg <= 32760:
        return epsg - 32700, False
    raise ValueError(f"EPSG:{epsg} is not a WGS84 UTM zone (expected 326xx or 327xx)")


def epsg_for_utm(zone: int, is_north: bool) -> int:
    """Inverse of :func:`utm_zone_from_epsg`."""
    if not 1 <= zone <= 60:
        raise ValueError(f"UTM zone must be 1..60, got {zone}")
    return (32600 if is_north else 32700) + zone


def central_meridian(zone: int) -> float:
    """Longitude (degrees) of the zone's central meridian: zone 45 -> 87 E."""
    return zone * 6 - 183.0


def lonlat_to_utm(lon: float, lat: float, epsg: int) -> tuple[float, float]:
    """WGS84 longitude/latitude (degrees) -> UTM (easting, northing) in metres."""
    zone, north = utm_zone_from_epsg(epsg)
    lam = math.radians(lon - central_meridian(zone))
    phi = math.radians(lat)

    # Conformal latitude via t = sinh(atanh(sin phi) - e * atanh(e * sin phi)).
    sin_phi = math.sin(phi)
    t = math.sinh(math.atanh(sin_phi) - _E * math.atanh(_E * sin_phi))
    xi_p = math.atan2(t, math.cos(lam))
    eta_p = math.atanh(math.sin(lam) / math.sqrt(1.0 + t * t))

    xi = xi_p + sum(
        a * math.sin(2 * j * xi_p) * math.cosh(2 * j * eta_p)
        for j, a in enumerate(_ALPHA, start=1)
    )
    eta = eta_p + sum(
        a * math.cos(2 * j * xi_p) * math.sinh(2 * j * eta_p)
        for j, a in enumerate(_ALPHA, start=1)
    )
    easting = _E0 + _K0 * _A_HAT * eta
    northing = (0.0 if north else _N0_SOUTH) + _K0 * _A_HAT * xi
    return easting, northing


def utm_to_lonlat(easting: float, northing: float, epsg: int) -> tuple[float, float]:
    """UTM (easting, northing) in metres -> WGS84 ``(lon, lat)`` in degrees."""
    zone, north = utm_zone_from_epsg(epsg)
    xi = (northing - (0.0 if north else _N0_SOUTH)) / (_K0 * _A_HAT)
    eta = (easting - _E0) / (_K0 * _A_HAT)

    xi_p = xi - sum(
        b * math.sin(2 * j * xi) * math.cosh(2 * j * eta)
        for j, b in enumerate(_BETA, start=1)
    )
    eta_p = eta - sum(
        b * math.cos(2 * j * xi) * math.sinh(2 * j * eta)
        for j, b in enumerate(_BETA, start=1)
    )
    chi = math.asin(math.sin(xi_p) / math.cosh(eta_p))
    phi = chi + sum(d * math.sin(2 * j * chi) for j, d in enumerate(_DELTA, start=1))
    lam = math.atan2(math.sinh(eta_p), math.cos(xi_p))
    return central_meridian(zone) + math.degrees(lam), math.degrees(phi)


def lonlat_to_utm_array(lon, lat, epsg: int):
    """Vectorised :func:`lonlat_to_utm` for numpy arrays -> ``(easting, northing)`` arrays.

    Same maths, evaluated on whole grids at once; the dashboard uses it to
    resample UTM rasters onto a lon/lat grid without GDAL.  numpy is imported
    lazily so the rest of this module stays standard-library only.
    """
    import numpy as np

    zone, north = utm_zone_from_epsg(epsg)
    lam = np.radians(np.asarray(lon, dtype=float) - central_meridian(zone))
    phi = np.radians(np.asarray(lat, dtype=float))
    sin_phi = np.sin(phi)
    t = np.sinh(np.arctanh(sin_phi) - _E * np.arctanh(_E * sin_phi))
    xi_p = np.arctan2(t, np.cos(lam))
    eta_p = np.arctanh(np.sin(lam) / np.sqrt(1.0 + t * t))
    xi = xi_p.copy()
    eta = eta_p.copy()
    for j, a in enumerate(_ALPHA, start=1):
        xi += a * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta += a * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    return _E0 + _K0 * _A_HAT * eta, (0.0 if north else _N0_SOUTH) + _K0 * _A_HAT * xi
