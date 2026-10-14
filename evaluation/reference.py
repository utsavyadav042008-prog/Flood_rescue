"""Read reference polygons (check-only) and burn them onto OUR raster grid.

The reference flood extent usually arrives as polygons in lon/lat.  To compare
it with our mask pixel by pixel we need it on the SAME grid.  This module does
that with numpy + ``matplotlib.path`` (point-in-polygon), so it needs neither
GDAL nor geopandas for GeoJSON files.

DATA RULE: files read here are check-only.  Never call this from the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline.errors import InputValidationError
from pipeline.geo_math import lonlat_to_utm

Ring = list          # [(lon, lat), ...]
Polygon = list       # [outer_ring, hole_ring, ...]


def _polygons_from_geometry(geometry: dict) -> list:
    kind, coords = geometry.get("type"), geometry.get("coordinates")
    if kind == "Polygon":
        return [[[tuple(p[:2]) for p in ring] for ring in coords]]
    if kind == "MultiPolygon":
        return [[[tuple(p[:2]) for p in ring] for ring in poly] for poly in coords]
    if kind == "GeometryCollection":
        return [p for g in geometry.get("geometries", []) for p in _polygons_from_geometry(g)]
    return []                                       # points / lines are ignored


def polygons_from_geojson(data: dict) -> list:
    """All polygons (outer ring + holes, lon/lat) of a Feature / FeatureCollection / geometry."""
    if data.get("type") == "FeatureCollection":
        return [p for f in data.get("features", []) for p in _polygons_from_geometry(f.get("geometry") or {})]
    if data.get("type") == "Feature":
        return _polygons_from_geometry(data.get("geometry") or {})
    return _polygons_from_geometry(data)


def load_reference_polygons(path: str | Path) -> list:
    """Polygons from a GeoJSON file; shapefile/GeoPackage need geopandas (optional)."""
    path = Path(path)
    if not path.exists():
        raise InputValidationError(f"Reference file not found: {path}")
    if path.suffix.lower() in (".geojson", ".json"):
        return polygons_from_geojson(json.loads(path.read_text(encoding="utf-8")))
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise InputValidationError(
            "Only GeoJSON is read without geopandas. Convert the file to GeoJSON (QGIS: Export > "
            "Save Features As) or install geopandas.") from exc
    return polygons_from_geojson(json.loads(gpd.read_file(path).to_crs(4326).to_json()))


def rasterize_polygons(polygons: list, transform: tuple, shape: tuple, epsg: int,
                       chunk: int = 400_000) -> np.ndarray:
    """Boolean array: True where a pixel CENTRE lies inside any polygon (holes respected).

    Only the window around each polygon is tested, and points are processed in
    chunks, so a few small polygons on a huge raster stay cheap and memory-safe.
    """
    try:
        from matplotlib.path import Path as MplPath
    except ImportError as exc:
        raise ImportError("matplotlib is required to rasterise reference polygons") from exc

    a, b, c, d, e, f = transform[:6]
    if abs(b) > 1e-9 or abs(d) > 1e-9:
        raise ValueError("rotated rasters are not supported")
    rows, cols = shape[:2]
    out = np.zeros((rows, cols), dtype=bool)

    for polygon in polygons:
        rings_xy = [np.array([lonlat_to_utm(lon, lat, epsg) for lon, lat in ring]) for ring in polygon if len(ring) >= 3]
        if not rings_xy:
            continue
        outer = rings_xy[0]
        col0 = max(int(np.floor((outer[:, 0].min() - c) / a)), 0)
        col1 = min(int(np.ceil((outer[:, 0].max() - c) / a)), cols - 1)
        row_a, row_b = (outer[:, 1].min() - f) / e, (outer[:, 1].max() - f) / e   # e < 0 for north-up
        row0, row1 = max(int(np.floor(min(row_a, row_b))), 0), min(int(np.ceil(max(row_a, row_b))), rows - 1)
        if col1 < col0 or row1 < row0:
            continue                                           # polygon entirely outside our raster
        rr, cc = np.meshgrid(np.arange(row0, row1 + 1), np.arange(col0, col1 + 1), indexing="ij")
        px = np.column_stack([(c + a * (cc.ravel() + 0.5)), (f + e * (rr.ravel() + 0.5))])
        inside = np.zeros(len(px), dtype=bool)
        for start in range(0, len(px), chunk):
            part = px[start:start + chunk]
            hit = MplPath(outer).contains_points(part)
            for hole in rings_xy[1:]:
                hit &= ~MplPath(hole).contains_points(part)
            inside[start:start + chunk] = hit
        window = out[row0:row1 + 1, col0:col1 + 1]
        window |= inside.reshape(window.shape)
    return out
