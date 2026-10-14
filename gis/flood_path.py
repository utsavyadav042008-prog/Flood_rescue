"""BONUS: trace a flood / debris-flow path downhill from any upstream point.

"Given any point upstream, trace the flood path down the valley using elevation
data and list the settlements along it."

Method (steepest descent on the Copernicus DEM, with pit breaching)
-------------------------------------------------------------------
1. Start at the DEM cell under the clicked point.
2. Step to the LOWEST of the 8 neighbours (D8), as water would.
3. Real DEMs have pits and flat reaches (30 m pixels, lakes, bridges, noise).
   When no neighbour is lower, search a growing window (up to ``breach_m``) for
   the nearest cell that IS lower and jump straight to it (breaching the dam).
   This keeps the path in the valley instead of stopping at the first puddle.
4. Stop at the edge of the DEM, after ``max_km``, or when nothing lower exists
   within ``breach_m`` (a closed basin / lake).
5. Buffer the path by ``corridor_m`` and list settlements inside the corridor,
   ordered by distance along the path, with the elevation drop to each.

Honest limits: a debris flow can overtop low ridges and avulse at bends, and a
glacial-lake outburst spreads far wider in flat reaches than any fixed corridor.
This is a quick screening tool, not a hydraulic model.

Usage:
    python -m gis.flood_path --run outputs/<run_folder> --lon 85.36 --lat 28.27
    python -m gis.flood_path --bbox 85.1 27.85 85.45 28.3 --lon 85.36 --lat 28.27  # DEM download only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class FloodPath:
    lonlat: list                       # [[lon, lat], ...] along the path
    elevations: list                   # metres, same length
    length_km: float
    drop_m: float
    stop_reason: str
    settlements: list = field(default_factory=list)   # dicts, ordered downstream

    def to_geojson(self) -> dict:
        feats = [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": self.lonlat},
                  "properties": {"kind": "flood_path", "length_km": self.length_km, "drop_m": self.drop_m,
                                 "stop_reason": self.stop_reason}}]
        for s in self.settlements:
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": s["lonlat"]},
                          "properties": {"kind": "settlement_on_path",
                                         **{k: v for k, v in s.items() if k != "lonlat"}}})
        return {"type": "FeatureCollection", "features": feats,
                "attribution": "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and "
                               "© Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS "
                               "by the European Union and ESA; all rights reserved. © OpenStreetMap contributors."}


def trace(dem: np.ndarray, start_rc: tuple, res_m: float, *, max_km: float = 80.0,
          breach_m: float = 1500.0) -> tuple:
    """Pure-numpy path on a metric DEM grid -> (list of (row, col), stop_reason)."""
    h, w = dem.shape
    z = np.where(np.isfinite(dem), dem, np.inf)
    r, c = start_rc
    if not (0 <= r < h and 0 <= c < w) or not np.isfinite(dem[r, c]):
        raise ValueError("the start point is outside the elevation model")
    path, visited = [(r, c)], {(r, c)}
    travelled, max_steps = 0.0, int(max_km * 1000 / res_m * 2)
    max_rad = max(2, int(breach_m / res_m))
    for _ in range(max_steps):
        here = z[r, c]
        best, best_z = None, here
        for dr, dc in _NEIGH:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and z[rr, cc] < best_z and (rr, cc) not in visited:
                best, best_z = (rr, cc), z[rr, cc]
        if best is None:                                   # pit or flat: breach
            for rad in range(2, max_rad + 1):
                r0, r1, c0, c1 = max(0, r - rad), min(h, r + rad + 1), max(0, c - rad), min(w, c + rad + 1)
                win = z[r0:r1, c0:c1]
                lower = np.argwhere(win < here)
                if len(lower):
                    d = np.hypot(lower[:, 0] + r0 - r, lower[:, 1] + c0 - c)
                    k = int(np.argmin(d - 1e-6 * (here - win[lower[:, 0], lower[:, 1]])))
                    best = (int(lower[k, 0] + r0), int(lower[k, 1] + c0))
                    break
                if r0 == 0 or c0 == 0 or r1 == h or c1 == w:
                    return path, "reached the edge of the elevation model"
            if best is None:
                return path, f"no lower ground within {breach_m:.0f} m (closed basin or lake)"
        travelled += math.hypot(best[0] - r, best[1] - c) * res_m
        r, c = best
        path.append(best)
        visited.add(best)
        if r in (0, h - 1) or c in (0, w - 1):
            return path, "reached the edge of the elevation model"
        if travelled >= max_km * 1000:
            return path, f"stopped after {max_km:.0f} km"
    return path, "step limit reached"


def _load_dem_for_run(run_dir: Path):
    """(dem array, transform, epsg) from extras/dem.tif written by the acquire stage."""
    import rasterio
    p = Path(run_dir) / "extras" / "dem.tif"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (run the 'acquire' stage first, or use --bbox)")
    with rasterio.open(p) as src:
        return src.read(1).astype("float32"), src.transform, src.crs.to_epsg()


def _load_dem_for_bbox(bbox: tuple, res_m: float = 30.0):
    """Download Copernicus GLO-30 for the bbox (T1's downloader) onto a UTM grid."""
    from acquisition.dem import CopernicusDemDownloader
    from acquisition.sentinel import AOI as T1AOI
    from preprocessing.alignment import build_grid, dem_to_grid
    from preprocessing.sentinel1 import buffered_aoi, utm_epsg
    from rasterio.transform import from_origin
    aoi = T1AOI.from_bbox(bbox)
    crs = f"EPSG:{utm_epsg((aoi.min_lon + aoi.max_lon) / 2, (aoi.min_lat + aoi.max_lat) / 2)}"
    grid = build_grid(aoi, crs, res_m)
    dem = CopernicusDemDownloader(REPO_ROOT / "data" / "raw" / "dem").load(buffered_aoi(aoi, 1.0).as_tuple())
    arr = dem_to_grid(dem, grid)
    return arr, from_origin(grid.x0, grid.y0, grid.res, grid.res), int(crs.split(":")[1])


def _settlements(path: Optional[Path]) -> list:
    if path is None or not Path(path).exists():
        return []
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for f in fc.get("features", []):
        g = f.get("geometry") or {}
        if g.get("type") == "Point":
            out.append((g["coordinates"][:2], f.get("properties") or {}))
    return out


def trace_flood_path(lon: float, lat: float, *, run_dir: Optional[Path] = None, bbox: Optional[tuple] = None,
                     settlements_path: Optional[Path] = None, corridor_m: float = 500.0,
                     max_km: float = 80.0, breach_m: float = 1500.0) -> FloodPath:
    from pipeline.geo_math import lonlat_to_utm, lonlat_to_utm_array, utm_to_lonlat
    if run_dir is not None:
        dem, transform, epsg = _load_dem_for_run(Path(run_dir))
        settlements_path = settlements_path or Path(run_dir) / "osm" / "settlements.geojson"
    elif bbox is not None:
        dem, transform, epsg = _load_dem_for_bbox(tuple(bbox))
    else:
        raise ValueError("give run_dir or bbox")
    res = abs(transform.a)
    x, y = lonlat_to_utm(lon, lat, epsg)
    col, row = (x - transform.c) / transform.a, (y - transform.f) / transform.e     # north-up grid
    cells, reason = trace(dem, (int(row), int(col)), res, max_km=max_km, breach_m=breach_m)

    xs = np.array([transform.c + (c + 0.5) * transform.a for _, c in cells])
    ys = np.array([transform.f + (r + 0.5) * transform.e for r, _ in cells])
    lonlat = [list(map(lambda v: round(v, 6), utm_to_lonlat(px, py, epsg))) for px, py in zip(xs, ys)]
    elev = [round(float(dem[r, c]), 1) for r, c in cells]
    seg = np.hypot(np.diff(xs), np.diff(ys))
    along = np.concatenate([[0.0], np.cumsum(seg)])

    found = []
    for (slon, slat), props in _settlements(settlements_path):
        sx, sy = lonlat_to_utm_array(np.array([slon]), np.array([slat]), epsg)
        d = np.hypot(xs - sx[0], ys - sy[0])
        k = int(np.argmin(d))
        if d[k] <= corridor_m:
            found.append({"name": props.get("name", ""), "name_ne": props.get("name_ne"), "id": props.get("id"),
                          "distance_along_path_km": round(float(along[k]) / 1000, 2),
                          "distance_from_path_m": round(float(d[k])), "elevation_drop_m": round(elev[0] - elev[k], 1),
                          "lonlat": [slon, slat]})
    found.sort(key=lambda s: s["distance_along_path_km"])
    return FloodPath(lonlat, elev, round(float(along[-1]) / 1000, 2), round(elev[0] - elev[-1], 1), reason, found)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Trace a flood/debris path downhill and list settlements on it.")
    ap.add_argument("--run", help="run folder (uses extras/dem.tif and osm/settlements.geojson)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="download the DEM instead")
    ap.add_argument("--settlements", help="settlements GeoJSON (with --bbox)")
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--corridor-m", type=float, default=500.0)
    ap.add_argument("--max-km", type=float, default=80.0)
    ap.add_argument("--out", help="write the path + settlements as GeoJSON here")
    a = ap.parse_args(argv)
    fp = trace_flood_path(a.lon, a.lat, run_dir=Path(a.run) if a.run else None, bbox=a.bbox,
                          settlements_path=Path(a.settlements) if a.settlements else None,
                          corridor_m=a.corridor_m, max_km=a.max_km)
    print(f"path: {fp.length_km} km, drop {fp.drop_m} m ({fp.stop_reason})")
    for s in fp.settlements:
        print(f"  {s['distance_along_path_km']:7.2f} km  {s['name']}  ({s['distance_from_path_m']} m from path)")
    out = Path(a.out) if a.out else (Path(a.run) / "extras" / "flood_path.geojson" if a.run else None)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fp.to_geojson(), indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
