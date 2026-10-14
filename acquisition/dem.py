"""
acquisition/dem.py  —  Copernicus DEM GLO-30 tiles (helper for PART 3)
======================================================================

WHAT THIS FILE DOES
-------------------
Gets elevation data for an AOI as ONE geographic (lon/lat) raster in metres:
    dem = source.load((min_lon, min_lat, max_lon, max_lat))   ->  LatLonRaster
Part 3 then resamples it onto the analysis grid and derives slope and radar
shadow masks from it. The DEM is also the input of the bonus flood-path tracing.

TWO SOURCES (same interface, pick one)
--------------------------------------
  LocalDemSource(path)        a single GeoTIFF or a folder of 1x1 degree tiles on
                              disk. Works OFFLINE: the safest choice for the demo.
  CopernicusDemDownloader()   downloads the 1x1 degree tiles you need into a cache
                              folder (once), then behaves like LocalDemSource.

NON-OBVIOUS TRICKS
------------------
  TRICK 1  Tiles are found by NAME, computed from coordinates. One tile covers
           1 x 1 degree, named after its SOUTH-WEST corner, so the tiles needed
           for any bbox follow from floor/ceil. No catalogue search required.
  TRICK 2  Exact-integer boundaries: a bbox whose max edge is exactly on a whole
           degree (e.g. 86.0) must NOT pull in the next tile. ceil(max) - 1.
  TRICK 3  Tiles share one pixel lattice, so mosaicking is an integer paste (no
           resampling, no blur). The paste rounds, and reports the worst
           sub-pixel error so an unexpected lattice cannot pass silently.
  TRICK 4  Download to ``.part`` then atomic rename (a half-file never looks
           complete), with retries. A missing tile (HTTP 404, e.g. open sea) is
           a warning, not a crash.
  TRICK 5  Reader and HTTP session are injectable => tested offline with fakes.

DATA RULES / ATTRIBUTION
------------------------
Copernicus DEM is an allowed input. The brief requires the statement in
``DEM_ATTRIBUTION`` in every submission; Part 3 writes it into metadata.json.
Heights are above the EGM2008 geoid (orthometric), which is fine for slope.

The download URL pattern below is the public AWS open-data copy of the same
product. I could not test it from here: verify one URL in a browser first, or
use LocalDemSource with tiles downloaded from the Copernicus Data Space.
"""
from __future__ import annotations

import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import requests

log = logging.getLogger("acquisition.dem")

DEM_ATTRIBUTION = (
    "Produced using Copernicus WorldDEM-30 \u00a9 DLR e.V. 2010\u20132014 and \u00a9 Airbus "
    "Defence and Space GmbH 2014\u20132018 provided under COPERNICUS by the European Union "
    "and ESA; all rights reserved.")


class DemError(Exception):
    """Elevation data could not be obtained."""


# ----------------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------------

@dataclass
class LatLonRaster:
    """A single-band raster on a regular lon/lat lattice (what DEM tiles are).

    data   (rows, cols) float32 metres, NaN = no data
    lon0   longitude of the WEST edge of column 0
    lat0   latitude of the NORTH edge of row 0
    dlon   pixel width in degrees  (> 0)
    dlat   pixel height in degrees (> 0; rows go south)

    Pixel centre (row r, col c) is at lon0 + (c+0.5)*dlon, lat0 - (r+0.5)*dlat.
    Pixels need NOT be square in degrees (they are 1 arc-second both ways only
    south of 50 N), which is why this is not the metre-based GeoRaster of Part 2.
    """
    data: np.ndarray
    lon0: float
    lat0: float
    dlon: float
    dlat: float

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def bounds(self) -> tuple:
        """(min_lon, min_lat, max_lon, max_lat)"""
        return (self.lon0, self.lat0 - self.height * self.dlat,
                self.lon0 + self.width * self.dlon, self.lat0)


# ----------------------------------------------------------------------------
# Tile naming (TRICK 1, 2)
# ----------------------------------------------------------------------------

def tile_name(lat_floor: int, lon_floor: int) -> str:
    """Name of the 1x1 degree GLO-30 tile whose SOUTH-WEST corner is
    (lat_floor, lon_floor):  Copernicus_DSM_COG_10_N27_00_E085_00_DEM
    ('10' = 1 arc-second = 30 m)."""
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"


def tiles_needed(bbox: Sequence[float]) -> list:
    """All (lat_floor, lon_floor) tiles intersecting bbox = (min_lon, min_lat, max_lon, max_lat).

    TRICK 2: ranges end at ceil(max) exclusive, so a bbox ending exactly on a
    whole degree does not request a useless extra tile."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lons = range(math.floor(min_lon), max(math.ceil(max_lon), math.floor(min_lon) + 1))
    lats = range(math.floor(min_lat), max(math.ceil(max_lat), math.floor(min_lat) + 1))
    return [(la, lo) for la in lats for lo in lons]


# ----------------------------------------------------------------------------
# Reading + mosaic
# ----------------------------------------------------------------------------

def read_latlon_tile(path) -> LatLonRaster:
    """Read a DEM GeoTIFF (EPSG:4326) with rasterio. The only rasterio use here."""
    try:
        import rasterio
    except ImportError as exc:                                   # pragma: no cover
        raise DemError("rasterio is required to read DEM tiles: pip install rasterio") from exc
    with rasterio.open(path) as src:
        t = src.transform
        if abs(t.b) > 1e-12 or abs(t.d) > 1e-12 or t.e > 0:
            raise DemError(f"{path}: expected a north-up lon/lat raster")
        data = src.read(1).astype("float32")
        if src.nodata is not None and not np.isnan(src.nodata):
            data[data == src.nodata] = np.nan
        return LatLonRaster(data, t.c, t.f, abs(t.a), abs(t.e))


def mosaic_latlon(tiles: Sequence[LatLonRaster]) -> tuple:
    """Stitch tiles on the same lattice. Returns (raster, max_fractional_offset_px).

    TRICK 3: integer paste, first tile wins in overlaps, NaN filled from others.
    """
    if not tiles:
        raise DemError("no DEM tiles to mosaic")
    ref = tiles[0]
    for t in tiles[1:]:
        if abs(t.dlon - ref.dlon) > 1e-12 or abs(t.dlat - ref.dlat) > 1e-12:
            raise DemError("DEM tiles have different pixel sizes (AOI crosses a latitude "
                           "band where tile resolution changes); not supported.")
    if len(tiles) == 1:
        return ref, 0.0
    offs, worst = [], 0.0
    for t in tiles:
        cx, cy = (t.lon0 - ref.lon0) / ref.dlon, (ref.lat0 - t.lat0) / ref.dlat
        ix, iy = int(round(cx)), int(round(cy))
        worst = max(worst, abs(cx - ix), abs(cy - iy))
        offs.append((ix, iy))
    min_x, min_y = min(o[0] for o in offs), min(o[1] for o in offs)
    width = max(o[0] + t.width for o, t in zip(offs, tiles)) - min_x
    height = max(o[1] + t.height for o, t in zip(offs, tiles)) - min_y
    canvas = np.full((height, width), np.nan, dtype=np.float32)
    for (ix, iy), t in zip(offs, tiles):
        view = canvas[iy - min_y: iy - min_y + t.height, ix - min_x: ix - min_x + t.width]
        empty = np.isnan(view)
        view[empty] = t.data[empty]
    return LatLonRaster(canvas, ref.lon0 + min_x * ref.dlon, ref.lat0 - min_y * ref.dlat,
                        ref.dlon, ref.dlat), worst


# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------

class LocalDemSource:
    """DEM from disk: a single GeoTIFF, or a folder containing 1x1 degree tiles
    (searched recursively by the tile-name token, e.g. 'N27_00_E085_00')."""

    def __init__(self, path, reader: Callable = read_latlon_tile):
        self.path, self.reader = Path(path), reader

    def describe(self) -> str:
        return str(self.path)

    def tile_files(self, bbox) -> list:
        files = []
        for lat, lon in tiles_needed(bbox):
            token = tile_name(lat, lon).split("COG_10_")[1].split("_DEM")[0]   # N27_00_E085_00
            hits = sorted(self.path.rglob(f"*{token}*.tif"))
            if hits:
                files.append(hits[0])
            else:
                log.warning("DEM tile %s not found in %s", token, self.path)
        return files

    def load(self, bbox) -> LatLonRaster:
        if self.path.is_file():
            return self.reader(self.path)
        files = self.tile_files(bbox)
        if not files:
            raise DemError(f"No DEM tiles covering {tuple(bbox)} found in {self.path}")
        tiles = [self.reader(f) for f in files]
        dem, worst = mosaic_latlon(tiles)
        if worst > 0.05:
            log.warning("DEM tiles are %.2f px off a common lattice; pasted with rounding.", worst)
        return dem


class CopernicusDemDownloader:
    """Downloads the needed 1x1 degree tiles once into ``cache_dir`` (TRICK 4)."""

    def __init__(self, cache_dir="data/raw/dem",
                 base_url: str = "https://copernicus-dem-30m.s3.amazonaws.com",
                 session: Optional[requests.Session] = None,
                 reader: Callable = read_latlon_tile,
                 max_retries: int = 4, retry_base_delay_s: float = 2.0, timeout_s: float = 60.0):
        self.cache_dir = Path(cache_dir)
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.reader = reader
        self.max_retries, self.delay, self.timeout = max_retries, retry_base_delay_s, timeout_s

    def describe(self) -> str:
        return f"{self.base_url} (cache {self.cache_dir})"

    def _download_tile(self, name: str) -> Optional[Path]:
        final = self.cache_dir / f"{name}.tif"
        if final.exists() and final.stat().st_size > 0:
            return final
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        part = final.with_suffix(".tif.part")
        url = f"{self.base_url}/{name}/{name}.tif"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, stream=True, timeout=self.timeout)
                try:
                    if resp.status_code == 404:
                        log.warning("DEM tile %s does not exist (open sea?). Skipping.", name)
                        return None
                    if resp.status_code == 429 or resp.status_code >= 500:
                        raise requests.ConnectionError(f"HTTP {resp.status_code}")
                    if resp.status_code != 200:
                        raise DemError(f"HTTP {resp.status_code} for {url}")
                    with open(part, "wb") as fh:
                        for chunk in resp.iter_content(1 << 20):
                            if chunk:
                                fh.write(chunk)
                finally:
                    resp.close()
                os.replace(part, final)                           # atomic
                return final
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == self.max_retries:
                    raise DemError(f"Could not download {name}: {exc}") from exc
                time.sleep(self.delay * 2 ** (attempt - 1) * (1 + 0.25 * random.random()))
        return None

    def load(self, bbox) -> LatLonRaster:
        files = [self._download_tile(tile_name(la, lo)) for la, lo in tiles_needed(bbox)]
        files = [f for f in files if f]
        if not files:
            raise DemError(f"No DEM tiles available for {tuple(bbox)}")
        dem, worst = mosaic_latlon([self.reader(f) for f in files])
        if worst > 0.05:
            log.warning("DEM tiles are %.2f px off a common lattice; pasted with rounding.", worst)
        return dem
