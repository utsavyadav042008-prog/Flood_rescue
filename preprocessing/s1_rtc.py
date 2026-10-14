"""
preprocessing/s1_rtc.py  —  Sentinel-1 from Microsoft Planetary Computer (no SNAP)
==================================================================================

WHY THIS EXISTS (added during integration)
------------------------------------------
T1's main path (``acquisition/sentinel.py`` + ``preprocessing/sentinel1.py``)
downloads full 1-2 GB GRD products from the Copernicus Data Space and runs ESA
SNAP on them.  That is the reference processing chain, but it needs a SNAP
install, a CDSE account and time.  For a live demo on an area chosen by the
judges, this backend reads the SAME Sentinel-1 acquisitions already processed
to radiometrically terrain-corrected backscatter (collection
``sentinel-1-rtc``), streaming ONLY the AOI window of each Cloud-Optimised
GeoTIFF.  Allowed by the data rules: it is Sentinel-1 data.

WHAT IS REUSED FROM T1 (so both backends behave identically downstream)
-----------------------------------------------------------------------
* ``select_pair``      same-track rule, relaxation ladder, coverage, warnings
* ``build_grid``       the shared 10 m UTM lattice
* ``to_db`` / ``erode_valid`` / ``qa_report``   same cleaning and QA
* the output is a ``PreInputs`` for ``preprocessing.alignment.build_analysis_package``

HONEST DIFFERENCES vs the SNAP chain
------------------------------------
* RTC gamma0 (terrain-flattened) instead of sigma0 + Refined Lee: less speckle
  filtering, flatter mountain slopes.  Thresholds are still in dB.
* The catalogue may lag behind ESA by days; very recent scenes can be missing.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from acquisition.sentinel import (AOI, AcquisitionConfig, AcquisitionError, Scene, ScenePair,
                                  _NAME_RE, _parse_dt, select_pair)
from preprocessing.alignment import GridSpec, PreInputs, build_grid
from preprocessing.sentinel1 import GeoRaster, PreprocessConfig, RasterioIO, erode_valid, qa_report, to_db, utm_epsg

log = logging.getLogger("preprocessing.s1_rtc")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
BACKEND_VERSION = "1"


class RtcError(AcquisitionError):
    """Planetary Computer search or read failed."""


def _rings(geom: dict) -> list:
    if not geom:
        return []
    if geom.get("type") == "Polygon":
        return [geom["coordinates"][0]]
    if geom.get("type") == "MultiPolygon":
        return [p[0] for p in geom["coordinates"]]
    return []


def item_to_scene(item) -> Optional[Scene]:
    """STAC item -> T1's Scene (only dual-pol VV+VH IW products are kept)."""
    props = item.properties
    if "vv" not in item.assets or "vh" not in item.assets:
        return None
    m = _NAME_RE.match(item.id)
    platform = m["platform"] if m else ("S1" + str(props.get("platform", "?"))[-1:].upper())
    try:
        start = _parse_dt(props.get("start_datetime") or props["datetime"])
        end = _parse_dt(props.get("end_datetime") or props.get("datetime"))
        rel = int(props["sat:relative_orbit"])
        direction = str(props["sat:orbit_state"]).upper()
    except (KeyError, TypeError, ValueError):
        return None
    rings = _rings(item.geometry)
    if not rings:
        return None
    return Scene(product_id=item.id, name=item.id, platform=platform, start=start, end=end,
                 rel_orbit=rel, direction=direction, rings=rings,
                 abs_orbit=int(m["abs"]) if m else props.get("sat:absolute_orbit"),
                 datatake_id=m["dt"] if m else None)


def search(aoi: AOI, event_date: date, cfg: AcquisitionConfig) -> tuple:
    """(scenes, {product_id: item}) for the widest rung of the relaxation ladder."""
    try:
        import planetary_computer
        import pystac_client
    except ImportError as exc:
        raise RtcError("The Planetary Computer backend needs `pip install pystac-client "
                       "planetary-computer`.") from exc
    max_w = max(r.window_days for r in cfg.ladder)
    today = datetime.now(timezone.utc).date()
    start, end = event_date - timedelta(days=max_w), min(event_date + timedelta(days=max_w), today)
    try:
        catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
        items = list(catalog.search(collections=[COLLECTION], bbox=list(aoi.as_tuple()),
                                    datetime=f"{start.isoformat()}/{end.isoformat()}").items())
    except Exception as exc:
        raise RtcError(f"Planetary Computer search failed: {exc}") from exc
    scenes, by_id = [], {}
    for it in items:
        s = item_to_scene(it)
        if s is not None and s.product_id not in by_id:
            scenes.append(s)
            by_id[s.product_id] = it
    log.info("rtc search: %d items, %d usable dual-pol scenes", len(items), len(scenes))
    return scenes, by_id


def read_on_grid(href: str, grid: GridSpec) -> np.ndarray:
    """One COG band resampled (bilinear, linear power units) onto the grid; NaN = no data."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "4",
           "GDAL_HTTP_RETRY_DELAY": "2"}
    with rasterio.Env(**env), rasterio.open(href) as src:
        with WarpedVRT(src, crs=grid.crs, transform=from_origin(grid.x0, grid.y0, grid.res, grid.res),
                       width=grid.width, height=grid.height, resampling=Resampling.bilinear,
                       src_nodata=src.nodata, nodata=np.nan, dtype="float32") as vrt:
            arr = vrt.read(1).astype("float32")
    arr[~np.isfinite(arr) | (arr <= 0)] = np.nan
    return arr


def acquisition_to_db(acq, by_id: dict, grid: GridSpec, cfg: PreprocessConfig) -> GeoRaster:
    """All slices of one pass -> 2-band dB raster on the grid (first slice wins in overlaps)."""
    vv = np.full((grid.height, grid.width), np.nan, np.float32)
    vh = vv.copy()
    for s in acq.slices:
        item = by_id[s.product_id]
        for arr, band in ((vv, "vv"), (vh, "vh")):
            piece = read_on_grid(item.assets[band].href, grid)
            empty = np.isnan(arr)
            arr[empty] = piece[empty]
    valid = np.isfinite(vv) & np.isfinite(vh)
    valid = erode_valid(valid, cfg.edge_guard_px)
    lo, hi = cfg.clip_db
    data = np.stack([np.clip(to_db(np.where(valid, vv, np.nan)), lo, hi),
                     np.clip(to_db(np.where(valid, vh, np.nan)), lo, hi)]).astype(np.float32)
    return GeoRaster(data, grid.x0, grid.y0, grid.res, grid.crs)


def prepare(aoi: AOI, event_date: date, out_root: Path, acq_cfg: Optional[AcquisitionConfig] = None,
            pre_cfg: Optional[PreprocessConfig] = None, res_m: float = 10.0) -> tuple:
    """Search, choose a same-track pair (T1 logic), read it, write pre tifs.
    Returns (ScenePair, PreInputs)."""
    acq_cfg = acq_cfg or AcquisitionConfig()
    pre_cfg = pre_cfg or PreprocessConfig()
    scenes, by_id = search(aoi, event_date, acq_cfg)
    pair: ScenePair = select_pair(scenes, aoi, event_date, acq_cfg)
    crs = f"EPSG:{utm_epsg((aoi.min_lon + aoi.max_lon) / 2, (aoi.min_lat + aoi.max_lat) / 2)}"
    grid = build_grid(aoi, crs, res_m)
    phash = hashlib.sha1(f"rtc{BACKEND_VERSION}|{grid.to_dict()}|{pre_cfg.edge_guard_px}|{pre_cfg.clip_db}"
                         .encode()).hexdigest()[:10]
    out_dir = Path(out_root) / f"{aoi.key()}_{event_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    io = RasterioIO()
    paths, warnings = {}, list(pair.warnings)
    for role, acq in (("before", pair.before), ("after", pair.after)):
        ids = "-".join(sorted(s.product_id[-28:] for s in acq.slices))
        path = out_dir / f"rtc_{role}_{phash}_{hashlib.sha1(ids.encode()).hexdigest()[:8]}.tif"
        if not path.exists():
            r = acquisition_to_db(acq, by_id, grid, pre_cfg)
            _, qa_warn = qa_report(r, pre_cfg)
            warnings += [f"[{role}] {w}" for w in qa_warn]
            io.write(path, r, descriptions=["VV_dB", "VH_dB"],
                     tags={"units": "dB", "source": "sentinel-1-rtc (Planetary Computer)", "role": role})
        paths[role] = path
    pre = PreInputs(paths["before"], paths["after"], crs=crs, params_hash=f"rtc-{phash}", warnings=warnings)
    return pair, pre
