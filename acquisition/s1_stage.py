"""Stage 'acquire' - a same-orbit Sentinel-1 before/after pair on one UTM grid.

Glue around T1's Parts 1-4:
    Part 1  acquisition/sentinel.py       search + same-track pair selection (+ download)
    Part 2  preprocessing/sentinel1.py    ESA SNAP chain -> VV/VH dB GeoTIFFs
    Part 3  preprocessing/alignment.py    common 10 m grid, Copernicus DEM, slope,
                                          terrain (layover/shadow) and permanent-water masks
    Part 4  preprocessing/sentinel2.py    optional optical evidence when the sky is clear

Backends (``sentinel1.backend`` in config/default.yaml):
    cdse_snap           Copernicus Data Space download + SNAP (reference chain; needs SNAP
                        and CDSE_USERNAME / CDSE_PASSWORD)
    planetary_computer  Sentinel-1 RTC streamed from Microsoft Planetary Computer
                        (no SNAP, no account; preprocessing/s1_rtc.py)
    local               your own preprocessed before/after GeoTIFFs (offline demos)
    auto                cdse_snap if SNAP and credentials are present, else planetary_computer

Writes: before.tif, after.tif (band 1 VV dB, band 2 VH dB, NaN = no data),
        metadata.json, and extras/ (dem, slope, terrain_mask, permanent_water,
        valid_mask, scene_pair.json, optical/...) used by infer, figures and the
        flood-path bonus.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

from pipeline.config import REPO_ROOT
from pipeline.contracts import Artifact
from pipeline.errors import AcquisitionError
from pipeline.fileio import write_geotiff, write_json
from pipeline.run import RunPaths
from pipeline.schemas import AOI, Provenance, parse_date

log = logging.getLogger("floodresq.acquire")

S1_ATTRIBUTION = "Contains modified Copernicus Sentinel data 2026."
EXTRA_RASTERS = ("dem", "slope", "terrain_mask", "permanent_water", "valid_mask")


def _data(*parts: str) -> Path:
    return REPO_ROOT.joinpath("data", *parts)


def _snap_available() -> bool:
    try:
        from preprocessing.sentinel1 import find_gpt
        find_gpt(os.environ.get("SNAP_GPT"))
        return True
    except Exception:
        return False


def choose_backend(s1cfg: dict) -> str:
    backend = str(s1cfg.get("backend", "auto")).lower()
    if backend != "auto":
        return backend
    if _snap_available() and os.environ.get("CDSE_USERNAME") and os.environ.get("CDSE_PASSWORD"):
        return "cdse_snap"
    return "planetary_computer"


def _acq_config(cfg: dict):
    from acquisition.sentinel import AcquisitionConfig, Relaxation
    s1 = cfg.get("sentinel1", {})
    w = int(s1.get("search_window_days", 40))
    cov = float(s1.get("min_coverage", 0.95))
    return AcquisitionConfig(
        ladder=(Relaxation(w, cov, "standard"), Relaxation(2 * w, cov, "wider time window"),
                Relaxation(2 * w, float(s1.get("min_coverage_relaxed", 0.6)), "partial AOI coverage")),
        max_aoi_km2=float(cfg.get("limits", {}).get("max_aoi_km2", 2500)),
        search_cache_dir=_data("cache", "search"))


def _local_pair(aoi, event: date, s1cfg: dict):
    """A ScenePair describing user-supplied GeoTIFFs (for the 'local' backend)."""
    from acquisition.sentinel import Acquisition, Scene, ScenePair
    loc = s1cfg.get("local", {}) or {}
    b_date, a_date = parse_date(loc.get("before_date"), "before_date"), parse_date(loc.get("after_date"), "after_date")
    orbit, direction = int(loc.get("relative_orbit", 0)), str(loc.get("orbit_direction", "UNKNOWN")).upper()
    ring = [[aoi.min_lon, aoi.min_lat], [aoi.max_lon, aoi.min_lat], [aoi.max_lon, aoi.max_lat],
            [aoi.min_lon, aoi.max_lat], [aoi.min_lon, aoi.min_lat]]

    def scene(role, d):
        t = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        return Scene(product_id=f"local-{role}", name=Path(str(loc.get(role))).name, platform="S1?",
                     start=t, end=t, rel_orbit=orbit, direction=direction, rings=[ring])
    return ScenePair(aoi=aoi, event_date=event, before=Acquisition([scene("before", b_date)]),
                     after=Acquisition([scene("after", a_date)]), coverage=1.0, relaxation="local files",
                     warnings=["Sentinel-1 pair supplied as local files: orbit/date metadata come from config."])


def _prepare(backend: str, aoi, event: date, cfg: dict) -> tuple:
    """-> (ScenePair, PreInputs, source text)."""
    from preprocessing.alignment import PreInputs
    from preprocessing.sentinel1 import PreprocessConfig
    s1 = cfg.get("sentinel1", {})
    pre_cfg = PreprocessConfig(snap_mem_gb=int(s1.get("snap_mem_gb", 8)),
                               dem_path=s1.get("snap_dem_path") or None)
    if backend == "cdse_snap":
        from acquisition.sentinel import get_satellite_data
        from preprocessing.sentinel1 import preprocess_pair
        pair = get_satellite_data(aoi.as_tuple(), event, download=True, out_dir=_data("raw", "s1"),
                                  processed_dir=_data("processed"), config=_acq_config(cfg))
        pre = PreInputs.from_preprocessed(preprocess_pair(pair, pre_cfg, out_root=_data("interim", "s1")))
        return pair, pre, ("Sentinel-1 IW GRDH (Copernicus Data Space) processed with ESA SNAP: orbit, "
                           "border/thermal noise removal, sigma0 calibration, Refined Lee, terrain correction")
    if backend == "planetary_computer":
        from preprocessing import s1_rtc
        pair, pre = s1_rtc.prepare(aoi, event, _data("interim", "s1_rtc"), _acq_config(cfg), pre_cfg)
        return pair, pre, "Sentinel-1 IW RTC gamma0 (Microsoft Planetary Computer, sentinel-1-rtc)"
    if backend == "local":
        loc = s1.get("local", {}) or {}
        for role in ("before", "after"):
            if not loc.get(role) or not Path(loc[role]).exists():
                raise AcquisitionError(f"sentinel1.local.{role} is not set or the file does not exist.")
        pair = _local_pair(aoi, event, s1)
        return pair, PreInputs(Path(loc["before"]), Path(loc["after"]), params_hash="local",
                               warnings=list(pair.warnings)), "Sentinel-1 (user-supplied local GeoTIFFs)"
    raise AcquisitionError(f"Unknown sentinel1.backend '{backend}' (use auto, cdse_snap, "
                           "planetary_computer or local).")


def _dem_source(cfg: dict):
    from acquisition.dem import CopernicusDemDownloader, LocalDemSource
    local = (cfg.get("dem", {}) or {}).get("local_path")
    if local:
        p = Path(local)
        return LocalDemSource(p if p.is_absolute() else REPO_ROOT / p)
    return CopernicusDemDownloader(_data("raw", "dem"))


def _copy_raster(src: Path, dst: Path, prov: Provenance) -> None:
    import rasterio
    with rasterio.open(src) as s:
        arr = s.read()
        epsg = s.crs.to_epsg()
        transform = s.transform
    write_geotiff(dst, arr, transform=transform, epsg=epsg, provenance=prov,
                  nodata=float("nan") if arr.dtype.kind == "f" else None)


def run_acquire(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None:
    from acquisition.sentinel import AOI as T1AOI
    from acquisition.sentinel import AcquisitionError as T1AcquisitionError

    backend = choose_backend(cfg.get("sentinel1", {}))
    t1_aoi = T1AOI.from_bbox(aoi.bbox)
    try:
        from preprocessing.alignment import AlignmentConfig, build_analysis_package
        pair, pre, source = _prepare(backend, t1_aoi, event_date, cfg)
        pkg = build_analysis_package(pair, pre, AlignmentConfig(), _dem_source(cfg),
                                     processed_root=_data("processed"))
    except AcquisitionError:
        raise
    except T1AcquisitionError as exc:
        raise AcquisitionError(str(exc).split("\n")[0], str(exc)) from exc
    except ImportError as exc:
        raise AcquisitionError(f"A package needed for Sentinel-1 ({backend}) is missing: {exc}",
                               str(exc)) from exc
    except Exception as exc:                       # preprocessing / alignment / DEM problems
        raise AcquisitionError(f"Sentinel-1 preparation failed ({backend}): {exc}", repr(exc)) from exc

    warnings = list(pkg.warnings)
    optical_status = {"status": "disabled"}
    if cfg.get("sentinel2", {}).get("enabled", False):
        try:
            import numpy as np
            import rasterio
            from preprocessing.sentinel2 import build_optical_layers
            with rasterio.open(pkg.files["permanent_water"]) as src:
                perm = src.read(1) > 0
            res = build_optical_layers(pair, pkg.grid, pkg.out_dir, permanent_water=perm)
            optical_status = {"status": res.status, "reason": res.reason}
            if res.status != "used":
                warnings.append(f"Sentinel-2: {res.reason}")
        except Exception as exc:                   # optional by design: never fatal
            optical_status = {"status": "unavailable", "reason": str(exc)}
            warnings.append(f"Sentinel-2 optical layers unavailable: {exc}")

    crs = pkg.grid.crs
    prov = Provenance(producer=f"acquisition.s1_stage[{backend}]", source=source, crs=crs,
                      event_date=event_date.isoformat())
    _copy_raster(pkg.files["before"], run.path(Artifact.BEFORE), prov)
    _copy_raster(pkg.files["after"], run.path(Artifact.AFTER), prov)

    extras = run.root / "extras"
    extras.mkdir(exist_ok=True)
    for name in EXTRA_RASTERS:
        if name in pkg.files and Path(pkg.files[name]).exists():
            shutil.copyfile(pkg.files[name], extras / f"{name}.tif")
    optical_dir = Path(pkg.out_dir)
    if optical_status.get("status") in ("used", "partial"):
        (extras / "optical").mkdir(exist_ok=True)
        for f in optical_dir.glob("s2_*.tif"):
            shutil.copyfile(f, extras / "optical" / f.name)
        for f in optical_dir.glob("quicklook_*.png"):
            shutil.copyfile(f, extras / "optical" / f.name)
    (extras / "scene_pair.json").write_text(json.dumps(pair.to_dict(), indent=2), encoding="utf-8")

    package_meta = json.loads(Path(pkg.metadata_path).read_text())
    write_json(run.path(Artifact.METADATA), {
        "source": source, "backend": backend,
        "before_date": pair.before.date.isoformat(), "after_date": pair.after.date.isoformat(),
        "relative_orbit_before": pair.before.track_key[0], "relative_orbit_after": pair.after.track_key[0],
        "orbit_direction": pair.before.track_key[1],
        "platform_before": pair.before.platform, "platform_after": pair.after.platform,
        "products_before": [s.name for s in pair.before.slices],
        "products_after": [s.name for s in pair.after.slices],
        "gap_days": pair.gap_days, "aoi_coverage": round(pair.coverage, 4), "selection_rung": pair.relaxation,
        "crs": crs, "pixel_size_m": pkg.grid.res, "bands": ["VV_dB", "VH_dB"],
        "dem_source": package_meta.get("dem_source"), "terrain": package_meta.get("terrain"),
        "permanent_water": package_meta.get("permanent_water"), "qa": package_meta.get("qa"),
        "sentinel2": optical_status, "warnings": warnings,
        "attributions": package_meta.get("attributions", [S1_ATTRIBUTION]),
    }, prov)
