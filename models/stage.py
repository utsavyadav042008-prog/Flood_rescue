"""Stage 'infer' - flood/debris mask from the Sentinel-1 before/after pair.

Reads:  before.tif, after.tif (2 bands VV, VH in dB, same UTM grid)
        extras/permanent_water.tif, extras/terrain_mask.tif, extras/slope.tif,
        extras/optical/s2_evidence.tif   (all optional, written by 'acquire')
Writes: flood_mask.tif (uint8 0/1), flood_confidence.tif (float32 0..1),
        extras/inference_meta.json

Method, in order of preference:
1. T2's U-Net (``models/inference.py`` FloodPredictor) from ``model.package``
   (a verified model_package folder) or ``model.weights_path`` (.pt), if
   PyTorch is installed.  Threshold = the one frozen on validation.
2. Fallback: ``models/change_detection.py`` (log-ratio + debris), clearly
   labelled as NOT the AI model in provenance and in the summary warnings.

Post-processing shared by both:
* permanent water (river channel before the event) is never "new flood";
* pixels flagged as radar layover/shadow are removed (cfg infer.mask_unreliable_terrain);
* optional union with Sentinel-2 change evidence where the sky was clear.
The contract wants a strict 0/1 mask, so "no data" (255 in T2's mask) becomes 0;
the no-data share is recorded in inference_meta.json instead.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

from pipeline.config import REPO_ROOT
from pipeline.contracts import Artifact
from pipeline.errors import ModelError
from pipeline.fileio import write_geotiff, write_json
from pipeline.run import RunPaths
from pipeline.schemas import AOI, Provenance

log = logging.getLogger("floodresq.infer")

#: terrain_mask bits (preprocessing/alignment.py): layover 2, local shadow 4, cast shadow 8
UNRELIABLE_TERRAIN_BITS = 2 | 4 | 8


def extras_dir(run: RunPaths) -> Path:
    return run.root / "extras"


def _read_extra(run: RunPaths, name: str, band: int = 1):
    import rasterio
    p = extras_dir(run) / name
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        return src.read(band)


def _resolve(path_text: str) -> Optional[Path]:
    if not path_text:
        return None
    p = Path(path_text)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_unet(cfg: dict):
    """FloodPredictor or None (with the reason)."""
    mcfg = cfg.get("model", {})
    if not mcfg.get("use_unet", True):
        return None, "U-Net disabled in config (model.use_unet: false)"
    package = _resolve(str(mcfg.get("package", "") or ""))
    weights = _resolve(str(mcfg.get("weights_path", "") or ""))
    source = package if package is not None and package.is_dir() else \
        (weights if weights is not None and weights.is_file() else None)
    if source is None:
        return None, (f"no trained model found (looked for {mcfg.get('package')} and "
                      f"{mcfg.get('weights_path')})")
    try:
        import torch  # noqa: F401
    except ImportError:
        return None, "PyTorch is not installed (pip install -r requirements-ml.txt)"
    from models.inference import FloodPredictor
    try:
        kw = {"tta": bool(mcfg.get("tta", False))}
        pred = (FloodPredictor.from_package(source, mcfg.get("device"), **kw) if source.is_dir()
                else FloodPredictor.from_checkpoint(source, mcfg.get("device"), **kw))
    except Exception as exc:                                # damaged package, wrong format ...
        return None, f"model could not be loaded: {exc}"
    return pred, ""


def run_infer(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None:
    import numpy as np
    import rasterio

    icfg = cfg.get("infer", {})
    warnings: list = []
    meta: dict = {}

    def read_pair(path):
        with rasterio.open(path) as src:
            arr = src.read().astype("float32")
            return arr, src.transform, src.crs.to_epsg(), abs(src.transform.a)

    before, transform, epsg, pixel_m = read_pair(run.path(Artifact.BEFORE))
    after, *_ = read_pair(run.path(Artifact.AFTER))
    if before.shape != after.shape:
        raise ModelError("before.tif and after.tif do not have the same size / bands.")
    vv_only = before.shape[0] < 2
    if vv_only:          # the contract guarantees VV only; VH is optional
        before, after = np.concatenate([before, before]), np.concatenate([after, after])
        warnings.append("Only the VV band is available; VH-based cues are not used.")
    before, after = before[:2], after[:2]
    for arr in (before, after):
        arr[arr <= -9998] = np.nan                         # defensive: numeric nodata markers

    predictor, why_not = (None, "the U-Net needs VV and VH, only VV is available") if vv_only else _load_unet(cfg)
    if predictor is not None:
        from models.inference import FLOOD_IDX, postprocess
        pred = predictor.predict_arrays(before, after, unit="db")
        thr, thr_src = predictor.resolve_threshold(icfg.get("threshold"))
        mask255, conf = postprocess(pred.prob, pred.invalid, thr,
                                    int(icfg.get("min_blob_pixels", 8)), int(icfg.get("fill_hole_pixels", 8)))
        mask = mask255 == 1
        conf = pred.prob[FLOOD_IDX].astype("float32")
        invalid = pred.invalid
        method = f"unet:{cfg.get('model', {}).get('version', 'unknown')}"
        meta.update({"threshold": thr, "threshold_source": thr_src, "model": predictor.model_info,
                     "median_vv_change_db": pred.median_shift_db})
    else:
        from models.change_detection import ChangeConfig, detect
        warnings.append(f"AI model not used ({why_not}); the flood map comes from the non-AI "
                        f"change-detection baseline.")
        ccfg = ChangeConfig(drop_db=float(icfg.get("drop_db", 3.0)), water_db=float(icfg.get("water_db", -15.0)),
                            debris=bool(icfg.get("detect_debris", True)),
                            change_db=float(icfg.get("debris_change_db", 4.0)),
                            river_db=float(icfg.get("river_db", -18.0)),
                            smooth_px=int(icfg.get("smooth_px", 2)),
                            debris_max_slope_deg=float(icfg.get("debris_max_slope_deg", 15.0)),
                            debris_max_dist_m=float(icfg.get("debris_max_dist_m", 1000.0)),
                            min_pixels=int(icfg.get("min_blob_pixels", 10)))
        res = detect(before, after, slope=_read_extra(run, "slope.tif"),
                     permanent_water=_read_extra(run, "permanent_water.tif"), pixel_m=pixel_m, cfg=ccfg)
        mask, conf = res["mask"], res["confidence"]
        invalid = ~(np.isfinite(before).all(axis=0) & np.isfinite(after).all(axis=0))
        method = "change-detection-baseline"
        meta.update(res["info"])

    # ---- shared post-processing ---------------------------------------------------------
    perm = _read_extra(run, "permanent_water.tif")
    if perm is not None and perm.shape == mask.shape and icfg.get("exclude_permanent_water", True):
        meta["removed_permanent_water_px"] = int((mask & (perm > 0)).sum())
        mask &= ~(perm > 0)
    terrain = _read_extra(run, "terrain_mask.tif")
    if terrain is not None and terrain.shape == mask.shape and icfg.get("mask_unreliable_terrain", True):
        bad = (terrain.astype("uint8") & UNRELIABLE_TERRAIN_BITS) > 0
        meta["removed_unreliable_terrain_px"] = int((mask & bad).sum())
        mask &= ~bad
    if icfg.get("use_optical_evidence", True):
        ev_path = extras_dir(run) / "optical" / "s2_evidence.tif"
        if ev_path.exists():
            with rasterio.open(ev_path) as src:
                ev = src.read()
            if ev.shape[1:] == mask.shape:
                optical = (ev[0] > 0) | ((ev[1] > 0) if ev.shape[0] > 1 else False)
                meta["added_from_sentinel2_px"] = int((optical & ~mask).sum())
                mask |= optical & ~invalid
                conf = np.where(optical & (conf < 0.6), 0.6, conf).astype("float32")
                method += "+sentinel2"

    mask = mask & ~invalid
    conf = np.nan_to_num(conf, nan=0.0)
    conf[invalid] = 0.0
    conf = np.clip(conf, 0.0, 1.0).astype("float32")

    prov = Provenance(producer="models.stage", source=f"Sentinel-1 before/after; method {method}",
                      crs=f"EPSG:{epsg}", event_date=event_date.isoformat(), model_version=method)
    write_geotiff(run.path(Artifact.FLOOD_MASK), mask.astype("uint8"), transform=transform, epsg=epsg,
                  provenance=prov)
    write_geotiff(run.path(Artifact.FLOOD_CONFIDENCE), conf, transform=transform, epsg=epsg, provenance=prov)

    valid_n = int((~invalid).sum())
    meta.update({"method": method, "warnings": warnings, "no_data_fraction": round(float(invalid.mean()), 4),
                 "flood_pixels": int(mask.sum()), "valid_pixels": valid_n,
                 "flood_fraction_of_valid": round(int(mask.sum()) / valid_n, 4) if valid_n else None})
    extras_dir(run).mkdir(exist_ok=True)
    write_json(extras_dir(run) / "inference_meta.json", json.loads(json.dumps(meta, default=str)))
