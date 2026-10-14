"""Non-AI flood + debris change detection (fallback when no trained U-Net is available).

Used by ``models/stage.py`` when the model package / weights or PyTorch are
missing, and reported as such (``model_version`` = ``change-detection-baseline``,
plus a warning in the summary), so a fallback result is never presented as
the AI model's.

Signals on a same-orbit Sentinel-1 pair (VV, dB), computed on a locally
averaged (``smooth_px``) image because single radar pixels are speckle:

* NEW WATER   VV darkens by at least ``drop_db`` (Otsu-guarded, the same rule as
              T2's ``eval_metrics.logratio_baseline``) AND is water-dark afterwards.
* DEBRIS      the local MAGNITUDE of change |after - before| is at least
              ``change_db`` on gentle slopes close to the river.  A debris flow
              mixes rough bright rubble with dark ponded water at the pixel scale,
              so the signed change can average out while its magnitude cannot.
              Restricted to the valley floor (slope, distance to water) because
              strong change elsewhere is usually vegetation, snow or construction.

"Close to the river" uses the permanent-water mask from the acquire stage, or,
if it is missing, a proxy: pixels already very dark (< ``river_db``) BEFORE the
event that form connected channels.

Confidence is a smooth function of how far the change exceeds its threshold
(a logistic); it is a heuristic score, NOT a calibrated probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from models.eval_metrics import dilate, otsu_threshold, remove_small_blobs


@dataclass
class ChangeConfig:
    drop_db: float = 3.0            # minimum smoothed VV darkening for new water
    water_db: float = -15.0         # smoothed after-image VV must be darker than this to be water
    debris: bool = True
    change_db: float = 4.0          # minimum smoothed |VV change| for debris / ground disturbance
    river_db: float = -18.0         # "already water before" proxy when no permanent-water mask exists
    debris_max_slope_deg: float = 15.0
    debris_max_dist_m: float = 1000.0
    smooth_px: int = 2              # half-width of the averaging window (2 -> 5 x 5 pixels)
    min_pixels: int = 10


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def nan_smooth(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over a (2k+1)^2 window ignoring NaN; NaN where the whole window is NaN."""
    from scipy import ndimage
    if k <= 0:
        return a.astype(np.float32)
    ok = np.isfinite(a)
    size = 2 * k + 1
    s = ndimage.uniform_filter(np.where(ok, a, 0.0).astype(np.float64), size=size, mode="nearest")
    n = ndimage.uniform_filter(ok.astype(np.float64), size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 1e-9, s / np.maximum(n, 1e-9), np.nan)
    return out.astype(np.float32)


def detect(before: np.ndarray, after: np.ndarray, *, invalid: Optional[np.ndarray] = None,
           slope: Optional[np.ndarray] = None, permanent_water: Optional[np.ndarray] = None,
           pixel_m: float = 10.0, cfg: Optional[ChangeConfig] = None) -> dict:
    """before/after [2,H,W] dB (VV, VH).  Returns {"mask", "confidence", "water", "debris", "info"}."""
    cfg = cfg or ChangeConfig()
    before = np.asarray(before, np.float32)
    after = np.asarray(after, np.float32)
    inv = ~(np.isfinite(before[0]) & np.isfinite(after[0]))
    if invalid is not None:
        inv |= np.asarray(invalid, bool)
    vb = np.where(inv, np.nan, before[0])
    va = np.where(inv, np.nan, after[0])
    d = va - vb
    d_s = nan_smooth(d, cfg.smooth_px)
    a_s = nan_smooth(va, cfg.smooth_px)
    absd_s = nan_smooth(np.abs(d), cfg.smooth_px)
    valid = ~inv & np.isfinite(d_s)

    otsu = otsu_threshold(d_s[valid]) if valid.any() else float("nan")
    drop = min(-abs(cfg.drop_db), otsu) if np.isfinite(otsu) else -abs(cfg.drop_db)
    water = valid & (d_s <= drop) & (np.nan_to_num(a_s, nan=0.0) < cfg.water_db)
    water = remove_small_blobs(water, cfg.min_pixels)

    river_source = "permanent_water mask"
    if permanent_water is not None:
        river = np.asarray(permanent_water, bool)
    else:
        river = remove_small_blobs(np.nan_to_num(vb, nan=0.0) < cfg.river_db, max(cfg.min_pixels, 20))
        river_source = f"dark-before proxy (< {cfg.river_db:g} dB)"

    debris = np.zeros_like(water)
    if cfg.debris:
        cand = valid & ~water & (np.nan_to_num(absd_s, nan=0.0) >= cfg.change_db)
        if slope is not None:
            cand &= np.nan_to_num(np.asarray(slope, np.float32), nan=90.0) <= cfg.debris_max_slope_deg
        anchor = water | river
        if anchor.any():
            px = max(1, int(round(cfg.debris_max_dist_m / max(pixel_m, 1e-6))))
            cand &= dilate(anchor, min(px, 300))
        else:
            cand[:] = False             # no water anywhere: strong change alone is not called debris
        debris = remove_small_blobs(cand, cfg.min_pixels)

    conf_w = _sigmoid((drop - np.nan_to_num(d_s, nan=0.0)) / 1.5)
    conf_d = _sigmoid((np.nan_to_num(absd_s, nan=0.0) - cfg.change_db) / 1.5)
    mask = (water | debris) & valid
    conf = np.where(water, conf_w, np.where(debris, conf_d, np.minimum(conf_w, conf_d)))
    conf = np.where(mask, np.maximum(conf, 0.5), np.minimum(conf, 0.49)).astype(np.float32)
    conf[inv] = 0.0
    return {"mask": mask, "confidence": np.clip(conf, 0, 1), "water": water, "debris": debris,
            "info": {"method": "smoothed log-ratio change detection (Otsu-guarded) + debris change magnitude",
                     "drop_threshold_db": round(float(drop), 2), "change_threshold_db": cfg.change_db,
                     "smooth_window_px": 2 * cfg.smooth_px + 1, "river_anchor": river_source,
                     "water_pixels": int(water.sum()), "debris_pixels": int(debris.sum()),
                     "invalid_fraction": round(float(inv.mean()), 4)}}
