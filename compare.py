"""
compare.py  --  Part 5: comparing OUR flood map with a reference map (pure numpy)
=================================================================================

Repo location: models/compare.py

Used for the Trishuli case study: our system's flood mask versus the official
Copernicus EMS activation EMSR927.  IMPORTANT DATA RULE: EMSR927 is for CHECKING
ONLY.  Nothing in this file may feed back into the model, the threshold or any
setting (case_study.py enforces that order; see ems_reference.py).

Everything here is plain numpy and fully unit-tested.  It answers four questions:

    1. How much do the maps agree?           compare_masks()      IoU, precision, recall, areas
    2. How sure are we of that number?       block_bootstrap_ci() one event only -> resample SPATIAL BLOCKS
    3. Is the disagreement just misregistration?  shift_sensitivity()   IoU vs shifting the reference
    4. Where does it fail (mountains?)       error_by_slope(), categorize_errors (Part 3)

"TRICK Tn" marks non-obvious ideas (same list in 05_case_study_limitations_and_report.md).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "part3_code", _HERE.parent / "part4_code"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_metrics import (Counts, _metric_arrays, categorize_errors,  # noqa: E402
                          relaxed_counts, relaxed_precision_recall)

# agreement-map codes
AGREE_DRY, AGREE_FLOOD, OURS_ONLY, REF_ONLY, OUTSIDE = 0, 1, 2, 3, 255


# =============================================================================
# 1. AGREEMENT
# =============================================================================
def agreement_map(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """uint8 map: 0 both dry, 1 both flood, 2 ours only (possible false alarm),
    3 reference only (possible miss), 255 outside the comparison area."""
    out = np.full(pred.shape, OUTSIDE, np.uint8)
    out[valid & ~pred & ~ref] = AGREE_DRY
    out[valid & pred & ref] = AGREE_FLOOD
    out[valid & pred & ~ref] = OURS_ONLY
    out[valid & ~pred & ref] = REF_ONLY
    return out


def comparison_valid_mask(pred_mask: np.ndarray, footprint: Optional[np.ndarray] = None,
                          ref_valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Pixels where BOTH maps have an opinion.

    TRICK T1 (compare only inside the reference's footprint):
        An EMS map covers a specific area of interest and acquisition.  Flood
        OUTSIDE that area is not "dry", it is simply unmapped.  Counting our
        flood pixels there as false alarms would unfairly crush precision.
        So the comparison is restricted to
              our valid pixels (mask != 255)  AND  reference footprint.
        If no footprint is supplied the whole grid is used and the report must
        say that precision may be understated.
    """
    v = pred_mask != 255
    if footprint is not None:
        v &= footprint.astype(bool)
    if ref_valid is not None:
        v &= ref_valid.astype(bool)
    return v


# =============================================================================
# 2. UNCERTAINTY FROM A SINGLE EVENT
# =============================================================================
def block_bootstrap_ci(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray, metric: str = "iou",
                       block: int = 32, n_boot: int = 2000, seed: int = 0,
                       alpha: float = 0.05) -> Dict[str, float]:
    """95% interval for a pooled metric by resampling square BLOCKS of the map.

    TRICK T2 (spatial block bootstrap):
        Part 3 resampled EVENTS, but the case study is ONE event, so that is
        impossible.  Resampling single pixels would be absurdly optimistic
        (neighbouring pixels are nearly identical).  Compromise: cut the map
        into block x block tiles (default 32 px = 320 m), keep one small
        (TP, FP, FN) triple per tile, then draw tiles with replacement and
        re-sum.  This respects spatial correlation up to the tile size.
        Vectorized: one [n_boot, n_tiles] index matrix, one fancy index, one sum.
        It captures sampling variability only; it cannot capture reference
        errors, date mismatch or misregistration (those need shift_sensitivity
        and the limitations section).
    """
    h, w = pred.shape
    ph, pw = (-h) % block, (-w) % block

    def blocks(a: np.ndarray) -> np.ndarray:
        a = np.pad(a, ((0, ph), (0, pw)), constant_values=False)
        H, W = a.shape
        return a.reshape(H // block, block, W // block, block).sum(axis=(1, 3)).ravel()

    tp, fp, fn = blocks(pred & ref & valid), blocks(pred & ~ref & valid), blocks(~pred & ref & valid)
    keep = (tp + fp + fn) > 0                      # tiles with any flood activity matter
    arr = np.stack([tp, fp, fn], axis=1)[keep].astype(np.float64)
    point = _metric_arrays(tp.sum(), fp.sum(), fn.sum())[metric]
    if len(arr) < 2:
        return {"point": float(point), "low": float("nan"), "high": float("nan"), "n_blocks": int(len(arr))}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    s = arr[idx].sum(axis=1)
    vals = _metric_arrays(s[:, 0], s[:, 1], s[:, 2])[metric]
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "low": float(lo), "high": float(hi), "n_blocks": int(len(arr))}


# =============================================================================
# 3. IS IT MISREGISTRATION?
# =============================================================================
def shift_sensitivity(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray,
                      max_shift: int = 3) -> Dict[str, object]:
    """IoU between our mask and the reference SHIFTED by (dy, dx) pixels.

    TRICK T3 (separate "wrong" from "misaligned"):
        Two maps of the same flood from different satellites/products can be
        offset by a pixel or two (resampling, orthorectification, DEM).  If the
        IoU peaks at, say, (dy=0, dx=2) instead of (0,0), a good part of the
        "error" is a georeferencing offset, not a model mistake.  We tabulate
        IoU for every shift in [-max_shift, max_shift]^2 using slicing (no
        np.roll, which would wrap flood around the image edge).
        Report BOTH the (0,0) IoU (honest, unshifted) and the best shifted IoU
        (diagnostic): never quote the shifted number as the result.
        CAVEAT (the 'aperture problem'): a shift ALONG a slanted feature such as a
        diagonal river looks almost the same as a different shift across it, so the
        best (dy, dx) can be off by a pixel in one axis. Treat it as a clue that a
        misregistration exists, not as a measurement of its exact size.
    """
    h, w = pred.shape
    m = max_shift
    grid = np.full((2 * m + 1, 2 * m + 1), np.nan)
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            ys, ye = max(0, -dy), h - max(0, dy)
            xs, xe = max(0, -dx), w - max(0, dx)
            p = pred[ys:ye, xs:xe]
            r = ref[ys + dy:ye + dy, xs + dx:xe + dx]          # reference sampled at (y+dy, x+dx)
            v = valid[ys:ye, xs:xe] & valid[ys + dy:ye + dy, xs + dx:xe + dx]
            tp, fp, fn = (p & r & v).sum(), (p & ~r & v).sum(), (~p & r & v).sum()
            grid[dy + m, dx + m] = _metric_arrays(tp, fp, fn)["iou"]
    if np.all(np.isnan(grid)):
        best = (m, m)
    else:
        # TRICK T3b: several shifts can tie (a straight river matches equally at every shift ALONG
        # it). Among all cells within 1e-9 of the maximum choose the SMALLEST shift: the simplest
        # explanation that fits the data, and it makes the result independent of scan order.
        top = np.nanmax(grid)
        cand = np.argwhere(np.isclose(grid, top, atol=1e-9, equal_nan=False))
        best = tuple(min(cand, key=lambda ij: (ij[0] - m) ** 2 + (ij[1] - m) ** 2))
    return {"iou_grid": grid, "iou_at_zero": float(grid[m, m]), "best_shift_dy_dx": (int(best[0] - m), int(best[1] - m)),
            "iou_best": float(grid[best]), "max_shift": m}


# =============================================================================
# 4. WHERE DOES IT FAIL?
# =============================================================================
def error_by_slope(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray, slope_deg: np.ndarray,
                   bins: Sequence[float] = (0, 5, 15, 30, 90)) -> List[Dict[str, float]]:
    """Precision/recall/IoU per terrain-steepness bin (slope from the DEM, Teammate 1).

    TRICK T4 (turn "mountains are hard" into a number):
        Radar shadow and layover happen on steep slopes.  Instead of asserting
        that our map is worse there, we measure it: one row per slope bin with
        its pixel count, so the limitations section can say "recall drops from
        X on flat ground to Y above 30 degrees".  Bins with no flood return NaN.
    """
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = valid & (slope_deg >= lo) & (slope_deg < hi if hi < bins[-1] else slope_deg <= hi)
        c = Counts(int((pred & ref & sel).sum()), int((pred & ~ref & sel).sum()),
                   int((~pred & ref & sel).sum()), int((~pred & ~ref & sel).sum()))
        rows.append({"slope_from": float(lo), "slope_to": float(hi), "pixels": int(sel.sum()),
                     "flood_pixels_ref": c.tp + c.fn, "precision": c.precision, "recall": c.recall, "iou": c.iou})
    return rows


# =============================================================================
# 5. EVERYTHING TOGETHER
# =============================================================================
@dataclass
class Comparison:
    counts: Counts
    iou_ci: Dict[str, float]
    f1_ci: Dict[str, float]
    relaxed_precision: float
    relaxed_recall: float
    area_ours_km2: Optional[float]
    area_ref_km2: Optional[float]
    area_ratio: Optional[float]
    n_valid_pixels: int
    footprint_used: bool
    shift: Dict[str, object]
    error_causes: Dict[str, Dict[str, int]]
    slope_table: Optional[List[Dict[str, float]]] = None
    notes: List[str] = field(default_factory=list)


def compare_masks(pred_mask: np.ndarray, ref: np.ndarray, *, footprint: Optional[np.ndarray] = None,
                  before_vv: Optional[np.ndarray] = None, after_vv: Optional[np.ndarray] = None,
                  slope_deg: Optional[np.ndarray] = None, pixel_size_m: Optional[float] = None,
                  relaxed_px: int = 2, block: int = 32, max_shift: int = 3) -> Tuple[Comparison, np.ndarray]:
    """Full comparison of our mask (0/1/255) with a reference flood mask (bool).

    Returns (Comparison, agreement_map).  Nothing here changes the prediction.
    """
    pred = pred_mask == 1
    ref = ref.astype(bool)
    valid = comparison_valid_mask(pred_mask, footprint)
    c = Counts(int((pred & ref & valid).sum()), int((pred & ~ref & valid).sum()),
               int((~pred & ref & valid).sum()), int((~pred & ~ref & valid).sum()))

    truth = np.where(ref, 2, 0).astype(np.uint8)
    truth[~valid] = 255
    rp, rr = relaxed_precision_recall(relaxed_counts(pred & valid, truth, relaxed_px))

    if before_vv is not None and after_vv is not None:
        causes = categorize_errors(pred & valid, truth, np.nan_to_num(before_vv, nan=-30.0),
                                   np.nan_to_num(after_vv - before_vv, nan=0.0), boundary_px=relaxed_px)
    else:
        causes = {"fp": {}, "fn": {}}

    px_km2 = (pixel_size_m ** 2) / 1e6 if pixel_size_m else None
    a_o = (pred & valid).sum() * px_km2 if px_km2 else None
    a_r = (ref & valid).sum() * px_km2 if px_km2 else None
    notes: List[str] = []
    if footprint is None:
        notes.append("No reference footprint supplied: flood outside the reference's mapped area counts as a "
                     "false alarm, so precision may be understated.")
    if c.tp + c.fn == 0:
        notes.append("The reference contains no flood inside the comparison area.")
    comp = Comparison(
        counts=c, iou_ci=block_bootstrap_ci(pred, ref, valid, "iou", block),
        f1_ci=block_bootstrap_ci(pred, ref, valid, "f1", block),
        relaxed_precision=rp, relaxed_recall=rr, area_ours_km2=a_o, area_ref_km2=a_r,
        area_ratio=(a_o / a_r) if (a_o is not None and a_r) else None,
        n_valid_pixels=int(valid.sum()), footprint_used=footprint is not None,
        shift=shift_sensitivity(pred, ref, valid, max_shift), error_causes=causes,
        slope_table=error_by_slope(pred, ref, valid, slope_deg) if slope_deg is not None else None,
        notes=notes)
    return comp, agreement_map(pred, ref, valid)
