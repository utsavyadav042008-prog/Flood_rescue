"""
eval_metrics.py  --  Part 3 of Teammate 2's workflow: THE MATHS OF EVALUATION
=============================================================================

Repo location: models/eval_metrics.py   (numpy + scipy only, no PyTorch)

NOTE ON PROVENANCE
------------------
This file was missing from the Person_2 branch (evaluate.py, inference.py,
compare.py and tests/test_eval_metrics.py all import it).  It was rebuilt
during integration from those call sites and from the behaviour pinned down
by ``tests/test_eval_metrics.py``.  If the original turns up, diff the two.

WHAT LIVES HERE
---------------
    Counts / counts_from_arrays      TP FP FN TN for the FLOOD class (2) and the
                                     metrics derived from them (NaN = undefined)
    ThresholdSweep                   histogram of probabilities -> counts at EVERY
                                     threshold in one pass (PR curve, best F1, AP)
    bootstrap_ci / macro_average     uncertainty over EVENTS (not pixels)
    otsu_threshold, logratio_baseline, fit_drop_threshold
                                     the non-AI change-detection baseline
    dilate / erode / remove_small_blobs
    categorize_errors                WHY a pixel is wrong (river, edge, shadow ...)
    relaxed_counts                   precision/recall forgiving a 1-2 px shift
    pad_to_multiple / crop_padding   U-Net input sizes
    freeze_protocol / load_protocol  the validation threshold, written once
    GroupResult + markdown/json      tables for the report

Label convention (same as dataset.py): 0 no water, 1 permanent water, 2 flood,
255 ignore.  Permanent water counts as a NEGATIVE for the flood class.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

FLOOD = 2
PERM_WATER = 1
IGNORE = 255

#: dB cut-offs used to explain errors (heuristics; see README "Testing status").
DARK_BEFORE_DB = -18.0      # "already dark before" = radar shadow / smooth surface
WEAK_CHANGE_DB = 3.0        # a real flood normally darkens VV by more than this
#: oracle F1 this much above the frozen-threshold F1 => report "distribution shift"
SHIFT_GAP = 0.10
#: fewer events than this => the confidence interval is not trustworthy
FEW_EVENTS = 5


# =============================================================================
# 1. COUNTS
# =============================================================================
def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


@dataclass
class Counts:
    """Pixel counts for the flood class.  Metrics are NaN when undefined
    (e.g. IoU of a scene with no flood in truth AND prediction)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn)

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def iou(self) -> float:
        return _ratio(self.tp, self.tp + self.fp + self.fn)

    @property
    def f1(self) -> float:
        return _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    def metric(self, name: str) -> float:
        return float(getattr(self, name))

    def to_dict(self) -> Dict[str, object]:
        d: Dict[str, object] = {"tp": int(self.tp), "fp": int(self.fp), "fn": int(self.fn), "tn": int(self.tn)}
        for k in ("precision", "recall", "iou", "f1"):
            v = getattr(self, k)
            d[k] = None if math.isnan(v) else float(v)
        return d

    as_dict = to_dict          # name used by the case study (Part 5)


def _metric_arrays(tp, fp, fn) -> Dict[str, np.ndarray]:
    """Vectorised metrics for arrays (or scalars) of tp/fp/fn; NaN where undefined."""
    tp, fp, fn = (np.asarray(a, dtype=np.float64) for a in (tp, fp, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "precision": np.where(tp + fp > 0, tp / (tp + fp), np.nan),
            "recall": np.where(tp + fn > 0, tp / (tp + fn), np.nan),
            "iou": np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan),
            "f1": np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan),
        }
    return out


def counts_from_arrays(pred, truth) -> Counts:
    """``pred`` bool-like [..], ``truth`` labels 0/1/2/255 [..] -> Counts.

    Pixels with truth 255 are ignored; truth 1 (permanent water) is a negative."""
    pred = np.asarray(pred).astype(bool)
    truth = np.asarray(truth)
    valid = truth != IGNORE
    pos = (truth == FLOOD) & valid
    neg = valid & ~pos
    return Counts(int((pred & pos).sum()), int((pred & neg).sum()),
                  int((~pred & pos).sum()), int((~pred & neg).sum()))


# =============================================================================
# 2. THRESHOLD SWEEP
# =============================================================================
class ThresholdSweep:
    """Counts at every threshold from ONE pass over the data.

    TRICK: keep two histograms of the predicted probability, one for truly
    flooded pixels (``pos``) and one for the rest (``neg``).  "Predicted flood at
    threshold t" means prob >= t, i.e. every bin from t upward, so the counts at
    any threshold are two cumulative sums.  Memory: 2 x bins integers, whatever
    the size of the test set.
    """

    def __init__(self, bins: int = 200) -> None:
        self.bins = int(bins)
        self.pos = np.zeros(self.bins, np.int64)
        self.neg = np.zeros(self.bins, np.int64)

    def _bin(self, prob: np.ndarray) -> np.ndarray:
        return np.clip(np.floor(prob * self.bins).astype(np.int64), 0, self.bins - 1)

    def update(self, prob, truth) -> None:
        prob = np.asarray(prob, dtype=np.float64)
        truth = np.asarray(truth)
        valid = (truth != IGNORE) & np.isfinite(prob)
        idx = self._bin(prob[valid])
        is_pos = truth[valid] == FLOOD
        self.pos += np.bincount(idx[is_pos], minlength=self.bins)
        self.neg += np.bincount(idx[~is_pos], minlength=self.bins)

    def __add__(self, other: "ThresholdSweep") -> "ThresholdSweep":
        if other.bins != self.bins:
            raise ValueError("cannot add sweeps with different bin counts")
        out = ThresholdSweep(self.bins)
        out.pos = self.pos + other.pos
        out.neg = self.neg + other.neg
        return out

    def _k(self, threshold: float) -> int:
        # thresholds on bin edges are exact; 1e-9 absorbs float noise (17/50*50 = 16.999..)
        return int(min(max(math.ceil(threshold * self.bins - 1e-9), 0), self.bins))

    def counts_at(self, threshold: float) -> Counts:
        k = self._k(threshold)
        return Counts(int(self.pos[k:].sum()), int(self.neg[k:].sum()),
                      int(self.pos[:k].sum()), int(self.neg[:k].sum()))

    def _all_counts(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """tp, fp, fn per threshold index k = 0..bins (threshold k/bins)."""
        tp = np.concatenate([np.cumsum(self.pos[::-1])[::-1], [0]])
        fp = np.concatenate([np.cumsum(self.neg[::-1])[::-1], [0]])
        fn = self.pos.sum() - tp
        thr = np.arange(self.bins + 1) / self.bins
        return thr, tp, fp, fn

    def curve(self) -> Dict[str, np.ndarray]:
        thr, tp, fp, fn = self._all_counts()
        m = _metric_arrays(tp, fp, fn)
        return {"threshold": thr, "precision": m["precision"], "recall": m["recall"],
                "f1": m["f1"], "iou": m["iou"]}

    def best(self, metric: str = "f1") -> Tuple[float, float]:
        """(threshold, value) maximising ``metric`` over thresholds 1/bins .. (bins-1)/bins.

        With clean data a whole range of thresholds ties; argmax would pick the
        lowest one, right on top of the negatives.  We take the MIDDLE of the
        plateau, which is the most robust choice for unseen data."""
        c = self.curve()
        vals = np.nan_to_num(c[metric][1:self.bins], nan=-1.0)
        if vals.max() < 0:
            return 0.5, float("nan")
        plateau = np.flatnonzero(vals >= vals.max() - 1e-12)
        k = int(plateau[len(plateau) // 2]) + 1
        return float(k / self.bins), float(c[metric][k])

    def average_precision(self) -> float:
        """Area under the precision-recall curve (step-wise, thresholds high -> low)."""
        if self.pos.sum() == 0:
            return float("nan")
        c = self.curve()
        p, r = c["precision"][::-1], c["recall"][::-1]          # threshold decreasing
        ap, prev_r = 0.0, 0.0
        for pi, ri in zip(p, r):
            if np.isnan(ri):
                continue
            if ri > prev_r:
                ap += (ri - prev_r) * (0.0 if np.isnan(pi) else pi)
                prev_r = ri
        return float(ap)


# =============================================================================
# 3. UNCERTAINTY OVER EVENTS
# =============================================================================
def bootstrap_ci(units: Sequence[Counts], metric: str = "iou", n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """95 % percentile interval of the POOLED metric, resampling whole units (events).

    Vectorised: one [n_boot, n_units] index matrix, one fancy index, one sum.
    With fewer than 2 units an interval is meaningless -> low/high are NaN."""
    units = list(units)
    total = sum(units, Counts())
    point = total.metric(metric)
    out = {"point": point, "low": float("nan"), "high": float("nan"), "n_units": len(units)}
    if len(units) < 2:
        return out
    arr = np.array([[u.tp, u.fp, u.fn] for u in units], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(units), size=(n_boot, len(units)))
    s = arr[idx].sum(axis=1)
    vals = _metric_arrays(s[:, 0], s[:, 1], s[:, 2])[metric]
    if np.all(np.isnan(vals)):
        return out
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    out["low"], out["high"] = float(min(lo, point)), float(max(hi, point))
    return out


def macro_average(units: Sequence[Counts], metric: str = "iou") -> float:
    """Mean of the per-unit metric, skipping units where it is undefined."""
    vals = [u.metric(metric) for u in units]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


# =============================================================================
# 4. MORPHOLOGY
# =============================================================================
def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow a boolean mask by ``px`` pixels (3x3 square structuring element)."""
    from scipy import ndimage
    mask = np.asarray(mask, bool)
    if px <= 0 or not mask.any():
        return mask.copy()
    return ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=px)


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    """Shrink a boolean mask by ``px`` pixels.  The image border does NOT erode
    (border_value=1): the edge of a crop is not the edge of the flood."""
    from scipy import ndimage
    mask = np.asarray(mask, bool)
    if px <= 0:
        return mask.copy()
    return ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool), iterations=px, border_value=1)


def remove_small_blobs(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Delete connected groups (4-connectivity) of True pixels smaller than ``min_pixels``."""
    from scipy import ndimage
    mask = np.asarray(mask, bool)
    if min_pixels <= 1 or not mask.any():
        return mask.copy()
    labels, n = ndimage.label(mask)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[labels]


# =============================================================================
# 5. THE NON-AI BASELINE (log-ratio change detection)
# =============================================================================
def otsu_threshold(values, bins: int = 256) -> float:
    """Otsu's threshold of a 1-D sample (maximises between-class variance)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean0 = m0 / w0
        mean1 = (m0[-1] - m0) / w1
        between = np.nan_to_num(w0 * w1 * (mean0 - mean1) ** 2)
    plateau = np.flatnonzero(between >= between.max() * (1.0 - 1e-9))
    return float(centres[int(plateau[len(plateau) // 2])])


def logratio_baseline(before: np.ndarray, after: np.ndarray, invalid: np.ndarray,
                      mode: str = "otsu", drop_db: float = 3.0, min_pixels: int = 8) -> np.ndarray:
    """Flood = VV darkened by at least ``drop_db`` (after - before, in dB).

    ``mode="fixed"``  threshold = -drop_db
    ``mode="otsu"``   threshold = Otsu on the change histogram, but never weaker
                      than -drop_db.  That guard matters: on a flood-free scene
                      Otsu still "finds" a split inside the noise; demanding a
                      real drop of at least ``drop_db`` makes it return nothing.
    Permanent water is not flagged by construction (dark before AND after =
    no change).  Specks smaller than ``min_pixels`` are removed.
    """
    change = np.asarray(after[0], np.float64) - np.asarray(before[0], np.float64)
    valid = ~np.asarray(invalid, bool) & np.isfinite(change)
    if mode == "fixed":
        thr = -abs(drop_db)
    elif mode == "otsu":
        t = otsu_threshold(change[valid]) if valid.any() else float("nan")
        thr = -abs(drop_db) if not np.isfinite(t) else min(t, -abs(drop_db))
    else:
        raise ValueError("mode must be 'fixed' or 'otsu'")
    flood = valid & (change <= thr)
    return remove_small_blobs(flood, min_pixels)


def fit_drop_threshold(tuples: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                       candidates: Optional[Sequence[float]] = None,
                       min_pixels: int = 8) -> Tuple[float, float]:
    """Choose the baseline's dB drop on VALIDATION data: (drop_db, pooled IoU).

    ``tuples`` = [(before, after, invalid, truth), ...].  Among equally good
    drops the middle one is returned (same plateau rule as everywhere else)."""
    cands = np.asarray(candidates if candidates is not None else np.arange(0.5, 15.01, 0.5), float)
    totals = [Counts() for _ in cands]
    for before, after, invalid, truth in tuples:
        for i, d in enumerate(cands):
            pred = logratio_baseline(before, after, invalid, "fixed", float(d), min_pixels)
            totals[i] = totals[i] + counts_from_arrays(pred, truth)
    ious = np.array([np.nan_to_num(c.iou, nan=-1.0) for c in totals])
    plateau = np.flatnonzero(ious >= ious.max() - 1e-9)
    k = int(plateau[len(plateau) // 2])
    return float(cands[k]), float(totals[k].iou)


# =============================================================================
# 6. WHY IS IT WRONG?
# =============================================================================
FP_CAUSES = ("permanent_water", "boundary", "dark_before", "other")
FN_CAUSES = ("boundary", "weak_change", "other")


def empty_error_counts() -> Dict[str, Dict[str, int]]:
    return {"fp": {k: 0 for k in FP_CAUSES}, "fn": {k: 0 for k in FN_CAUSES}}


def add_error_counts(a: Dict[str, Dict[str, int]], b: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    out = empty_error_counts()
    for side in ("fp", "fn"):
        for src in (a.get(side, {}), b.get(side, {})):
            for k, v in src.items():
                out[side][k] = out[side].get(k, 0) + int(v)
    return out


def categorize_errors(pred: np.ndarray, truth: np.ndarray, before_vv: np.ndarray,
                      change_vv: np.ndarray, boundary_px: int = 2) -> Dict[str, Dict[str, int]]:
    """Give every wrong pixel exactly ONE cause (first matching rule wins).

    False alarms (pred flood, truth not flood):
        permanent_water  truth says river/lake (model confused old and new water)
        boundary         within ``boundary_px`` of real flood (edge uncertainty)
        dark_before      VV already < -18 dB before (radar shadow / smooth surface)
        other
    Missed flood (truth flood, pred not):
        boundary         within ``boundary_px`` of the flood edge
        weak_change      VV darkened by less than 3 dB (little radar signal)
        other
    """
    pred = np.asarray(pred, bool)
    truth = np.asarray(truth)
    valid = truth != IGNORE
    flood = (truth == FLOOD) & valid
    before_vv = np.asarray(before_vv, np.float64)
    change_vv = np.asarray(change_vv, np.float64)
    near_flood = dilate(flood, boundary_px) & ~flood
    flood_edge = flood & ~erode(flood, boundary_px)

    out = empty_error_counts()
    fp = pred & valid & ~flood
    left = fp.copy()
    for name, rule in (("permanent_water", truth == PERM_WATER), ("boundary", near_flood),
                       ("dark_before", before_vv < DARK_BEFORE_DB)):
        hit = left & rule
        out["fp"][name] = int(hit.sum())
        left &= ~hit
    out["fp"]["other"] = int(left.sum())

    fn = ~pred & flood
    left = fn.copy()
    for name, rule in (("boundary", flood_edge), ("weak_change", change_vv > -WEAK_CHANGE_DB)):
        hit = left & rule
        out["fn"][name] = int(hit.sum())
        left &= ~hit
    out["fn"]["other"] = int(left.sum())
    return out


# =============================================================================
# 7. RELAXED (SHIFT-TOLERANT) METRICS
# =============================================================================
def relaxed_counts(pred: np.ndarray, truth: np.ndarray, px: int = 2) -> Dict[str, int]:
    """Counts for precision/recall that forgive a misregistration of ``px`` pixels.

    precision side: predicted pixels lying within px of a true flood pixel
    recall side:    true flood pixels lying within px of a predicted pixel
    (two different sets; that is the point)."""
    truth = np.asarray(truth)
    valid = truth != IGNORE
    pred = np.asarray(pred, bool) & valid
    flood = (truth == FLOOD) & valid
    return {"pred_total": int(pred.sum()), "pred_near_truth": int((pred & dilate(flood, px)).sum()),
            "truth_total": int(flood.sum()), "truth_near_pred": int((flood & dilate(pred, px)).sum())}


def relaxed_precision_recall(c: Dict[str, int]) -> Tuple[float, float]:
    return (_ratio(c["pred_near_truth"], c["pred_total"]), _ratio(c["truth_near_pred"], c["truth_total"]))


# =============================================================================
# 8. PADDING (the U-Net needs sizes divisible by 32)
# =============================================================================
def pad_to_multiple(x: np.ndarray, m: int = 32) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad the last two axes at the bottom/right with zeros (= channel mean after
    normalisation).  Returns (padded, (pad_rows, pad_cols))."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph == 0 and pw == 0:
        return x, (0, 0)
    widths = [(0, 0)] * (x.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(x, widths, mode="constant"), (ph, pw)


def crop_padding(x: np.ndarray, pad: Tuple[int, int]) -> np.ndarray:
    ph, pw = pad
    h, w = x.shape[-2:]
    return x[..., :h - ph, :w - pw]


# =============================================================================
# 9. THE FROZEN PROTOCOL
# =============================================================================
def freeze_protocol(path, threshold: float, metric: str, value: float,
                    val_events: Sequence[str], checkpoint: str) -> Dict[str, object]:
    """Write the validation-chosen threshold ONCE; everything later reads it."""
    proto = {"threshold": float(threshold), "selection_metric": metric,
             "selection_value": None if value is None or (isinstance(value, float) and math.isnan(value))
             else float(value),
             "val_events": sorted(val_events), "checkpoint": str(checkpoint),
             "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(proto, indent=2))
    return proto


def load_protocol(path) -> Dict[str, object]:
    proto = json.loads(Path(path).read_text())
    if "threshold" not in proto:
        raise ValueError(f"{path} has no 'threshold'")
    return proto


# =============================================================================
# 10. RESULTS TABLES
# =============================================================================
@dataclass
class GroupResult:
    name: str
    kind: str
    threshold: float
    counts: Counts
    macro: Dict[str, float] = field(default_factory=dict)
    ci_iou: Dict[str, float] = field(default_factory=dict)
    ci_f1: Dict[str, float] = field(default_factory=dict)
    n_events: int = 0
    n_scenes: int = 0
    n_flood_scenes: int = 0
    errors: Dict[str, Dict[str, int]] = field(default_factory=empty_error_counts)
    relaxed_precision: float = float("nan")
    relaxed_recall: float = float("nan")
    average_precision: Optional[float] = None
    oracle: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["counts"] = self.counts.to_dict()
        return _clean_nan(d)


def _clean_nan(o):
    if isinstance(o, dict):
        return {k: _clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_nan(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _f(v, nd: int = 3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):.{nd}f}"


def results_markdown(results: Sequence[GroupResult]) -> str:
    """One row per (group, method) plus honesty notes under the table."""
    lines = ["| Group | Method | Thr | Precision | Recall | IoU [95% CI] | F1 | Rel. P/R | AP | Events |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    notes: List[str] = []
    warned_groups = set()
    for r in results:
        c = r.counts
        lines.append(
            f"| {r.name} | {r.kind} | {_f(r.threshold, 2)} | {_f(c.precision)} | {_f(c.recall)} | "
            f"{_f(c.iou)} [{_f(r.ci_iou.get('low'))}, {_f(r.ci_iou.get('high'))}] | {_f(c.f1)} | "
            f"{_f(r.relaxed_precision, 2)}/{_f(r.relaxed_recall, 2)} | {_f(r.average_precision)} | {r.n_events} |")
        if r.n_events < FEW_EVENTS and r.name not in warned_groups:
            warned_groups.add(r.name)
            notes.append(f"- `{r.name}`: only {r.n_events} event(s); the confidence interval is wide or "
                         f"undefined, so treat the numbers as indicative.")
        if r.oracle and not math.isnan(c.f1) and r.oracle.get("f1") is not None \
                and r.oracle["f1"] - c.f1 > SHIFT_GAP:
            notes.append(f"- `{r.name}`: best possible F1 on this group is {_f(r.oracle['f1'])} at threshold "
                         f"{_f(r.oracle['threshold'], 2)} versus {_f(c.f1)} with the frozen threshold: a sign of "
                         f"distribution shift (the model's confidence means something different here).")
    return "\n".join(lines + ([""] + notes if notes else []))


def errors_markdown(results: Sequence[GroupResult]) -> str:
    """Error causes per group as shares of all false alarms / all misses."""
    lines: List[str] = []
    for side, causes, title in (("fp", FP_CAUSES, "False alarms"), ("fn", FN_CAUSES, "Missed flood")):
        lines += [f"**{title}**", "", "| Group | Pixels | " + " | ".join(causes) + " |",
                  "|---|---|" + "---|" * len(causes)]
        for r in results:
            d = r.errors.get(side, {})
            tot = sum(d.values())
            cells = [f"{100 * d.get(k, 0) / tot:.0f}%" if tot else "-" for k in causes]
            lines.append(f"| {r.name} | {tot} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def results_to_json(results: Sequence[GroupResult]) -> str:
    return json.dumps([r.to_dict() for r in results], indent=2)
