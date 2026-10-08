"""
train_utils.py  --  Part 2 helpers that need NO PyTorch
=======================================================

Repo location: models/train_utils.py

WHY A SEPARATE FILE?
    train.py needs PyTorch (and a GPU for real runs).  Everything in THIS file is
    plain Python/numpy, so it can be unit-tested anywhere in a second
    (test_train_utils.py).  Bugs in the boring parts (config typos, LR schedule,
    metric maths, bookkeeping) are the ones that silently ruin experiments, so
    they get proper tests.

CONTENTS (in reading order)
    1. TrainConfig + load_config   typo-proof settings, command-line overrides
    2. lr_factor                   warmup + cosine learning-rate shape
    3. ConfusionMatrix             dataset-level IoU / F1 / precision / recall
    4. masked_dice_numpy           reference Dice loss (used to test the torch one)
    5. BestTracker                 best-checkpoint + early stopping
    6. make_checkpoint / experiment log helpers
"""
from __future__ import annotations

import csv
import difflib
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

IGNORE_INDEX = 255          # same value as dataset.py (Part 1); loss must use it too


# =============================================================================
# 1. CONFIG
# =============================================================================
@dataclass
class TrainConfig:
    """Every training knob in ONE place.  Defaults = the baseline from Part 2."""
    # ---- model ----
    encoder: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"   # None = random init
    in_channels: int = 6
    num_classes: int = 3
    # ---- data (files produced by Part 1) ----
    data_root: str = "data/KuroSiwo"
    unit: str = "db"                              # "db" or "linear"
    splits_file: str = "config/splits.json"
    norm_stats: str = "config/norm_stats.json"
    class_weights_file: str = "config/class_weights.json"
    use_class_weights: bool = True
    aug_noise_db: float = 0.5
    # ---- optimisation ----
    batch_size: int = 8
    epochs: int = 30
    lr: float = 1e-3                              # LR of the decoder / head
    encoder_lr_mult: float = 0.1                  # encoder LR = lr * this
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    grad_clip: float = 1.0                        # 0 = off
    amp: bool = True                              # mixed precision (GPU only)
    num_workers: int = 2
    patience: int = 8                             # early stop; 0 = never stop early
    seed: int = 42
    # ---- outputs ----
    flood_class: int = 2
    out_dir: str = "models/weights"
    run_name: str = "baseline"
    experiments_csv: str = "models/experiments.csv"


_OPTIONAL_NONE_FIELDS = {"encoder_weights"}


def _coerce(name: str, text: str, default):
    """Turn a command-line string into the type of the field's default value."""
    if name in _OPTIONAL_NONE_FIELDS and text.lower() == "none":
        return None
    if isinstance(default, bool):                 # check bool BEFORE int (bool is an int)
        if text.lower() in ("1", "true", "yes", "y"):
            return True
        if text.lower() in ("0", "false", "no", "n"):
            return False
        raise ValueError(f"'{name}' expects true/false, got '{text}'")
    if default is None:
        return text
    return type(default)(text)


def load_config(path: Optional[str] = None, overrides: Sequence[str] = ()) -> TrainConfig:
    """YAML file -> TrainConfig, then apply overrides like  ["lr=3e-4", "epochs=5"].

    TRICK T11a (typo-proof config):
        Plain `TrainConfig(**yaml_dict)` is already strict, but a hand-rolled
        `cfg.get("lrr", 1e-3)` style loader silently ignores typos and you train
        with the default while believing you changed it.  Here ANY unknown key
        raises an error with a "did you mean ...?" hint, for both the YAML file
        and the command line.  A wasted 40-minute Colab run is expensive.
    """
    known = {f.name: f for f in fields(TrainConfig)}
    values: Dict[str, object] = {}

    def check(key: str) -> None:
        if key not in known:
            hint = difflib.get_close_matches(key, list(known), n=1)
            extra = f"  Did you mean '{hint[0]}'?" if hint else ""
            raise ValueError(f"unknown config key '{key}'.{extra}")

    if path is not None:
        import yaml
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for k, v in raw.items():
            check(k)
            values[k] = v
    defaults = TrainConfig()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like key=value")
        k, v = item.split("=", 1)
        check(k)
        values[k] = _coerce(k, v, getattr(defaults, k))
    return TrainConfig(**values)


def config_diff(cfg: TrainConfig) -> Dict[str, object]:
    """Fields that differ from the defaults.  Fills the "what changed" column of
    the experiment table automatically (TRICK T11b): you never have to remember
    to write down what you tried, and the note can never disagree with the run."""
    base = asdict(TrainConfig())
    now = asdict(cfg)
    skip = {"run_name", "out_dir", "experiments_csv", "data_root"}
    return {k: v for k, v in now.items() if k not in skip and base[k] != v}


# =============================================================================
# 2. LEARNING-RATE SHAPE
# =============================================================================
def lr_factor(step: int, total_steps: int, warmup_steps: int,
              min_factor: float = 0.01) -> float:
    """Multiplier on the base learning rate at optimisation step `step`.

        factor
        1.0 |      /‾‾‾‾‾‾‾‾‾‾‾‾\\___
            |     /                 ‾‾‾‾‾‾‾‾___
        0.01|____/                              ‾‾
            +--warmup--+------ cosine decay ------> step

    TRICK T2 (per STEP, and never exactly zero):
        * We update the schedule every batch, not every epoch, so the curve is
          smooth even when an "epoch" is only 30 batches.
        * Step 0 returns 1/warmup_steps (not 0).  A learning rate of exactly 0
          on the first batch wastes it, and PyTorch's LambdaLR evaluates the
          function at step 0 when it is created.
        * Why warm up at all?  The encoder starts from pretrained weights and
          BatchNorm statistics that were learned on photos.  A big first step
          can wreck them before the network has seen a single radar patch.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / span))
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


# =============================================================================
# 3. METRICS FROM A CONFUSION MATRIX
# =============================================================================
class ConfusionMatrix:
    """Accumulates a [C, C] matrix over the WHOLE validation set.

    Rows = true class, columns = predicted class.  Part 3 (evaluation) reuses it.

    TRICK T6a (bincount instead of loops):
        For every pixel compute  index = true * C + predicted  and count how
        often each index occurs with ONE np.bincount call; reshaped to [C, C]
        that IS the confusion matrix.  A 3-class matrix over millions of pixels
        takes milliseconds.  `ignore_index` pixels are dropped first.

    TRICK T6b (dataset-level, not batch-level):
        IoU is computed ONCE from the summed matrix, not averaged over batches.
        Averaging per-batch IoU is wrong: a batch with no flood has IoU 0/0, and
        a batch with a tiny flood has a wildly noisy IoU, so the average would
        depend on how the batches happened to be cut.

    TRICK T6c (undefined = NaN, not 0 and not 1):
        If a class is absent from both truth and prediction, its IoU is
        undefined.  We return NaN so averages (np.nanmean) skip it instead of
        being dragged up or down by a made-up number.
    """

    def __init__(self, num_classes: int = 3, ignore_index: int = IGNORE_INDEX):
        self.n = num_classes
        self.ignore = ignore_index
        self.mat = np.zeros((num_classes, num_classes), np.int64)

    def update(self, pred, target) -> None:
        pred = np.asarray(pred).ravel().astype(np.int64)
        target = np.asarray(target).ravel().astype(np.int64)
        keep = ((target != self.ignore) & (target >= 0) & (target < self.n)
                & (pred >= 0) & (pred < self.n))
        idx = target[keep] * self.n + pred[keep]
        self.mat += np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def class_metrics(self, c: int) -> Dict[str, float]:
        m = self.mat
        tp = float(m[c, c])
        fp = float(m[:, c].sum() - m[c, c])
        fn = float(m[c, :].sum() - m[c, c])
        nan = float("nan")
        return {
            "precision": tp / (tp + fp) if tp + fp > 0 else nan,
            "recall": tp / (tp + fn) if tp + fn > 0 else nan,
            "iou": tp / (tp + fp + fn) if tp + fp + fn > 0 else nan,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else nan,
        }

    def summary(self, flood_class: int = 2) -> Dict[str, object]:
        per_class = [self.class_metrics(c) for c in range(self.n)]
        total = self.mat.sum()
        f = per_class[flood_class]
        with np.errstate(all="ignore"):
            miou = float(np.nanmean([p["iou"] for p in per_class])) if total else float("nan")
        return {
            "flood_iou": f["iou"], "flood_f1": f["f1"],
            "flood_precision": f["precision"], "flood_recall": f["recall"],
            "miou": miou,
            "pixel_acc": float(np.trace(self.mat) / total) if total else float("nan"),
            "per_class_iou": [p["iou"] for p in per_class],
            "n_pixels": int(total),
        }


# =============================================================================
# 4. REFERENCE DICE LOSS (numpy)
# =============================================================================
def masked_dice_numpy(probs: np.ndarray, target: np.ndarray,
                      ignore_index: int = IGNORE_INDEX, smooth: float = 1.0) -> float:
    """Dice LOSS (1 - dice) from probabilities, written in numpy.

    It exists ONLY as an independent reference.  `python train.py selftest`
    feeds the same random numbers to the PyTorch MaskedDiceLoss and to this
    function and checks they agree, which catches silly tensor-shape bugs in the
    torch version without needing a trained model.

    probs  : [B, C, H, W] softmax probabilities     target : [B, H, W] class ids
    See MaskedDiceLoss in train.py for what each design choice means.
    """
    _, c, _, _ = probs.shape
    valid = target != ignore_index
    t = np.where(valid, target, 0)
    onehot = np.moveaxis(np.eye(c, dtype=probs.dtype)[t], -1, 1)
    vf = valid[:, None].astype(probs.dtype)
    p, o = probs * vf, onehot * vf
    dims = (0, 2, 3)
    inter = (p * o).sum(dims)
    denom = p.sum(dims) + o.sum(dims)
    dice = (2 * inter + smooth) / (denom + smooth)
    present = o.sum(dims) > 0
    return float(1.0 - dice[present].mean()) if present.any() else 0.0


# =============================================================================
# 5. BEST-CHECKPOINT / EARLY-STOPPING TRACKER
# =============================================================================
class BestTracker:
    """Remembers the best validation score and counts epochs without progress.

    `update(score, epoch)` -> (improved, should_stop)

    TRICK T7 (NaN-safe):
        If the validation set contains no flood pixels at all, flood IoU is NaN.
        A naive `score > best` is False for NaN, so no checkpoint would ever be
        saved.  We map NaN to -inf: it never counts as "best", never crashes the
        comparison, and train.py falls back to saving the last model with a
        warning.  `min_delta` stops us from saving a new "best" for a 0.00001
        gain that is only noise.
    """

    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.best_epoch = -1
        self.bad_epochs = 0

    def update(self, score: float, epoch: int) -> Tuple[bool, bool]:
        s = float("-inf") if (score is None or math.isnan(score)) else float(score)
        improved = s > self.best + self.min_delta if math.isfinite(self.best) else s > self.best
        if improved:
            self.best, self.best_epoch, self.bad_epochs = s, epoch, 0
        else:
            self.bad_epochs += 1
        stop = self.patience > 0 and self.bad_epochs >= self.patience
        return improved, stop


# =============================================================================
# 6. CHECKPOINT METADATA + EXPERIMENT LOG
# =============================================================================
def make_checkpoint(model_state, cfg: TrainConfig, norm: Dict, metrics: Dict,
                    epoch: int, class_names: Dict[int, str]) -> Dict:
    """Assemble what goes into best.pt:  {"model_state": tensors, "meta_json": str}.

    TRICK T8 (everything travels together, and it loads SAFELY):
        * The normalization mean/std and the model settings are stored INSIDE the
          checkpoint, so weights can never be paired with the wrong
          norm_stats.json (a silent accuracy killer in Part 4).
        * All metadata is ONE JSON *string*, so the file contains only tensors and
          a str.  That lets torch.load(..., weights_only=True) open it, which
          refuses to run arbitrary pickled code.  Important because we share
          checkpoints through Drive.
    """
    meta = {
        "config": asdict(cfg),
        "norm": norm,                              # {"mean": [...], "std": [...], "meta": {...}}
        "metrics": {k: (None if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in metrics.items()},
        "epoch": epoch,
        "class_names": {str(k): v for k, v in class_names.items()},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {"model_state": model_state, "meta_json": json.dumps(meta)}


def append_jsonl(path, record: Dict) -> None:
    """Append one JSON line (one epoch of logs).  Append-only = survives crashes."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


EXPERIMENT_FIELDS = ["run_name", "changed", "best_epoch", "val_flood_iou", "val_flood_f1",
                     "val_precision", "val_recall", "epochs_run", "minutes", "started"]


def append_experiment_row(csv_path, row: Dict) -> None:
    """Add one run to experiments.csv (header written on first use)."""
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EXPERIMENT_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def experiments_markdown(csv_path) -> str:
    """Render experiments.csv as a markdown table to paste into the report."""
    rows = list(csv.DictReader(open(csv_path)))
    cols = ["run_name", "changed", "val_flood_iou", "val_flood_f1", "best_epoch"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)
