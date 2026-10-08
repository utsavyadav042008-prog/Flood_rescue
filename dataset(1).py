"""
dataset.py  --  Part 1 of Teammate 2's workflow: DATA & INPUTS
==============================================================

Repo location: models/dataset.py

WHAT THIS FILE DOES
-------------------
It turns a folder of flood "patches" (small radar image crops + answer masks)
into batches that a PyTorch model can train on.

    patch file (.npz)
        |  read_patch_npz()      read + convert to dB + clean bad pixels
        v
    RawPatch (before, after, label, invalid)
        |  augment_patch()       flips / rotations / light noise  (training only)
        v
        |  build_input()         stack 6 channels + normalize
        v
    x: float32 [6, H, W]         y: int64 [H, W]   (0 no water, 1 permanent water,
                                                    2 flood, 255 = ignore)

THE 6 INPUT CHANNELS  (all in dB, then normalized)
    0 before VV | 1 before VH | 2 after VV | 3 after VH | 4 change VV | 5 change VH

THE 3 CLASSES
    0 = no water, 1 = permanent water (normal river/lake), 2 = FLOOD
    255 = ignore (invalid pixel; the loss function skips it)

HOW TO READ THIS FILE
---------------------
Sections are numbered 1..12 and appear in the order data flows through them.
Every non-obvious idea is tagged  "TRICK Tn"  and explained in the
docstring right there.  (The same list is in 01_data_and_inputs.md.)

THE ONLY PART YOU MUST ADAPT TO THE REAL DATASET
------------------------------------------------
Section 3 (`read_patch_npz` and `DEFAULT_LABEL_LUT`).  I wrote them from an
ASSUMED file layout.  After reading the Kuro Siwo README on Day 1, fix only
that section; everything below it keeps working.

QUICK START
-----------
    python dataset.py demo --out demo_out            # fake data, proves it all runs
    python dataset.py scan --root /data/KuroSiwo_npz --out config
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# TRICK T9: torch is imported *optionally*.  All the real logic in this file is
# plain numpy, so it can be unit-tested on a laptop/CI machine that has no
# PyTorch.  Only the very last step (numpy -> tensor) needs torch.
try:  # pragma: no cover
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # lets `class FloodDataset(Dataset)` still be defined
    _HAS_TORCH = False


# =============================================================================
# 1. CONSTANTS  (the "decisions" from Part 1 live here, in ONE place)
# =============================================================================
NO_WATER, PERM_WATER, FLOOD = 0, 1, 2
IGNORE_INDEX = 255          # must equal ignore_index in the loss function (Part 2)
NUM_CLASSES = 3
CLASS_NAMES = {NO_WATER: "no_water", PERM_WATER: "permanent_water", FLOOD: "flood"}

CHANNEL_NAMES = ["before_VV", "before_VH", "after_VV", "after_VH",
                 "change_VV", "change_VH"]
N_CHANNELS = len(CHANNEL_NAMES)

# Radar backscatter in dB is almost always between about -30 (very dark, smooth
# water) and +5 (very bright, e.g. buildings / steep slopes facing the satellite).
# Anything outside is clipped so a few crazy pixels cannot wreck the statistics.
DB_MIN, DB_MAX = -30.0, 5.0
CHANGE_CLIP = 15.0          # the (after - before) channel is clipped to +-15 dB
EPS_LINEAR = 1e-6           # floor before log10 so we never take log(0)

# Maps the dataset's own label numbers -> OUR class numbers.
# ASSUMPTION (verify in the Kuro Siwo README!): 0 no water, 1 permanent water,
# 2 flood, 3 invalid.  Code 3 is deliberately NOT listed -> it becomes 255.
DEFAULT_LABEL_LUT: Dict[int, int] = {0: NO_WATER, 1: PERM_WATER, 2: FLOOD}


# =============================================================================
# 2. SMALL SIGNAL HELPERS
# =============================================================================
def to_db(linear: np.ndarray) -> np.ndarray:
    """Convert linear backscatter to decibels:  dB = 10 * log10(linear).

    Why dB?  Radar values span several orders of magnitude (0.0001 .. 1).
    dB squeezes that into a friendly range (-40 .. 0) where a neural network
    trains much better, and where "water is ~10 dB darker" is a simple statement.
    NaN stays NaN (np.maximum propagates NaN), which we exploit in section 3.
    """
    return 10.0 * np.log10(np.maximum(linear, EPS_LINEAR))


def remap_labels(label: np.ndarray, lut: Optional[Dict[int, int]] = None) -> np.ndarray:
    """Translate the dataset's label codes into our class codes (0/1/2/255).

    TRICK T3a (look-up-table remap with a SAFE default):
        We build a 256-entry table pre-filled with 255 ("ignore"), then only
        overwrite the codes we know about.  Result: any unexpected code (e.g.
        the dataset's "invalid" = 3, or a corrupted 17) silently becomes
        "ignore" instead of being mistaken for a real class.  Indexing a numpy
        table with the label array is also fully vectorized (no Python loop).
    """
    lut = DEFAULT_LABEL_LUT if lut is None else lut
    table = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for src, dst in lut.items():
        table[src] = dst
    lab = np.asarray(label)
    out = table[np.clip(lab, 0, 255).astype(np.uint8)]
    out[(lab < 0) | (lab > 255)] = IGNORE_INDEX   # clip() would hide these
    return out


# =============================================================================
# 3. READING ONE PATCH   <-- the only dataset-specific part. ADAPT HERE.
# =============================================================================
@dataclass
class RawPatch:
    """One clean, in-memory patch.  Everything downstream uses this container.

    before, after : float32 [2, H, W]   dB, band 0 = VV, band 1 = VH, clipped
    label         : uint8   [H, W]      values in {0, 1, 2, 255}
    invalid       : bool    [H, W]      True where the IMAGE had no usable data
    """
    before: np.ndarray
    after: np.ndarray
    label: np.ndarray
    invalid: np.ndarray


def make_raw_patch(before, after, label, unit: str = "db",
                   label_lut: Optional[Dict[int, int]] = None) -> RawPatch:
    """Validate shapes, convert units, find bad pixels, clip.  Returns RawPatch.

    Every reader (real data, synthetic data, tests) ends by calling THIS
    function, so cleaning rules exist in exactly one place.

    TRICK T10 (linear zeros mean "no data"):
        If the data is linear, a pixel that is exactly 0 (or negative) is almost
        always the black border SAR scenes have, not "very dark water".  We turn
        those into NaN BEFORE the dB conversion, otherwise log10(0) would turn a
        no-data border into a fake "-60 dB = deep water" stripe.
    """
    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)
    label = np.asarray(label)
    if before.ndim != 3 or before.shape[0] != 2 or before.shape != after.shape:
        raise ValueError(f"before/after must both be [2,H,W]; got {before.shape} and {after.shape}")
    if label.shape != before.shape[1:]:
        raise ValueError(f"label shape {label.shape} != image size {before.shape[1:]}")

    if unit == "linear":
        before = to_db(np.where(before > 0, before, np.nan))
        after = to_db(np.where(after > 0, after, np.nan))
    elif unit != "db":
        raise ValueError("unit must be 'db' or 'linear'")

    # A pixel is invalid if ANY of its 4 numbers (2 dates x 2 bands) is NaN/inf.
    invalid = ~np.isfinite(before).all(axis=0) | ~np.isfinite(after).all(axis=0)

    def clean(a):
        a = np.nan_to_num(a, nan=DB_MIN, posinf=DB_MAX, neginf=DB_MIN)
        return np.clip(a, DB_MIN, DB_MAX).astype(np.float32)

    return RawPatch(clean(before), clean(after), remap_labels(label, label_lut), invalid)


def read_patch_npz(path, unit: str = "db",
                   label_lut: Optional[Dict[int, int]] = None) -> RawPatch:
    """Read one patch stored as a .npz file.

    ASSUMED LAYOUT (change if the real dataset differs):
        before : [2, H, W]  VV, VH before the flood
        after  : [2, H, W]  VV, VH after the flood
        label  : [H, W]     integer class codes

    If Kuro Siwo ships .tif/.npy or one array with 6 bands, rewrite ONLY the
    three lines inside `with` and keep the final `return make_raw_patch(...)`.
    """
    with np.load(path) as z:
        before, after, label = z["before"], z["after"], z["label"]
    return make_raw_patch(before, after, label, unit=unit, label_lut=label_lut)


# =============================================================================
# 4. BUILDING THE 6-CHANNEL INPUT
# =============================================================================
def stack_channels(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Stack [before(2), after(2), change(2)] -> float32 [6, H, W]  (un-normalized).

    TRICK T1 (why subtract in dB?):
        Subtracting two dB images equals dividing the two LINEAR images:
            after_dB - before_dB  =  10*log10(after / before)      (a "log-ratio")
        Radar brightness = (what the ground is) x (how the slope faces the
        satellite).  The slope factor is the same on both dates IF both images
        come from the same orbit track, so it cancels in the ratio and only the
        real ground change is left.  This is exactly why the challenge insists on
        same-orbit pairs, and why the model gets the change as its own channel.

    This function is also used by `scan_dataset`, so the statistics are computed
    on exactly the same channel definition the model sees (no mismatch possible).
    """
    change = np.clip(after - before, -CHANGE_CLIP, CHANGE_CLIP)
    return np.concatenate([before, after, change], axis=0).astype(np.float32)


@dataclass
class Normalizer:
    """Per-channel (x - mean) / std.  Stats come from TRAINING data only.

    `meta` keeps provenance (how many files/pixels, a fingerprint of the file
    list) so we can later detect "these stats belong to a different split".
    """
    mean: np.ndarray    # [6]
    std: np.ndarray     # [6]
    meta: dict

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean[:, None, None]) / self.std[:, None, None]

    def save(self, path) -> None:
        payload = {"channels": CHANNEL_NAMES, "mean": self.mean.tolist(),
                   "std": self.std.tolist(), "db_range": [DB_MIN, DB_MAX],
                   "change_clip": CHANGE_CLIP, **self.meta}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path) -> "Normalizer":
        d = json.loads(Path(path).read_text())
        if d["channels"] != CHANNEL_NAMES:
            raise ValueError("norm_stats.json was made with a different channel layout")
        meta = {k: v for k, v in d.items()
                if k not in ("channels", "mean", "std", "db_range", "change_clip")}
        return cls(np.array(d["mean"], np.float32), np.array(d["std"], np.float32), meta)


def build_input(before: np.ndarray, after: np.ndarray, invalid: np.ndarray,
                normalizer: Normalizer) -> np.ndarray:
    """RawPatch pieces -> normalized float32 [6, H, W] model input.

    TRICK T3b (neutral value for missing pixels):
        Invalid pixels were filled with DB_MIN (a very dark, extreme value) just
        to avoid NaN.  After normalization we overwrite them with 0, which is the
        channel AVERAGE, i.e. "no information".  Otherwise the network could
        learn "huge-negative stripe = water" from the black border of a scene.
        (The loss ignores these pixels too, see FloodDataset.__getitem__.)

    Inference (Part 4) must call this same function so train/test inputs match.
    """
    x = normalizer(stack_channels(before, after)).astype(np.float32)
    x[:, invalid] = 0.0
    return x


# =============================================================================
# 5. STATISTICS + CLASS COUNTS  (one pass over the training files)
# =============================================================================
class RunningStats:
    """Streaming per-channel mean/std without loading the dataset into RAM.

    TRICK T6a: we only keep, per channel, three running numbers:
        N (pixel count), S = sum(x), SS = sum(x^2)
    then   mean = S/N   and   var = SS/N - mean^2.
    Sums are kept in float64 so the subtraction does not lose precision (the
    textbook weakness of this formula).  dB values are small (|x| <= 35), so
    float64 is plenty accurate here.  Only VALID pixels are counted.
    """
    def __init__(self, n_channels: int = N_CHANNELS):
        self.n = 0
        self.s = np.zeros(n_channels, np.float64)
        self.ss = np.zeros(n_channels, np.float64)

    def update(self, x: np.ndarray, valid: np.ndarray) -> None:
        v = x[:, valid].astype(np.float64)          # [C, number_of_valid_pixels]
        self.n += v.shape[1]
        self.s += v.sum(axis=1)
        self.ss += (v * v).sum(axis=1)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n == 0:
            raise ValueError("no valid pixels seen; check the data reader")
        mean = self.s / self.n
        std = np.sqrt(np.maximum(self.ss / self.n - mean ** 2, 1e-6))
        return mean.astype(np.float32), std.astype(np.float32)


def files_fingerprint(files: Iterable) -> str:
    """Short ID of a file list, identical on every teammate's computer.

    TRICK T8: only "event_folder/file_name" is hashed, not the absolute path
    (/home/you/... differs per machine).  Stored inside norm_stats.json, so if
    the split changes later we can tell the stats are stale.
    """
    keys = sorted(f"{Path(f).parent.name}/{Path(f).name}" for f in files)
    return hashlib.md5("\n".join(keys).encode()).hexdigest()[:12]


def scan_dataset(files: Sequence, reader: Callable[[object], RawPatch],
                 verbose: bool = True) -> Tuple[Normalizer, np.ndarray]:
    """Single pass over TRAINING files -> (Normalizer, class pixel counts[3]).

    TRICK T6b: stats AND class counts come from the same read of every file.
    Reading thousands of radar patches from Drive/Colab is the slow part, so we
    never loop twice.  Counts are used for class weights in Part 2 (the flood
    class is rare, so the loss must care about it more).
    Statistics are measured WITHOUT augmentation (clean data); the small
    augmentation noise (0.5 dB) is negligible compared with the channel std.
    """
    stats = RunningStats()
    counts = np.zeros(NUM_CLASSES, np.int64)
    for i, f in enumerate(files):
        p = reader(f)
        valid = ~p.invalid
        stats.update(stack_channels(p.before, p.after), valid)
        lab = p.label[valid & (p.label != IGNORE_INDEX)]
        counts += np.bincount(lab, minlength=NUM_CLASSES)[:NUM_CLASSES]
        if verbose and (i + 1) % 200 == 0:
            print(f"  scanned {i + 1}/{len(files)} patches")
    mean, std = stats.finalize()
    meta = {"n_files": len(files), "n_valid_pixels": int(stats.n),
            "files_fingerprint": files_fingerprint(files),
            "class_counts": counts.tolist()}
    return Normalizer(mean, std, meta), counts


def class_weights_from_counts(counts: Sequence[int], mode: str = "sqrt_inv") -> List[float]:
    """Loss weights from pixel counts.  Rare class -> bigger weight.

    mode="sqrt_inv": weight = 1/sqrt(frequency), rescaled to average 1.
    Full inverse frequency (mode="inv") is too aggressive when flood is ~1% of
    pixels: the model starts hallucinating floods everywhere.  The square root
    is a gentler middle ground.  A class with zero pixels gets weight 0 (and a
    warning is wise: it means your split has no examples of it).
    """
    c = np.asarray(counts, np.float64)
    freq = c / max(c.sum(), 1.0)
    with np.errstate(divide="ignore"):
        w = 1.0 / (np.sqrt(freq) if mode == "sqrt_inv" else freq)
    w[~np.isfinite(w)] = 0.0
    nz = w[w > 0]
    if nz.size:
        w = w / nz.mean()
    return w.tolist()


# =============================================================================
# 6. SPLITTING BY FLOOD EVENT  (the most important rule of Part 1)
# =============================================================================
def event_of(path) -> str:
    """Which flood event a patch belongs to = the name of its parent folder.
    Change this one function if the real dataset names events differently."""
    return Path(path).parent.name


def _rank_key(event: str, seed: int) -> str:
    return hashlib.md5(f"{seed}:{event}".encode()).hexdigest()


def split_by_event(files: Sequence, n_val: int = 1, n_test: int = 1, seed: int = 42,
                   force_test: Sequence[str] = (), force_val: Sequence[str] = (),
                   force_train: Sequence[str] = (),
                   event_fn: Callable = event_of) -> Dict[str, List[str]]:
    """Split patch files into train/val/test so that NO event is in two splits.

    Why by event?  Neighbouring patches of the same flood look almost identical.
    A random patch split would let the model "see the answer" in training and
    produce fake-high scores.

    `force_test` is how you keep Himalayan/mountain events out of training
    (Part 3's unseen-Himalayan test).  Forced events count toward n_test/n_val.

    TRICK T4 (rank by hash instead of shuffling):
        The remaining events are sorted by md5(seed + event name) and dealt out:
        first the test events, then val, rest train.  Compared with
        random.shuffle:
          * identical on every machine and every run (no RNG state to share),
          * adding a NEW event later does not reshuffle the old ones much,
          * you get EXACT counts (hashing each event into a bucket could leave
            val empty when you only have 6 events),
          * unlike Python's built-in hash(), md5 does not change between runs
            (hash() of strings is randomized by PYTHONHASHSEED).
    """
    by_event: Dict[str, List[str]] = {}
    for f in sorted(str(x) for x in files):
        by_event.setdefault(event_fn(f), []).append(f)

    forced = {"test": list(force_test), "val": list(force_val), "train": list(force_train)}
    assign: Dict[str, str] = {}
    for split, evs in forced.items():
        for e in evs:
            if e not in by_event:
                raise ValueError(f"forced event '{e}' not found. Events: {sorted(by_event)}")
            if e in assign:
                raise ValueError(f"event '{e}' forced into two splits")
            assign[e] = split

    free = sorted((e for e in by_event if e not in assign), key=lambda e: _rank_key(e, seed))
    need_test = max(0, n_test - len(forced["test"]))
    need_val = max(0, n_val - len(forced["val"]))
    if need_test + need_val > len(free):
        raise ValueError(f"not enough events ({len(by_event)}) for n_val={n_val}, n_test={n_test}")
    for e in free[:need_test]:
        assign[e] = "test"
    for e in free[need_test:need_test + need_val]:
        assign[e] = "val"
    for e in free[need_test + need_val:]:
        assign[e] = "train"

    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for e, split in assign.items():
        out[split].extend(by_event[e])
    if not out["train"]:
        raise ValueError("no training events left; lower n_val/n_test")
    for k in out:
        out[k].sort()
    verify_no_leakage(out, event_fn)
    return out


def verify_no_leakage(splits: Dict[str, List[str]], event_fn: Callable = event_of) -> None:
    """Raise if any event appears in more than one split.  Cheap; call it always."""
    ev = {k: {event_fn(f) for f in v} for k, v in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        both = ev[a] & ev[b]
        if both:
            raise AssertionError(f"LEAK: events {sorted(both)} are in both {a} and {b}")


def save_splits(splits: Dict[str, List[str]], path, root,
                event_fn: Callable = event_of) -> None:
    """Write splits.json using paths RELATIVE to `root` (TRICK T8), so the file
    works on every teammate's machine, wherever they put the data."""
    root = Path(root)
    rel = {k: [str(Path(f).relative_to(root)) for f in v] for k, v in splits.items()}
    events = {k: sorted({event_fn(f) for f in v}) for k, v in splits.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"events": events, "files": rel}, indent=2))


def load_splits(path, root) -> Dict[str, List[str]]:
    d = json.loads(Path(path).read_text())
    return {k: [str(Path(root) / f) for f in v] for k, v in d["files"].items()}


# =============================================================================
# 7. AUGMENTATION  (training only, pure numpy)
# =============================================================================
@dataclass
class AugmentConfig:
    flip: bool = True            # random horizontal / vertical flip
    rot90: bool = True           # random 0/90/180/270 degree rotation
    noise_db: float = 0.5        # std of extra speckle-like noise in dB (0 = off)
    crop_size: Optional[int] = None   # random square crop; None = keep full patch


def augment_patch(p: RawPatch, rng: np.random.Generator, cfg: AugmentConfig) -> RawPatch:
    """Randomly transform a patch.  Only physically sensible operations:
    flips, 90-degree rotations, crop, small noise.  NO colour/brightness jitter
    (that is for photos; for radar the absolute dB level is the signal).

    TRICK T2 (augment BEFORE the change channels exist):
        This runs on the raw before/after dB images, not on the finished 6
        channels.  So the geometry change is identical for before, after, label
        and invalid mask (one random draw for all of them), and the noise is
        added to the two dates independently, and THEN `build_input` computes the
        change channels.  The change channels therefore stay consistent with the
        noisy images, and the model learns to ignore speckle-like differences
        instead of treating them as real change.

    TRICK T7 (np.ascontiguousarray at the end):
        np.flip/np.rot90 return "views" with negative strides, and
        torch.from_numpy() refuses those ("negative strides not supported").
        Copying into contiguous memory fixes it.
    """
    before, after, label, invalid = p.before, p.after, p.label, p.invalid

    if cfg.crop_size is not None:
        h, w = label.shape
        cs = cfg.crop_size
        if cs > min(h, w):
            raise ValueError(f"crop_size {cs} larger than patch {h}x{w}")
        top = int(rng.integers(0, h - cs + 1))
        left = int(rng.integers(0, w - cs + 1))
        sl = (..., slice(top, top + cs), slice(left, left + cs))
        before, after, label, invalid = before[sl], after[sl], label[sl], invalid[sl]

    if cfg.rot90:
        k = int(rng.integers(0, 4))
        if k:
            before = np.rot90(before, k, axes=(-2, -1))
            after = np.rot90(after, k, axes=(-2, -1))
            label = np.rot90(label, k, axes=(-2, -1))
            invalid = np.rot90(invalid, k, axes=(-2, -1))
    if cfg.flip:
        for axis in (-1, -2):                       # horizontal, then vertical
            if rng.random() < 0.5:
                before, after = np.flip(before, axis), np.flip(after, axis)
                label, invalid = np.flip(label, axis), np.flip(invalid, axis)

    if cfg.noise_db > 0:
        keep = (~invalid).astype(np.float32)        # leave missing pixels alone
        before = before + rng.normal(0, cfg.noise_db, before.shape).astype(np.float32) * keep
        after = after + rng.normal(0, cfg.noise_db, after.shape).astype(np.float32) * keep
        before, after = np.clip(before, DB_MIN, DB_MAX), np.clip(after, DB_MIN, DB_MAX)

    c = np.ascontiguousarray
    return RawPatch(c(before, dtype=np.float32), c(after, dtype=np.float32), c(label), c(invalid))


# =============================================================================
# 8. THE PYTORCH DATASET
# =============================================================================
class FloodDataset(Dataset):
    """Yields (x, y) = (float32 [6,H,W], int64 [H,W]).

    files      : list of patch paths (one of the lists from `split_by_event`)
    normalizer : fitted on the TRAIN split (`scan_dataset`), reused for val/test
    augment    : AugmentConfig for training, None for val/test
    reader     : function path -> RawPatch (default: read_patch_npz)

    Without PyTorch installed it returns numpy arrays instead of tensors (T9).
    """

    def __init__(self, files: Sequence, normalizer: Normalizer,
                 augment: Optional[AugmentConfig] = None,
                 reader: Callable[[object], RawPatch] = read_patch_npz,
                 seed: int = 42, return_meta: bool = False):
        self.files = [str(f) for f in files]
        self.normalizer = normalizer
        self.augment = augment
        self.reader = reader
        self.seed = seed
        self.epoch = 0
        self.return_meta = return_meta

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch: `train_ds.set_epoch(epoch)`.
        Changes the augmentations each epoch while staying reproducible."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        # TRICK T5 (one private random generator per (seed, epoch, sample)):
        #   The classic PyTorch bug: with num_workers > 0 every worker process
        #   inherits the SAME numpy random state, so several workers produce
        #   identical "random" flips.  Instead of any shared/global RNG, each
        #   sample builds its own generator from (seed, epoch, index).
        #   Result: reproducible, different every epoch, correct with any number
        #   of workers, and it needs no worker_init_fn.
        rng = np.random.default_rng([self.seed, self.epoch, int(i)])

        p = self.reader(self.files[i])
        if self.augment is not None:
            p = augment_patch(p, rng, self.augment)

        x = build_input(p.before, p.after, p.invalid, self.normalizer)
        y = p.label.copy()
        y[p.invalid] = IGNORE_INDEX      # missing image pixels must not be graded

        if _HAS_TORCH:
            x_out = torch.from_numpy(x)
            y_out = torch.from_numpy(y.astype(np.int64))
        else:
            x_out, y_out = x, y.astype(np.int64)
        if self.return_meta:
            return x_out, y_out, {"path": self.files[i], "event": event_of(self.files[i])}
        return x_out, y_out


# =============================================================================
# 9. SANITY CHECK + PLOT   (the "done-when" of Part 1)
# =============================================================================
def sanity_check(ds: FloodDataset, n: int = 20) -> dict:
    """Fail loudly if samples are malformed.  Returns a small report dict."""
    n = min(n, len(ds))
    invalid_frac, flood_frac = [], []
    for i in range(n):
        x, y = ds[i][:2]
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape[0] == N_CHANNELS and x.ndim == 3, f"bad x shape {x.shape}"
        assert y.shape == x.shape[1:], f"mask {y.shape} vs image {x.shape[1:]}"
        assert x.dtype == np.float32, f"x dtype {x.dtype}"
        assert np.isfinite(x).all(), "NaN/inf in input"
        bad = set(np.unique(y).tolist()) - {0, 1, 2, IGNORE_INDEX}
        assert not bad, f"unexpected label values {bad}"
        invalid_frac.append(float((y == IGNORE_INDEX).mean()))
        flood_frac.append(float((y == FLOOD).mean()))
    return {"checked": n, "mean_ignored_fraction": float(np.mean(invalid_frac)),
            "mean_flood_fraction": float(np.mean(flood_frac)), "x_shape": tuple(x.shape)}


def plot_samples(ds: FloodDataset, out_png, n: int = 6) -> None:
    """Save before / after / change / mask side by side to eyeball alignment.
    Use a NON-augmented dataset for this.  Flood (red) must sit on top of the
    dark/changed area; if it is shifted or flipped, the loader has a bug."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    n = min(n, len(ds))
    cmap = ListedColormap(["#dddddd", "#2b6cb0", "#e53e3e", "#000000"])  # no/perm/flood/ignore
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.5 * n), squeeze=False)
    for r in range(n):
        x, y = (np.asarray(t) for t in ds[r][:2])
        y_plot = np.where(y == IGNORE_INDEX, 3, y)
        for c, (img, title, cm) in enumerate([
                (x[0], "before VV", "gray"), (x[2], "after VV", "gray"),
                (x[4], "change VV", "RdBu"), (y_plot, "mask", cmap)]):
            kw = dict(vmin=0, vmax=3, interpolation="nearest") if title == "mask" else {}
            axes[r, c].imshow(img, cmap=cm, **kw)
            axes[r, c].set_title(title if r == 0 else "", fontsize=9)
            axes[r, c].axis("off")
    fig.suptitle("grey=no water, blue=permanent water, red=FLOOD, black=ignored")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# =============================================================================
# 10. SYNTHETIC DATA  (so the whole pipeline can be tested without downloads)
# =============================================================================
def make_synthetic_events(root, n_events: int = 5, patches_per_event: int = 6,
                          size: int = 64, seed: int = 0) -> List[Path]:
    """Create fake flood patches: a river (permanent water) plus, after the
    event, a wide dark flood band around it, plus a NaN no-data stripe.
    Layout: root/event_k/patch_j.npz  with keys before, after, label."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    yy, xx = np.mgrid[0:size, 0:size]
    paths = []
    for e in range(n_events):
        folder = root / f"event_{e:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        for j in range(patches_per_event):
            centre = size // 2 + rng.integers(-8, 8) + 5 * np.sin(yy / 8.0)
            dist = np.abs(xx - centre)
            river, flood = dist < 2.5, (dist < 12) & (dist >= 2.5)
            has_flood = rng.random() < 0.8

            def scene(with_flood):
                vv = -11 + rng.normal(0, 1.5, (size, size))        # land + speckle
                vv[river] = -22
                if with_flood:
                    vv[flood] = -21
                vh = vv - 6 + rng.normal(0, 1.0, (size, size))
                return np.stack([vv, vh]).astype(np.float32)

            before, after = scene(False), scene(has_flood)
            label = np.zeros((size, size), np.uint8)
            label[river] = PERM_WATER
            if has_flood:
                label[flood] = FLOOD
            before[:, :, -2:] = np.nan                  # fake no-data border
            after[:, :, -2:] = np.nan
            label[:, -2:] = 3                           # dataset's "invalid" code
            f = folder / f"patch_{j:02d}.npz"
            np.savez(f, before=before, after=after, label=label)
            paths.append(f)
    return paths


# =============================================================================
# 11. COMMAND-LINE ENTRY POINTS
# =============================================================================
def _prepare(files, root, out_dir, reader, n_val, n_test, force_test, seed=42):
    """Shared by `demo` and `scan`: split -> stats -> save the 3 config files."""
    splits = split_by_event(files, n_val=n_val, n_test=n_test, seed=seed,
                            force_test=force_test)
    print({k: len(v) for k, v in splits.items()}, "patches per split")
    normalizer, counts = scan_dataset(splits["train"], reader)
    weights = class_weights_from_counts(counts)
    out_dir = Path(out_dir)
    save_splits(splits, out_dir / "splits.json", root)
    normalizer.save(out_dir / "norm_stats.json")
    (out_dir / "class_weights.json").write_text(json.dumps(
        {"class_counts": counts.tolist(), "weights": weights}, indent=2))
    print("class pixel counts [no, perm, flood]:", counts.tolist())
    print("class weights:", [round(w, 3) for w in weights])
    return splits, normalizer


def run_demo(out_dir: str) -> None:
    """End-to-end run on synthetic data. If this prints 'PART 1 OK', the code works."""
    out = Path(out_dir)
    root = out / "synthetic"
    files = make_synthetic_events(root)
    reader = functools.partial(read_patch_npz, unit="db")
    splits, norm = _prepare(files, root, out / "config", reader, 1, 1, ())

    train_ds = FloodDataset(splits["train"], norm, AugmentConfig(), reader)
    val_ds = FloodDataset(splits["val"], norm, None, reader)
    print("train sanity:", sanity_check(train_ds))
    print("val sanity  :", sanity_check(val_ds))
    plot_samples(val_ds, out / "check_alignment.png")
    print(f"saved {out / 'check_alignment.png'}  <- open it and confirm red sits on the dark area")
    print("PART 1 OK")


def run_scan(args) -> None:
    root = Path(args.root)
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files under {root}")
    reader = functools.partial(read_patch_npz, unit=args.unit)
    _prepare(files, root, args.out, reader, args.n_val, args.n_test, args.force_test)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run everything on synthetic data")
    d.add_argument("--out", default="demo_out")
    s = sub.add_parser("scan", help="split + statistics on the real dataset")
    s.add_argument("--root", required=True, help="folder holding event_*/patch.npz")
    s.add_argument("--out", default="config")
    s.add_argument("--unit", default="db", choices=["db", "linear"])
    s.add_argument("--n-val", type=int, default=1)
    s.add_argument("--n-test", type=int, default=1)
    s.add_argument("--force-test", nargs="*", default=[],
                   help="event folder names to keep OUT of training (e.g. Himalayan events)")
    a = ap.parse_args(argv)
    run_demo(a.out) if a.cmd == "demo" else run_scan(a)


if __name__ == "__main__":
    main()
