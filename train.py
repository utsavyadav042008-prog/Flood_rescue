"""
train.py  --  Part 2 of Teammate 2's workflow: MODEL & TRAINING
===============================================================

Repo location: models/train.py        (needs: train_utils.py and Part 1's dataset.py)

WHAT THIS FILE DOES
-------------------
Builds a U-Net, trains it on the Part 1 data, picks the best epoch by
validation FLOOD IoU, and saves one self-contained checkpoint (best.pt).

    config/train.yaml ──► TrainConfig
    Part 1 files (splits.json, norm_stats.json, class_weights.json)
            │
            ▼
    FloodDataset ─► DataLoader ─► U-Net(6 ch -> 3 classes) ─► SegLoss
                                       ▲                          │
                              AdamW (2 learning rates)  ◄── backward + clip
                                       │
        every epoch: evaluate() ─► ConfusionMatrix ─► flood IoU ─► BestTracker
                                       │
                                       ▼
        models/weights/<run>/best.pt, last.pt, log.jsonl, config.json
        models/experiments.csv  (one row per run, filled automatically)

HOW TO READ THIS FILE
---------------------
Sections 1..9 follow the order things happen.  "TRICK Tn" marks anything that
is not obvious; each is explained where it appears and in 02_model_and_training.md.

COMMANDS
--------
    python train.py selftest                 # fake data, CPU, ~1-3 min: proves it works
    python train.py forward-check            # Day 2: shapes come out right
    python train.py overfit  --config config/train.yaml     # Day 2: bug detector
    python train.py train    --config config/train.yaml --set lr=3e-4 epochs=20
    python train.py table                    # print experiments.csv as markdown
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
except ImportError:  # pragma: no cover
    raise SystemExit("PyTorch is required for train.py:  pip install torch")

# Make `dataset` (Part 1) and `train_utils` importable whether files sit together
# in models/ (final repo) or in the part1_code/ part2_code/ folders (this handout).
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "part1_code"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dataset import (AugmentConfig, CLASS_NAMES, FloodDataset, IGNORE_INDEX,  # noqa: E402
                     Normalizer, class_weights_from_counts, load_splits,
                     make_synthetic_events, read_patch_npz, scan_dataset,
                     split_by_event)
from train_utils import (BestTracker, ConfusionMatrix, TrainConfig,  # noqa: E402
                         append_experiment_row, append_jsonl, config_diff,
                         experiments_markdown, load_config, lr_factor,
                         make_checkpoint, masked_dice_numpy)


# =============================================================================
# 1. SMALL UTILITIES
# =============================================================================
def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch.  Runs become repeatable *approximately*:
    we do not force deterministic GPU kernels (that would slow training)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> "torch.device":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: "nn.Module") -> int:
    return sum(p.numel() for p in model.parameters())


def atomic_torch_save(obj, path) -> None:
    """Save to a temp file, then rename.

    TRICK T8b: Colab/Kaggle sessions can die at any moment.  If that happens in
    the middle of torch.save, a normal save leaves a half-written, unreadable
    best.pt - possibly overwriting your only good one.  os.replace() is atomic:
    the path always holds either the complete old file or the complete new one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# =============================================================================
# 2. THE MODEL
# =============================================================================
def build_model(cfg: TrainConfig, pretrained: bool = True) -> "nn.Module":
    """U-Net with a ResNet encoder from segmentation-models-pytorch (smp).

    Input  [B, 6, H, W]  ->  output [B, 3, H, W]  (raw scores = "logits", one per class).

    * in_channels=6: smp rebuilds the encoder's first convolution for 6 inputs
      and reuses the pretrained RGB filters for them (it cycles and rescales
      them - see smp docs).  Our inputs are already roughly zero-mean/unit-std,
      the same scale ImageNet photos have after normalization, so the
      pretrained filters are not fed wildly different magnitudes.
    * Does ImageNet pretraining help on radar?  Often a little, sometimes not.
      That is an experiment (row 3 of the experiment table), not an assumption.

    TRICK T9: `pretrained=False` skips the ImageNet download.  load_checkpoint()
    and the selftest use it: when we are about to overwrite the weights anyway,
    downloading them (and failing on a judge's offline laptop) is pointless.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise SystemExit("pip install segmentation-models-pytorch")
    return smp.Unet(
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights if pretrained else None,
        in_channels=cfg.in_channels,
        classes=cfg.num_classes,
    )


def check_forward(cfg: TrainConfig, size: int = 64,
                  device: Optional["torch.device"] = None) -> Tuple[tuple, int]:
    """Day-2 deliverable: random input in, correct-shaped output out.

    GOTCHA: the encoder halves the image 5 times (2^5 = 32), so H and W must be
    multiples of 32 (224 = 7 x 32 is fine).  Part 4 (inference) must pad odd
    sizes up to a multiple of 32 and crop afterwards.
    """
    assert size % 32 == 0, "image size must be a multiple of 32 for this U-Net"
    device = device or get_device()
    model = build_model(cfg, pretrained=False).to(device).eval()
    x = torch.randn(2, cfg.in_channels, size, size, device=device)
    with torch.no_grad():
        out = model(x)
    expected = (2, cfg.num_classes, size, size)
    assert tuple(out.shape) == expected, f"got {tuple(out.shape)}, expected {expected}"
    return tuple(out.shape), count_parameters(model)


def make_param_groups(model: "nn.Module", lr: float, encoder_lr_mult: float,
                      weight_decay: float) -> List[dict]:
    """Split parameters into up to 4 groups: {encoder, decoder} x {decay, no-decay}.

    TRICK T1 (two learning rates): the encoder is pretrained and only needs
    gentle adjusting (lr x 0.1); the decoder and the 3-class head start random and
    need the full rate.  One rate for both either wrecks the encoder or trains
    the decoder too slowly.

    TRICK T3 (no weight decay on biases and BatchNorm): weight decay pulls
    weights toward 0.  For 1-D parameters (biases, BatchNorm scale/shift) that
    shrinks things that are NOT overfitting-prone and hurts accuracy slightly.
    Convention: decay only the weight matrices/kernels (ndim > 1).

    The `lr` stored in each group is its BASE rate.  LambdaLR multiplies every
    group's own base rate by the same factor, so the encoder:decoder ratio stays
    10:1 through warmup and cosine decay (TRICK T2b).
    """
    enc_ids = {id(p) for p in model.encoder.parameters()}
    buckets: Dict[Tuple[bool, bool], list] = {(e, d): [] for e in (True, False) for d in (True, False)}
    for p in model.parameters():
        if p.requires_grad:
            buckets[(id(p) in enc_ids, p.ndim > 1)].append(p)
    groups = []
    for (is_enc, decay), params in buckets.items():
        if params:
            groups.append({"params": params,
                           "lr": lr * (encoder_lr_mult if is_enc else 1.0),
                           "weight_decay": weight_decay if decay else 0.0})
    return groups


# =============================================================================
# 3. THE LOSS  (cross-entropy + Dice, both ignoring invalid pixels)
# =============================================================================
class MaskedDiceLoss(nn.Module):
    """Soft Dice loss that skips ignored pixels.  Cares about OVERLAP, not pixel count.

    Why Dice at all?  Flood may be 1-5% of pixels.  Cross-entropy alone is
    happy with "predict no-water everywhere" (95% right).  Dice measures how
    well the flood region OVERLAPS the true flood region, so the rare class
    cannot be ignored.

    TRICK T5 (three details that make it stable):
      (a) Dice is computed over the WHOLE BATCH per class, not per image.
          A per-image Dice on an image with no flood is 0/0 - noisy gradients.
      (b) Ignored pixels (value 255) are multiplied out of both the predictions
          and the targets, so they contribute nothing.
      (c) Only classes that are PRESENT in the batch's targets are averaged.
          Otherwise an absent class scores Dice ~ 0 and adds a constant,
          meaningless penalty.  The cost of (c): Dice alone does not punish
          false alarms for an absent class - the cross-entropy term does.
    A numpy twin (train_utils.masked_dice_numpy) is compared against this in the
    selftest.
    """

    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX, smooth: float = 1.0):
        super().__init__()
        self.c, self.ignore, self.smooth = num_classes, ignore_index, smooth

    def forward(self, logits: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
        probs = logits.float().softmax(dim=1)                       # [B,C,H,W]
        valid = target != self.ignore                               # [B,H,W]
        t = torch.where(valid, target, torch.zeros_like(target))
        onehot = F.one_hot(t, self.c).permute(0, 3, 1, 2).to(probs.dtype)
        vf = valid.unsqueeze(1).to(probs.dtype)
        p, o = probs * vf, onehot * vf
        dims = (0, 2, 3)
        inter = (p * o).sum(dims)
        denom = p.sum(dims) + o.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        present = o.sum(dims) > 0
        if not bool(present.any()):
            return logits.sum() * 0.0          # nothing to grade; keeps the graph alive
        return 1.0 - dice[present].mean()


class SegLoss(nn.Module):
    """total = ce_weight * weighted CrossEntropy + dice_weight * MaskedDice.

    TRICK T4a: logits are cast to float32 here.  The network runs in half
    precision under autocast, but softmax/log inside a loss can overflow or
    underflow in float16.  Model in fp16 (fast), loss in fp32 (safe).
    `class_weights` (from Part 1) make rare-class mistakes cost more in the CE
    part; ignore_index=255 removes invalid pixels from it.
    """

    def __init__(self, num_classes: int, class_weights: Optional[List[float]] = None,
                 ce_weight: float = 1.0, dice_weight: float = 1.0,
                 ignore_index: int = IGNORE_INDEX):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("cw", w)                  # moves to GPU with .to(device)
        self.ce_weight, self.dice_weight, self.ignore = ce_weight, dice_weight, ignore_index
        self.dice = MaskedDiceLoss(num_classes, ignore_index)

    def forward(self, logits: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
        logits = logits.float()
        total = logits.sum() * 0.0
        if self.ce_weight > 0:
            total = total + self.ce_weight * F.cross_entropy(
                logits, target, weight=self.cw, ignore_index=self.ignore)
        if self.dice_weight > 0:
            total = total + self.dice_weight * self.dice(logits, target)
        return total


# =============================================================================
# 4. ONE TRAINING EPOCH
# =============================================================================
def make_scaler(enabled: bool):
    """Mixed-precision gradient scaler; works on old and new PyTorch versions."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, scaler,
                    device, cfg: TrainConfig) -> Dict[str, float]:
    """Run every batch once.  Returns {"loss": mean loss, "skipped": bad batches}.

    Per batch:  forward (fp16) -> loss (fp32) -> backward -> unscale -> clip ->
                optimizer step -> LR step.

    TRICK T4b (clip AFTER unscale): with mixed precision the gradients are
    multiplied by a big "scale" factor to avoid fp16 underflow.  Clipping before
    `scaler.unscale_` would compare the scaled norm with your threshold of 1.0
    and clip almost every step.  Unscale first, then clip.

    TRICK T4c (skip a poisoned batch): one batch with all pixels ignored (loss =
    NaN) would put NaN into every weight and ruin the whole run.  Non-finite
    losses are skipped and counted instead; if `skipped` grows large, something
    is wrong with the data.
    """
    model.train()
    use_amp = cfg.amp and device.type == "cuda"
    total, n, skipped = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        loss = loss_fn(logits, y)                      # outside autocast -> fp32
        if not torch.isfinite(loss):
            skipped += 1
            scheduler.step()                           # keep schedule aligned with data seen
            continue
        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += float(loss.item())
        n += 1
    return {"loss": total / max(n, 1), "skipped": skipped}


# =============================================================================
# 5. EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device, num_classes: int,
             loss_fn=None) -> Tuple[ConfusionMatrix, float]:
    """Predict every validation batch; accumulate ONE confusion matrix.

    The class with the highest score wins (argmax).  Part 4 will instead use
    "probability of flood >= threshold", which is why the threshold is tuned
    later on validation data - not here.
    Returns (confusion matrix, mean validation loss).
    """
    model.eval()
    cm = ConfusionMatrix(num_classes, IGNORE_INDEX)
    total, n = 0.0, 0
    for x, y in loader:
        logits = model(x.to(device))
        if loss_fn is not None:
            v = loss_fn(logits, y.to(device))
            if torch.isfinite(v):
                total += float(v.item())
                n += 1
        cm.update(logits.argmax(dim=1).cpu().numpy(), y.numpy())
    return cm, total / max(n, 1)


# =============================================================================
# 6. THE BUG DETECTOR: OVERFIT A TINY SET
# =============================================================================
def overfit_test(cfg: TrainConfig, dataset, n: int = 16, steps: int = 80,
                 device: Optional["torch.device"] = None,
                 class_weights: Optional[List[float]] = None) -> Dict[str, object]:
    """Train on only `n` patches for `steps` steps; the loss MUST collapse.

    This is the single most useful debugging step in deep learning:
      * if the loss falls to a fraction of its start -> data, labels, loss and
        optimizer are wired correctly (a model can always memorize 16 images);
      * if it does not -> there is a BUG (labels misaligned with images, wrong
        ignore value, bad normalization...), not a "model quality" problem.
    Pass a NON-augmented dataset.  No pretrained weights (fast, offline).
    """
    device = device or get_device()
    ds = Subset(dataset, list(range(min(n, len(dataset)))))
    loader = DataLoader(ds, batch_size=max(2, min(4, len(ds))), shuffle=True, num_workers=0)
    model = build_model(cfg, pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = SegLoss(cfg.num_classes, class_weights, cfg.ce_weight, cfg.dice_weight).to(device)
    losses: List[float] = []
    model.train()
    while len(losses) < steps:
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                continue
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            if len(losses) >= steps:
                break
    first = float(np.mean(losses[:3]))
    last = float(np.mean(losses[-5:]))
    return {"first_loss": first, "last_loss": last, "passed": last < 0.4 * first}


# =============================================================================
# 7. FULL TRAINING RUN
# =============================================================================
def run_training(cfg: TrainConfig, train_files, val_files, normalizer: Normalizer,
                 class_weights: Optional[List[float]] = None, reader=None,
                 verbose: bool = True) -> Dict[str, object]:
    """The whole recipe.  Returns a dict with best epoch/metrics and the run folder.

    Choosing the best epoch uses validation FLOOD IoU, never the loss: the loss
    mixes three classes and can improve while the flood class gets worse.
    """
    seed_everything(cfg.seed)
    device = get_device()
    reader = reader or functools.partial(read_patch_npz, unit=cfg.unit)

    train_ds = FloodDataset(train_files, normalizer,
                            AugmentConfig(noise_db=cfg.aug_noise_db), reader, seed=cfg.seed)
    val_ds = FloodDataset(val_files, normalizer, None, reader, seed=cfg.seed)

    # TRICK T10: drop_last=True removes a final batch of size 1.  BatchNorm
    # cannot compute statistics from one sample and training gets noisy/crashes.
    # The seeded Generator makes the shuffling order repeatable.
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=len(train_ds) > cfg.batch_size, generator=gen,
                              num_workers=cfg.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin)

    model = build_model(cfg, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        make_param_groups(model, cfg.lr, cfg.encoder_lr_mult, cfg.weight_decay))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.epochs * steps_per_epoch
    warm_steps = int(cfg.warmup_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total_steps, warm_steps))
    scaler = make_scaler(cfg.amp and device.type == "cuda")
    loss_fn = SegLoss(cfg.num_classes, class_weights if cfg.use_class_weights else None,
                      cfg.ce_weight, cfg.dice_weight).to(device)

    tracker = BestTracker(cfg.patience)
    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    norm_dict = {"mean": normalizer.mean.tolist(), "std": normalizer.std.tolist(),
                 "meta": normalizer.meta}

    if verbose:
        print(f"device={device}  params={count_parameters(model):,}  "
              f"train={len(train_ds)} val={len(val_ds)}  steps/epoch={steps_per_epoch}")
    t0 = time.time()
    best_metrics: Dict[str, object] = {}
    epochs_run = 0
    for epoch in range(cfg.epochs):
        train_ds.set_epoch(epoch)                     # Part 1: new augmentations each epoch
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, scaler, device, cfg)
        cm, val_loss = evaluate(model, val_loader, device, cfg.num_classes, loss_fn)
        m = cm.summary(cfg.flood_class)
        improved, stop = tracker.update(m["flood_iou"], epoch)
        epochs_run = epoch + 1
        ck = make_checkpoint(model.state_dict(), cfg, norm_dict, m, epoch, CLASS_NAMES)
        if improved:
            best_metrics = m
            atomic_torch_save(ck, run_dir / "best.pt")
        atomic_torch_save(ck, run_dir / "last.pt")
        append_jsonl(run_dir / "log.jsonl", {
            "epoch": epoch, "train_loss": tr["loss"], "skipped": tr["skipped"],
            "val_loss": val_loss, "lr_decoder": optimizer.param_groups[-1]["lr"],
            **{k: (None if isinstance(v, float) and v != v else v) for k, v in m.items()
               if k != "per_class_iou"}})
        if verbose:
            print(f"epoch {epoch + 1:>3}/{cfg.epochs}  loss {tr['loss']:.4f}  val_loss {val_loss:.4f}  "
                  f"flood IoU {m['flood_iou']:.4f}  F1 {m['flood_f1']:.4f}  "
                  f"P {m['flood_precision']:.3f}  R {m['flood_recall']:.3f}"
                  f"{'  <- best' if improved else ''}"
                  f"{'  (skipped %d bad batches)' % tr['skipped'] if tr['skipped'] else ''}")
        if stop:
            if verbose:
                print(f"early stop: no gain for {cfg.patience} epochs")
            break

    if not (run_dir / "best.pt").exists():            # flood IoU was NaN every epoch
        os.replace(run_dir / "last.pt", run_dir / "best.pt")
        print("WARNING: validation set had no flood pixels; saved the LAST epoch as best.pt")

    append_experiment_row(cfg.experiments_csv, {
        "run_name": cfg.run_name, "changed": json.dumps(config_diff(cfg)),
        "best_epoch": tracker.best_epoch,
        "val_flood_iou": _r(best_metrics.get("flood_iou")),
        "val_flood_f1": _r(best_metrics.get("flood_f1")),
        "val_precision": _r(best_metrics.get("flood_precision")),
        "val_recall": _r(best_metrics.get("flood_recall")),
        "epochs_run": epochs_run, "minutes": round((time.time() - t0) / 60, 1),
        "started": time.strftime("%Y-%m-%d %H:%M")})
    return {"run_dir": str(run_dir), "best_epoch": tracker.best_epoch,
            "best_flood_iou": tracker.best, "best_metrics": best_metrics}


def _r(v) -> str:
    return "" if v is None or v != v else f"{float(v):.4f}"


# =============================================================================
# 8. LOADING A SAVED MODEL  (used by Part 4)
# =============================================================================
def load_checkpoint(path, device="cpu") -> Tuple["nn.Module", Dict]:
    """best.pt -> (model in eval mode, meta dict containing config + norm stats).

    Uses weights_only=True when PyTorch supports it: the file holds only tensors
    and a JSON string (TRICK T8), so nothing can execute on load.  Only the
    config fields that still exist are used, so old checkpoints keep loading
    after TrainConfig grows.
    """
    try:
        ck = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover  (old PyTorch)
        ck = torch.load(path, map_location=device)
    meta = json.loads(ck["meta_json"])
    names = {f.name for f in fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in meta["config"].items() if k in names})
    model = build_model(cfg, pretrained=False)          # TRICK T9: no download
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval(), meta


# =============================================================================
# 9. COMMAND LINE
# =============================================================================
def _load_real_inputs(cfg: TrainConfig):
    """Read the three Part 1 files and return (splits, normalizer, class weights)."""
    splits = load_splits(cfg.splits_file, cfg.data_root)
    norm = Normalizer.load(cfg.norm_stats)
    cw = json.loads(Path(cfg.class_weights_file).read_text())["weights"]
    return splits, norm, cw


def run_selftest() -> None:
    """End-to-end proof on synthetic data (CPU, no downloads, 1-3 minutes).

    Checks: (1) torch Dice == numpy Dice  (2) forward shapes  (3) the overfit
    test passes  (4) a 2-epoch training run writes a checkpoint  (5) the
    checkpoint reloads and predicts the right shape.  If this prints
    'PART 2 OK' on your machine, the plumbing is correct.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = TrainConfig(encoder="resnet18", encoder_weights=None, epochs=2, batch_size=4,
                          num_workers=0, amp=False, patience=0, run_name="selftest",
                          out_dir=str(tmp / "weights"), experiments_csv=str(tmp / "exp.csv"))
        files = make_synthetic_events(tmp / "syn", n_events=5, patches_per_event=4, size=64)
        reader = functools.partial(read_patch_npz, unit="db")
        splits = split_by_event(files, n_val=1, n_test=1)
        norm, counts = scan_dataset(splits["train"], reader, verbose=False)
        cw = class_weights_from_counts(counts)

        # 1. torch Dice vs numpy Dice on identical numbers
        val_ds = FloodDataset(splits["val"], norm, None, reader)
        xb, yb = (torch.stack(t) for t in zip(*[val_ds[i] for i in range(4)]))
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(4, 3, 64, 64, generator=g)
        t_val = float(MaskedDiceLoss(3)(logits, yb))
        n_val = masked_dice_numpy(logits.softmax(1).numpy().astype(np.float64), yb.numpy())
        assert abs(t_val - n_val) < 1e-4, f"Dice mismatch torch={t_val} numpy={n_val}"
        print(f"[1/5] Dice torch == numpy ({t_val:.5f})")

        # 2. forward pass shape
        shape, n_par = check_forward(cfg, size=64)
        print(f"[2/5] forward pass OK: output {shape}, {n_par:,} parameters")

        # 3. overfit test
        res = overfit_test(cfg, val_ds, n=8, steps=60, class_weights=cw)
        print(f"[3/5] overfit: loss {res['first_loss']:.3f} -> {res['last_loss']:.3f}")
        assert res["passed"], "overfit test FAILED: loss did not collapse -> there is a bug"

        # 4. mini training run
        out = run_training(cfg, splits["train"], splits["val"], norm, cw, reader, verbose=False)
        best = Path(out["run_dir"]) / "best.pt"
        assert best.exists()
        print(f"[4/5] training run OK, best.pt written, best flood IoU {out['best_flood_iou']:.3f}")

        # 5. reload + predict
        model, meta = load_checkpoint(best)
        with torch.no_grad():
            pred = model(xb[:1]).argmax(1)
        assert tuple(pred.shape) == (1, 64, 64)
        assert meta["norm"]["mean"] == norm.mean.tolist()
        print("[5/5] checkpoint reloads; norm stats travel inside it")
    print("PART 2 OK")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Part 2: model & training")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="synthetic end-to-end check (CPU)")
    for name, helptext in (("forward-check", "random tensor through the model"),
                           ("overfit", "overfit 16 real patches (bug detector)"),
                           ("train", "full training run")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--config", default=None, help="YAML file (e.g. config/train.yaml)")
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                       help="override settings, e.g. --set lr=3e-4 epochs=5")
    t = sub.add_parser("table", help="print experiments.csv as a markdown table")
    t.add_argument("--csv", default="models/experiments.csv")
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        run_selftest()
    elif a.cmd == "table":
        print(experiments_markdown(a.csv))
    else:
        cfg = load_config(a.config, a.set)
        if a.cmd == "forward-check":
            print(check_forward(cfg, size=224))
        else:
            splits, norm, cw = _load_real_inputs(cfg)
            reader = functools.partial(read_patch_npz, unit=cfg.unit)
            if a.cmd == "overfit":
                ds = FloodDataset(splits["train"], norm, None, reader)
                print(overfit_test(cfg, ds, class_weights=cw))
            else:
                print(run_training(cfg, splits["train"], splits["val"], norm, cw, reader))


if __name__ == "__main__":
    main()
