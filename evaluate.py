"""
evaluate.py  --  Part 3 of Teammate 2's workflow: EVALUATION & METRICS
======================================================================

Repo location: models/evaluate.py   (needs eval_metrics.py + Part 1's dataset.py;
                                     Part 2's train.py only for real checkpoints)

WHAT THIS FILE DOES
-------------------
Measures HONESTLY how good the model is, in four levels:

    validation  ->  used ONCE to choose the probability threshold, then frozen
    test        ->  flood events the model never saw
    himalayan   ->  mountain-valley scenes the model never saw (the case-study
                    situation; the challenge scores this explicitly)
    baseline    ->  the same numbers for a simple non-AI method, for comparison

    scenes (Part 1 files)
        │  iter_scenes()
        ▼
    predict_fn(scene) -> flood probability map       <- model OR baseline OR demo
        │
        ├─ select_threshold()   VALIDATION only -> best threshold -> frozen to disk
        ▼
    evaluate_group(threshold=frozen) -> GroupResult
        │   counts + per-event bootstrap CI + error causes + relaxed metrics
        ▼
    results.md / results.json / figures (PR curves, error maps)

HOW TO READ THIS FILE
---------------------
Sections 1..8 follow the data flow.  "TRICK Tn" marks non-obvious ideas
(explained in 03_evaluation_and_metrics.md).  The maths lives in eval_metrics.py.

COMMANDS
--------
    python evaluate.py demo --out demo_eval        # synthetic, no model needed
    python evaluate.py run --checkpoint models/weights/baseline/best.pt \
        --splits config/splits.json --data-root /data/KuroSiwo_npz \
        --himalayan-events <event_folder> ... --out outputs/eval
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "part1_code", _HERE.parent / "part2_code"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dataset import (IGNORE_INDEX, Normalizer, build_input, event_of,  # noqa: E402
                     load_splits, make_synthetic_events, read_patch_npz)
from eval_metrics import (Counts, GroupResult, ThresholdSweep, add_error_counts,  # noqa: E402
                          bootstrap_ci, categorize_errors, counts_from_arrays, crop_padding,
                          empty_error_counts, errors_markdown, fit_drop_threshold,
                          freeze_protocol, logratio_baseline, macro_average, pad_to_multiple,
                          relaxed_counts, relaxed_precision_recall, results_markdown,
                          results_to_json)

FLOOD = 2


# =============================================================================
# 1. SCENES
# =============================================================================
@dataclass
class Scene:
    """One patch prepared for evaluation.

    truth is the label with ignored pixels set to 255 INCLUDING pixels where the
    radar image itself had no data (Part 1 `invalid`), so nobody is graded on
    pixels the model could not possibly see.
    """
    name: str
    event: str
    before: np.ndarray      # [2,H,W] dB
    after: np.ndarray       # [2,H,W] dB
    truth: np.ndarray       # [H,W] uint8: 0/1/2 or 255
    invalid: np.ndarray     # [H,W] bool


def iter_scenes(files: Sequence, reader: Callable, event_fn: Callable = event_of) -> Iterator[Scene]:
    """Lazily read patches one at a time (a generator), so a whole test set never
    has to fit in memory."""
    for f in files:
        p = reader(f)
        truth = p.label.copy()
        truth[p.invalid] = IGNORE_INDEX
        yield Scene(Path(f).name, event_fn(f), p.before, p.after, truth, p.invalid)


# =============================================================================
# 2. PREDICTORS  (anything that maps Scene -> flood probability [H,W])
# =============================================================================
Predictor = Callable[[Scene], np.ndarray]


def baseline_predictor(mode: str = "otsu", drop_db: float = 3.0, min_pixels: int = 8) -> Predictor:
    """The non-AI log-ratio method as a predictor (0.0 / 1.0 'probabilities')."""
    def predict(s: Scene) -> np.ndarray:
        return logratio_baseline(s.before, s.after, s.invalid, mode, drop_db, min_pixels).astype(np.float32)
    return predict


def make_checkpoint_predictor(ckpt_path, device: str = "cpu", flood_class: Optional[int] = None) -> Predictor:
    """Wrap a Part 2 checkpoint (best.pt) as a Predictor.  Imports PyTorch lazily.

    TRICK T7 (the checkpoint brings its own normalization):
        Part 2 stores the normalization mean/std INSIDE best.pt.  We rebuild the
        Normalizer from there, so evaluation can never use a stale
        norm_stats.json.  Input is built with Part 1's `build_input` - the very
        function used in training - and padded to a multiple of 32 (the U-Net
        needs it) then cropped back.
    """
    import torch
    from train import load_checkpoint
    model, meta = load_checkpoint(ckpt_path, device)
    cls = meta["config"].get("flood_class", 2) if flood_class is None else flood_class
    nm = meta["norm"]
    norm = Normalizer(np.array(nm["mean"], np.float32), np.array(nm["std"], np.float32), nm.get("meta", {}))

    @torch.no_grad()
    def predict(s: Scene) -> np.ndarray:
        x = build_input(s.before, s.after, s.invalid, norm)
        x, pad = pad_to_multiple(x, 32)
        logits = model(torch.from_numpy(np.ascontiguousarray(x))[None].to(device))
        prob = logits.softmax(dim=1)[0, cls].cpu().numpy()
        prob = crop_padding(prob, pad)
        prob[s.invalid] = 0.0
        return prob.astype(np.float32)
    return predict


def _box_blur(a: np.ndarray, k: int = 1) -> np.ndarray:
    p = np.pad(a.astype(np.float32), k, mode="edge")
    h, w = a.shape
    out = np.zeros((h, w), np.float32)
    for dy in range(2 * k + 1):
        for dx in range(2 * k + 1):
            out += p[dy:dy + h, dx:dx + w]
    return out / (2 * k + 1) ** 2


def noisy_oracle_predictor(noise_by_event: Callable[[str], float], seed: int = 0) -> Predictor:
    """DEMO ONLY: a fake 'model' = blurred truth + noise.  Lets us test the entire
    evaluation pipeline with no PyTorch and no trained weights.  Noise is seeded
    from the scene name (md5, not Python's randomized hash()), so it is the same
    on every run and every machine."""
    def predict(s: Scene) -> np.ndarray:
        h = int(hashlib.md5(f"{seed}:{s.event}:{s.name}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        base = _box_blur((s.truth == FLOOD).astype(np.float32), 1)
        prob = base + rng.normal(0, noise_by_event(s.event), base.shape)
        prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
        prob[s.invalid] = 0.0
        return prob
    return predict


# =============================================================================
# 3. CHOOSING THE THRESHOLD  (validation only)
# =============================================================================
def select_threshold(val_scenes: Iterable[Scene], predict_fn: Predictor,
                     metric: str = "f1", bins: int = 200) -> Tuple[float, float, ThresholdSweep]:
    """Run the model over VALIDATION scenes and pick the best probability threshold.

    Returns (threshold, metric value at it, sweep).  This is the ONLY function
    that chooses a threshold.  Everything else (test, Himalayan, the case study)
    receives the number as an argument - see evaluate_group (TRICK T6).
    The model is run once here and once more inside evaluate_group for the
    validation row; that doubles validation inference time and is deliberate:
    it keeps the two functions independent and simple.
    """
    sweep = ThresholdSweep(bins)
    for s in val_scenes:
        sweep.update(predict_fn(s), s.truth)
    thr, val = sweep.best(metric)
    return thr, val, sweep


# =============================================================================
# 4. EVALUATING ONE GROUP
# =============================================================================
@dataclass
class GroupExtras:
    """Heavier by-products of evaluate_group (kept out of the table)."""
    sweep: ThresholdSweep
    per_event: Dict[str, Counts]
    worst: List[Tuple[float, Scene, np.ndarray]] = field(default_factory=list)


def evaluate_group(name: str, scenes: Iterable[Scene], predict_fn: Predictor, *,
                   threshold: float, kind: str = "model", bins: int = 200,
                   boundary_px: int = 2, relaxed_px: int = 2, n_examples: int = 3,
                   with_oracle: bool = True) -> Tuple[GroupResult, GroupExtras]:
    """Score ONE group of scenes with a FIXED threshold.

    TRICK T6 (no-tuning-on-test by construction):
        `threshold` is keyword-only and has NO default, so you cannot call this
        without deciding where the number came from, and the function never
        computes a threshold itself.  The only diagnostic it computes from the
        group's own data is the ORACLE (best possible threshold on this very
        group).  It is reported, never used:
            gap = oracle F1 - frozen F1
        A small gap = the validation threshold transfers.  A big gap on the
        Himalayan group = distribution shift (the model's confidence means
        something different in mountains).  That gap is a result worth reporting.

    TRICK T2b (one Counts object per EVENT): scenes are summed into their event,
    and the bootstrap later resamples events, not pixels or scenes.
    """
    sweep = ThresholdSweep(bins)
    per_event: Dict[str, Counts] = {}
    errors = empty_error_counts()
    relaxed = {"pred_total": 0, "pred_near_truth": 0, "truth_total": 0, "truth_near_pred": 0}
    n_scenes = n_flood = 0
    worst: List[Tuple[float, Scene, np.ndarray]] = []

    for s in scenes:
        prob = predict_fn(s)
        pred = prob >= threshold
        c = counts_from_arrays(pred, s.truth)
        per_event[s.event] = per_event.get(s.event, Counts()) + c
        sweep.update(prob, s.truth)
        errors = add_error_counts(errors, categorize_errors(
            pred, s.truth, s.before[0], s.after[0] - s.before[0], boundary_px))
        for k, v in relaxed_counts(pred, s.truth, relaxed_px).items():
            relaxed[k] += v
        n_scenes += 1
        if c.tp + c.fn > 0:                         # only scenes that contain flood can be "worst"
            n_flood += 1
            score = c.iou if c.iou == c.iou else 0.0
            worst.append((score, s, prob))
            worst.sort(key=lambda t: t[0])
            del worst[n_examples:]                  # keep only the n worst (bounded memory)

    units = list(per_event.values())
    total = sum(units, Counts())
    rp, rr = relaxed_precision_recall(relaxed)
    oracle = None
    if with_oracle and kind == "model" and n_scenes:
        ot, of1 = sweep.best("f1")
        oracle = {"threshold": ot, "f1": of1, "iou": sweep.counts_at(ot).iou}
    result = GroupResult(
        name=name, kind=kind, threshold=threshold, counts=total,
        macro={m: macro_average(units, m) for m in ("precision", "recall", "iou", "f1")},
        ci_iou=bootstrap_ci(units, "iou"), ci_f1=bootstrap_ci(units, "f1"),
        n_events=len(per_event), n_scenes=n_scenes, n_flood_scenes=n_flood, errors=errors,
        relaxed_precision=rp, relaxed_recall=rr,
        average_precision=sweep.average_precision() if kind == "model" else None, oracle=oracle)
    return result, GroupExtras(sweep, per_event, worst)


# =============================================================================
# 5. FIGURES
# =============================================================================
def plot_error_map(scene: Scene, prob: np.ndarray, threshold: float, out_png) -> None:
    """Four panels: AFTER radar | truth | prediction | errors.
    Errors: red = false alarm (FP), blue = missed flood (FN), green = correct (TP)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    pred = prob >= threshold
    t = (scene.truth == FLOOD)
    valid = scene.truth != IGNORE_INDEX
    err = np.zeros(scene.truth.shape, np.uint8)       # 0 background, 1 TP, 2 FP, 3 FN
    err[pred & t & valid] = 1
    err[pred & ~t & valid] = 2
    err[~pred & t & valid] = 3
    truth_plot = np.where(scene.truth == IGNORE_INDEX, 3, scene.truth)

    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    ax[0].imshow(scene.after[0], cmap="gray"); ax[0].set_title("after VV")
    ax[1].imshow(truth_plot, cmap=ListedColormap(["#dddddd", "#2b6cb0", "#e53e3e", "#000000"]),
                 vmin=0, vmax=3, interpolation="nearest"); ax[1].set_title("truth")
    ax[2].imshow(pred, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    ax[2].set_title(f"prediction (p ≥ {threshold:.2f})")
    ax[3].imshow(err, cmap=ListedColormap(["#eeeeee", "#38a169", "#e53e3e", "#2b6cb0"]),
                 vmin=0, vmax=3, interpolation="nearest")
    ax[3].set_title("green ok | red false alarm | blue missed")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{scene.event} / {scene.name}", fontsize=9)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def plot_pr_curves(sweeps: Dict[str, ThresholdSweep], thresholds: Dict[str, float], out_png) -> None:
    """Precision-recall curve per group; a dot marks the FROZEN threshold.
    If the dot sits far from the curve's 'knee' on the Himalayan curve, the
    validation threshold does not transfer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, sw in sweeps.items():
        c = sw.curve()
        ax.plot(c["recall"], c["precision"], label=f"{name} (AP {sw.average_precision():.2f})")
        cc = sw.counts_at(thresholds[name])
        ax.plot([cc.recall], [cc.precision], "o", color=ax.lines[-1].get_color())
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Precision-recall (dot = frozen threshold)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# =============================================================================
# 6. THE FULL PROTOCOL
# =============================================================================
def build_groups(splits: Dict[str, List[str]], himalayan_events: Sequence[str],
                 extra_himalayan_files: Sequence[str] = (),
                 event_fn: Callable = event_of) -> Dict[str, List[str]]:
    """splits.json -> {"validation": [...], "test": [...], "himalayan": [...]}.

    TRICK T8 (leak guard): a Himalayan event must come from the TEST split.  If
    it is found in train or val the function raises, because then the "unseen"
    claim would be false.  (In Part 1 you keep such events out of training with
    `--force-test`.)  Files in `extra_himalayan_files` (e.g. hand-labelled
    patches) are taken on trust: you must make sure they were never trained on.
    """
    himal = set(himalayan_events)
    for split in ("train", "val"):
        bad = himal & {event_fn(f) for f in splits.get(split, [])}
        if bad:
            raise ValueError(f"Himalayan event(s) {sorted(bad)} are in the '{split}' split; "
                             f"they would not be unseen. Re-run Part 1 with --force-test.")
    missing = himal - {event_fn(f) for f in splits.get("test", [])}
    if missing:
        raise ValueError(f"Himalayan event(s) {sorted(missing)} not found in the test split")
    test = [f for f in splits["test"] if event_fn(f) not in himal]
    himalayan = [f for f in splits["test"] if event_fn(f) in himal] + list(extra_himalayan_files)
    groups = {"validation": list(splits["val"]), "test": test}
    if himalayan:
        groups["himalayan"] = himalayan
    return groups


def run_evaluation(groups: Dict[str, List[str]], model_predict: Predictor, reader: Callable,
                   out_dir, checkpoint: str = "n/a", metric: str = "f1",
                   verbose: bool = True) -> List[GroupResult]:
    """The whole Part 3 protocol; writes everything into `out_dir`.

    Steps: (1) choose the threshold on `groups["validation"]` and freeze it,
    (2) fit the baseline's dB drop on validation too, (3) score the model and
    both baselines on every group, (4) write tables and figures.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenes = lambda name: iter_scenes(groups[name], reader)

    # (1) threshold from validation, frozen
    thr, val_score, _ = select_threshold(scenes("validation"), model_predict, metric)
    val_events = sorted({event_of(f) for f in groups["validation"]})
    freeze_protocol(out / "eval_protocol.json", thr, metric, val_score, val_events, checkpoint)
    if verbose:
        print(f"threshold chosen on validation by {metric}: {thr:.3f} (validation {metric} {val_score:.3f})")

    # (2) baseline parameter, also from validation only
    val_tuples = [(s.before, s.after, s.invalid, s.truth) for s in scenes("validation")]
    drop_db, drop_iou = fit_drop_threshold(val_tuples)
    del val_tuples
    if verbose:
        print(f"baseline drop fitted on validation: {drop_db:.1f} dB (IoU {drop_iou:.3f})")

    # (3) score everything
    results: List[GroupResult] = []
    sweeps, thresholds = {}, {}
    for gname in groups:
        label = "validation*" if gname == "validation" else gname
        res, ex = evaluate_group(label, scenes(gname), model_predict, threshold=thr, kind="model")
        results.append(res)
        sweeps[label], thresholds[label] = ex.sweep, thr
        for i, (score, s, prob) in enumerate(ex.worst):
            plot_error_map(s, prob, thr, out / "figures" / f"errors_{gname}_{i}.png")
        for bname, fn in (("baseline-fixed", baseline_predictor("fixed", drop_db)),
                          ("baseline-otsu", baseline_predictor("otsu"))):
            bres, _ = evaluate_group(label, scenes(gname), fn, threshold=0.5, kind=bname)
            results.append(bres)
    plot_pr_curves(sweeps, thresholds, out / "figures" / "pr_curves.png")

    # (4) write
    md = ["# Part 3 results", "",
          "`*` validation is the group the threshold (and baseline drop) were chosen on; "
          "it is NOT an honest score. Judge the model on `test` and `himalayan`.", "",
          results_markdown(results), "", "## Why does it fail?", "", errors_markdown(
              [r for r in results if r.kind == "model"])]
    (out / "results.md").write_text("\n".join(md))
    (out / "results.json").write_text(results_to_json(results))
    return results


# =============================================================================
# 7. COMMAND LINE
# =============================================================================
def run_demo(out_dir: str) -> None:
    """Whole protocol on synthetic data with a FAKE model (no PyTorch needed).
    The fake model is made noisier on the 'Himalayan' events, so the table must
    show a drop there; if it does, the pipeline measures what it should."""
    out = Path(out_dir)
    files = make_synthetic_events(out / "synthetic", n_events=7, patches_per_event=6, size=64)
    by_ev: Dict[str, List[str]] = {}
    for f in files:
        by_ev.setdefault(event_of(f), []).append(str(f))
    ev = sorted(by_ev)
    splits = {"train": by_ev[ev[6]], "val": by_ev[ev[0]],
              "test": sum((by_ev[e] for e in ev[1:6]), [])}
    groups = build_groups(splits, himalayan_events=ev[4:6])
    himal = set(ev[4:6])
    model = noisy_oracle_predictor(lambda e: 0.55 if e in himal else 0.20)
    reader = functools.partial(read_patch_npz, unit="db")
    results = run_evaluation(groups, model, reader, out / "eval", checkpoint="FAKE-demo-model")
    print()
    print((out / "eval" / "results.md").read_text())
    by = {(r.name, r.kind): r for r in results}
    assert by[("himalayan", "model")].counts.f1 < by[("test", "model")].counts.f1, "demo sanity failed"
    for f in ("results.md", "results.json", "eval_protocol.json", "figures/pr_curves.png"):
        assert (out / "eval" / f).exists(), f
    print("\nPART 3 OK")


def run_real(a) -> None:
    splits = load_splits(a.splits, a.data_root)
    extra = sorted(str(p) for p in Path(a.himalayan_extra_dir).rglob("*.npz")) if a.himalayan_extra_dir else []
    groups = build_groups(splits, a.himalayan_events, extra)
    reader = functools.partial(read_patch_npz, unit=a.unit)
    predict = make_checkpoint_predictor(a.checkpoint, a.device)
    results = run_evaluation(groups, predict, reader, a.out, checkpoint=a.checkpoint, metric=a.metric)
    print((Path(a.out) / "results.md").read_text())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Part 3: evaluation & metrics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="synthetic end-to-end run with a fake model")
    d.add_argument("--out", default="demo_eval")
    r = sub.add_parser("run", help="evaluate a real checkpoint")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--splits", default="config/splits.json")
    r.add_argument("--data-root", required=True)
    r.add_argument("--himalayan-events", nargs="*", default=[])
    r.add_argument("--himalayan-extra-dir", default=None,
                   help="folder of extra hand-labelled .npz patches never used in training")
    r.add_argument("--unit", default="db", choices=["db", "linear"])
    r.add_argument("--metric", default="f1", choices=["f1", "iou"])
    r.add_argument("--device", default="cpu")
    r.add_argument("--out", default="outputs/eval")
    a = ap.parse_args(argv)
    run_demo(a.out) if a.cmd == "demo" else run_real(a)


if __name__ == "__main__":
    main()
