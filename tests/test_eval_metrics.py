"""Tests for eval_metrics.py and evaluate.py (numpy only, no PyTorch).
    python test_eval_metrics.py        or        pytest test_eval_metrics.py
Repo location: tests/test_eval_metrics.py
"""
import functools
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent / "models", _HERE.parent / "evaluation"):
    sys.path.insert(0, str(_p))

from eval_metrics import (Counts, GroupResult, ThresholdSweep, bootstrap_ci, categorize_errors,  # noqa: E402
                          counts_from_arrays, crop_padding, dilate, erode, fit_drop_threshold,
                          freeze_protocol, load_protocol, logratio_baseline, macro_average,
                          otsu_threshold, pad_to_multiple, relaxed_counts,
                          relaxed_precision_recall, remove_small_blobs, results_markdown)
from evaluate import (Scene, build_groups, evaluate_group, iter_scenes,  # noqa: E402
                      noisy_oracle_predictor, select_threshold)
from dataset import make_synthetic_events, read_patch_npz, event_of  # noqa: E402
from train_utils import ConfusionMatrix  # noqa: E402

READER = functools.partial(read_patch_npz, unit="db")


# ---------- Counts ----------
def test_counts_known_example():
    truth = np.array([2, 2, 2, 0, 0, 255])
    pred = np.array([1, 1, 0, 0, 1, 1])
    c = counts_from_arrays(pred, truth)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 1)
    assert math.isclose(c.iou, 0.5) and math.isclose(c.f1, 4 / 6)


def test_counts_match_part2_confusion_matrix():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 3, (40, 40)); truth[:3] = 255
    pred = rng.random((40, 40)) > 0.6
    c = counts_from_arrays(pred, truth)
    cm = ConfusionMatrix(3)
    cm.update(np.where(pred, 2, 0), truth)      # flood vs "not flood" (class 0)
    m = cm.class_metrics(2)
    for k in ("precision", "recall", "iou", "f1"):
        assert math.isclose(getattr(c, k), m[k]), k


def test_counts_nan_when_undefined():
    c = Counts(0, 0, 0, 10)                      # no flood anywhere, none predicted
    assert all(math.isnan(getattr(c, k)) for k in ("precision", "recall", "iou", "f1"))
    assert math.isnan(Counts(0, 5, 0, 5).precision) is False and Counts(0, 5, 0, 5).iou == 0.0


def test_counts_add():
    assert Counts(1, 2, 3, 4) + Counts(1, 1, 1, 1) == Counts(2, 3, 4, 5)


# ---------- ThresholdSweep ----------
def _brute(prob, truth, thr):
    return counts_from_arrays(prob >= thr, truth)


def test_sweep_equals_bruteforce_at_every_threshold():
    rng = np.random.default_rng(1)
    bins = 50
    prob = (rng.integers(0, bins, (60, 60)) + 0.5) / bins        # probs at bin centres: no quantization error
    truth = rng.integers(0, 3, (60, 60)); truth[:2] = 255
    sw = ThresholdSweep(bins)
    sw.update(prob, truth)
    for k in (0, 5, 17, 25, 33, 49):
        thr = k / bins
        assert sw.counts_at(thr) == _brute(prob, truth, thr), k


def test_sweep_add_equals_joint_update():
    rng = np.random.default_rng(2)
    p1, p2 = rng.random((20, 20)), rng.random((20, 20))
    t1, t2 = rng.integers(0, 3, (20, 20)), rng.integers(0, 3, (20, 20))
    a, b, j = ThresholdSweep(), ThresholdSweep(), ThresholdSweep()
    a.update(p1, t1); b.update(p2, t2); j.update(p1, t1); j.update(p2, t2)
    s = a + b
    assert np.array_equal(s.pos, j.pos) and np.array_equal(s.neg, j.neg)


def test_sweep_best_threshold_and_perfect_ap():
    truth = np.array([[2, 2, 0, 0]])
    prob = np.array([[0.9, 0.8, 0.2, 0.1]])
    sw = ThresholdSweep(100)
    sw.update(prob, truth)
    thr, f1 = sw.best("f1")
    assert 0.2 < thr <= 0.8 and math.isclose(f1, 1.0)
    assert math.isclose(sw.average_precision(), 1.0)


def test_sweep_ignores_invalid_and_counts_permanent_water_as_negative():
    truth = np.array([[2, 1, 255]])
    sw = ThresholdSweep(10)
    sw.update(np.array([[0.95, 0.95, 0.95]]), truth)
    c = sw.counts_at(0.5)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 0, 0)


def test_sweep_best_takes_middle_of_plateau():
    truth = np.array([[2, 2, 0, 0]])
    prob = np.array([[0.95, 0.95, 0.05, 0.05]])
    sw = ThresholdSweep(100)
    sw.update(prob, truth)
    thr, _ = sw.best("f1")
    assert 0.3 < thr < 0.7


# ---------- bootstrap ----------
def test_bootstrap_contains_point_and_is_deterministic():
    units = [Counts(80, 10, 10, 900), Counts(60, 20, 20, 900), Counts(90, 5, 10, 900), Counts(40, 30, 30, 900)]
    a = bootstrap_ci(units, "iou", seed=3)
    b = bootstrap_ci(units, "iou", seed=3)
    assert a == b and a["low"] <= a["point"] <= a["high"] and a["n_units"] == 4


def test_bootstrap_narrower_with_more_identical_units_and_nan_for_one_unit():
    few = bootstrap_ci([Counts(80, 10, 10, 0), Counts(40, 30, 30, 0)], "iou")
    many = bootstrap_ci([Counts(80, 10, 10, 0), Counts(40, 30, 30, 0)] * 10, "iou")
    assert (many["high"] - many["low"]) < (few["high"] - few["low"])
    assert math.isnan(bootstrap_ci([Counts(5, 1, 1, 1)])["low"])


def test_bootstrap_vectorized_matches_manual_resample():
    units = [Counts(10, 1, 2, 0), Counts(3, 5, 4, 0), Counts(7, 2, 2, 0)]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 3, size=(5, 3))
    manual = []
    for row in idx:
        tot = sum((units[i] for i in row), Counts())
        manual.append(tot.iou)
    arr = np.array([[u.tp, u.fp, u.fn] for u in units], float)
    s = arr[idx].sum(axis=1)
    vec = s[:, 0] / (s[:, 0] + s[:, 1] + s[:, 2])
    assert np.allclose(manual, vec)


def test_macro_skips_nan_units():
    assert math.isclose(macro_average([Counts(5, 0, 5, 0), Counts(0, 0, 0, 9)], "iou"), 0.5)


# ---------- baseline ----------
def test_otsu_splits_bimodal_data():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(-10, 1, 5000), rng.normal(0, 1, 5000)])
    assert -8 < otsu_threshold(v) < -2


def test_baseline_finds_dark_change_and_ignores_unchanged():
    before = np.full((2, 20, 20), -10.0, np.float32)
    after = before.copy()
    after[0, 5:15, 5:15] = -22.0                    # 12 dB drop
    pred = logratio_baseline(before, after, np.zeros((20, 20), bool), "fixed", 5.0, min_pixels=4)
    assert pred[5:15, 5:15].all() and pred.sum() == 100


def test_baseline_otsu_guard_on_flood_free_scene():
    rng = np.random.default_rng(0)
    before = rng.normal(-10, 1, (2, 40, 40)).astype(np.float32)
    after = before + rng.normal(0, 0.5, (2, 40, 40)).astype(np.float32)
    assert logratio_baseline(before, after, np.zeros((40, 40), bool), "otsu").sum() == 0


def test_permanent_water_is_not_flagged_by_baseline():
    before = np.full((2, 20, 20), -10.0, np.float32); before[:, :, :4] = -22.0   # river
    after = before.copy(); after[0, 8:14, 8:14] = -22.0
    pred = logratio_baseline(before, after, np.zeros((20, 20), bool), "fixed", 5.0, min_pixels=4)
    assert not pred[:, :4].any() and pred[8:14, 8:14].all()


def test_remove_small_blobs():
    m = np.zeros((10, 10), bool); m[0, 0] = True; m[4:8, 4:8] = True
    out = remove_small_blobs(m, 5)
    assert not out[0, 0] and out[4:8, 4:8].all()


def test_fit_drop_threshold_recovers_the_true_drop_range():
    rng = np.random.default_rng(0)
    before = np.full((2, 30, 30), -10.0, np.float32)
    after = before.copy(); after[0, 10:20, 10:20] -= 8.0
    after += rng.normal(0, 0.3, after.shape).astype(np.float32)
    truth = np.zeros((30, 30), np.uint8); truth[10:20, 10:20] = 2
    d, iou = fit_drop_threshold([(before, after, np.zeros((30, 30), bool), truth)])
    assert 1.0 <= d <= 7.5 and iou > 0.95


# ---------- morphology ----------
def test_dilate_erode_basic():
    m = np.zeros((9, 9), bool); m[4, 4] = True
    assert dilate(m, 1).sum() == 9 and dilate(m, 2).sum() == 25
    big = np.zeros((9, 9), bool); big[2:7, 2:7] = True
    assert erode(big, 1).sum() == 9
    assert erode(np.ones((5, 5), bool), 2).all()    # image border does not erode


# ---------- error categories ----------
def test_error_categories_sum_to_total_errors_and_pick_right_cause():
    truth = np.zeros((20, 20), np.uint8)
    truth[5:15, 5:15] = 2                           # flood block
    truth[0, :] = 1                                 # a river row
    pred = np.zeros((20, 20), bool)
    pred[6:15, 5:15] = True                         # misses top row of flood (boundary FN)
    pred[0, 3] = True                               # FP on river
    pred[18, 18] = True                             # FP far away
    before = np.full((20, 20), -10.0); before[18, 18] = -25.0   # dark before => shadow-like
    change = np.zeros((20, 20))
    res = categorize_errors(pred, truth, before, change, boundary_px=2)
    assert res["fp"]["permanent_water"] == 1 and res["fp"]["dark_before"] == 1
    assert sum(res["fp"].values()) == 2
    assert res["fn"]["boundary"] == 10 and sum(res["fn"].values()) == 10


def test_weak_change_fn_category():
    truth = np.zeros((30, 30), np.uint8); truth[5:25, 5:25] = 2
    pred = np.zeros((30, 30), bool)
    res = categorize_errors(pred, truth, np.full((30, 30), -10.0), np.full((30, 30), -1.0), boundary_px=2)
    assert res["fn"]["weak_change"] > 0 and sum(res["fn"].values()) == 400


# ---------- relaxed metrics ----------
def test_relaxed_forgives_one_pixel_shift_but_strict_does_not():
    truth = np.zeros((20, 20), np.uint8); truth[5:15, 5:15] = 2
    pred = np.zeros((20, 20), bool); pred[6:16, 5:15] = True       # shifted by 1 px
    strict = counts_from_arrays(pred, truth)
    rp, rr = relaxed_precision_recall(relaxed_counts(pred, truth, 1))
    assert strict.iou < 0.85 and rp == 1.0 and rr == 1.0


# ---------- padding ----------
def test_pad_and_crop_roundtrip():
    x = np.random.rand(6, 45, 70).astype(np.float32)
    padded, pad = pad_to_multiple(x, 32)
    assert padded.shape == (6, 64, 96) and pad == (19, 26)
    assert np.array_equal(crop_padding(padded, pad), x)
    same, p0 = pad_to_multiple(x[:, :32, :64], 32)
    assert p0 == (0, 0)


# ---------- protocol ----------
def test_protocol_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        freeze_protocol(Path(tmp) / "p.json", 0.37, "f1", 0.8, ["b", "a"], "best.pt")
        p = load_protocol(Path(tmp) / "p.json")
        assert p["threshold"] == 0.37 and p["val_events"] == ["a", "b"]


# ---------- evaluate.py ----------
def _scenes(tmp, n_events=3):
    files = make_synthetic_events(Path(tmp) / "syn", n_events=n_events, patches_per_event=3, size=32)
    return list(iter_scenes(files, READER))


def test_threshold_is_required_keyword():
    with tempfile.TemporaryDirectory() as tmp:
        scenes = _scenes(tmp)
        try:
            evaluate_group("x", scenes, lambda s: np.zeros(s.truth.shape), kind="model")
        except TypeError:
            return
        raise AssertionError("evaluate_group must refuse to run without an explicit threshold")


def test_perfect_predictor_scores_one_and_random_scores_low():
    with tempfile.TemporaryDirectory() as tmp:
        scenes = _scenes(tmp)
        perfect = lambda s: (s.truth == 2).astype(np.float32)
        r, _ = evaluate_group("p", scenes, perfect, threshold=0.5)
        assert math.isclose(r.counts.iou, 1.0) and r.n_events == 3 and r.n_scenes == 9
        rng = np.random.default_rng(0)
        r2, _ = evaluate_group("r", scenes, lambda s: rng.random(s.truth.shape).astype(np.float32), threshold=0.5)
        assert r2.counts.iou < 0.5


def test_invalid_pixels_never_graded():
    with tempfile.TemporaryDirectory() as tmp:
        scenes = _scenes(tmp)
        s = scenes[0]
        assert (s.truth[s.invalid] == 255).all() and s.invalid.any()
        r, _ = evaluate_group("p", [s], lambda sc: np.ones(sc.truth.shape, np.float32), threshold=0.5)
        total = r.counts.tp + r.counts.fp + r.counts.fn + r.counts.tn
        assert total == int((s.truth != 255).sum())


def test_select_threshold_finds_good_threshold_on_noisy_oracle():
    with tempfile.TemporaryDirectory() as tmp:
        scenes = _scenes(tmp)
        thr, f1, _ = select_threshold(scenes, noisy_oracle_predictor(lambda e: 0.2))
        assert 0.2 < thr < 0.8 and f1 > 0.8


def test_oracle_gap_flags_distribution_shift():
    with tempfile.TemporaryDirectory() as tmp:
        scenes = _scenes(tmp)
        # a model whose probabilities are shifted down: frozen thr 0.5 is far from the best one
        shifted = lambda s: np.clip((s.truth == 2) * 0.3 + 0.0, 0, 1).astype(np.float32)
        r, _ = evaluate_group("h", scenes, shifted, threshold=0.5)
        assert r.oracle["f1"] - r.counts.f1 > 0.5
        assert "distribution shift" in results_markdown([r])


def test_worst_examples_are_bounded_and_sorted():
    with tempfile.TemporaryDirectory() as tmp:
        scenes = _scenes(tmp, n_events=4)
        _, ex = evaluate_group("w", scenes, noisy_oracle_predictor(lambda e: 0.3), threshold=0.5, n_examples=2)
        assert len(ex.worst) <= 2 and ex.worst[0][0] <= ex.worst[-1][0]


def test_build_groups_leak_guard():
    splits = {"train": ["/d/ev1/a.npz"], "val": ["/d/ev2/a.npz"], "test": ["/d/ev3/a.npz", "/d/ev4/a.npz"]}
    g = build_groups(splits, ["ev4"])
    assert [event_of(f) for f in g["himalayan"]] == ["ev4"] and [event_of(f) for f in g["test"]] == ["ev3"]
    for bad in (["ev1"], ["ev2"], ["nope"]):
        try:
            build_groups(splits, bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")


def test_results_markdown_warns_on_few_events_once_per_group():
    with tempfile.TemporaryDirectory() as tmp:
        scenes = _scenes(tmp, n_events=2)
        r, _ = evaluate_group("g", scenes, lambda s: (s.truth == 2).astype(np.float32), threshold=0.5)
        b, _ = evaluate_group("g", scenes, lambda s: (s.truth == 2).astype(np.float32),
                              threshold=0.5, kind="baseline-otsu")
        assert results_markdown([r, b]).count("only 2 event(s)") == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
