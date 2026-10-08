"""Tests for train_utils.py (no PyTorch needed).
    python test_train_utils.py      or      pytest test_train_utils.py
Repo location: tests/test_train_utils.py
"""
import math
import tempfile
from pathlib import Path

import numpy as np

from train_utils import (BestTracker, ConfusionMatrix, TrainConfig, append_experiment_row,
                         config_diff, experiments_markdown, load_config, lr_factor,
                         make_checkpoint, masked_dice_numpy)


# ---------- config ----------
def test_defaults_and_overrides_are_typed():
    cfg = load_config(None, ["lr=3e-4", "epochs=5", "amp=false", "encoder_weights=none"])
    assert cfg.lr == 3e-4 and cfg.epochs == 5 and cfg.amp is False
    assert cfg.encoder_weights is None and isinstance(cfg.epochs, int)


def test_typo_in_override_is_rejected_with_hint():
    try:
        load_config(None, ["lrr=0.1"])
    except ValueError as e:
        assert "did you mean 'lr'" in str(e).lower()
    else:
        raise AssertionError("typo should raise")


def test_typo_in_yaml_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.yaml"
        p.write_text("batchsize: 4\n")
        try:
            load_config(str(p))
        except ValueError as e:
            assert "batch_size" in str(e)
        else:
            raise AssertionError("typo should raise")


def test_yaml_values_load():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.yaml"
        p.write_text("batch_size: 4\nencoder_weights: null\n")
        cfg = load_config(str(p), ["epochs=2"])
        assert cfg.batch_size == 4 and cfg.encoder_weights is None and cfg.epochs == 2


def test_config_diff_lists_only_changes():
    cfg = load_config(None, ["lr=0.01", "run_name=x"])
    assert config_diff(cfg) == {"lr": 0.01}


# ---------- LR schedule ----------
def test_lr_factor_shape():
    total, warm = 100, 10
    assert math.isclose(lr_factor(0, total, warm), 0.1)          # never exactly 0
    assert math.isclose(lr_factor(9, total, warm), 1.0)          # warmup done
    assert math.isclose(lr_factor(10, total, warm), 1.0)         # cosine starts at peak
    assert lr_factor(55, total, warm) < lr_factor(20, total, warm)   # decaying
    assert math.isclose(lr_factor(100, total, warm), 0.01)       # ends at min_factor
    assert all(lr_factor(s, total, warm) > 0 for s in range(0, 150))  # never 0, never crashes past end


def test_lr_factor_without_warmup():
    assert math.isclose(lr_factor(0, 50, 0), 1.0)


# ---------- confusion matrix ----------
def test_confusion_matrix_known_example():
    # truth:  flood flood flood none none ignored
    # pred :  flood flood none  none flood flood
    target = np.array([2, 2, 2, 0, 0, 255])
    pred = np.array([2, 2, 0, 0, 2, 2])
    cm = ConfusionMatrix(3)
    cm.update(pred, target)
    f = cm.class_metrics(2)                 # tp=2 fp=1 fn=1
    assert math.isclose(f["precision"], 2 / 3) and math.isclose(f["recall"], 2 / 3)
    assert math.isclose(f["iou"], 2 / 4) and math.isclose(f["f1"], 4 / 6)
    assert cm.mat.sum() == 5                # the ignored pixel is not counted


def test_dataset_level_not_batch_level():
    cm = ConfusionMatrix(3)
    cm.update(np.array([2, 2]), np.array([2, 0]))   # batch 1: tp=1 fp=1
    cm.update(np.array([0, 0]), np.array([0, 0]))   # batch 2: no flood anywhere
    assert math.isclose(cm.class_metrics(2)["iou"], 0.5)


def test_absent_class_is_nan_not_zero_or_one():
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 0]), np.array([0, 0]))   # perfect, but no flood at all
    s = cm.summary(2)
    assert math.isnan(s["flood_iou"]) and math.isnan(s["flood_f1"])
    assert math.isclose(s["miou"], 1.0)             # nanmean skips the absent classes


def test_false_alarm_on_absent_class_is_zero_iou():
    cm = ConfusionMatrix(3)
    cm.update(np.array([2, 2]), np.array([0, 0]))   # predicts flood where there is none
    assert cm.class_metrics(2)["iou"] == 0.0


def test_confusion_matrix_works_on_2d_batches():
    cm = ConfusionMatrix(3)
    cm.update(np.random.randint(0, 3, (2, 8, 8)), np.random.randint(0, 3, (2, 8, 8)))
    assert cm.mat.sum() == 128


# ---------- Dice reference ----------
def _onehot_probs(target, c=3):
    t = np.where(target == 255, 0, target)
    return np.moveaxis(np.eye(c)[t], -1, 1)


def test_dice_perfect_prediction_is_near_zero():
    target = np.random.default_rng(0).integers(0, 3, (2, 8, 8))
    assert masked_dice_numpy(_onehot_probs(target), target) < 1e-9


def test_dice_wrong_prediction_is_large():
    target = np.full((1, 8, 8), 2)
    wrong = np.full((1, 8, 8), 0)
    assert masked_dice_numpy(_onehot_probs(wrong), target) > 0.8


def test_dice_ignores_invalid_pixels():
    target = np.random.default_rng(1).integers(0, 3, (1, 8, 8))
    probs = _onehot_probs(target)
    target_masked = target.copy()
    target_masked[0, :2, :] = 255
    probs_garbage = probs.copy()
    probs_garbage[:, :, :2, :] = 1.0 / 3            # garbage exactly where pixels are ignored
    assert abs(masked_dice_numpy(probs, target_masked)
               - masked_dice_numpy(probs_garbage, target_masked)) < 1e-9


# ---------- tracker ----------
def test_tracker_improves_then_stops():
    t = BestTracker(patience=2)
    assert t.update(0.3, 0) == (True, False)
    assert t.update(0.5, 1) == (True, False)
    assert t.update(0.4, 2) == (False, False)
    assert t.update(0.45, 3) == (False, True)
    assert t.best == 0.5 and t.best_epoch == 1


def test_tracker_nan_never_best_and_no_crash():
    t = BestTracker(patience=0)
    for e in range(3):
        assert t.update(float("nan"), e) == (False, False)
    assert t.update(0.2, 3) == (True, False)


def test_tracker_ignores_tiny_gains():
    t = BestTracker(patience=5, min_delta=0.01)
    t.update(0.50, 0)
    assert t.update(0.505, 1)[0] is False


# ---------- checkpoint + experiment log ----------
def test_checkpoint_is_tensors_plus_json_string():
    import json
    ck = make_checkpoint({"w": [1, 2]}, TrainConfig(), {"mean": [0.0], "std": [1.0], "meta": {}},
                         {"flood_iou": float("nan"), "miou": 0.5}, 3, {0: "a"})
    assert set(ck) == {"model_state", "meta_json"} and isinstance(ck["meta_json"], str)
    meta = json.loads(ck["meta_json"])
    assert meta["metrics"]["flood_iou"] is None and meta["epoch"] == 3
    assert meta["config"]["encoder"] == "resnet34"


def test_experiment_csv_roundtrip_and_markdown():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "exp.csv"
        append_experiment_row(p, {"run_name": "a", "changed": "{}", "val_flood_iou": "0.5",
                                  "val_flood_f1": "0.6", "best_epoch": 4})
        append_experiment_row(p, {"run_name": "b", "changed": '{"lr": 0.01}', "val_flood_iou": "0.4",
                                  "val_flood_f1": "0.5", "best_epoch": 2})
        md = experiments_markdown(p)
        assert md.count("\n") == 3 and "| b |" in md


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
