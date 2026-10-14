"""Tests for dataset.py.  Run either way:
    python test_dataset.py          (no pytest needed)
    pytest test_dataset.py
Repo location: tests/test_dataset.py   (adjust the import to `from models.dataset import ...`)
"""
import functools
import tempfile
from pathlib import Path

import numpy as np
import sys  # noqa: E402
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "models"))

from dataset import (
    AugmentConfig, FloodDataset, IGNORE_INDEX, N_CHANNELS, Normalizer, RawPatch,
    augment_patch, build_input, class_weights_from_counts, make_raw_patch,
    make_synthetic_events, read_patch_npz, remap_labels, scan_dataset,
    split_by_event, to_db, verify_no_leakage,
)

READER = functools.partial(read_patch_npz, unit="db")


def _synthetic(tmp):
    return make_synthetic_events(Path(tmp) / "syn", n_events=6, patches_per_event=4, size=32)


def test_to_db_known_values():
    assert np.isclose(to_db(np.array([1.0]))[0], 0.0)
    assert np.isclose(to_db(np.array([0.1]))[0], -10.0)


def test_unknown_label_codes_become_ignore():
    out = remap_labels(np.array([[0, 1, 2, 3, 17, -1]]))
    assert out.tolist() == [[0, 1, 2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]]


def test_linear_zero_is_nodata_not_deep_water():
    before = np.full((2, 4, 4), 0.1, np.float32)
    after = before.copy()
    after[:, 0, 0] = 0.0                       # black border pixel
    p = make_raw_patch(before, after, np.zeros((4, 4), np.uint8), unit="linear")
    assert p.invalid[0, 0] and p.invalid.sum() == 1


def test_split_is_disjoint_deterministic_and_exact():
    with tempfile.TemporaryDirectory() as tmp:
        files = _synthetic(tmp)
        a = split_by_event(files, n_val=1, n_test=2, seed=1)
        b = split_by_event(files, n_val=1, n_test=2, seed=1)
        assert a == b                                           # deterministic
        verify_no_leakage(a)                                    # disjoint
        n_events = lambda s: len({Path(f).parent.name for f in a[s]})
        assert (n_events("train"), n_events("val"), n_events("test")) == (3, 1, 2)


def test_force_test_keeps_event_out_of_training():
    with tempfile.TemporaryDirectory() as tmp:
        files = _synthetic(tmp)
        s = split_by_event(files, n_val=1, n_test=1, force_test=["event_03"])
        assert all("event_03" in f for f in s["test"])
        assert not any("event_03" in f for f in s["train"] + s["val"])


def test_input_shape_and_invalid_pixels_are_neutral():
    with tempfile.TemporaryDirectory() as tmp:
        files = _synthetic(tmp)
        norm, _ = scan_dataset(files, READER, verbose=False)
        p = READER(files[0])
        x = build_input(p.before, p.after, p.invalid, norm)
        assert x.shape == (N_CHANNELS, 32, 32) and x.dtype == np.float32
        assert p.invalid.any() and (x[:, p.invalid] == 0).all()


def test_normalized_training_data_has_mean0_std1():
    with tempfile.TemporaryDirectory() as tmp:
        files = _synthetic(tmp)
        norm, counts = scan_dataset(files, READER, verbose=False)
        xs = []
        for f in files:
            p = READER(f)
            xs.append(build_input(p.before, p.after, p.invalid, norm)[:, ~p.invalid])
        xs = np.concatenate(xs, axis=1)
        assert np.allclose(xs.mean(axis=1), 0, atol=1e-3)
        assert np.allclose(xs.std(axis=1), 1, atol=1e-2)
        assert counts.sum() > 0


def test_stats_json_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        files = _synthetic(tmp)
        norm, _ = scan_dataset(files, READER, verbose=False)
        norm.save(Path(tmp) / "n.json")
        back = Normalizer.load(Path(tmp) / "n.json")
        assert np.allclose(back.mean, norm.mean) and back.meta["files_fingerprint"]


def test_augmentation_moves_image_and_label_together():
    # Make image encode the label so any mismatch in the transform shows up.
    rng = np.random.default_rng(0)
    label = rng.integers(0, 3, (16, 16)).astype(np.uint8)
    before = np.stack([label, label]).astype(np.float32) - 10
    p = RawPatch(before, before.copy(), label, np.zeros((16, 16), bool))
    cfg = AugmentConfig(noise_db=0.0)
    for seed in range(20):
        q = augment_patch(p, np.random.default_rng(seed), cfg)
        assert np.array_equal(q.before[0] + 10, q.label.astype(np.float32))
        assert all(a.flags["C_CONTIGUOUS"] for a in (q.before, q.after, q.label, q.invalid))


def test_same_sample_same_epoch_is_reproducible_and_epochs_differ():
    with tempfile.TemporaryDirectory() as tmp:
        files = _synthetic(tmp)
        norm, _ = scan_dataset(files, READER, verbose=False)
        ds = FloodDataset(files, norm, AugmentConfig(), READER, seed=7)
        x1 = np.asarray(ds[3][0]); x2 = np.asarray(ds[3][0])
        assert np.array_equal(x1, x2)
        ds.set_epoch(1)
        assert not np.array_equal(x1, np.asarray(ds[3][0]))


def test_class_weights_rare_class_gets_more_weight():
    w = class_weights_from_counts([9000, 900, 100])
    assert w[2] > w[1] > w[0]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
