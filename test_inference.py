"""Tests for Part 4 (numpy only, no PyTorch, no rasterio).
    python test_inference.py        or        pytest test_inference.py
Repo location: tests/test_inference.py
"""
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "part1_code", _HERE.parent / "part3_code"):
    sys.path.insert(0, str(_p))

from dataset import Normalizer, build_input, make_raw_patch  # noqa: E402
from evaluate import Scene  # noqa: E402
from geo_io import (GridMismatchError, RasterMeta, check_same_grid, make_npz_raster,  # noqa: E402
                    read_raster, validate_outputs, write_raster)
from inference import (FloodPredictor, detect_unit, fake_model_fn, fill_small_holes,  # noqa: E402
                       make_synthetic_scene, postprocess, sanity_warnings)
from pack_model import (PackageError, package_model, read_package, render_model_card,  # noqa: E402
                        sha256_file, verify_package)
from tiling import (blend_window, pad_to_min, predict_tiled, safe_call, tile_starts,  # noqa: E402
                    with_tta)

IDENT = Normalizer(np.zeros(6, np.float32), np.ones(6, np.float32), {})


def _predictor(**kw):
    return FloodPredictor(fake_model_fn, IDENT, **kw)


def _meta(h, w):
    return RasterMeta(w, h, "EPSG:32645", (10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0), 10.0)


# ---------- tiling ----------
def test_tile_starts_cover_axis_and_last_snaps_to_edge():
    for length in (224, 225, 300, 450, 1000, 5000):
        s = tile_starts(length, 224, 192)
        assert s[0] == 0 and s[-1] == length - 224
        covered = np.zeros(length, bool)
        for st in s:
            covered[st:st + 224] = True
        assert covered.all(), length
    assert tile_starts(100, 224, 192) == [0]


def test_blend_window_properties():
    w = blend_window(224, 32)
    assert w.shape == (224, 224) and w.min() > 0 and w.max() == 1.0
    assert w[112, 112] == 1.0 and np.allclose(w, w.T) and np.allclose(w, w[::-1, ::-1])
    assert w[0, 0] < w[16, 16] < w[40, 40] == 1.0
    assert (blend_window(64, 0) == 1).all()


def _per_pixel(batch):
    return fake_model_fn(batch)


def test_tiled_prediction_equals_whole_image_for_per_pixel_model():
    rng = np.random.default_rng(0)
    for h, w in ((224, 224), (230, 300), (450, 331), (500, 500)):
        x = rng.normal(-10, 4, (6, h, w)).astype(np.float32)
        direct = fake_model_fn(x[None])[0]
        for overlap, bs in ((32, 3), (0, 1), (64, 8)):
            got = predict_tiled(lambda y0, x0: x[:, y0:y0 + 224, x0:x0 + 224], _per_pixel,
                                h, w, 3, 224, overlap, bs)
            assert np.allclose(got, direct, atol=1e-5), (h, w, overlap, bs)


def test_constant_model_gives_constant_output():
    got = predict_tiled(lambda y, x: np.zeros((6, 224, 224), np.float32),
                        lambda b: np.ones((len(b), 3, 224, 224), np.float32), 400, 333, 3)
    assert np.allclose(got, 1.0)


def test_predict_tiled_input_validation():
    z = lambda y, x: np.zeros((6, 224, 224), np.float32)
    ok = lambda b: np.zeros((len(b), 3, 224, 224), np.float32)
    for args in (dict(h=100, w=300), dict(h=300, w=300, tile=200), dict(h=300, w=300, overlap=224)):
        a = dict(h=300, w=300, tile=224, overlap=32)
        a.update(args)
        try:
            predict_tiled(z, ok, a["h"], a["w"], 3, a["tile"], a["overlap"])
        except ValueError:
            continue
        raise AssertionError(args)
    try:
        predict_tiled(z, lambda b: np.zeros((len(b), 2, 224, 224)), 300, 300, 3)
    except ValueError:
        return
    raise AssertionError("wrong model output shape must be rejected")


def test_pad_to_min():
    x = np.random.rand(6, 50, 60).astype(np.float32)
    p, pad = pad_to_min(x, 224)
    assert p.shape == (6, 224, 224) and pad == (174, 164) and np.array_equal(p[:, :50, :60], x)
    same, pad0 = pad_to_min(np.zeros((6, 300, 300)), 224)
    assert pad0 == (0, 0)
    assert pad_to_min(np.zeros((1, 1)), 4)[0].shape == (4, 4)


def test_safe_call_splits_on_oom_and_matches_plain_result():
    calls = []

    def flaky(batch):
        calls.append(len(batch))
        if len(batch) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        return batch * 2

    b = np.arange(7 * 4, dtype=np.float32).reshape(7, 4)
    assert np.array_equal(safe_call(flaky, b), b * 2)
    assert max(calls[1:]) < 7


def test_safe_call_reraises_other_errors_and_unsplittable_batches():
    def bad(batch):
        raise RuntimeError("something else")
    for fn, batch in ((bad, np.zeros((4, 1))),
                      (lambda b: (_ for _ in ()).throw(RuntimeError("out of memory")), np.zeros((1, 1)))):
        try:
            safe_call(fn, batch)
        except RuntimeError:
            continue
        raise AssertionError("should re-raise")


def test_tta_equals_plain_for_flip_symmetric_model_but_not_for_asymmetric():
    x = np.random.default_rng(1).normal(-10, 3, (2, 6, 32, 32)).astype(np.float32)
    assert np.allclose(with_tta(fake_model_fn)(x), fake_model_fn(x), atol=1e-6)

    def shift_model(b):                       # not flip-equivariant: looks at the left neighbour
        p = fake_model_fn(b)
        return np.concatenate([p[..., :1], p[..., :-1]], axis=-1)
    assert not np.allclose(with_tta(shift_model)(x), shift_model(x), atol=1e-3)


# ---------- inference helpers ----------
def test_detect_unit():
    db = np.random.default_rng(0).normal(-12, 4, (2, 50, 50)).astype(np.float32)
    assert detect_unit(db) == "db" and detect_unit(10 ** (db / 10)) == "linear"
    try:
        detect_unit(np.full((2, 4, 4), np.nan))
    except ValueError:
        return
    raise AssertionError("all-NaN must raise")


def test_fill_small_holes():
    m = np.ones((30, 30), bool)
    m[10, 10] = False                         # tiny enclosed hole -> filled
    m[20:26, 20:26] = False                   # 36 px hole -> kept
    m[0, 5] = False                           # touches border -> kept
    out = fill_small_holes(m, 8)
    assert out[10, 10] and not out[22, 22] and not out[0, 5]


def test_postprocess_threshold_veto_nodata_and_specks():
    h = w = 40
    prob = np.zeros((3, h, w), np.float32)
    prob[0] = 1.0
    prob[2, 5:20, 5:20], prob[0, 5:20, 5:20] = 0.9, 0.1          # real flood block
    prob[2, 30, 30], prob[0, 30, 30] = 0.9, 0.1                  # 1-pixel speck
    prob[1, 25:35, 5:10], prob[2, 25:35, 5:10], prob[0, 25:35, 5:10] = 0.6, 0.35, 0.05   # river-ish
    invalid = np.zeros((h, w), bool); invalid[:, -2:] = True
    mask, conf = postprocess(prob, invalid, 0.3, min_blob_pixels=4, fill_hole_pixels=4)
    assert mask[5:20, 5:20].all()                                # flood kept
    assert mask[30, 30] == 0                                     # speck removed
    assert mask[25:35, 5:10].max() == 0                          # veto: river not flood although p_flood >= 0.3
    assert (mask[:, -2:] == 255).all() and (conf[:, -2:] == 0).all()
    assert set(np.unique(mask).tolist()) <= {0, 1, 255} and conf.dtype == np.float32
    assert conf[30, 30] > 0.8                                    # raw confidence still shows the speck


def test_sanity_warnings():
    assert sanity_warnings(0.0, 0.2, 0.1) == []
    assert any("orbit" in m for m in sanity_warnings(0.0, 6.0, 0.1))
    assert any("suspiciously" in m for m in sanity_warnings(0.0, 0.0, 0.8))
    assert any("no flood" in m for m in sanity_warnings(0.0, 0.0, 0.0))
    assert any("no data" in m for m in sanity_warnings(0.7, 0.0, 0.1))


def test_tilewise_input_equals_global_input_with_real_normalizer():
    before, after, _ = make_synthetic_scene(260, 300, seed=1)
    raw = make_raw_patch(before, after, np.zeros((260, 300), np.uint8))
    norm = Normalizer(np.arange(6, dtype=np.float32), np.linspace(1, 3, 6).astype(np.float32), {})
    whole = build_input(raw.before, raw.after, raw.invalid, norm)
    y, x = 30, 47
    tile = build_input(raw.before[:, y:y + 224, x:x + 224], raw.after[:, y:y + 224, x:x + 224],
                       raw.invalid[y:y + 224, x:x + 224], norm)
    assert np.array_equal(tile, whole[:, y:y + 224, x:x + 224])


# ---------- predict_arrays ----------
def test_predict_arrays_shapes_small_and_odd_images():
    for h, w in ((50, 60), (224, 224), (301, 457)):
        b, a, _ = make_synthetic_scene(h, w)
        p = _predictor().predict_arrays(b, a)
        assert p.prob.shape == (3, h, w) and p.invalid.shape == (h, w)
        assert np.allclose(p.prob.sum(axis=0), 1.0, atol=1e-4)


def test_linear_input_gives_same_prediction_as_db():
    b, a, _ = make_synthetic_scene(120, 140)
    pdb = _predictor().predict_arrays(b, a)
    plin = _predictor().predict_arrays(10 ** (b / 10), 10 ** (a / 10))
    assert pdb.unit == "db" and plin.unit == "linear"
    assert np.allclose(pdb.prob, plin.prob, atol=1e-3)


def test_predict_arrays_rejects_bad_shapes():
    b, a, _ = make_synthetic_scene(60, 60)
    for bad in ((b[0], a[0]), (b, a[:, :50]), (np.stack([b[0]] * 3), np.stack([a[0]] * 3))):
        try:
            _predictor().predict_arrays(*bad)
        except ValueError:
            continue
        raise AssertionError("bad shape must be rejected")


def test_threshold_resolution():
    p = _predictor(default_threshold=0.42, threshold_source="pkg")
    assert p.resolve_threshold(None) == (0.42, "pkg") and p.resolve_threshold(0.7) == (0.7, "argument")
    for bad in (0.0, 1.0, -0.2, 3):
        try:
            p.resolve_threshold(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


# ---------- files / contract ----------
def _write_pair(tmp, h=260, w=330, shift_db=0.0):
    before, after, truth = make_synthetic_scene(h, w)
    make_npz_raster(Path(tmp) / "before.npz", before, _meta(h, w))
    make_npz_raster(Path(tmp) / "after.npz", after + shift_db, _meta(h, w))
    return truth


def test_predict_files_honours_the_contract():
    with tempfile.TemporaryDirectory() as tmp:
        truth = _write_pair(tmp)
        res = _predictor().predict_files(Path(tmp) / "before.npz", Path(tmp) / "after.npz", Path(tmp) / "o")
        assert validate_outputs(res["mask"], res["confidence"], Path(tmp) / "before.npz") == []
        mask = read_raster(res["mask"]).data[0]
        assert ((mask == 1) == (truth == 2)).sum() > 0.99 * truth.size
        meta = json.loads(Path(res["meta"]).read_text())
        assert meta["threshold"] == 0.5 and meta["inputs"]["unit"] == "db"
        assert meta["summary"]["flood_area_km2"] == res["summary"]["flood_pixels"] * 100 / 1e6
        assert "Copernicus" in meta["attribution"][0] and meta["tiling"]["tile"] == 224


def test_orbit_mismatch_produces_warning():
    with tempfile.TemporaryDirectory() as tmp:
        _write_pair(tmp, shift_db=-6.0)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res = _predictor().predict_files(Path(tmp) / "before.npz", Path(tmp) / "after.npz", Path(tmp) / "o")
        assert any("orbit" in w for w in res["warnings"])


def test_grid_mismatch_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        before, after, _ = make_synthetic_scene(100, 100)
        make_npz_raster(Path(tmp) / "b.npz", before, _meta(100, 100))
        shifted = RasterMeta(100, 100, "EPSG:32645", (10.0, 0.0, 500010.0, 0.0, -10.0, 3000000.0), 10.0)
        make_npz_raster(Path(tmp) / "a.npz", after, shifted)
        try:
            _predictor().predict_files(Path(tmp) / "b.npz", Path(tmp) / "a.npz", Path(tmp) / "o")
        except GridMismatchError as e:
            assert "transform" in str(e)
            return
        raise AssertionError("shifted grid must be refused")


def test_validate_outputs_detects_contract_violations():
    with tempfile.TemporaryDirectory() as tmp:
        _write_pair(tmp, 100, 100)
        ref, meta = Path(tmp) / "before.npz", _meta(100, 100)
        bad = np.full((100, 100), 7, np.uint8)
        write_raster(Path(tmp) / "m.npz", bad, meta, np.uint8, 255)
        write_raster(Path(tmp) / "c.npz", np.full((100, 100), 1.5, np.float32), meta, np.float32)
        probs = validate_outputs(Path(tmp) / "m.npz", Path(tmp) / "c.npz", ref)
        assert any("unexpected values" in p for p in probs) and any("outside [0,1]" in p for p in probs)
        try:
            write_raster(Path(tmp) / "x.npz", np.zeros((5, 5)), meta, np.uint8)
        except ValueError:
            return
        raise AssertionError("wrong shape must be rejected")


def test_as_scene_predictor_for_part3():
    b, a, truth = make_synthetic_scene(64, 64)
    inv = np.isnan(b).any(axis=0)
    t = truth.copy(); t[inv] = 255
    scene = Scene("s", "e", np.nan_to_num(b, nan=-30.0), np.nan_to_num(a, nan=-30.0), t, inv)
    prob = _predictor().as_scene_predictor()(scene)
    assert prob.shape == (64, 64) and prob[inv].max() == 0.0 and prob[t == 2].mean() > 0.8


# ---------- packaging ----------
def test_sha256_known_value():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x"; p.write_bytes(b"abc")
        assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def _fake_results():
    return [{"name": "test", "kind": "model", "counts": {"precision": 0.9, "recall": 0.8, "iou": 0.73, "f1": 0.85},
             "ci_iou": {"low": 0.7, "high": 0.76}, "n_events": 3,
             "errors": {"fp": {"boundary": 8, "other": 2}, "fn": {"weak_change": 5, "boundary": 5}}},
            {"name": "himalayan", "kind": "model", "counts": {"precision": None, "recall": 0.5, "iou": 0.4, "f1": 0.5},
             "ci_iou": {"low": None, "high": None}, "n_events": 1,
             "errors": {"fp": {"other": 0}, "fn": {"other": 0}}}]


def test_package_build_verify_tamper_and_model_card():
    with tempfile.TemporaryDirectory() as tmp:
        ck = Path(tmp) / "c.pt"; ck.write_bytes(b"weights")
        proto = Path(tmp) / "p.json"; proto.write_text(json.dumps({"threshold": 0.41, "selection_metric": "f1"}))
        res = Path(tmp) / "r.json"; res.write_text(json.dumps(_fake_results()))
        pkg = package_model(ck, proto, Path(tmp) / "pkg", res,
                            meta={"config": {"encoder": "resnet34", "in_channels": 6, "num_classes": 3},
                                  "epoch": 7, "norm": {"meta": {"n_files": 99}}})
        assert verify_package(pkg) == [] and read_package(pkg)["threshold"] == 0.41
        card = (pkg / "model_card.md").read_text()
        assert "0.410" in card and "0.730 [0.700, 0.760]" in card and "n/a" in card     # numbers copied by code
        assert "Contains modified Copernicus Sentinel data 2026." in card and "99 patches" in card
        assert "mostly `boundary`" in card
        (pkg / "best.pt").write_bytes(b"other weights")
        assert any("SHA-256" in p for p in verify_package(pkg))
        try:
            read_package(pkg)
        except PackageError:
            (pkg / "best.pt").unlink()
            assert any("missing" in p for p in verify_package(pkg))
            return
        raise AssertionError("tampered package must be refused")


def test_model_card_without_results():
    card = render_model_card({"config": {}, "norm": {}}, {"threshold": 0.5}, None, "ab" * 32)
    assert "No results.json" in card


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
