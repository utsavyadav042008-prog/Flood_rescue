"""Tests for Part 5 (numpy only; no PyTorch, rasterio or network).
    python test_part5.py        or        pytest test_part5.py
Repo location: tests/test_part5.py
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "part1_code", _HERE.parent / "part3_code", _HERE.parent / "part4_code"):
    sys.path.insert(0, str(_p))

import audit as A  # noqa: E402
import case_study as CS  # noqa: E402
from compare import (AGREE_FLOOD, OURS_ONLY, OUTSIDE, REF_ONLY, agreement_map, block_bootstrap_ci,  # noqa: E402
                     compare_masks, comparison_valid_mask, error_by_slope, shift_sensitivity)
from ems_reference import (DataRuleViolation, LockError, audit_no_reference_leak, load_reference,  # noqa: E402
                           lock_prediction, rasterize_geojson, verify_lock)
from geo_io import RasterMeta, make_npz_raster  # noqa: E402
from report_tools import (TemplateError, audit_attribution, build_attribution_doc, build_limitations,  # noqa: E402
                          build_qa, fill_template, flatten_case, flatten_results,
                          model_vs_baseline_statement, unfilled_placeholders)

META = RasterMeta(100, 80, "EPSG:32645", (10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0), 10.0)


def _px(c, r):
    return [500000.0 + 10 * c, 3000000.0 - 10 * r]


def _poly(c0, r0, c1, r1):
    return [_px(c0, r0), _px(c1, r0), _px(c1, r1), _px(c0, r1), _px(c0, r0)]


# ---------------- compare ----------------
def test_agreement_codes_and_valid_mask():
    pred = np.array([[1, 1, 0, 0, 1]], bool)
    ref = np.array([[1, 0, 1, 0, 1]], bool)
    valid = np.array([[1, 1, 1, 1, 0]], bool)
    a = agreement_map(pred, ref, valid)
    assert a.tolist() == [[AGREE_FLOOD, OURS_ONLY, REF_ONLY, 0, OUTSIDE]]
    mask = np.array([[1, 255, 0]], np.uint8)
    foot = np.array([[1, 1, 0]], bool)
    assert comparison_valid_mask(mask, foot).tolist() == [[True, False, False]]


def _band(h=60, w=80, x0=30, x1=45):
    m = np.zeros((h, w), bool)
    m[:, x0:x1] = True
    return m


def test_footprint_stops_unmapped_flood_counting_as_false_alarm():
    ref = _band(x0=30, x1=45)
    pred_mask = (_band(x0=30, x1=45) | _band(x0=60, x1=75)).astype(np.uint8)   # extra flood far right
    foot = np.zeros(ref.shape, bool); foot[:, :50] = True
    full, _ = compare_masks(pred_mask, ref)
    cut, _ = compare_masks(pred_mask, ref, footprint=foot)
    assert full.counts.precision < 0.6 and math.isclose(cut.counts.precision, 1.0)
    assert any("footprint" in n for n in full.notes) and not cut.notes


def test_block_bootstrap_properties():
    rng = np.random.default_rng(0)
    ref = _band(120, 160, 50, 100)
    pred = ref ^ (rng.random(ref.shape) < 0.03)
    valid = np.ones(ref.shape, bool)
    a = block_bootstrap_ci(pred, ref, valid, "iou", block=16, seed=1)
    b = block_bootstrap_ci(pred, ref, valid, "iou", block=16, seed=1)
    assert a == b and a["low"] <= a["point"] <= a["high"] and a["n_blocks"] > 4
    tiny = block_bootstrap_ci(np.ones((8, 8), bool), np.ones((8, 8), bool), np.ones((8, 8), bool), block=32)
    assert math.isnan(tiny["low"]) and tiny["n_blocks"] == 1


def test_block_bootstrap_wider_than_naive_for_correlated_errors():
    ref = _band(128, 128, 20, 90)
    pred = ref.copy(); pred[:, 20:26] = False           # one contiguous miss
    valid = np.ones(ref.shape, bool)
    coarse = block_bootstrap_ci(pred, ref, valid, "recall", block=64)
    fine = block_bootstrap_ci(pred, ref, valid, "recall", block=4)
    assert (coarse["high"] - coarse["low"]) > (fine["high"] - fine["low"])


def test_shift_sensitivity_finds_known_offset():
    pred = _band(60, 80, 30, 45)
    ref = _band(60, 80, 32, 47)                          # reference is 2 px to the right
    s = shift_sensitivity(pred, ref, np.ones(pred.shape, bool), 3)
    assert tuple(s["best_shift_dy_dx"]) == (0, 2) and math.isclose(s["iou_best"], 1.0)
    assert s["iou_at_zero"] < 0.85


def test_shift_sensitivity_does_not_wrap_around_edges():
    pred = np.zeros((20, 20), bool); pred[:, 0] = True
    ref = np.zeros((20, 20), bool); ref[:, -1] = True    # np.roll would make these match
    s = shift_sensitivity(pred, ref, np.ones((20, 20), bool), 2)
    assert (s["iou_best"] == 0.0) or math.isnan(s["iou_best"])


def test_error_by_slope_bins():
    h, w = 40, 40
    slope = np.zeros((h, w)); slope[:, 20:] = 35.0       # flat left, steep right
    ref = np.zeros((h, w), bool); ref[:, 5:15] = True; ref[:, 25:35] = True
    pred = ref.copy(); pred[:, 25:35] = False            # miss everything on steep ground
    rows = error_by_slope(pred, ref, np.ones((h, w), bool), slope)
    flat, steep = rows[0], rows[-1]
    assert math.isclose(flat["recall"], 1.0) and math.isclose(steep["recall"], 0.0)
    assert flat["pixels"] == h * 20 and steep["pixels"] == h * 20


def test_compare_masks_perfect_and_area():
    ref = _band()
    c, agree = compare_masks(ref.astype(np.uint8), ref, pixel_size_m=10.0)
    assert math.isclose(c.counts.iou, 1.0) and math.isclose(c.area_ours_km2, ref.sum() * 100 / 1e6)
    assert math.isclose(c.area_ratio, 1.0) and (agree == AGREE_FLOOD).sum() == ref.sum()


# ---------------- ems_reference ----------------
def test_rasterize_square_with_hole_is_exact():
    gj = {"type": "Feature", "geometry": {"type": "Polygon",
          "coordinates": [_poly(20, 10, 60, 50), _poly(30, 20, 40, 30)]}}
    m = rasterize_geojson(gj, META, "EPSG:32645")
    assert m.sum() == 40 * 40 - 10 * 10 and not m[25, 35] and m[15, 25]


def test_rasterize_nesting_overlap_and_partly_outside():
    f1 = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [_poly(10, 10, 30, 30)]}}
    f2 = {"type": "Feature", "geometry": {"type": "MultiPolygon",
          "coordinates": [[_poly(20, 20, 40, 40)], [_poly(90, 70, 130, 100)]]}}   # second part leaves the grid
    pt = {"type": "Feature", "geometry": {"type": "Point", "coordinates": _px(5, 5)}}
    m = rasterize_geojson({"type": "FeatureCollection", "features": [f1, f2, pt]}, META, "EPSG:32645")
    assert m.sum() == 400 + 400 - 100 + 10 * 10                        # union, plus the clipped 10x10 corner
    assert not m[5, 5]


def test_rasterize_crs_mismatch_needs_pyproj_or_works():
    gj = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [_poly(1, 1, 5, 5)]}}
    try:
        import pyproj  # noqa: F401
        return                                        # real reprojection is not unit-tested here
    except ImportError:
        pass
    try:
        rasterize_geojson(gj, META, "EPSG:4326")
    except ImportError as e:
        assert "pyproj" in str(e)
        return
    raise AssertionError("expected ImportError")


def test_load_reference_union_of_geojson_and_raster_and_grid_check():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "a.geojson").write_text(json.dumps({"type": "Feature", "geometry": {"type": "Polygon",
                                                   "coordinates": [_poly(10, 10, 20, 20)]}}))
        r = np.zeros((1, 80, 100), np.float32); r[0, 50:60, 50:60] = 1
        make_npz_raster(tmp / "b.npz", r, META)
        m = load_reference([tmp / "a.geojson", tmp / "b.npz"], META, "EPSG:32645")
        assert m.sum() == 200
        other = RasterMeta(50, 40, META.crs, META.transform, 10.0)
        make_npz_raster(tmp / "c.npz", np.zeros((1, 40, 50), np.float32), other)
        try:
            load_reference([tmp / "c.npz"], META)
        except ValueError:
            return
        raise AssertionError("grid mismatch must be refused")


def test_leak_audit_by_name_and_by_content():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref = tmp / "emsr927_flood.geojson"; ref.write_text("{}")
        disguised = tmp / "before.tif"; disguised.write_text("{}")
        clean = tmp / "clean.tif"; clean.write_text("different")
        v = audit_no_reference_leak([clean], [ref], [])
        assert v == []
        assert any("emsr" in x for x in audit_no_reference_leak([ref], [], []))
        assert any("byte-identical" in x for x in audit_no_reference_leak([disguised], [ref], []))
        assert any("unosat" in x for x in audit_no_reference_leak([clean], [], ["/data/UNOSAT_flood/a.npz"]))


def test_prediction_lock_detects_tampering():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        f = tmp / "mask.npz"; f.write_bytes(b"prediction")
        lock_prediction({"mask": str(f)}, tmp / "lock.json", 0.4, "package")
        assert verify_lock(tmp / "lock.json")["threshold"] == 0.4
        f.write_bytes(b"edited after seeing the reference")
        try:
            verify_lock(tmp / "lock.json")
        except LockError as e:
            assert "changed after the lock" in str(e)
            return
        raise AssertionError("tampering must be detected")


# ---------------- report_tools ----------------
def _row(group, kind, iou, n=3, low=None, high=None):
    return {"name": group, "kind": kind, "threshold": 0.5,
            "counts": {"precision": 0.8, "recall": 0.7, "iou": iou, "f1": 0.75},
            "ci_iou": {"low": iou - 0.05 if low is None else low, "high": iou + 0.05 if high is None else high},
            "n_events": n, "n_scenes": 6, "relaxed_precision": 0.9, "relaxed_recall": 0.9, "average_precision": 0.8,
            "oracle": {"threshold": 0.3, "f1": 0.9},
            "errors": {"fp": {"boundary": 3, "other": 1}, "fn": {"weak_change": 9, "boundary": 1}}}


RESULTS = [_row("validation*", "model", 0.9), _row("test", "model", 0.8), _row("test", "baseline-fixed", 0.7),
           _row("himalayan", "model", 0.6, n=2), _row("himalayan", "baseline-otsu", 0.5, n=2)]


def test_flatten_and_fill_template():
    flat = flatten_results(RESULTS)
    assert flat["validation.model.iou"] == 0.9 and flat["test.baseline_fixed.iou"] == 0.7
    assert flat["himalayan.model.fn_top_cause"] == "weak_change" and math.isclose(flat["himalayan.model.fn_top_share"], 0.9)
    assert fill_template("IoU {{test.model.iou:.2f}} on {{test.model.n_events}} events", flat) == "IoU 0.80 on 3 events"
    assert fill_template("{{x}}", {"x": None}) == "n/a" and fill_template("{{x}}", {"x": float("nan")}) == "n/a"
    try:
        fill_template("{{test.model.iuo:.2f}}", flat)
    except TemplateError as e:
        assert "test.model.iou" in str(e)           # "did you mean"
    else:
        raise AssertionError("typo must raise")
    assert "[not available yet]" in fill_template("{{nope}}", flat, strict=False)
    assert unfilled_placeholders("a {{b.c}} d {{e:.1f}}") == ["b.c", "e"]


def test_model_vs_baseline_statement_three_outcomes():
    clear = [_row("g", "model", 0.8, low=0.75, high=0.85), _row("g", "baseline-fixed", 0.6, low=0.55, high=0.65)]
    overlap = [_row("g", "model", 0.62, low=0.45, high=0.8), _row("g", "baseline-fixed", 0.6, low=0.4, high=0.78)]
    worse = [_row("g", "model", 0.5), _row("g", "baseline-otsu", 0.6)]
    assert "clearly better" in model_vs_baseline_statement(clear, "g")
    assert "not clearly distinguishable" in model_vs_baseline_statement(overlap, "g")
    assert "does NOT beat" in model_vs_baseline_statement(worse, "g")
    assert "No model-vs-baseline" in model_vs_baseline_statement(clear, "zzz")


def _case(best=0.9, zero=0.7, footprint=True):
    return {"counts": {"iou": zero, "precision": 0.8, "recall": 0.85, "f1": 0.82}, "iou_ci": {"low": 0.65, "high": 0.75},
            "relaxed_precision": 0.9, "relaxed_recall": 0.9, "area_ours_km2": 5.0, "area_ref_km2": 4.0, "area_ratio": 1.25,
            "shift": {"iou_at_zero": zero, "iou_best": best, "best_shift_dy_dx": [0, 2]}, "threshold": 0.45,
            "locked_at": "t", "footprint_used": footprint}


def test_limitations_generated_from_numbers_and_tagged():
    t = build_limitations(RESULTS, _case(footprint=False), {"warnings": ["median VV change is +4.0 dB"]})
    assert "[measured]" in t and "[by design]" in t and "minutes ahead" in t
    assert "IoU 0.60" in t and "only 2 event(s)" in t                    # numbers + few-events warning
    assert "miscalibrated" in t                                          # oracle gap 0.15 > 0.05
    assert "weak_change" in t and "No reference footprint" in t and "1.25 times" in t
    assert "suggests part of the disagreement is a georeferencing offset" in t
    assert "median VV change is +4.0 dB" in t


def test_limitations_without_himalayan_or_shift_gain():
    t = build_limitations([_row("test", "model", 0.8), _row("test", "baseline-fixed", 0.7)], _case(best=0.7, zero=0.7))
    assert "UNVERIFIED" in t and "does not improve the IoU" in t and "georeferencing offset" not in t


def test_attribution_doc_and_audit():
    doc = build_attribution_doc(sen1floods11=True)
    for s in ("Contains modified Copernicus Sentinel data 2026.", "© OpenStreetMap contributors.", "Bountos", "Bonafilia",
              "Copernicus Emergency Management Service data"):
        assert s in doc
    with tempfile.TemporaryDirectory() as tmp:
        ok, bad = Path(tmp) / "ok.md", Path(tmp) / "bad.md"
        ok.write_text(doc); bad.write_text("nothing here")
        res = audit_attribution([ok, bad, Path(tmp) / "missing.md"])
        assert res[str(ok)] == [] and len(res[str(bad)]) == 3 and len(res[str(Path(tmp) / "missing.md")]) == 3


def test_qa_filled_and_flags_missing():
    q = build_qa(RESULTS, _case())
    assert "IoU is 0.60 (95% CI 0.55 to 0.65)" in q and "0.45" in q and "does NOT beat" not in q
    q2 = build_qa([_row("test", "model", 0.8)])
    assert "[not available yet]" in q2 and "{{" not in q2


def test_flatten_case_keys():
    f = flatten_case(_case())
    assert f["case.iou"] == 0.7 and f["case.best_shift"] == "[0, 2]" and f["case.threshold"] == 0.45


# ---------------- audit ----------------
def test_audit_limitations_and_report_and_leak():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        good = tmp / "lim.md"; good.write_text(build_limitations(RESULTS))
        assert A.check_limitations(good).status == A.PASS
        thin = tmp / "thin.md"; thin.write_text("We are great.")
        assert A.check_limitations(thin).status == A.FAIL and A.check_limitations(None).status == A.FAIL
        rep = tmp / "r.md"; rep.write_text("IoU {{test.model.iou:.2f}}")
        assert A.check_report(rep)[0].status == A.FAIL
        rep.write_text("IoU 0.80")
        assert A.check_report(rep)[0].status == A.PASS
        splits = tmp / "s.json"
        splits.write_text(json.dumps({"files": {"train": ["ev1/a.npz", "EMSR927/x.npz"]}}))
        assert A.check_no_reference_leak(str(splits), None).status == A.FAIL
        splits.write_text(json.dumps({"files": {"train": ["ev1/a.npz"]}}))
        assert A.check_no_reference_leak(str(splits), None).status == A.PASS


def test_audit_flags_images_not_made_by_our_code():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "figures_manifest.json").write_text(json.dumps({"figures": {}}))
        (tmp / "photo.png").write_bytes(b"x")
        out = A.check_figures(str(tmp))
        assert out[0].status == A.WARN and "photo.png" in out[0].detail and "victims" in out[0].detail


def test_end_to_end_demo_then_audit_then_tamper():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        CS.run_demo(str(tmp / "d"))
        case_dir = tmp / "d" / "case"
        readme = tmp / "README.md"; readme.write_text("pip install -r requirements.txt\npython x.py --package model_package\n"
                          + (case_dir / "docs" / "attribution.md").read_text())
        report = tmp / "report.md"
        report.write_text((case_dir / "docs" / "attribution.md").read_text() + "\nIoU 0.80\n")
        args = dict(results=str(tmp / "d" / "results.json"), case=str(case_dir / "comparison.json"), readme=str(readme),
                    report=str(report), limitations=str(case_dir / "docs" / "limitations.md"),
                    attribution=str(case_dir / "docs" / "attribution.md"), figures=str(case_dir / "figures"))
        checks = A.run_audit(args)
        fails = [c for c in checks if c.status == A.FAIL]
        assert not fails, [(c.name, c.detail) for c in fails]
        assert sum(c.status == A.MANUAL for c in checks) == len(A.MANUAL_ITEMS)
        assert "passed" in A.render(checks)
        # editing the prediction after the lock must be caught
        mask = case_dir / "prediction" / "flood_mask.npz"
        mask.write_bytes(mask.read_bytes() + b"0")
        again = A.run_audit(args)
        assert any(c.status == A.FAIL and "locked" in c.name for c in again)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
