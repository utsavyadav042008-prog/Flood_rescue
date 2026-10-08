"""
case_study.py  --  Part 5 of Teammate 2's workflow: CASE STUDY, LIMITATIONS & REPORT
====================================================================================

Repo location: models/case_study.py
Needs: compare.py, ems_reference.py, figures.py, report_tools.py, audit.py (this folder),
       Part 4 (inference, geo_io, pack_model), Part 3 (eval_metrics).

WHAT THIS FILE DOES
-------------------
Runs the Trishuli case study in the order the data rule requires, then produces
everything the report needs.

    before.tif, after.tif  (Teammate 1)             EMSR927 files  (CHECKING ONLY)
            │                                              │
   1 audit_no_reference_leak ──────────────────────────────┤  inputs must not be EMS data
   2 predict with the FROZEN threshold (package)           │
   3 lock_prediction  (hash + time)  ◄── sealed envelope   │  reference not opened yet
   4 ─────────────────────────────────────────────────────►  load_reference / footprint
   5 verify_lock  (prediction unchanged?)
   6 compare_masks  -> IoU, CI, shift, causes, slope table
   7 figures (with attribution) + comparison.json + case_summary.md
   8 limitations.md, attribution.md, qa_prep.md, demo_script.md  (numbers from files)

Then run  audit.py  on Day 13.

COMMANDS
    python case_study.py demo --out demo_case          # synthetic, proves the whole chain
    python case_study.py run --before b.tif --after a.tif --package model_package \
         --reference emsr927_flood.geojson [more layers] --footprint emsr927_aoi.geojson \
         --eval-results outputs/eval/results.json --out outputs/case
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "part3_code", _HERE.parent / "part4_code"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import figures as F  # noqa: E402
from compare import Comparison, compare_masks  # noqa: E402
from ems_reference import (audit_no_reference_leak, DataRuleViolation, load_footprint,  # noqa: E402
                           load_reference, lock_prediction, verify_lock)
from geo_io import RasterMeta, make_npz_raster, read_raster, write_raster  # noqa: E402
from report_tools import (build_attribution_doc, build_demo_script, build_limitations,  # noqa: E402
                          build_qa, flatten_case, flatten_results, model_vs_baseline_statement)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return _jsonable(o.item())
    if isinstance(o, float) and o != o:
        return None
    return o


def run_case_study(before_path, after_path, out_dir, predictor, reference_paths: Sequence,
                   footprint_path: Optional[str] = None, eval_results_path: Optional[str] = None,
                   slope_path: Optional[str] = None, ref_crs: Optional[str] = "EPSG:4326",
                   extra_listed_paths: Sequence = (), allow_manual_threshold: bool = False,
                   verbose: bool = True) -> Dict:
    """Run steps 1-8 of the module docstring.  `predictor` is a Part 4 FloodPredictor.

    Returns the comparison dictionary that is also written to comparison.json.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. data-rule tripwires (BEFORE anything is read from the reference)
    violations = audit_no_reference_leak([before_path, after_path], reference_paths, extra_listed_paths)
    if violations:
        raise DataRuleViolation("; ".join(violations))

    # 2. predict with the frozen threshold
    res = predictor.predict_files(before_path, after_path, out / "prediction", threshold=None)
    meta = json.loads(Path(res["meta"]).read_text())
    if meta["threshold_source"] == "argument" and not allow_manual_threshold:
        raise DataRuleViolation("the case study must use the frozen validation threshold, not a hand-picked one")

    # 3. seal the prediction BEFORE the reference is opened
    lock_path = out / "prediction_lock.json"
    lock = lock_prediction({"mask": res["mask"], "confidence": res["confidence"], "meta": res["meta"]},
                           lock_path, meta["threshold"], meta["threshold_source"])
    if verbose:
        print(f"prediction locked at {lock['locked_at']} (threshold {meta['threshold']:.3f}, {meta['threshold_source']})")

    # 4. only now touch the reference
    mask_r = read_raster(res["mask"])
    gmeta: RasterMeta = mask_r.meta
    ref = load_reference(reference_paths, gmeta, ref_crs)
    footprint = load_footprint(footprint_path, gmeta, ref_crs) if footprint_path else None

    # 5. the prediction must be exactly what was locked
    verify_lock(lock_path)

    # 6. compare
    rb, ra = read_raster(before_path), read_raster(after_path)
    slope = read_raster(slope_path).data[0] if slope_path else None
    comp, agree = compare_masks(mask_r.data[0].astype(np.uint8), ref, footprint=footprint,
                                before_vv=rb.data[0], after_vv=ra.data[0], slope_deg=slope,
                                pixel_size_m=gmeta.pixel_size_m)

    # 7. figures + files
    ext = Path(res["mask"]).suffix
    write_raster(out / f"agreement_map{ext}", agree, gmeta, np.uint8, nodata=255,
                 tags={"codes": "0 both dry, 1 both flood, 2 ours only, 3 reference only, 255 not compared"})
    man = F.FigureManifest(out / "figures" / "figures_manifest.json")
    conf = read_raster(res["confidence"]).data[0]
    F.save_figure(F.fig_before_after(rb.data[0], ra.data[0]), out / "figures" / "F1_before_after.png", man,
                  "Sentinel-1 VV before and after the event")
    F.save_figure(F.fig_overlay(ra.data[0], mask_r.data[0], conf), out / "figures" / "F2_prediction.png", man,
                  "Predicted flood and model confidence")
    F.save_figure(F.fig_agreement(agree, comp.counts.iou), out / "figures" / "F3_agreement_emsr927.png", man,
                  "Agreement with EMSR927", uses_ems=True)
    F.save_figure(F.fig_failures(ra.data[0], agree), out / "figures" / "F4_failures.png", man,
                  "Largest disagreements with EMSR927", uses_ems=True)
    F.save_figure(F.fig_shift(comp.shift), out / "figures" / "F5_shift_sensitivity.png", man,
                  "IoU vs shifted reference", uses_ems=True)

    case = _jsonable(asdict(comp))
    case["counts"] = _jsonable(comp.counts.as_dict())          # asdict() drops the metric properties
    case.update({"threshold": meta["threshold"], "threshold_source": meta["threshold_source"],
                 "locked_at": lock["locked_at"], "lock_path": str(lock_path),
                 "input_files": [str(before_path), str(after_path)],
                 "reference_files": [str(p) for p in reference_paths],
                 "inference_warnings": meta.get("warnings", []),
                 "date_compared": time.strftime("%Y-%m-%d %H:%M:%S")})
    (out / "comparison.json").write_text(json.dumps(case, indent=2))
    (out / "case_summary.md").write_text(_summary_md(case, comp))

    # 8. documents whose numbers come from files
    docs = out / "docs"
    docs.mkdir(exist_ok=True)
    results = json.loads(Path(eval_results_path).read_text()) if eval_results_path and Path(eval_results_path).exists() else []
    (docs / "limitations.md").write_text(build_limitations(results, case, meta))
    (docs / "attribution.md").write_text(build_attribution_doc())
    (docs / "qa_prep.md").write_text(build_qa(results, case))
    (docs / "demo_script.md").write_text(build_demo_script(results, case))
    (out / "case_values.json").write_text(json.dumps(_jsonable({**flatten_results(results), **flatten_case(case)}), indent=2))
    return case


def _summary_md(case: Dict, comp: Comparison) -> str:
    c = case["counts"]
    ci = case["iou_ci"]
    s = case["shift"]
    L = ["# Case study: comparison with EMSR927 (checking only)", "",
         f"- Prediction locked **{case['locked_at']}** with threshold **{case['threshold']:.3f}** "
         f"({case['threshold_source']}); the reference was opened afterwards.",
         f"- IoU **{c['iou']:.3f}** (block-bootstrap 95% interval {ci['low']:.3f} to {ci['high']:.3f}), "
         f"precision {c['precision']:.3f}, recall {c['recall']:.3f}, F1 {c['f1']:.3f}.",
         f"- Relaxed (±2 px) precision/recall: {case['relaxed_precision']:.3f} / {case['relaxed_recall']:.3f}.",
         f"- Unshifted IoU {s['iou_at_zero']:.3f}; best shifted IoU {s['iou_best']:.3f} at shift {s['best_shift_dy_dx']} "
         f"(diagnostic only, not our result).",
         f"- Compared pixels: {case['n_valid_pixels']:,}; reference footprint used: {case['footprint_used']}."]
    if case.get("area_ours_km2") is not None:
        L.append(f"- Flooded area: ours {case['area_ours_km2']:.2f} km², reference {case['area_ref_km2']:.2f} km².")
    for k in ("fp", "fn"):
        if case["error_causes"].get(k):
            L.append(f"- {'False alarms' if k == 'fp' else 'Missed flood'} by cause: {case['error_causes'][k]}")
    L += [f"- Note: {n}" for n in case.get("notes", [])]
    L += ["", "Contains modified Copernicus Sentinel data 2026.",
          f"Reference: {F.EMS_CREDIT} (EMSR927), used for checking only."]
    return "\n".join(L)


# =============================================================================
# DEMO
# =============================================================================
def run_demo(out_dir: str) -> None:
    """Synthetic end-to-end run: fake per-pixel model + a hand-made 'EMS' GeoJSON that is
    deliberately offset by 2 pixels and covers only part of the scene."""
    from inference import FloodPredictor, fake_model_fn, make_synthetic_scene
    from dataset import Normalizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    before, after, truth = make_synthetic_scene(300, 450)
    h, w = truth.shape
    meta = RasterMeta(w, h, "EPSG:32645", (10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0), 10.0)
    make_npz_raster(out / "before.npz", before, meta)
    make_npz_raster(out / "after.npz", after, meta)

    # fake "EMSR927": two flood bands either side of the river (permanent water excluded, as EMS does),
    # shifted +2 px in x, written in MAP coordinates; footprint = left 80 % of the scene
    yy = np.arange(0, h, 2)
    centre = w * 0.45 + 25 * np.sin(yy / 40.0) + 2
    px = lambda c, r: [500000.0 + 10.0 * c, 3000000.0 - 10.0 * r]

    def band(lo: float, hi: float):
        outer = [px(c + lo, r) for c, r in zip(centre, yy)]
        inner = [px(c + hi, r) for c, r in zip(centre[::-1], yy[::-1])]
        return outer + inner + [outer[0]]

    (out / "fake_emsr927_flood.geojson").write_text(json.dumps(
        {"type": "Feature", "properties": {}, "geometry": {"type": "MultiPolygon",
         "coordinates": [[band(-22, -4)], [band(4, 22)]]}}))
    foot = [px(0, 0), px(int(w * 0.8), 0), px(int(w * 0.8), h), px(0, h), px(0, 0)]
    (out / "fake_emsr927_aoi.geojson").write_text(json.dumps(
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [foot]}}))

    ident = Normalizer(np.zeros(6, np.float32), np.ones(6, np.float32), {})
    predictor = FloodPredictor(fake_model_fn, ident, threshold_source="package demo (frozen on validation)",
                               model_info={"checkpoint": "FAKE demo model"})
    # a tiny fake Part 3 result so the generated documents have numbers
    fake_eval = [_fake_row("test", "model", 0.82, 0.9, 0.88, 3), _fake_row("test", "baseline-fixed", 0.70, 0.8, 0.75, 3),
                 _fake_row("himalayan", "model", 0.58, 0.7, 0.62, 2), _fake_row("himalayan", "baseline-fixed", 0.50, 0.6, 0.6, 2)]
    (out / "results.json").write_text(json.dumps(fake_eval))

    case = run_case_study(out / "before.npz", out / "after.npz", out / "case", predictor,
                          [out / "fake_emsr927_flood.geojson"], out / "fake_emsr927_aoi.geojson",
                          out / "results.json", ref_crs="EPSG:32645")
    c, s = case["counts"], case["shift"]
    print(f"IoU vs fake reference: {c['iou']:.3f}  (unshifted {s['iou_at_zero']:.3f}, best shifted "
          f"{s['iou_best']:.3f} at {s['best_shift_dy_dx']}), footprint used: {case['footprint_used']}")
    assert tuple(s["best_shift_dy_dx"]) != (0, 0) and s["iou_best"] > s["iou_at_zero"] + 0.05, "shift diagnostic failed"
    print("(the fake reference was offset by +2 px; on a slanted river the best shift is ambiguous along the "
          "river direction, see compare.shift_sensitivity)")
    assert case["footprint_used"] and abs(case["area_ratio"] - 1) < 0.2
    for f in ("comparison.json", "case_summary.md", "prediction_lock.json", "docs/limitations.md", "docs/qa_prep.md",
              "figures/F3_agreement_emsr927.png", "figures/figures_manifest.json"):
        assert (out / "case" / f).exists(), f

    # the data rule: a reference renamed to look like an input must be caught
    import shutil
    shutil.copy(out / "fake_emsr927_flood.geojson", out / "innocent_name.geojson")
    try:
        run_case_study(out / "innocent_name.geojson", out / "after.npz", out / "case2", predictor,
                       [out / "fake_emsr927_flood.geojson"], ref_crs="EPSG:32645")
    except DataRuleViolation as e:
        print("leak correctly refused:", str(e)[:90], "...")
    else:
        raise AssertionError("renamed reference file was not detected")
    print("\nPART 5 OK")


def _fake_row(group, kind, iou, rec, prec, n_ev):
    f1 = 2 * prec * rec / (prec + rec)
    return {"name": group, "kind": kind, "threshold": 0.5,
            "counts": {"precision": prec, "recall": rec, "iou": iou, "f1": f1},
            "ci_iou": {"low": iou - 0.05, "high": iou + 0.05}, "n_events": n_ev, "n_scenes": n_ev * 6,
            "relaxed_precision": prec, "relaxed_recall": rec, "average_precision": 0.9,
            "oracle": {"threshold": 0.4, "f1": f1 + 0.01},
            "errors": {"fp": {"boundary": 6, "other": 4}, "fn": {"weak_change": 7, "boundary": 3}}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Part 5: case study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--out", default="demo_case")
    r = sub.add_parser("run")
    r.add_argument("--before", required=True)
    r.add_argument("--after", required=True)
    r.add_argument("--package", default="model_package")
    r.add_argument("--reference", nargs="+", required=True, help="EMSR927 flood/debris layers (GeoJSON, raster)")
    r.add_argument("--footprint", default=None, help="EMSR927 area of interest (GeoJSON)")
    r.add_argument("--slope", default=None, help="slope raster in degrees (Teammate 1)")
    r.add_argument("--eval-results", default=None)
    r.add_argument("--splits", default=None, help="Part 1 splits.json (checked for EMS paths)")
    r.add_argument("--ref-crs", default="EPSG:4326")
    r.add_argument("--device", default=None)
    r.add_argument("--out", default="outputs/case")
    a = ap.parse_args(argv)
    if a.cmd == "demo":
        run_demo(a.out)
        return
    from inference import FloodPredictor
    predictor = FloodPredictor.from_package(a.package, a.device)
    listed = []
    if a.splits:
        listed = [f for v in json.loads(Path(a.splits).read_text()).get("files", {}).values() for f in v]
    case = run_case_study(a.before, a.after, a.out, predictor, a.reference, a.footprint, a.eval_results,
                          a.slope, a.ref_crs, listed)
    print(Path(a.out, "case_summary.md").read_text())


if __name__ == "__main__":
    main()
