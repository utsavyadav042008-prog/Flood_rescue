"""
report_tools.py  --  Part 5: writing the honest parts of the report from real numbers
=====================================================================================

Repo location: models/report_tools.py        (pure python, fully tested)

The judges score "report, understanding in Q&A and honesty about limitations"
(15%).  The most common way to lose those points is a number in the report that
no longer matches the results (retrained model, re-tuned threshold) or a
limitation that was never actually measured.  This file removes that risk: text
that carries numbers is GENERATED from results.json (Part 3) and the case-study
output, never retyped.

CONTENTS
    1. flatten_results / flatten_case   results.json -> {"test.model.iou": 0.71, ...}
    2. fill_template                    {{test.model.iou:.3f}} placeholders
    3. model_vs_baseline_statement      does the AI really beat the simple method?
    4. build_limitations                docs/limitations.md
    5. build_attribution_doc / audit_attribution
    6. build_qa / build_demo_script     Q&A drill and the 30-second demo segment
"""
from __future__ import annotations

import difflib
import importlib.metadata as md
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "models"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pack_model import ATTRIBUTION  # noqa: E402

EMS_CREDIT = "European Union, Copernicus Emergency Management Service data"
KURO_SIWO_CITATION = ("Bountos, N. I., Sdraka, M., Zavras, A., Karasante, I., Karavias, A., Herekakis, T., "
                      "Thanasou, A., Michail, D., Papoutsis, I. (2024). Kuro Siwo: 33 billion m² under the water! "
                      "A global multi-temporal satellite dataset for rapid flood mapping. NeurIPS 2024 Datasets "
                      "and Benchmarks Track.")
SEN1FLOODS11_CITATION = ("Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). Sen1Floods11: A "
                         "georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. "
                         "CVPR Workshops 2020.")


class TemplateError(ValueError):
    """A placeholder in a template has no value."""


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


# =============================================================================
# 1. FLATTENING
# =============================================================================
def _top(d: Dict[str, int]):
    tot = sum(d.values())
    if not tot:
        return None, None
    k, n = max(d.items(), key=lambda kv: kv[1])
    return k, n / tot


def flatten_results(results: Sequence[Dict]) -> Dict[str, object]:
    """Part 3 results.json -> flat dict with keys  <group>.<method>.<field>.

    group  : 'validation' (the '*' is dropped), 'test', 'himalayan'
    method : 'model', 'baseline_fixed', 'baseline_otsu'
    field  : precision recall iou f1 iou_low iou_high n_events n_scenes rel_precision rel_recall ap
             threshold oracle_f1 oracle_threshold fp_top_cause fp_top_share fn_top_cause fn_top_share
    e.g.  {{himalayan.model.recall:.2f}}
    """
    out: Dict[str, object] = {}
    for r in results:
        g = r["name"].replace("*", "").strip().lower()
        m = r["kind"].replace("-", "_")
        base = f"{g}.{m}."
        c, ci = r["counts"], r.get("ci_iou", {})
        vals = {"precision": c.get("precision"), "recall": c.get("recall"), "iou": c.get("iou"), "f1": c.get("f1"),
                "iou_low": ci.get("low"), "iou_high": ci.get("high"), "n_events": r.get("n_events"),
                "n_scenes": r.get("n_scenes"), "rel_precision": r.get("relaxed_precision"),
                "rel_recall": r.get("relaxed_recall"), "ap": r.get("average_precision"),
                "threshold": r.get("threshold")}
        if r.get("oracle"):
            vals.update(oracle_f1=r["oracle"].get("f1"), oracle_threshold=r["oracle"].get("threshold"))
        for kind in ("fp", "fn"):
            cause, share = _top(r.get("errors", {}).get(kind, {}))
            vals[f"{kind}_top_cause"], vals[f"{kind}_top_share"] = cause, share
        out.update({base + k: v for k, v in vals.items()})
    return out


def flatten_case(case: Dict) -> Dict[str, object]:
    """case_study comparison.json -> keys 'case.iou', 'case.area_ratio', ..."""
    c, sh = case["counts"], case.get("shift", {})
    ci = case.get("iou_ci", {})
    vals = {"iou": c["iou"], "precision": c["precision"], "recall": c["recall"], "f1": c["f1"],
            "iou_low": ci.get("low"), "iou_high": ci.get("high"),
            "rel_precision": case.get("relaxed_precision"), "rel_recall": case.get("relaxed_recall"),
            "area_ours_km2": case.get("area_ours_km2"), "area_ref_km2": case.get("area_ref_km2"),
            "area_ratio": case.get("area_ratio"), "iou_at_zero": sh.get("iou_at_zero"),
            "iou_best_shifted": sh.get("iou_best"), "best_shift": str(sh.get("best_shift_dy_dx")),
            "threshold": case.get("threshold"), "locked_at": case.get("locked_at"),
            "footprint_used": case.get("footprint_used")}
    return {f"case.{k}": v for k, v in vals.items()}


# =============================================================================
# 2. TEMPLATES
# =============================================================================
_PLACEHOLDER = re.compile(r"\{\{\s*([\w.\-]+)\s*(?::([^{}]+))?\}\}")


def fill_template(text: str, values: Dict[str, object], strict: bool = True,
                  missing_text: str = "[not available yet]") -> str:
    """Replace {{key}} or {{key:.3f}} with values.  None/NaN print as 'n/a'.

    TRICK T10 (the report cannot contain a stale number):
        Write the report with placeholders, e.g.
            "On unseen Himalayan scenes the model reaches IoU {{himalayan.model.iou:.2f}}"
        and fill it from results.json when you build the final version.  With
        strict=True an unknown key raises TemplateError listing the key and its
        closest matches, so a typo or a missing evaluation can never turn into a
        silently blank or invented figure.  audit.py also fails if '{{' survives
        in the final report.
    """
    missing: List[str] = []

    def sub(m: "re.Match") -> str:
        key, spec = m.group(1), m.group(2)
        if key not in values:
            missing.append(key)
            return missing_text
        v = values[key]
        if _nan(v):
            return "n/a"
        if spec and isinstance(v, (int, float)) and not isinstance(v, bool):
            return format(v, spec.strip())
        return str(v)

    out = _PLACEHOLDER.sub(sub, text)
    if missing and strict:
        hints = {k: difflib.get_close_matches(k, list(values), n=2) for k in sorted(set(missing))}
        raise TemplateError("unknown placeholder(s): " + "; ".join(
            f"{k}" + (f" (did you mean {', '.join(h)}?)" if h else "") for k, h in hints.items()))
    return out


def unfilled_placeholders(text: str) -> List[str]:
    return sorted(set(m.group(1) for m in _PLACEHOLDER.finditer(text)))


# =============================================================================
# 3. DOES THE AI BEAT THE BASELINE?
# =============================================================================
def model_vs_baseline_statement(results: Sequence[Dict], group: str) -> str:
    """One honest sentence comparing the U-Net with the best non-AI baseline on a group.

    TRICK T11 (confidence intervals decide the wording, not the point estimate):
        'Model IoU 0.62 vs baseline 0.60' is NOT a win if the intervals overlap
        widely (few events).  The sentence is chosen by rule:
          model interval entirely above the baseline's  -> "clearly better"
          intervals overlap                             -> "not clearly distinguishable"
          model point estimate below the baseline       -> "does NOT beat it"
        and it always states the number of events.  Say what the data supports.
    """
    rows = {r["kind"]: r for r in results if r["name"].replace("*", "").strip().lower() == group}
    if "model" not in rows or not any(k.startswith("baseline") for k in rows):
        return f"No model-vs-baseline comparison is available for '{group}'."
    mod = rows["model"]
    base = max((r for k, r in rows.items() if k.startswith("baseline")),
               key=lambda r: -1 if _nan(r["counts"].get("iou")) else r["counts"]["iou"])
    mi, bi = mod["counts"].get("iou"), base["counts"].get("iou")
    if _nan(mi) or _nan(bi):
        return f"IoU is undefined for '{group}' (no flood in the reference)."
    n = mod.get("n_events", "?")
    m_low, b_high = mod["ci_iou"].get("low"), base["ci_iou"].get("high")
    head = f"On '{group}' ({n} event{'s' if n != 1 else ''}) the U-Net reaches IoU {mi:.2f} versus {bi:.2f} for the best baseline ({base['kind']}); "
    if mi < bi:
        return head + "the U-Net does NOT beat the simple baseline here, and the report says so."
    if not _nan(m_low) and not _nan(b_high) and m_low > b_high:
        return head + "the U-Net is clearly better (its 95% interval lies above the baseline's)."
    return head + "the intervals overlap, so the difference is not clearly distinguishable with this little data."


# =============================================================================
# 4. LIMITATIONS
# =============================================================================
def _fmt(v, nd=2):
    return "n/a" if _nan(v) else f"{v:.{nd}f}"


def _shift_sentence(c: Dict[str, object]) -> str:
    """L14: worded by what the shift diagnostic actually found."""
    z, best = c["case.iou_at_zero"], c["case.iou_best_shifted"]
    if _nan(z) or _nan(best):
        return "- **L14 [measured]** The shift diagnostic could not be computed."
    if best - z > 0.02:
        return (f"- **L14 [measured]** The IoU is {_fmt(z)} unshifted but {_fmt(best)} when the reference is shifted "
                f"(best shift {c['case.best_shift']} pixels), which suggests part of the disagreement is a "
                f"georeferencing offset between the two products rather than model error. Only the unshifted "
                f"number is our result.")
    return (f"- **L14 [measured]** Shifting the reference by up to a few pixels does not improve the IoU "
            f"({_fmt(z)} unshifted, {_fmt(best)} best), so misregistration does not explain the disagreement.")


def build_limitations(results: Sequence[Dict], case: Optional[Dict] = None,
                      inference_meta: Optional[Dict] = None) -> str:
    """docs/limitations.md.  Sentences with numbers are generated; the fixed
    sentences (L1...) are true by design and each has an id so Teammate 4 can cite it.

    TRICK T12 (measured vs asserted):
        Every limitation is marked [measured] (a number from our own results)
        or [by design] (a fact about the data/method that no experiment of ours
        proves).  Judges respect the difference, and you will remember in Q&A
        which claims you can back with a figure.
    """
    flat = flatten_results(results)
    L: List[str] = ["# Limitations: what this system cannot do", "",
                    "_Educational prototype, not an operational tool. Each item is tagged "
                    "**[measured]** (a number from our own evaluation) or **[by design]** (a property of the "
                    "data or method)._", ""]

    L += ["## About the model", ""]
    L += ["- **L1 [by design]** The training labels mark flood WATER. Debris and mud are not labelled, so debris "
          "detection is indirect (through the change channels) and weaker than water detection.",
          "- **L2 [by design]** Training data is mostly non-Himalayan. Radar shadow and layover on steep valley "
          "walls can look like water.",
          "- **L3 [by design]** Pixels are 10 m. Flood corridors only a few pixels wide can be missed or merged.",
          "- **L4 [by design]** Wet snow, ice and other smooth surfaces can resemble water in radar.",
          "- **L5 [by design]** The output is a probability; the threshold trades false alarms against misses.", ""]

    h = "himalayan.model."
    if f"{h}iou" in flat:
        top_fn, share = flat.get(h + "fn_top_cause"), flat.get(h + "fn_top_share")
        L.append(f"- **L6 [measured]** On unseen Himalayan scenes ({flat[h + 'n_events']} event(s)) the model reaches "
                 f"IoU {_fmt(flat[h + 'iou'])} [95% CI {_fmt(flat.get(h + 'iou_low'))} to {_fmt(flat.get(h + 'iou_high'))}], "
                 f"precision {_fmt(flat[h + 'precision'])}, recall {_fmt(flat[h + 'recall'])}."
                 + (f" The most frequent miss is `{top_fn}` ({100 * share:.0f}% of missed pixels)." if top_fn else ""))
        if (flat[h + "n_events"] or 0) < 5:
            L.append(f"- **L7 [measured]** That is only {flat[h + 'n_events']} event(s): the interval is unreliable and "
                     f"the result must not be generalized to all Himalayan valleys.")
        gap = (flat.get(h + "oracle_f1") or float("nan")) - (flat.get(h + "f1") or float("nan"))
        if not math.isnan(gap) and gap > 0.05:
            L.append(f"- **L8 [measured]** The validation threshold is {gap:.2f} F1 below the best possible threshold on "
                     f"Himalayan data: the model's confidence is miscalibrated in mountains (distribution shift).")
    else:
        L.append("- **L6 [measured]** No Himalayan test group was evaluated, so generalization to mountain valleys "
                 "is UNVERIFIED.")
    L.append(f"- **L9 [measured]** {model_vs_baseline_statement(results, 'himalayan' if f'{h}iou' in flat else 'test')}")
    L.append("")

    L += ["## About the satellite data", "",
          "- **L10 [by design]** Sentinel satellites revisit an area only every few days, so the system maps damage "
          "AFTER an event and cannot warn of a sudden glacier collapse minutes ahead.",
          "- **L11 [by design]** Change detection needs a BEFORE and AFTER image from the same orbit track; images from "
          "different tracks cannot be compared pixel by pixel. The system warns about suspicious pairs but cannot read "
          "the orbit number from a GeoTIFF.",
          "- **L12 [by design]** Radar sees through cloud; we do not use optical imagery for the flood mask, so "
          "optical detail (e.g. mud colour) is not exploited.", ""]

    if case is not None:
        c = flatten_case(case)
        L += ["## About the case-study comparison (EMSR927)", "",
              f"- **L13 [measured]** Against EMSR927 our map reaches IoU {_fmt(c['case.iou'])} "
              f"[block-bootstrap 95% interval {_fmt(c['case.iou_low'])} to {_fmt(c['case.iou_high'])}], precision "
              f"{_fmt(c['case.precision'])}, recall {_fmt(c['case.recall'])}. The interval covers sampling variability "
              f"only, not reference errors.",
              _shift_sentence(c),
              "- **L15 [by design]** The reference was mapped on a specific date and with different sensors/methods; "
              "floods recede, so some disagreement is real change over time, not an error of either map."]
        if not case.get("footprint_used", True):
            L.append("- **L16 [measured]** No reference footprint was available, so flood outside the area EMS mapped "
                     "counts as a false alarm; precision is understated.")
        if c["case.area_ratio"] not in (None, "None") and not _nan(c["case.area_ratio"]):
            L.append(f"- **L17 [measured]** Our flooded area is {_fmt(c['case.area_ratio'])} times the reference area "
                     f"({_fmt(c['case.area_ours_km2'])} vs {_fmt(c['case.area_ref_km2'])} km²).")
        L.append("")

    if inference_meta and inference_meta.get("warnings"):
        L += ["## Warnings raised by the system on the case-study input", ""]
        L += [f"- {w}" for w in inference_meta["warnings"]] + [""]

    L += ["## About use", "",
          "- **L18 [by design]** 'Damaged' means 'overlaps the satellite-derived flood or debris area', not verified "
          "destruction; 'cut off' is inferred from the pre-event OpenStreetMap road network (which may be incomplete).",
          "- **L19 [by design]** This is an educational prototype. Do not use it for rescue decisions.", ""]
    return "\n".join(L)


# =============================================================================
# 5. ATTRIBUTION
# =============================================================================
def _version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


def build_attribution_doc(kuro_siwo: bool = True, sen1floods11: bool = False, ems_shown: bool = True) -> str:
    """docs/attribution.md: required attribution lines, training-data citations,
    EMS credit, and the exact software versions used."""
    L = ["# Attribution and citations", "", "## Required in every submission", ""]
    L += [f"> {a}" for a in ATTRIBUTION] + [""]
    L += ["## Training data", ""]
    if kuro_siwo:
        L += [f"- {KURO_SIWO_CITATION}", "  (Check author list and title against the Kuro Siwo repository README before submitting.)"]
    if sen1floods11:
        L += [f"- {SEN1FLOODS11_CITATION}", "  (Check against the Sen1Floods11 repository README.)"]
    L += ["", "## Reference data (checking only, never an input)", ""]
    L += [f"- {EMS_CREDIT} (activation EMSR927)." if ems_shown else "- (not shown)", ""]
    L += ["## Software", ""]
    L += [f"- {p} {_version(p)}" for p in ("torch", "segmentation-models-pytorch", "numpy", "scipy", "rasterio",
                                          "matplotlib", "Pillow")]
    return "\n".join(L) + "\n"


def audit_attribution(paths: Iterable, require_ems: bool = False) -> Dict[str, List[str]]:
    """For each text file return the REQUIRED strings it lacks (empty list = fine)."""
    need = list(ATTRIBUTION) + ([EMS_CREDIT] if require_ems else [])
    out: Dict[str, List[str]] = {}
    for p in paths:
        text = Path(p).read_text(errors="ignore") if Path(p).exists() else ""
        out[str(p)] = [n for n in need if n not in text] if text else need
    return out


# =============================================================================
# 6. Q&A DRILL AND DEMO SCRIPT
# =============================================================================
QA_TEMPLATE = [
    ("Why a U-Net?", "It labels every pixel, works with modest data, is a proven choice for satellite "
                     "segmentation and is easy to debug. We kept it simple on purpose and compared it with a non-AI baseline."),
    ("Why radar and not optical?", "Radar sees through monsoon cloud, which is when floods happen. Optical imagery is "
                                   "not used for the flood mask."),
    ("Why six input channels?", "Before and after (VV, VH each) plus the two change channels. The change channels hand "
                                "the model the 'what changed' signal directly; subtracting dB values is a ratio, so terrain brightness cancels on same-orbit pairs."),
    ("How do you know it generalizes?", "Splits are by flood event, never by patch. On unseen Himalayan scenes "
                                        "({{himalayan.model.n_events}} event(s)) IoU is {{himalayan.model.iou:.2f}} "
                                        "(95% CI {{himalayan.model.iou_low:.2f}} to {{himalayan.model.iou_high:.2f}}); on other unseen floods {{test.model.iou:.2f}}."),
    ("What does IoU mean and what is yours?", "Overlap between predicted and true flood divided by their union. "
                                              "Test {{test.model.iou:.2f}}, Himalayan {{himalayan.model.iou:.2f}}, case study {{case.iou:.2f}} against EMSR927."),
    ("Does the AI beat a simple method?", "{{__baseline_statement__}}"),
    ("Why is permanent water a separate class?", "A river is always dark in radar. Without a separate class the model "
                                                  "would call every river a flood."),
    ("What about debris?", "The training labels mark water, not debris, so debris detection is indirect and weaker. "
                           "Its most frequent miss in Himalayan data is '{{himalayan.model.fn_top_cause}}'."),
    ("Did you use EMS data to train or tune?", "No. EMSR927 was used only for checking, after our prediction was "
                                              "locked ({{case.locked_at}}). The threshold {{case.threshold:.2f}} was fixed on validation data beforehand."),
    ("How would it fail on a new valley?", "Radar shadow on steep walls, corridors only a few pixels wide, missing "
                                           "same-orbit pairs, and debris-dominated floods. These are in our limitations section."),
    ("Could it warn before a flood?", "No. Sentinel revisits every few days; we map damage after an event."),
    ("What would you do with more time?", "Label Himalayan scenes, add a DEM-based shadow mask, include optical "
                                           "evidence, and calibrate confidence for mountain terrain."),
]


def build_qa(results: Sequence[Dict], case: Optional[Dict] = None) -> str:
    """qa_prep.md: likely judge questions with answers filled from real numbers
    ('[not available yet]' where an evaluation has not been run)."""
    vals = flatten_results(results)
    if case:
        vals.update(flatten_case(case))
    group = "himalayan" if "himalayan.model.iou" in vals else "test"
    vals["__baseline_statement__"] = model_vs_baseline_statement(results, group)
    L = ["# Q&A preparation: practise answering aloud", ""]
    for i, (q, a) in enumerate(QA_TEMPLATE, 1):
        L += [f"**{i}. {q}**", "", fill_template(a, vals, strict=False), ""]
    return "\n".join(L)


def build_demo_script(results: Sequence[Dict], case: Optional[Dict] = None) -> str:
    """The ~30-second AI segment of the 3-minute video, numbers filled in."""
    vals = flatten_results(results)
    if case:
        vals.update(flatten_case(case))
    text = ("# Demo segment: the AI (about 30 seconds)\n\n"
            "1. (0:00) Show BEFORE then AFTER radar. Say: \"Radar sees through monsoon cloud. The river is a thin dark line before; after the flood it widens.\"\n"
            "2. (0:08) Show the predicted flood overlay. Say: \"A U-Net compares the two images and marks flooded pixels, with a confidence for each.\"\n"
            "3. (0:16) Show the agreement map. Say: \"Against the official EMSR927 map we reach IoU {{case.iou:.2f}}; "
            "on unseen Himalayan scenes {{himalayan.model.iou:.2f}}.\"\n"
            "4. (0:24) Say one limitation: \"It cannot warn ahead of an event, and steep shadow can fool it.\"\n\n"
            "Rules: no images of victims; show the attribution line on screen.\n")
    return fill_template(text, vals, strict=False)
