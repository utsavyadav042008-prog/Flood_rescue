"""
audit.py  --  Part 5: the Day-13 AI/model audit as a script
===========================================================

Repo location: models/audit.py          (pure python, fully tested)

Run it the evening before submission and again just before judging:

    python audit.py --package model_package --results outputs/eval/results.json \
        --case outputs/case/comparison.json --readme README.md --report report.md \
        --limitations docs/limitations.md --attribution docs/attribution.md \
        --figures figures --splits config/splits.json --model-card model_package/model_card.md

Every check ends as  PASS / FAIL / WARN / MANUAL / SKIP.  FAIL makes the exit
code non-zero.  MANUAL items are things no script can judge (a human must tick them).

TRICK T13 (the audit checks facts, not intentions): each check looks at the actual
files (hashes, text, numbers) instead of asking "did you do X?", so a forgotten
step shows up as a red line rather than as a surprise in front of the judges.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "part4_code"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ems_reference import LockError, audit_no_reference_leak, verify_lock  # noqa: E402
from pack_model import ATTRIBUTION, verify_package  # noqa: E402
from report_tools import (EMS_CREDIT, audit_attribution, flatten_results,  # noqa: E402
                          unfilled_placeholders)

PASS, FAIL, WARN, MANUAL, SKIP = "PASS", "FAIL", "WARN", "MANUAL", "SKIP"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


def _read(p) -> Optional[str]:
    return Path(p).read_text(errors="ignore") if p and Path(p).exists() else None


# =============================================================================
# THE CHECKS
# =============================================================================
def check_package(package: Optional[str]) -> Check:
    if not package:
        return Check("model package verifies", SKIP, "no --package given")
    probs = verify_package(package)
    return Check("model package verifies", FAIL if probs else PASS, "; ".join(probs) or "manifest hashes match")


def check_results(results_path: Optional[str]) -> List[Check]:
    if not results_path or not Path(results_path).exists():
        return [Check("Part 3 results exist", FAIL, "results.json not found")]
    results = json.loads(Path(results_path).read_text())
    flat = flatten_results(results)
    out = [Check("Part 3 results exist", PASS, f"{len(results)} rows")]
    for g in ("test", "himalayan"):
        out.append(Check(f"results include '{g}' model row", PASS if f"{g}.model.iou" in flat else FAIL,
                         "" if f"{g}.model.iou" in flat else f"no '{g}' group evaluated"))
    out.append(Check("results include a non-AI baseline", PASS if any(k.startswith("test.baseline") for k in flat) else FAIL))
    n = flat.get("himalayan.model.n_events")
    if n is not None:
        out.append(Check("Himalayan test has enough events for a CI", WARN if n < 5 else PASS,
                         f"{n} event(s); state this in the report" if n < 5 else f"{n} events"))
    return out


def check_case(case_path: Optional[str], inference_meta: Optional[str] = None) -> List[Check]:
    if not case_path or not Path(case_path).exists():
        return [Check("case study ran", FAIL, "comparison.json not found")]
    case = json.loads(Path(case_path).read_text())
    out = [Check("case study ran", PASS)]
    lock = case.get("lock_path")
    if lock and Path(lock).exists():
        try:
            verify_lock(lock)
            out.append(Check("prediction unchanged since it was locked", PASS, f"locked {case.get('locked_at')}"))
        except LockError as e:
            out.append(Check("prediction unchanged since it was locked", FAIL, str(e)))
    else:
        out.append(Check("prediction unchanged since it was locked", FAIL, "no lock file"))
    src = case.get("threshold_source", "")
    out.append(Check("case-study threshold was the frozen one", FAIL if src == "argument" else PASS, src))
    out.append(Check("reference footprint was used", PASS if case.get("footprint_used") else WARN,
                     "" if case.get("footprint_used") else "precision may be understated; say so in the report"))
    return out


def check_no_reference_leak(splits: Optional[str], case_path: Optional[str]) -> Check:
    listed: List[str] = []
    refs: List[str] = []
    inputs: List[str] = []
    if splits and Path(splits).exists():
        d = json.loads(Path(splits).read_text())
        listed += [f for v in d.get("files", {}).values() for f in v]
    if case_path and Path(case_path).exists():
        c = json.loads(Path(case_path).read_text())
        inputs += c.get("input_files", [])
        refs += c.get("reference_files", [])
    if not (listed or inputs):
        return Check("no EMS/UNOSAT data used as input", SKIP, "no splits/case file given")
    v = audit_no_reference_leak(inputs, refs, listed)
    return Check("no EMS/UNOSAT data used as input", FAIL if v else PASS, "; ".join(v) or
                 f"checked {len(listed)} listed training paths and {len(inputs)} inputs")


def check_text_files(files: Dict[str, Optional[str]], require_ems_in: Optional[List[str]] = None) -> List[Check]:
    out = []
    present = {k: v for k, v in files.items() if v}
    if not present:
        return [Check("attribution lines present", SKIP, "no text files given")]
    miss = audit_attribution(present.values(), require_ems=False)
    for label, path in present.items():
        m = miss[str(path)]
        out.append(Check(f"required attribution in {label}", FAIL if m else PASS,
                         ("missing: " + " | ".join(s[:50] + "..." for s in m)) if m else ""))
    for label in require_ems_in or []:
        p = files.get(label)
        if p:
            out.append(Check(f"EMS credit in {label}", PASS if EMS_CREDIT in (_read(p) or "") else FAIL))
    return out


def check_citations(files: List[Optional[str]]) -> Check:
    text = " ".join(_read(f) or "" for f in files if f)
    ok = "Bountos" in text
    return Check("training dataset cited (Kuro Siwo, Bountos et al.)", PASS if ok else FAIL,
                 "" if ok else "add the citation to attribution.md / report")


def check_limitations(path: Optional[str]) -> Check:
    t = (_read(path) or "").lower()
    if not t:
        return Check("limitations section exists", FAIL, "file missing or empty")
    need = {"revisit": "satellite revisit / cannot warn ahead", "debris": "debris", "shadow": "radar shadow",
            "himalayan": "Himalayan generalization", "educational": "educational prototype"}
    miss = [d for k, d in need.items() if k not in t]
    return Check("limitations cover the key points", FAIL if miss else PASS, "missing: " + ", ".join(miss) if miss else "")


def check_report(report: Optional[str]) -> List[Check]:
    text = _read(report)
    if text is None:
        return [Check("report placeholders filled", SKIP, "no --report given")]
    left = unfilled_placeholders(text)
    out = [Check("report placeholders filled", FAIL if left else PASS,
                 ("unfilled: " + ", ".join(left)) if left else "no {{...}} left")]
    if str(report).lower().endswith(".pdf"):
        out.append(_pdf_pages(report))
    return out


def _pdf_pages(path) -> Check:
    try:
        from pypdf import PdfReader
        n = len(PdfReader(path).pages)
        return Check("report is at most 6 pages", PASS if n <= 6 else FAIL, f"{n} pages")
    except ImportError:
        return Check("report is at most 6 pages", MANUAL, "install pypdf to check automatically")


def check_figures(figures_dir: Optional[str], manifest_name: str = "figures_manifest.json") -> List[Check]:
    if not figures_dir or not Path(figures_dir).exists():
        return [Check("figures come from our code", SKIP, "no --figures folder")]
    d = Path(figures_dir)
    mpath = d / manifest_name
    known = json.loads(mpath.read_text())["figures"] if mpath.exists() else {}
    import hashlib
    imgs = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    unknown = [p.name for p in imgs if p.name not in known]
    changed = [p.name for p in imgs if p.name in known and
               hashlib.sha256(p.read_bytes()).hexdigest() != known[p.name]["sha256"]]
    out = [Check("all figures are in the manifest", WARN if unknown else PASS,
                 f"{len(unknown)} image(s) not generated by our code: {', '.join(unknown[:6])}. A human must confirm "
                 f"none shows victims." if unknown else f"{len(imgs)} figure(s) with provenance")]
    if changed:
        out.append(Check("figures unchanged since generation", WARN, f"edited after generation: {', '.join(changed)}"))
    return out


def check_readme(readme: Optional[str]) -> Check:
    t = (_read(readme) or "").lower()
    if not t:
        return Check("README explains how to run", FAIL, "README missing")
    need = ["pip install", "python", "model_package"]
    miss = [n for n in need if n not in t]
    return Check("README explains how to run", WARN if miss else PASS, ("could not find: " + ", ".join(miss)) if miss else "")


MANUAL_ITEMS = [
    "3-minute demo video: runs end-to-end, is at most 3 minutes, and shows the attribution line",
    "No images of victims anywhere (video, slides, report, dashboard)",
    "Every number said in the video matches results.json / comparison.json",
    "Team has rehearsed the Q&A (qa_prep.md) and everyone can explain IoU, the threshold, and the limitations",
    "The system runs LIVE on a fresh area and date (try one that is not Trishuli) with no network surprises",
    "EMS/UNOSAT data was never used as input or to choose any setting",
]


# =============================================================================
# RUNNER
# =============================================================================
def run_audit(a: Dict[str, Optional[str]]) -> List[Check]:
    checks: List[Check] = [check_package(a.get("package"))]
    checks += check_results(a.get("results"))
    checks += check_case(a.get("case"))
    checks.append(check_no_reference_leak(a.get("splits"), a.get("case")))
    # limitations.md is a fragment that feeds the report, so it does not need the credits itself;
    # README, report, attribution doc and model card do.
    texts = {k: a.get(k) for k in ("readme", "report", "attribution", "model_card")}
    checks += check_text_files(texts, require_ems_in=["report", "attribution"])
    checks.append(check_citations([a.get("attribution"), a.get("report"), a.get("model_card")]))
    checks.append(check_limitations(a.get("limitations")))
    checks += check_report(a.get("report"))
    checks += check_figures(a.get("figures"))
    checks.append(check_readme(a.get("readme")))
    checks += [Check(item, MANUAL) for item in MANUAL_ITEMS]
    return checks


def render(checks: List[Check]) -> str:
    icon = {PASS: "✅", FAIL: "❌", WARN: "⚠️", MANUAL: "☐", SKIP: "·"}
    lines = ["# Day-13 audit", ""]
    for c in checks:
        lines.append(f"- {icon[c.status]} **{c.status}** {c.name}" + (f": {c.detail}" if c.detail else ""))
    n = {s: sum(c.status == s for c in checks) for s in (PASS, FAIL, WARN, MANUAL, SKIP)}
    lines += ["", f"**{n[PASS]} passed, {n[FAIL]} failed, {n[WARN]} warnings, {n[MANUAL]} manual, {n[SKIP]} skipped**"]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Part 5: Day-13 audit")
    for k in ("package", "results", "case", "readme", "report", "limitations", "attribution", "figures",
              "splits", "model-card"):
        ap.add_argument(f"--{k}", default=None)
    ap.add_argument("--out", default=None, help="write audit_report.md here")
    a = vars(ap.parse_args(argv))
    out = a.pop("out")
    a["model_card"] = a.pop("model-card", None) or a.get("model_card")
    checks = run_audit(a)
    text = render(checks)
    print(text)
    if out:
        Path(out).write_text(text)
    raise SystemExit(1 if any(c.status == FAIL for c in checks) else 0)


if __name__ == "__main__":
    main()
