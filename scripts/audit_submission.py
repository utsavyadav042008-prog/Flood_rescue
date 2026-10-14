"""Final submission audit (Day 13).  Automates what a machine can check and lists the rest.

    python scripts/audit_submission.py                 # full audit incl. the test suite
    python scripts/audit_submission.py --skip-tests    # quick
    python scripts/audit_submission.py --json

Status per check:  PASS | FAIL | WARN | MANUAL (a human must look).  Exit code 1 if any FAIL.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import load_config                                    # noqa: E402
from pipeline.contracts import check_run                                   # noqa: E402
from pipeline.errors import PipelineError                                  # noqa: E402
from pipeline.rules import scan_for_forbidden_inputs, GUARDED_FOLDERS      # noqa: E402
from pipeline.run import RunPaths                                          # noqa: E402
from reports.format import ATTRIBUTION_LINES                               # noqa: E402

PASS, FAIL, WARN, MANUAL = "PASS", "FAIL", "WARN", "MANUAL"
EMS_CREDIT = "European Union, Copernicus Emergency Management Service data"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")
MIN_LIMITATIONS_CHARS = 800
LIMITATION_KEYWORDS = ("revisit", "radar shadow", "OpenStreetMap", "operational")
README_SECTIONS = ("install", "quick start", "attribution", "limitations", "case study", "tests")


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _attribution_in(text: str) -> list:
    """Attribution lines (first 50 characters, whitespace-normalised) missing from ``text``."""
    flat = " ".join(text.split())
    return [line for line in ATTRIBUTION_LINES if " ".join(line.split())[:50] not in flat]


def check_readme(root: Path) -> Check:
    text = _read(root / "README.md").lower()
    if not text:
        return Check("README", FAIL, "README.md is missing")
    missing = [s for s in README_SECTIONS if s not in text]
    return Check("README", PASS if not missing else FAIL,
                 "all sections present" if not missing else f"missing sections: {', '.join(missing)}")


def check_attribution(root: Path) -> Check:
    problems = []
    for rel in ("README.md", "docs/attribution.md"):
        lacking = _attribution_in(_read(root / rel))
        if lacking:
            problems.append(f"{rel}: {len(lacking)} line(s) missing")
    footer = _read(root / "dashboard" / "components.py")
    if "ATTRIBUTION_LINES" not in footer:
        problems.append("dashboard footer does not use ATTRIBUTION_LINES")
    return Check("Required attribution", FAIL if problems else PASS, "; ".join(problems) or "README, docs, dashboard")


def check_citations(root: Path) -> Check:
    attr = _read(root / "docs" / "attribution.md")
    problems = []
    if "Bountos" not in attr or "2024" not in attr:
        problems.append("Kuro Siwo citation (Bountos et al., 2024) missing")
    mentioned = "Sen1Floods11" in (attr + _read(root / "README.md") + _read(root / "docs" / "data_sources.md"))
    if mentioned and ("Bonafilia" not in attr or "2020" not in attr):
        problems.append("Sen1Floods11 is mentioned but Bonafilia et al., 2020 is not cited")
    if EMS_CREDIT not in attr:
        problems.append("reference-map credit line missing")
    return Check("Dataset citations and credits", FAIL if problems else PASS, "; ".join(problems) or "present")


def check_limitations(root: Path) -> Check:
    text = _read(root / "docs" / "limitations.md")
    if len(text) < MIN_LIMITATIONS_CHARS:
        return Check("Limitations document", FAIL, f"docs/limitations.md is a stub ({len(text)} characters)")
    lacking = [k for k in LIMITATION_KEYWORDS if k.lower() not in text.lower()]
    return Check("Limitations document", WARN if lacking else PASS,
                 f"does not mention: {', '.join(lacking)}" if lacking else "complete")


def check_data_rules(root: Path) -> Check:
    hits = scan_for_forbidden_inputs(root, allow=["dashboard/case_study.py"])
    imports = []
    for folder in GUARDED_FOLDERS:
        for py in sorted((root / folder).rglob("*.py")) if (root / folder).exists() else []:
            if re.search(r"^\s*(from|import)\s+evaluation\b", _read(py), re.MULTILINE):
                imports.append(py.relative_to(root).as_posix())
    problems = [f"{p}:{n} mentions '{t}'" for p, n, t in hits] + [f"{p} imports evaluation" for p in imports]
    return Check("Data rules (no check-only data as input)", FAIL if problems else PASS,
                 "; ".join(problems[:5]) or "no reference data referenced by pipeline code")


def check_demo_run(root: Path) -> Check:
    demo = root / "data" / "sample" / "trishuli_run"
    if not (demo / "run.json").exists():
        return Check("Cached demo run", FAIL, "data/sample/trishuli_run missing (python scripts/export_demo_run.py <run>)")
    try:
        results = check_run(RunPaths(demo), forbid_mock=True)
    except PipelineError as exc:
        return Check("Cached demo run", FAIL, exc.user_message)
    errors = [str(i) for issues in results.values() for i in issues if i.severity == "error"]
    warns = [str(i) for issues in results.values() for i in issues if i.severity == "warning"]
    if errors:
        return Check("Cached demo run", FAIL, f"{len(errors)} contract error(s), first: {errors[0]}")
    return Check("Cached demo run", WARN if warns else PASS,
                 f"{len(warns)} warning(s), first: {warns[0]}" if warns else "valid, not mock")


def check_case_study(root: Path) -> Check:
    path = root / "evaluation" / "case_study_results.json"
    try:
        data = json.loads(_read(path))
    except ValueError:
        return Check("Case-study results", FAIL, "evaluation/case_study_results.json missing or invalid")
    iou = (data.get("flood_extent") or {}).get("iou")
    if not isinstance(iou, (int, float)):
        return Check("Case-study results", FAIL, "no numeric IoU in the results")
    if not data.get("notes"):
        return Check("Case-study results", WARN, "no notes: explain where and why we disagree (honesty is graded)")
    return Check("Case-study results", PASS, f"IoU {iou:.2f}, {len(data['notes'])} note(s)")


def check_report(root: Path, report: Path) -> Check:
    if not report.exists():
        return Check("Written report (max 6 pages)", FAIL, f"{report.relative_to(root)} missing")
    try:
        from pypdf import PdfReader
    except ImportError:
        return Check("Written report (max 6 pages)", MANUAL, "pypdf not installed: count the pages yourself")
    reader = PdfReader(str(report))
    pages = len(reader.pages)
    text = " ".join((p.extract_text() or "") for p in reader.pages).lower()
    if pages > 6:
        return Check("Written report (max 6 pages)", FAIL, f"{pages} pages")
    if "limitation" not in text:
        return Check("Written report (max 6 pages)", FAIL, f"{pages} pages but no 'Limitations' section found")
    return Check("Written report (max 6 pages)", PASS, f"{pages} page(s), has a limitations section")


def check_video(root: Path) -> Check:
    folder = root / "submission"
    videos = [p for p in folder.glob("*") if p.suffix.lower() in VIDEO_EXTENSIONS] if folder.exists() else []
    if not videos:
        return Check("Demo video", FAIL, "no video in submission/ (keep it out of git if it is large)")
    return Check("Demo video", MANUAL, f"{videos[0].name} ({videos[0].stat().st_size / 1_048_576:.0f} MB): "
                                       "confirm it is under 3:00 and shows no images of victims")


def check_hygiene(root: Path) -> Check:
    if not (root / ".git").exists():
        return Check("Repository hygiene", WARN, "not a git repository here; run the audit in your clone")
    res = subprocess.run([sys.executable, str(root / "scripts" / "check_repo_hygiene.py"), "--all"],
                         cwd=root, capture_output=True, text=True)
    return Check("Repository hygiene", PASS if res.returncode == 0 else FAIL,
                 "no big files, raw data, checkpoints or secrets" if res.returncode == 0 else res.stdout.strip()[:300])


def check_requirements(root: Path) -> Check:
    lines = [l.strip() for l in _read(root / "requirements.txt").splitlines() if l.strip() and not l.startswith("#")]
    unpinned = [l for l in lines if not re.search(r"[=<>~]=?", l)]
    if not lines:
        return Check("Pinned requirements", FAIL, "requirements.txt is empty or missing")
    return Check("Pinned requirements", WARN if unpinned else PASS,
                 f"{len(unpinned)} unpinned (e.g. {unpinned[0]}); run pip freeze after a clean install"
                 if unpinned else "all pinned")


def check_todos(root: Path) -> Check:
    """Unfinished placeholders left in the public documents."""
    hits = []
    for rel in ("README.md", "docs/limitations.md", "docs/attribution.md", "docs/data_sources.md"):
        for n, line in enumerate(_read(root / rel).splitlines(), start=1):
            if re.search(r"\bTODO\b|<fill|\bTBD\b|XXX", line):
                hits.append(f"{rel}:{n}")
    return Check("No placeholders left in public docs", WARN if hits else PASS,
                 ", ".join(hits[:6]) + (" ..." if len(hits) > 6 else "") if hits else "none found")


def check_tests(root: Path) -> Check:
    res = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                         cwd=root, capture_output=True, text=True)
    tail = (res.stderr.strip().splitlines() or ["(no output)"])[-1]
    summary = next((l for l in reversed(res.stderr.splitlines()) if l.startswith(("OK", "FAILED"))), tail)
    skipped = re.search(r"skipped=(\d+)", summary)
    if res.returncode != 0:
        return Check("Test suite", FAIL, summary)
    return Check("Test suite", WARN if skipped else PASS,
                 f"{summary}: skipped tests usually mean a missing optional package" if skipped else summary)


def check_tag(root: Path) -> Check:
    if not (root / ".git").exists():
        return Check("Release tag", MANUAL, "tag the final commit: git tag v1.0-submission")
    res = subprocess.run(["git", "tag", "--list", "v1.0-submission"], cwd=root, capture_output=True, text=True)
    return Check("Release tag", PASS if "v1.0-submission" in res.stdout else WARN, "git tag v1.0-submission")


MANUAL_ITEMS = (
    ("No images of victims", "check the dashboard screenshots, the report and the video"),
    ("Fresh clone works", "clone into a new folder, follow ONLY the README, run the demo"),
    ("Live evaluation rehearsal", "docs/JUDGING_REHEARSAL.md: two unseen areas/dates, one with no usable imagery"),
    ("Numbers agree", "dashboard cards == situation report == report tables (scripts/report_facts.py)"),
)


def run_audit(root: Path = REPO_ROOT, *, skip_tests: bool = False, report: Path = None) -> list:
    report = report or root / "docs" / "report" / "report.pdf"
    checks = [check_readme(root), check_attribution(root), check_citations(root), check_limitations(root),
              check_todos(root), check_data_rules(root), check_demo_run(root), check_case_study(root), check_report(root, report),
              check_video(root), check_hygiene(root), check_requirements(root)]
    checks.append(Check("Test suite", MANUAL, "skipped (--skip-tests)") if skip_tests else check_tests(root))
    checks.append(check_tag(root))
    checks += [Check(n, MANUAL, d) for n, d in MANUAL_ITEMS]
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--report", help="path of the report PDF (default docs/report/report.pdf)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    checks = run_audit(skip_tests=args.skip_tests, report=Path(args.report) if args.report else None)
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        for c in checks:
            print(f"[{c.status:^6}] {c.name}: {c.detail}")
        counts = {s: sum(c.status == s for c in checks) for s in (PASS, WARN, FAIL, MANUAL)}
        print("\n" + "  ".join(f"{k}: {v}" for k, v in counts.items()))
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
