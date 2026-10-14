"""Print the numbers for the 6-page report as Markdown, straight from the system's own files.

    python scripts/report_facts.py data/sample/trishuli_run

Paste the output into the report instead of retyping numbers, so the report can
never disagree with the dashboard (same source: summary.json and
evaluation/case_study_results.json).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.contracts import Artifact                  # noqa: E402
from pipeline.fileio import read_json                    # noqa: E402
from pipeline.run import RunPaths                        # noqa: E402
from reports.format import ATTRIBUTION_LINES, NA, fmt    # noqa: E402

RESULTS = REPO_ROOT / "evaluation" / "case_study_results.json"


def _pct(v) -> str:
    return NA if v is None else f"{100 * v:.0f}%"


def facts_markdown(run: RunPaths, case_study_path: Path = RESULTS) -> str:
    f = fmt(read_json(run.path(Artifact.SUMMARY)))
    lines = ["## Key results (from summary.json)", "", "| Quantity | Value |", "|---|---|",
             f"| Area | {f['aoi_name']} |", f"| Event date | {f['event_date']} |",
             f"| Imagery | before {f['before_date']} / after {f['after_date']}, orbit {f['sentinel1_orbit']} |",
             f"| OSM snapshot | {f['osm_snapshot_date']} |", f"| Model | {f['model_version']} |",
             f"| Flooded / debris area | {f['flood_area_km2']} km2 |",
             f"| Road segments affected | {f['affected_roads']} ({f['affected_road_length_km']} km) |",
             f"| Bridges affected | {f['affected_bridges']} |",
             f"| Buildings potentially affected | {f['affected_buildings']} |",
             f"| Settlements with no road connection | {f['cutoff_settlements']} |",
             f"| Mean model confidence | {f['mean_confidence']} |", ""]
    try:
        cs = read_json(case_study_path)
    except (OSError, ValueError):
        cs = None
    lines.append("## Case-study comparison (check-only)")
    lines.append("")
    if cs is None:
        lines.append("_No case-study results yet. Run `python -m evaluation.case_study ...`._")
    else:
        fe = cs.get("flood_extent", {})
        lines += [f"Reference: {cs.get('reference')}. Credit: {cs.get('credit', '')}", "",
                  "| IoU | Precision | Recall | F1 |", "|---|---|---|---|",
                  "| " + " | ".join(NA if fe.get(k) is None else f"{fe[k]:.2f}"
                                    for k in ("iou", "precision", "recall", "f1")) + " |", "",
                  "| Objects | In reference | Also found by us | Only by us |", "|---|---|---|---|"]
        for key in ("roads", "bridges", "buildings"):
            i = cs.get("infrastructure", {}).get(key, {})
            lines.append(f"| {key} | {i.get('reference_flagged', NA)} | {i.get('also_flagged', NA)} | "
                         f"{i.get('extra_flagged', NA)} |")
        lines += ["", "Notes:"] + [f"- {n}" for n in cs.get("notes", [])]
    lines += ["", "## Attribution (copy exactly)", ""] + [f"> {a}" for a in ATTRIBUTION_LINES]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    args = ap.parse_args(argv)
    try:
        print(facts_markdown(RunPaths.open(args.run_dir)))
    except Exception as exc:
        print(f"ERROR: {getattr(exc, 'user_message', exc)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
