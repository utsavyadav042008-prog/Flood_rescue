"""Regenerate the situation report of an existing run (no pipeline re-run).

    python scripts/make_report.py outputs/trishuli_2026-08-26_ab12cd34
    python scripts/make_report.py data/sample/mock_run --print          # also show the text
    python scripts/make_report.py <run_dir> --lang en                   # English only

Handy on Day 12 when you are polishing wording in ``reports/strings.py``:
edit, run this, open ``situation_report.pdf``; seconds instead of a full run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import load_config                  # noqa: E402
from pipeline.contracts import Artifact                  # noqa: E402
from pipeline.errors import PipelineError                # noqa: E402
from pipeline.fileio import read_json                    # noqa: E402
from pipeline.run import RunPaths                        # noqa: E402
from pipeline.schemas import Summary                     # noqa: E402
from reports.generator import write_reports              # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate a run's situation report from summary.json.")
    ap.add_argument("run_dir")
    ap.add_argument("--lang", choices=("en", "ne", "all"), default="all")
    ap.add_argument("--print", dest="show", action="store_true", help="print the English text report")
    args = ap.parse_args(argv)

    try:
        run = RunPaths.open(args.run_dir)
        summary = Summary.from_dict(read_json(run.path(Artifact.SUMMARY)))
    except PipelineError as exc:
        print(f"ERROR: {exc.user_message}")
        return 1
    except (OSError, ValueError):
        print("ERROR: this run has no readable summary.json (run the pipeline first).")
        return 1

    cfg = load_config()
    cfg["report"]["languages"] = ["en"] if args.lang == "en" else ["en", "ne"]
    for path in write_reports(run, summary, cfg):
        print(f"wrote {path}")
    if args.show:
        print()
        print(run.path(Artifact.REPORT_TXT).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
