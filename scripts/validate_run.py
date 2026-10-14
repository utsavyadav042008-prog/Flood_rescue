"""Check a run folder against the data contract.

Usage (Git Bash, repo root)::

    python scripts/validate_run.py outputs/trishuli_2026-08-26_ab12cd34
    python scripts/validate_run.py data/sample/mock_run --stage damage --stage connectivity
    python scripts/validate_run.py <run_dir> --strict --forbid-mock     # Day-13 final audit

Exit code 0 = no blocking problems, 1 = problems found.
Teammates: run this on your own stage output BEFORE telling T4 "it's done".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.contracts import CONTRACTS, ERROR, STAGE_ORDER, check_run   # noqa: E402
from pipeline.errors import PipelineError                                  # noqa: E402
from pipeline.run import RunPaths                                          # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate a FloodResQ run folder.")
    ap.add_argument("run_dir")
    ap.add_argument("--stage", action="append", choices=STAGE_ORDER,
                    help="validate only this stage (repeatable); default: every stage that has output")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    ap.add_argument("--forbid-mock", action="store_true",
                    help="fail if any file carries is_mock=true (use before submitting)")
    args = ap.parse_args(argv)

    try:
        run = RunPaths.open(args.run_dir)
    except PipelineError as exc:
        print(f"ERROR: {exc.user_message}")
        return 1

    # An explicit --stage means "I expect output": missing files then count as errors.
    results = check_run(run, args.stage, forbid_mock=args.forbid_mock,
                        only_existing=not args.stage)
    blocking = 0
    for stage, issues in results.items():
        owner = CONTRACTS[stage].owner
        bad = [i for i in issues if i.severity == ERROR or args.strict]
        blocking += len(bad)
        print(f"{'FAIL' if bad else 'ok  '}  {stage:<13} ({owner})  "
              f"{len(issues)} issue{'s' if len(issues) != 1 else ''}")
        for issue in issues:
            print(f"      {issue}")
    if not results:
        print("No stage outputs found in this run folder yet.")
    print("\nRESULT:", "PASS" if blocking == 0 else f"FAIL ({blocking} blocking problem(s))")
    return 0 if blocking == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
