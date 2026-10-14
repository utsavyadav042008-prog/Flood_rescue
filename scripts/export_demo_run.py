"""Copy a finished run into ``data/sample/trishuli_run`` so "Load cached demo" works instantly.

    python scripts/export_demo_run.py outputs/<run_dir>

Refuses mock runs (so a demo can never silently be fake) and files too large
for git; skips dashboard cache files (``figures/_overlay_*``, ``_preview_*``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.contracts import Artifact, check_run              # noqa: E402
from pipeline.errors import InputValidationError, PipelineError  # noqa: E402
from pipeline.fileio import read_json                           # noqa: E402
from pipeline.run import RunPaths                               # noqa: E402

DEFAULT_DEST = REPO_ROOT / "data" / "sample" / "trishuli_run"
MAX_FILE_MB = 5


def export_demo_run(run_dir: Path, dest: Path = DEFAULT_DEST, *, allow_mock: bool = False,
                    max_mb: float = MAX_FILE_MB, include_extras: bool = False) -> list:
    """Copy the run; returns the list of copied relative paths.  Raises on mock data / oversize files."""
    run = RunPaths.open(run_dir)
    if not allow_mock:
        summary = {}
        try:
            summary = read_json(run.path(Artifact.SUMMARY))
        except (OSError, ValueError):
            pass
        if run.manifest().get("is_mock") or summary.get("is_mock"):
            raise InputValidationError("This run contains MOCK data; the demo must be a real run "
                                       "(use --allow-mock only for development).")
    files = [p for p in run.root.rglob("*") if p.is_file()
             and not p.name.startswith(("_overlay_", "_preview_")) and not p.name.endswith(".tmp")]
    if not include_extras:   # DEM / masks are large; the dashboard works without them (no flood-path tab)
        files = [p for p in files if not (p.relative_to(run.root).parts[0] == "extras" and p.suffix == ".tif")]
    too_big = [p for p in files if p.stat().st_size > max_mb * 1024 * 1024]
    if too_big:
        raise InputValidationError(
            "These files are larger than the git limit: "
            + ", ".join(f"{p.relative_to(run.root)} ({p.stat().st_size / 1_048_576:.1f} MB)" for p in too_big)
            + ". Re-run on a smaller area or crop the rasters.")
    if dest.exists():
        shutil.rmtree(dest)
    copied = []
    for p in files:
        target = dest / p.relative_to(run.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        copied.append(p.relative_to(run.root).as_posix())
    return sorted(copied)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--allow-mock", action="store_true")
    ap.add_argument("--include-extras", action="store_true",
                    help="also copy extras/*.tif (DEM, masks; needed by the flood-path tab; may be large)")
    args = ap.parse_args(argv)
    try:
        copied = export_demo_run(Path(args.run_dir), Path(args.dest), allow_mock=args.allow_mock,
                                 include_extras=args.include_extras)
    except PipelineError as exc:
        print(f"ERROR: {exc.user_message}")
        return 1
    print(f"Copied {len(copied)} files to {args.dest}")
    bad = [(s, i) for s, issues in check_run(RunPaths(Path(args.dest)), forbid_mock=not args.allow_mock).items()
           for i in issues if i.severity == "error"]
    for stage, issue in bad:
        print(f"  contract problem [{stage}]: {issue}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
