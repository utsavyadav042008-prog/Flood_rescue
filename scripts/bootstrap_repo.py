"""Day-1 setup: create the whole repo skeleton in one command.

    python scripts/bootstrap_repo.py            # create what is missing
    python scripts/bootstrap_repo.py --dry-run  # show what would happen
    python scripts/bootstrap_repo.py --force    # also overwrite hand-editable files

What it creates
---------------
* the folder tree (one package per teammate) with ``__init__.py`` files
* requirements files, ``.gitignore``, ``.env.example``, README stub
* ``config/default.yaml`` (generated from ``pipeline.config.DEFAULT_YAML``)
* ``config/example_aoi.json`` (PLACEHOLDER corridor box for the case study)
* one *stub* per teammate stage, auto-written from the contract table
* ``docs/contracts.md`` (always regenerated; never edit it by hand)

Safety rule (the "trick"): existing files are NEVER overwritten unless you
pass ``--force``.  Re-running the script after teammates started coding is
therefore harmless.  Only auto-generated docs (``docs/contracts.md``) are
refreshed every time, because they are derived from code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import DEFAULT_YAML                         # noqa: E402
from pipeline.contracts import CONTRACTS, contracts_markdown     # noqa: E402
from pipeline.schemas import AOI                                  # noqa: E402
from reports.format import ATTRIBUTION_LINES                      # noqa: E402

PACKAGES = {                      # folder -> owner
    "acquisition": "T1 (Sentinel/DEM) + T3 (OSM)",
    "preprocessing": "T1",
    "models": "T2",
    "gis": "T3",
    "pipeline": "T4",
    "dashboard": "T4",
    "reports": "T4",
    "evaluation": "T2/T3 compute, T4 displays",
}
#: Git-ignored folders: created empty on disk, never committed.
IGNORED_DIRS = ("data/raw", "data/processed", "data/reference", "outputs")
#: Folders that ARE committed, so they get a .gitkeep placeholder.
TRACKED_DIRS = ("data/sample", "models/release")

REQUIREMENTS = {
    "requirements.txt": """\
# Core + dashboard. Pin exact versions with `pip freeze` after Day 3.
numpy
scipy
pandas
pyyaml
rasterio
rioxarray
xarray
geopandas
shapely
pyproj
networkx
osmnx
pystac-client
planetary-computer
scikit-image
scikit-learn
matplotlib
pillow
requests
streamlit
folium
streamlit-folium
plotly
reportlab
""",
    "requirements-ml.txt": """\
# T2 only (large download). Keeps everyone else's install light.
torch
torchvision
segmentation-models-pytorch
albumentations
""",
    "requirements-dev.txt": """\
pytest
ruff
black
pypdf
pre-commit
""",
}

GITIGNORE = """\
# environments
.venv/
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
.ruff_cache/
# data and outputs (never commit these)
data/raw/
data/processed/
data/reference/
outputs/
# model checkpoints (final weights go to models/release/ after a team decision)
*.pt
*.pth
*.ckpt
!models/release/*.pt
# demo video (large; submitted separately, never committed)
submission/*.mp4
submission/*.mov
submission/*.mkv
submission/*.webm
# editor / OS
.vscode/
.idea/
.DS_Store
Thumbs.db
# atomic-write temp files (pipeline/fileio.py)
*.tmp
"""

ENV_EXAMPLE = """\
# Copy to .env (git-ignored) and fill in. NEVER commit real credentials.
COPERNICUS_USERNAME=
COPERNICUS_PASSWORD=
"""

# Exact wording required by the brief lives in ONE place (reports/format.py).
ATTRIBUTION = "\n\n".join(ATTRIBUTION_LINES) + "\n"

README = """\
# FloodResQ

Educational prototype: maps flood/debris extent, affected infrastructure and
cut-off settlements from Sentinel-1 radar, Copernicus DEM and pre-event
OpenStreetMap. Multimodal AI Hackathon 2026, Track B. Not an operational tool.

## Install (Windows 10, Git Bash)
```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
python scripts/install_git_hooks.py
```

## Check your setup
```bash
python scripts/make_mock_outputs.py          # fake run in data/sample/mock_run
python scripts/validate_run.py data/sample/mock_run
python -m unittest discover -s tests -t .    # or: pytest -q
```

## Data contract
See `docs/contracts.md` (generated from `pipeline/contracts.py`).

## Attribution
{attribution}
"""

DOC_STUBS = {
    "docs/data_sources.md": "# Data sources\n\nSentinel-1/2, Copernicus DEM, OSM (ohsome pre-event snapshot), Kuro Siwo.\n",
    "docs/limitations.md": "# Limitations\n\n(T4 fills this in; see Part 5 of the workflow.)\n",
    "docs/attribution.md": "# Attribution\n\n" + ATTRIBUTION,
}


def stub_source(stage: str) -> tuple[str, str]:
    """Return ``(relative_path, source_text)`` for a teammate stage entry point."""
    c = CONTRACTS[stage]
    module, func = c.entrypoint.split(":")
    reads = "\n".join(f"    {a.value}" for a in c.needs)
    writes = "\n".join(f"    {a.value}" for a in c.produces)
    text = f'''"""Stage '{stage}' - owner {c.owner}.

{c.summary}

STUB written by scripts/bootstrap_repo.py from pipeline/contracts.py.
Replace the body; keep the signature.

Reads (from the run folder):
{reads}
Writes (into the run folder):
{writes}

Use run.path(Artifact.X) for every file name, write with pipeline.fileio, and
check your work with:  python scripts/validate_run.py <run_dir> --stage {stage}
"""

from __future__ import annotations

from datetime import date

from pipeline.contracts import Artifact  # noqa: F401  (use: run.path(Artifact.FLOOD_MASK))
from pipeline.run import RunPaths
from pipeline.schemas import AOI


def {func}(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None:
    raise NotImplementedError("stage '{stage}' is not implemented yet (owner: {c.owner})")
'''
    return module.replace(".", "/") + ".py", text


def plan(root: Path) -> list[tuple[str, str, bool]]:
    """List ``(relative_path, content, always_refresh)`` for every file to create."""
    files: list[tuple[str, str, bool]] = []
    for name, content in REQUIREMENTS.items():
        files.append((name, content, False))
    files += [
        (".gitignore", GITIGNORE, False),
        (".env.example", ENV_EXAMPLE, False),
        ("README.md", README.format(attribution=ATTRIBUTION.strip()), False),
        ("config/default.yaml", DEFAULT_YAML, False),
        # PLACEHOLDER box: choose the real AOI from the brief's description of the
        # corridor, never from published damage-map extents (data rule).
        ("config/example_aoi.json",
         json.dumps(AOI(85.10, 27.85, 85.45, 28.30, name="Trishuli corridor (case study)").to_dict(),
                    indent=2) + "\n", False),
        ("docs/contracts.md", contracts_markdown(), True),
    ]
    files += [(p, c, False) for p, c in DOC_STUBS.items()]
    for folder, owner in PACKAGES.items():
        files.append((f"{folder}/__init__.py", f'"""Owner: {owner}."""\n', False))
    for stage, c in CONTRACTS.items():
        if c.owner in ("T1", "T2", "T3") and c.entrypoint:
            rel, text = stub_source(stage)
            files.append((rel, text, False))
    for d in TRACKED_DIRS:
        files.append((f"{d}/.gitkeep", "", False))
    return files


def bootstrap(root: Path = REPO_ROOT, force: bool = False, dry_run: bool = False) -> list[tuple[str, str]]:
    """Create the skeleton. Returns ``[(relative_path, status)]``.

    status is one of ``created``, ``skipped`` (already exists), ``updated``
    (regenerated or --force).
    """
    report = []
    for d in IGNORED_DIRS:
        if not dry_run:
            (root / d).mkdir(parents=True, exist_ok=True)
    for rel, content, always in plan(root):
        path = root / rel
        if path.exists() and not (force or always):
            report.append((rel, "skipped"))
            continue
        status = "updated" if path.exists() else "created"
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
        report.append((rel, status))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--dry-run", action="store_true", help="only print what would change")
    args = ap.parse_args(argv)
    report = bootstrap(force=args.force, dry_run=args.dry_run)
    for rel, status in report:
        if status != "skipped":
            print(f"{status:8} {rel}")
    skipped = sum(1 for _, s in report if s == "skipped")
    print(f"\n{len(report) - skipped} written, {skipped} already present (left untouched).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
