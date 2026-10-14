"""Mock stages: fake T1/T2/T3 outputs so the pipeline runs end to end today.

How it works
------------
A *template* mock run is generated once (``scripts/make_mock_outputs.py``) in
``outputs/_mock_template``.  A mock stage copies the files that stage is
contracted to produce from the template into the real run folder.  Two files
are regenerated instead of copied, because they depend on the user's date:

* ``metadata.json``       scene dates = event -6 days / +6 days (same orbit)
* ``osm/snapshot.json``   the configured pre-event OSM snapshot date

so the real validators (bracketing rule, OSM-before-event rule) stay honest
even for mocks.  Mock output always shows the *mock valley*, whatever AOI the
user typed; the summary says so through ``is_mock`` and a warning.

Everything downstream of the mocked stages (summarize, report, figures,
dashboard) is the REAL code, so mocks test the real aggregator.
"""

from __future__ import annotations

import shutil
import sys
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from pipeline.config import REPO_ROOT
from pipeline.contracts import CONTRACTS, Artifact
from pipeline.errors import AcquisitionError, GISError, ModelError, PipelineError
from pipeline.fileio import atomic_path, rasterio_available, read_json, write_json
from pipeline.rules import effective_osm_snapshot
from pipeline.run import RunPaths
from pipeline.schemas import AOI, Provenance

#: Where the template mock run lives (module variable so tests can redirect it).
TEMPLATE_DIR = REPO_ROOT / "outputs" / "_mock_template"

_ERRORS: dict[str, type[PipelineError]] = {
    "acquire": AcquisitionError, "infer": ModelError,
    "osm": GISError, "damage": GISError, "connectivity": GISError,
}
_RASTER_STAGES = ("acquire", "infer")
_template_lock = threading.Lock()     # acquire and osm run in parallel threads!


def ensure_template() -> RunPaths:
    """Build the template mock run if it is missing (thread-safe, once)."""
    with _template_lock:
        template = RunPaths(TEMPLATE_DIR)
        have_vectors = template.path(Artifact.DAMAGE_SUMMARY).exists()
        have_rasters = template.path(Artifact.FLOOD_MASK).exists()
        # Rebuild if rasterio has appeared since a vector-only template was made.
        if have_vectors and (have_rasters or not rasterio_available()):
            return template
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from scripts.make_mock_outputs import build_mock_run
        if TEMPLATE_DIR.exists():
            shutil.rmtree(TEMPLATE_DIR)
        return build_mock_run(TEMPLATE_DIR, with_rasters=rasterio_available())


def _copy(template: RunPaths, run: RunPaths, artifact: Artifact, stage: str) -> None:
    src = template.path(artifact)
    if not src.exists():
        if stage in _RASTER_STAGES:
            raise _ERRORS[stage](
                "Mock rasters need rasterio. Install it (pip install rasterio) "
                "or use conda-forge on Windows.",
                f"mock template has no {artifact.value}")
        raise _ERRORS[stage](f"Mock template is missing {artifact.value}.")
    dst = run.ensure_parent(artifact)
    with atomic_path(dst) as tmp:                    # atomic, like every other writer
        shutil.copyfile(src, tmp)


def _mock_provenance(stage: str, run: RunPaths) -> Provenance:
    manifest = run.manifest()
    return Provenance(producer=f"mock.{stage}", source="MOCK synthetic data (not real imagery)",
                      crs="", event_date=manifest["event_date"],
                      model_version=manifest.get("model_version", ""), is_mock=True)


def mock_stage(stage: str) -> Callable[[RunPaths, AOI, date, dict], None]:
    """Return the mock implementation of ``stage``."""
    if stage not in _ERRORS:
        raise ValueError(f"no mock for stage {stage!r}")
    contract = CONTRACTS[stage]

    def run_mock(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None:
        template = ensure_template()
        # Date-dependent files are rebuilt, not copied (see module docstring).
        regenerated: set[Artifact] = set()
        if stage == "acquire":
            meta = read_json(template.path(Artifact.METADATA))
            meta.pop("provenance", None)
            meta["before_date"] = (event_date - timedelta(days=6)).isoformat()
            meta["after_date"] = (event_date + timedelta(days=6)).isoformat()
            write_json(run.path(Artifact.METADATA), meta, _mock_provenance(stage, run))
            regenerated.add(Artifact.METADATA)
        if stage == "osm":
            write_json(run.ensure_parent(Artifact.OSM_SNAPSHOT),
                       {"snapshot_date": effective_osm_snapshot(cfg["osm"]["snapshot_date"], event_date).isoformat(),
                        "provider": cfg["osm"]["provider"]},
                       _mock_provenance(stage, run))
            regenerated.add(Artifact.OSM_SNAPSHOT)
        for artifact in contract.produces:
            if artifact not in regenerated:
                _copy(template, run, artifact, stage)

    run_mock.__name__ = f"mock_{stage}"
    return run_mock
