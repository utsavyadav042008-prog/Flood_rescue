"""Run folders: one directory per (AOI, event date, config, model).

``RunPaths`` is the object every stage receives.  It knows the run folder and
translates an :class:`~pipeline.contracts.Artifact` into a real path, so no
module ever builds a file name by hand::

    run.path(Artifact.FLOOD_MASK)      # <run_dir>/flood_mask.tif
    run.ensure_parent(Artifact.OSM_ROADS)  # makes <run_dir>/osm/ and returns the path

``run.json`` (the manifest) is the first file written.  It stores the AOI,
the event date and the config actually used, so a finished folder explains
itself and validators can find the event date without extra arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from pipeline.config import (
    REPO_ROOT, cache_sections, config_fingerprint, load_config,
)
from pipeline.contracts import CONTRACT_VERSION, Artifact
from pipeline.errors import InputValidationError
from pipeline.fileio import read_json, write_json
from pipeline.schemas import AOI, parse_date, parse_event_date, run_id, utc_now_iso

#: Sub-folders created inside every run folder.
SUBDIRS = ("osm", "figures")


@dataclass(frozen=True)
class RunPaths:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    # -- path helpers ----------------------------------------------------
    def path(self, artifact: Artifact | str) -> Path:
        """Absolute path of an artifact. Unknown names raise ``ValueError``."""
        return self.root / Artifact(artifact).value

    __getitem__ = path

    def exists(self, artifact: Artifact | str) -> bool:
        return self.path(artifact).exists()

    def ensure_parent(self, artifact: Artifact | str) -> Path:
        """Create the artifact's parent folder (e.g. ``osm/``) and return its path."""
        p = self.path(artifact)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # -- manifest ----------------------------------------------------------
    def manifest(self) -> dict:
        return read_json(self.path(Artifact.RUN_MANIFEST))

    @property
    def aoi(self) -> AOI:
        return AOI.from_dict(self.manifest()["aoi"])

    @property
    def event_date(self) -> date:
        return parse_date(self.manifest()["event_date"], "event date")

    # -- construction --------------------------------------------------------
    @classmethod
    def open(cls, root: str | Path) -> "RunPaths":
        """Open an existing run folder (must contain run.json)."""
        run = cls(Path(root))
        if not run.path(Artifact.RUN_MANIFEST).exists():
            raise InputValidationError(
                "This folder is not a FloodResQ run (run.json is missing).", str(root))
        return run

    @classmethod
    def create(
        cls,
        aoi: AOI,
        event_date: object,
        cfg: Optional[dict] = None,
        out_root: Optional[str | Path] = None,
        *,
        exact_dir: Optional[str | Path] = None,
        is_mock: bool = False,
    ) -> "RunPaths":
        """Validate inputs, create the run folder and write ``run.json``.

        Idempotent for normal runs: identical inputs map to the same folder
        (name ends with the deterministic run id), so a second call reuses it
        and keeps the original ``created_utc``.

        ``exact_dir`` bypasses the hashed folder name (used for the committed
        mock/demo runs); its manifest is always rewritten.
        """
        cfg = cfg or load_config()
        limits = cfg.get("limits", {})
        aoi.validate(max_km2=float(limits.get("max_aoi_km2", 2500)),
                     min_km2=float(limits.get("min_aoi_km2", 0)))
        event = parse_event_date(event_date)

        fingerprint = config_fingerprint(cfg)
        model_version = str(cfg.get("model", {}).get("version", ""))
        rid = run_id(aoi, event, fingerprint, model_version)

        if exact_dir is not None:
            root = Path(exact_dir)
        else:
            base = Path(out_root) if out_root else Path(cfg.get("output", {}).get("root", "outputs"))
            if not base.is_absolute():
                base = REPO_ROOT / base      # relative to the repo, never to the cwd
            # Reuse an existing folder with the same id even if the AOI was
            # renamed since (the name is cosmetic, the id is the cache key).
            existing = sorted(base.glob(f"*_{event.isoformat()}_{rid}")) if base.exists() else []
            root = existing[0] if existing else base / f"{aoi.slug()}_{event.isoformat()}_{rid}"

        run = cls(root)
        run.root.mkdir(parents=True, exist_ok=True)
        for sub in SUBDIRS:
            (run.root / sub).mkdir(exist_ok=True)

        manifest_path = run.path(Artifact.RUN_MANIFEST)
        if exact_dir is not None or not manifest_path.exists():
            write_json(manifest_path, {
                "contract_version": CONTRACT_VERSION,
                "run_id": rid,
                "aoi": aoi.to_dict(),
                "event_date": event.isoformat(),
                "config_fingerprint": fingerprint,
                "model_version": model_version,
                "created_utc": utc_now_iso(),
                "is_mock": is_mock,
                # The config sections that influenced this run, for reproducibility.
                "config": cache_sections(cfg),
            })
        return run
