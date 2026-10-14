"""Shared configuration: one YAML template, one dict, one fingerprint.

Single source of truth
----------------------
``DEFAULT_YAML`` is the text of ``config/default.yaml`` *including comments*.
``DEFAULT_CONFIG`` is parsed from that same string, so the code defaults and
the file written by ``scripts/bootstrap_repo.py`` can never drift apart.

Cache fingerprint
-----------------
``config_fingerprint`` hashes only the sections that change *results*
(``CACHE_SECTIONS``).  Editing ``output`` or ``dashboard`` settings therefore
does not invalidate cached runs.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import yaml

from pipeline.errors import ConfigError

#: Repository root, resolved from this file so scripts work from any cwd
#: (Git Bash users `cd` around a lot).
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_YAML = """\
# FloodResQ shared configuration  (owner: T4; change via pull request)
# Sections listed in pipeline/config.py:CACHE_SECTIONS affect cached results.
# Relative paths are resolved against the repository root.

limits:
  max_aoi_km2: 2500          # reject larger boxes (keeps live demos fast)
  min_aoi_km2: 0.5

sentinel1:                   # owner: T1
  # auto = cdse_snap if ESA SNAP + CDSE_USERNAME/CDSE_PASSWORD exist, else planetary_computer
  # cdse_snap          Copernicus Data Space GRD download + ESA SNAP (reference chain)
  # planetary_computer Sentinel-1 RTC streamed from Microsoft Planetary Computer (no SNAP)
  # local              your own preprocessed GeoTIFFs (see 'local' below)
  backend: auto
  # Same relative orbit => acquisitions are a multiple of this many days apart.
  # The brief says 12 days; with two satellites flying it can be 6.
  gap_multiple_days: 6
  search_window_days: 40     # +/- days around the event (doubled if nothing is found)
  min_coverage: 0.95         # share of the AOI both images must cover
  min_coverage_relaxed: 0.6
  snap_mem_gb: 8
  snap_dem_path: ""          # optional local DEM GeoTIFF for SNAP terrain correction
  local:                     # only for backend: local (GeoTIFFs in a UTM CRS, bands VV, VH in dB)
    before: ""
    after: ""
    before_date: ""
    after_date: ""
    relative_orbit: 0
    orbit_direction: ""

sentinel2:                   # owner: T1 (optional; needs CDSE credentials to download bands)
  enabled: false

dem:
  local_path: ""             # folder of Copernicus GLO-30 tiles or one GeoTIFF; empty = download

osm:
  provider: ohsome           # ohsome (named in the brief), automatic fallback to an Overpass attic query
  overpass_fallback: true
  # Must be strictly BEFORE the event date (data rule). Latest per the brief:
  snapshot_date: "2026-07-27"
  network_buffer_km: 5       # roads and destinations are fetched this far beyond the AOI

model:                       # owner: T2
  version: unet-baseline-0   # T2 bumps this whenever weights change
  use_unet: true
  package: models/release/model_package   # verified package folder (models/pack_model.py)
  weights_path: models/release/unet_best.pt
  device: auto
  tta: false

infer:                       # owner: T2 (drop/water/debris values are used only without a model)
  exclude_permanent_water: true
  mask_unreliable_terrain: true   # drop radar layover/shadow pixels (DEM-based)
  use_optical_evidence: true      # add Sentinel-2 new-water / vegetation-loss evidence if present
  min_blob_pixels: 10
  fill_hole_pixels: 8
  drop_db: 3.0
  water_db: -15.0
  detect_debris: true
  debris_change_db: 4.0         # smoothed |VV change| that counts as debris near the river
  river_db: -18.0                # 'already water before' proxy if no permanent-water mask
  smooth_px: 2                   # 5 x 5 averaging against speckle
  debris_max_slope_deg: 15.0
  debris_max_dist_m: 1000.0

damage:                      # owner: T3
  min_impact_fraction: 0.10  # share of a building/bridge that must overlap the mask
  major_impact_fraction: 0.50
  road_min_block_m: 20       # a wet stretch this long blocks a road segment
  edge_buffer_m: 10
  min_flood_patch_m2: 1000   # drop smaller flood specks before the overlay

connectivity:                # owner: T3
  block_flooded_bridges: true   # debris floods destroy bridges and their approaches
  include_footpaths: false      # the brief asks about ROAD connections
  max_snap_m: 1500
  detour_ratio: 1.5
  detour_extra_min: 30

pipeline:
  max_workers: 3             # stages that do not depend on each other run in parallel
  slow_stage_seconds: 120    # only a warning in run_log.json; never aborts a stage

report:                      # owner: T4 (not part of the cache fingerprint)
  languages: [en, ne]        # en is always written; ne = Nepali text + printable HTML
  nepali_reviewed: false     # set true ONLY after a native speaker has checked reports/strings.py

output:
  root: outputs              # relative paths are resolved against the repo root
  geojson_crs: EPSG:4326

dashboard:
  max_buildings_drawn: 5000
"""

DEFAULT_CONFIG: dict = yaml.safe_load(DEFAULT_YAML)

#: Sections whose values influence the produced files.
CACHE_SECTIONS = ("sentinel1", "sentinel2", "dem", "osm", "model", "infer", "damage", "connectivity")


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: ``override`` wins, nested dicts are merged key by key.

    This is why an old, shorter ``default.yaml`` keeps working after we add a
    new setting to ``DEFAULT_YAML``: missing keys fall back to the defaults.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_dotenv(path: Optional[str | Path] = None) -> None:
    """Read ``KEY=value`` lines from ``.env`` into ``os.environ`` (existing variables win).

    Tiny on purpose (no python-dotenv dependency).  ``COPERNICUS_USERNAME`` /
    ``COPERNICUS_PASSWORD`` are accepted as aliases of the ``CDSE_*`` names that
    the Sentinel download code reads.
    """
    import os
    env_path = Path(path) if path else REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
    for new, old in (("CDSE_USERNAME", "COPERNICUS_USERNAME"), ("CDSE_PASSWORD", "COPERNICUS_PASSWORD")):
        if not os.environ.get(new) and os.environ.get(old):
            os.environ[new] = os.environ[old]


def load_config(path: Optional[str | Path] = None, overrides: Optional[dict] = None) -> dict:
    """Defaults <- ``config/default.yaml`` (if present) <- ``overrides``.  Also loads ``.env``."""
    load_dotenv()
    cfg_path = Path(path) if path else REPO_ROOT / "config" / "default.yaml"
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if cfg_path.exists():
        try:
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{cfg_path.name} is not valid YAML.", str(exc)) from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"{cfg_path.name} must contain a mapping at the top level.")
        cfg = deep_merge(cfg, loaded)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def config_fingerprint(cfg: dict) -> str:
    """Full SHA-1 hex digest of the result-relevant config sections."""
    relevant = {k: cfg.get(k) for k in CACHE_SECTIONS}
    # default=str: YAML turns an unquoted 2026-07-27 into a ``date`` object,
    # which json cannot serialise; str() keeps the fingerprint stable.
    blob = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def cache_sections(cfg: dict) -> dict[str, Any]:
    """The subset of ``cfg`` that is stored in ``run.json`` for reproducibility."""
    return {k: copy.deepcopy(cfg.get(k)) for k in CACHE_SECTIONS}
