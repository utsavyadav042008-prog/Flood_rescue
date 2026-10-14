"""Stage 'summarize' (T4): merge every stage summary into ONE ``summary.json``.

``build_summary`` only READS files from the run folder; it never calls other
stages.  Missing inputs do not crash it: the matching fields stay ``None``
("not available"), and a warning explains why.  That is what lets the
pipeline degrade gracefully when e.g. OSM fails.

Numbers come from exactly three places (so the dashboard and the report can
never disagree):

* ``damage_summary.json`` and ``connectivity_summary.json``  (T3)
* the flood-mask / confidence rasters                       (T2, via rasterio)
* ``metadata.json`` / ``osm/snapshot.json`` / ``run.json``   (labels only)
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pipeline.contracts import Artifact
from pipeline.fileio import mask_area_km2, read_json, read_provenance, write_json
from pipeline.run import RunPaths
from pipeline.schemas import AOI, Provenance, Summary, utc_now_iso

#: Stages whose failure must be mentioned in the summary warnings.
#: (summarize/report/figures are T4's own stages and are not "data" stages.)
DATA_STAGES = ("acquire", "osm", "infer", "damage", "connectivity")

#: A "flood" covering more than this share of the AOI is suspicious.
SUSPICIOUS_FLOOD_FRACTION = 0.30


def _read(run: RunPaths, artifact: Artifact) -> Optional[dict]:
    """JSON file as dict, or ``None`` if missing/unreadable (never raises)."""
    try:
        data = read_json(run.path(artifact))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _number(data: Optional[dict], key: str, as_int: bool, warnings: list, label: str) -> Optional[float]:
    """Read ``data[key]`` as int/float; ``None`` plus a warning if absent or not numeric."""
    if data is None:
        return None
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        warnings.append(f"{label}: '{key}' is not a number ({value!r}); shown as not available.")
        return None
    return int(value) if as_int else round(float(value), 2)


def _raster_stats(run: RunPaths) -> tuple[Optional[float], Optional[float]]:
    """``(flood_area_km2, mean_confidence)``; either may be ``None``.

    mean_confidence = mean model probability over the pixels the mask calls
    flooded (so it answers "how sure is the model about what it flagged").
    Kept as its own function so tests can replace it without rasterio.
    """
    import numpy as np
    import rasterio

    mask_path, conf_path = run.path(Artifact.FLOOD_MASK), run.path(Artifact.FLOOD_CONFIDENCE)
    area = mask_area_km2(mask_path)
    mean_conf = None
    if conf_path.exists():
        with rasterio.open(mask_path) as m, rasterio.open(conf_path) as c:
            if (m.height, m.width) == (c.height, c.width):
                flooded = m.read(1) > 0
                if flooded.any():
                    mean_conf = round(float(c.read(1)[flooded].mean()), 3)
    return round(area, 2), mean_conf


def _orbit_text(meta: dict) -> str:
    orbit = meta.get("relative_orbit_before")
    direction = meta.get("orbit_direction", "")
    return f"{orbit} {direction}".strip() if orbit is not None else str(direction)


def _stage_warnings(run: RunPaths) -> list[str]:
    """Turn failed/skipped data stages recorded in run_log.json into readable warnings."""
    log = _read(run, Artifact.RUN_LOG)
    out = []
    for stage in (log or {}).get("stages", []):
        if stage.get("name") in DATA_STAGES and stage.get("status") in ("failed", "skipped"):
            out.append(f"Stage '{stage['name']}' {stage['status']}: {stage.get('message', '')}".strip())
    return out


def _stage_notes(run: RunPaths, meta: dict, snapshot: dict, connectivity: Optional[dict]) -> list[str]:
    """Caveats the data stages wrote into their own outputs (scene pair, OSM, model, routing)."""
    out: list[str] = []
    for w in meta.get("warnings", []) or []:
        out.append(f"Satellite data: {w}")
    for w in snapshot.get("warnings", []) or []:
        out.append(f"OpenStreetMap: {w}")
    try:
        infer = read_json(run.root / "extras" / "inference_meta.json")
        out += [f"Flood map: {w}" for w in infer.get("warnings", []) or []]
    except (OSError, ValueError):
        pass
    for w in (connectivity or {}).get("warnings", []) or []:
        out.append(f"Cut-off analysis: {w}")
    # keep the list readable on one page: drop exact duplicates, cap the length
    seen, unique = set(), []
    for w in out:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:12]


def build_summary(run: RunPaths) -> Summary:
    manifest = run.manifest()
    aoi = AOI.from_dict(manifest["aoi"])
    meta = _read(run, Artifact.METADATA) or {}
    snapshot = _read(run, Artifact.OSM_SNAPSHOT) or {}
    damage = _read(run, Artifact.DAMAGE_SUMMARY)
    connectivity = _read(run, Artifact.CONNECTIVITY_SUMMARY)
    warnings: list[str] = []

    s = Summary(
        aoi_name=aoi.name, event_date=manifest["event_date"],
        model_version=manifest.get("model_version", ""),
        before_date=str(meta.get("before_date", "")), after_date=str(meta.get("after_date", "")),
        sentinel1_orbit=_orbit_text(meta), crs=str(meta.get("crs", "")),
        osm_snapshot_date=str(snapshot.get("snapshot_date", "")), processing_date=utc_now_iso(),
    )

    # --- raster-derived numbers -------------------------------------------------
    if run.path(Artifact.FLOOD_MASK).exists():
        try:
            s.flood_area_km2, s.mean_confidence = _raster_stats(run)
        except Exception as exc:     # rasterio missing, unreadable file, geographic CRS ...
            warnings.append(f"Flood area could not be computed: {exc}")

    # --- T3 numbers ----------------------------------------------------------------
    s.affected_roads = _number(damage, "affected_roads", True, warnings, "damage_summary")
    s.affected_road_length_km = _number(damage, "affected_road_length_km", False, warnings, "damage_summary")
    s.affected_bridges = _number(damage, "affected_bridges", True, warnings, "damage_summary")
    s.affected_buildings = _number(damage, "affected_buildings", True, warnings, "damage_summary")
    s.cutoff_settlements = _number(connectivity, "cutoff_settlements", True, warnings, "connectivity_summary")
    if connectivity is not None:
        names = connectivity.get("cutoff_names")
        s.cutoff_names = [str(n) for n in names] if isinstance(names, list) else []

    # --- honesty flags -------------------------------------------------------------------
    inputs = (Artifact.METADATA, Artifact.OSM_SNAPSHOT, Artifact.DAMAGE_SUMMARY,
              Artifact.CONNECTIVITY_SUMMARY, Artifact.FLOOD_MASK)
    s.is_mock = bool(manifest.get("is_mock")) or any(
        (read_provenance(run.path(a)) or {}).get("is_mock") for a in inputs if run.path(a).exists())
    if s.is_mock:
        warnings.append("MOCK DATA: synthetic scenario, not a real result.")
    warnings += _stage_warnings(run)
    warnings += _stage_notes(run, meta, snapshot, connectivity)
    mask_prov = read_provenance(run.path(Artifact.FLOOD_MASK)) if run.path(Artifact.FLOOD_MASK).exists() else None
    if mask_prov and mask_prov.get("model_version"):
        s.model_version = str(mask_prov["model_version"])     # what actually produced the mask
    if s.flood_area_km2 == 0:
        warnings.append("No flooded pixels were detected in this area/date.")
    elif s.flood_area_km2 and s.flood_area_km2 > SUSPICIOUS_FLOOD_FRACTION * aoi.area_km2():
        warnings.append(f"Flooded area is {100 * s.flood_area_km2 / aoi.area_km2():.0f}% of the AOI; "
                        "unusually large, check for false positives (radar shadow, wet snow).")
    s.warnings = warnings
    return s


def run_summarize(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None:
    """Stage entry point (signature fixed by the contract)."""
    summary = build_summary(run)
    write_json(run.path(Artifact.SUMMARY), summary.to_dict(),
               Provenance(producer="pipeline.summary", source="Aggregated stage summaries",
                          crs=summary.crs, event_date=summary.event_date,
                          model_version=summary.model_version, is_mock=summary.is_mock))
