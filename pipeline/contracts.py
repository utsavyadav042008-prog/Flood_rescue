"""THE data contract: which stage reads which file, and what a valid file is.

This module is the heart of Part 1.  It answers three questions in code
instead of in a Word document:

1. WHAT files exist?            -> ``Artifact`` (enum of every file name)
2. WHO makes / needs them?      -> ``CONTRACTS`` (one ``StageContract`` per stage)
3. IS a file valid?             -> ``check_artifact`` / ``check_stage``

Teammates never type file names; they use ``run.path(Artifact.FLOOD_MASK)``.
The validators return *all* problems at once (not just the first), grouped
into a histogram, and never need ``rasterio``/``geopandas`` for the
vector/JSON checks.

Severity levels
---------------
``error``    the file breaks the contract (pipeline stops, or degrades)
``warning``  suspicious but usable (e.g. missing provenance stamp);
             promoted to an error with ``strict=True`` (final audit, Day 13)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import Enum
from typing import TYPE_CHECKING, Callable, Iterable, Optional

from pipeline.errors import (
    ContractError,
    DataRuleViolation,
    InputValidationError,
)
from pipeline.fileio import rasterio_available, read_json, read_provenance
from pipeline.rules import assert_osm_snapshot_pre_event
from pipeline.schemas import AOI, Summary, parse_date

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import
    from pipeline.run import RunPaths

#: Bump when a file name / required field changes; stored in run.json.
CONTRACT_VERSION = "1.0"

ERROR = "error"
WARNING = "warning"

#: Words allowed in ``impact_status``.  "destroyed" is deliberately absent:
#: overlap with a flood mask does not prove structural destruction.
ALLOWED_IMPACT_STATUS = ("affected", "impacted", "potentially_damaged")
ALLOWED_DESTINATION_TYPES = ("hospital", "town")

#: Coordinates are range-checked on at most this many features per file.
COORD_SAMPLE_TARGET = 2000


# ==========================================================================
# 1. WHAT files exist
# ==========================================================================
class Artifact(str, Enum):
    """Every file a run folder may contain (value = path relative to the run folder)."""

    RUN_MANIFEST = "run.json"
    # --- acquire (T1) ---
    BEFORE = "before.tif"
    AFTER = "after.tif"
    METADATA = "metadata.json"
    # --- osm (T3) ---
    OSM_SNAPSHOT = "osm/snapshot.json"
    OSM_ROADS = "osm/roads.geojson"
    OSM_BRIDGES = "osm/bridges.geojson"
    OSM_BUILDINGS = "osm/buildings.geojson"
    OSM_SETTLEMENTS = "osm/settlements.geojson"
    OSM_HOSPITALS = "osm/hospitals.geojson"      # hospitals AND towns (destination_type)
    # --- infer (T2) ---
    FLOOD_MASK = "flood_mask.tif"
    FLOOD_CONFIDENCE = "flood_confidence.tif"
    # --- damage (T3) ---
    AFFECTED_ROADS = "affected_roads.geojson"
    AFFECTED_BRIDGES = "affected_bridges.geojson"
    AFFECTED_BUILDINGS = "affected_buildings.geojson"
    DAMAGE_SUMMARY = "damage_summary.json"
    # --- connectivity (T3) ---
    CUTOFF_SETTLEMENTS = "cutoff_settlements.geojson"
    CONNECTIVITY_SUMMARY = "connectivity_summary.json"
    # --- T4 ---
    SUMMARY = "summary.json"
    REPORT_TXT = "situation_report.txt"
    REPORT_PDF = "situation_report.pdf"
    REPORT_HTML = "situation_report.html"
    REPORT_NE_TXT = "situation_report_ne.txt"
    REPORT_NE_HTML = "situation_report_ne.html"
    FIG_BEFORE = "figures/before.png"
    FIG_AFTER = "figures/after.png"
    FIG_FLOOD = "figures/flood_map.png"
    FIG_DAMAGE = "figures/damage_map.png"
    FIG_CONNECTIVITY = "figures/connectivity_map.png"
    RUN_LOG = "run_log.json"


class Kind(str, Enum):
    RASTER = "raster"
    GEOJSON = "geojson"
    JSON = "json"
    TEXT = "text"
    PDF = "pdf"
    PNG = "png"


_LINES = ("LineString", "MultiLineString")
_POLYS = ("Polygon", "MultiPolygon")
_POINT_OR_AREA = ("Point",) + _POLYS


@dataclass(frozen=True)
class ArtifactSpec:
    """What a valid file of this kind must contain."""

    kind: Kind
    description: str
    required_keys: tuple = ()          # JSON: top-level keys
    required_props: tuple = ()         # GeoJSON: feature property names
    geometry_types: tuple = ()         # GeoJSON: allowed geometry types
    provenance: bool = True            # expect a provenance stamp?


_AFFECTED_PROPS = ("object_id", "object_type", "impact_status", "impact_fraction", "confidence")

SPECS: dict[Artifact, ArtifactSpec] = {
    Artifact.RUN_MANIFEST: ArtifactSpec(
        Kind.JSON, "Run identity: AOI, event date, config used, contract version.",
        required_keys=("contract_version", "run_id", "aoi", "event_date",
                       "config_fingerprint", "model_version", "created_utc", "config"),
        provenance=False),
    Artifact.BEFORE: ArtifactSpec(
        Kind.RASTER, "Pre-event Sentinel-1 backscatter (band 1 = VV in dB), metric CRS."),
    Artifact.AFTER: ArtifactSpec(
        Kind.RASTER, "Post-event Sentinel-1, SAME grid and orbit track as before.tif."),
    Artifact.METADATA: ArtifactSpec(
        Kind.JSON, "Scene facts for the report and the same-orbit check.",
        required_keys=("source", "before_date", "after_date", "relative_orbit_before",
                       "relative_orbit_after", "orbit_direction", "crs")),
    Artifact.OSM_SNAPSHOT: ArtifactSpec(
        Kind.JSON, "Which pre-event OSM snapshot was used (data-rule evidence).",
        required_keys=("snapshot_date", "provider")),
    Artifact.OSM_ROADS: ArtifactSpec(
        Kind.GEOJSON, "Road segments (graph edges) before the event.",
        required_props=("id", "highway_type", "bridge", "name"), geometry_types=_LINES),
    Artifact.OSM_BRIDGES: ArtifactSpec(
        Kind.GEOJSON, "Bridge ways before the event.",
        required_props=("id", "name"), geometry_types=_LINES),
    Artifact.OSM_BUILDINGS: ArtifactSpec(
        Kind.GEOJSON, "Building footprints before the event.",
        required_props=("id", "type"), geometry_types=_POLYS),
    Artifact.OSM_SETTLEMENTS: ArtifactSpec(
        Kind.GEOJSON, "Villages / towns whose connectivity is analysed.",
        required_props=("id", "name"), geometry_types=_POINT_OR_AREA),
    Artifact.OSM_HOSPITALS: ArtifactSpec(
        Kind.GEOJSON, "Destinations: hospitals and towns.",
        required_props=("id", "name", "destination_type"), geometry_types=_POINT_OR_AREA),
    Artifact.FLOOD_MASK: ArtifactSpec(
        Kind.RASTER, "Binary flood/debris mask (0/1, no nodata), same grid as after.tif."),
    Artifact.FLOOD_CONFIDENCE: ArtifactSpec(
        Kind.RASTER, "Model probability of the flood class, float 0..1, same grid."),
    Artifact.AFFECTED_ROADS: ArtifactSpec(
        Kind.GEOJSON, "Road segments overlapping the flood mask.",
        required_props=_AFFECTED_PROPS, geometry_types=_LINES),
    Artifact.AFFECTED_BRIDGES: ArtifactSpec(
        Kind.GEOJSON, "Bridges overlapping the flood mask.",
        required_props=_AFFECTED_PROPS, geometry_types=_LINES),
    Artifact.AFFECTED_BUILDINGS: ArtifactSpec(
        Kind.GEOJSON, "Buildings overlapping the flood mask.",
        required_props=_AFFECTED_PROPS, geometry_types=_POLYS),
    Artifact.DAMAGE_SUMMARY: ArtifactSpec(
        Kind.JSON, "Counts and lengths of affected objects.",
        required_keys=("affected_roads", "affected_road_length_km",
                       "affected_bridges", "affected_buildings")),
    Artifact.CUTOFF_SETTLEMENTS: ArtifactSpec(
        Kind.GEOJSON, "Settlements with no road path to any destination.",
        required_props=("settlement_id", "name", "reason", "nearest_destination",
                        "blocking_segment_ids"),
        geometry_types=_POINT_OR_AREA),
    Artifact.CONNECTIVITY_SUMMARY: ArtifactSpec(
        Kind.JSON, "Cut-off counts and names.",
        required_keys=("cutoff_settlements", "cutoff_names", "settlements_total",
                       "destinations_used")),
    Artifact.SUMMARY: ArtifactSpec(
        Kind.JSON, "The single object read by dashboard AND report.",
        required_keys=tuple(f.name for f in fields(Summary))),
    Artifact.REPORT_TXT: ArtifactSpec(Kind.TEXT, "One-page situation report.", provenance=False),
    Artifact.REPORT_PDF: ArtifactSpec(Kind.PDF, "PDF of the report (English).", provenance=False),
    Artifact.REPORT_HTML: ArtifactSpec(Kind.TEXT, "Printable HTML of the report (English).", provenance=False),
    Artifact.REPORT_NE_TXT: ArtifactSpec(Kind.TEXT, "Nepali report, plain text.", provenance=False),
    Artifact.REPORT_NE_HTML: ArtifactSpec(Kind.TEXT, "Nepali report, printable HTML.", provenance=False),
    Artifact.FIG_BEFORE: ArtifactSpec(Kind.PNG, "Presentation figure.", provenance=False),
    Artifact.FIG_AFTER: ArtifactSpec(Kind.PNG, "Presentation figure.", provenance=False),
    Artifact.FIG_FLOOD: ArtifactSpec(Kind.PNG, "Presentation figure.", provenance=False),
    Artifact.FIG_DAMAGE: ArtifactSpec(Kind.PNG, "Presentation figure.", provenance=False),
    Artifact.FIG_CONNECTIVITY: ArtifactSpec(Kind.PNG, "Presentation figure.", provenance=False),
    Artifact.RUN_LOG: ArtifactSpec(Kind.JSON, "Stage statuses and timings.", provenance=False),
}


# ==========================================================================
# 2. WHO makes / needs them
# ==========================================================================
@dataclass(frozen=True)
class StageContract:
    """One pipeline stage.

    ``entrypoint`` is ``"module:function"``.  Every entry-point function has
    the SAME signature::

        def run_xxx(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None

    It reads its ``needs`` from ``run.path(...)`` and writes its ``produces``
    there.  ``required=True`` means the pipeline aborts if the stage fails.
    """

    name: str
    owner: str
    summary: str
    entrypoint: Optional[str]
    needs: tuple
    produces: tuple
    required: bool = False
    optional: tuple = ()               # extra outputs that are checked only if present


A = Artifact
CONTRACTS: dict[str, StageContract] = {c.name: c for c in (
    StageContract("init", "T4", "Create the run folder and run.json.", None,
                  needs=(), produces=(A.RUN_MANIFEST,), required=True),
    StageContract("acquire", "T1", "Fetch + align a same-orbit Sentinel-1 before/after pair.",
                  "acquisition.s1_stage:run_acquire",
                  needs=(A.RUN_MANIFEST,), produces=(A.BEFORE, A.AFTER, A.METADATA), required=True),
    StageContract("osm", "T3", "Download PRE-event OSM layers (runs in parallel with acquire).",
                  "acquisition.osm:run_osm",
                  needs=(A.RUN_MANIFEST,),
                  produces=(A.OSM_SNAPSHOT, A.OSM_ROADS, A.OSM_BRIDGES, A.OSM_BUILDINGS,
                            A.OSM_SETTLEMENTS, A.OSM_HOSPITALS)),
    StageContract("infer", "T2", "Segment flood/debris from the before/after pair.",
                  "models.stage:run_infer",
                  needs=(A.BEFORE, A.AFTER), produces=(A.FLOOD_MASK, A.FLOOD_CONFIDENCE),
                  required=True),
    StageContract("damage", "T3", "Intersect the mask with roads, bridges, buildings.",
                  "gis.damage:run_damage",
                  needs=(A.FLOOD_MASK, A.OSM_ROADS, A.OSM_BRIDGES, A.OSM_BUILDINGS),
                  produces=(A.AFFECTED_ROADS, A.AFFECTED_BRIDGES, A.AFFECTED_BUILDINGS,
                            A.DAMAGE_SUMMARY)),
    StageContract("connectivity", "T3", "Remove affected edges, find cut-off settlements.",
                  "gis.connectivity:run_connectivity",
                  needs=(A.OSM_ROADS, A.OSM_SETTLEMENTS, A.OSM_HOSPITALS,
                         A.AFFECTED_ROADS, A.AFFECTED_BRIDGES),
                  produces=(A.CUTOFF_SETTLEMENTS, A.CONNECTIVITY_SUMMARY)),
    StageContract("summarize", "T4", "Merge all summaries into summary.json.",
                  "pipeline.summary:run_summarize",
                  needs=(A.METADATA, A.OSM_SNAPSHOT, A.FLOOD_MASK, A.FLOOD_CONFIDENCE,
                         A.DAMAGE_SUMMARY, A.CONNECTIVITY_SUMMARY),
                  produces=(A.SUMMARY,), required=True),
    StageContract("report", "T4", "Render the one-page situation report.",
                  "reports.generator:run_report",
                  needs=(A.SUMMARY,), produces=(A.REPORT_TXT,),
                  optional=(A.REPORT_PDF, A.REPORT_HTML, A.REPORT_NE_TXT, A.REPORT_NE_HTML)),
    StageContract("figures", "T4", "Render presentation PNGs.",
                  "pipeline.figures:run_figures",
                  needs=(A.BEFORE, A.AFTER, A.FLOOD_MASK, A.AFFECTED_ROADS, A.CUTOFF_SETTLEMENTS),
                  produces=(A.FIG_BEFORE, A.FIG_AFTER, A.FIG_FLOOD, A.FIG_DAMAGE,
                            A.FIG_CONNECTIVITY)),
)}
del A

STAGE_ORDER = tuple(CONTRACTS)


# ==========================================================================
# 3. IS a file valid?
# ==========================================================================
@dataclass(frozen=True)
class Issue:
    severity: str
    artifact: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper():7}] {self.artifact}: {self.message}"


class _Histogram:
    """Counts identical problems instead of printing one line per feature.

    A 50 000-building file with one bad property would otherwise produce
    50 000 identical messages.  Instead: ``"... (50000 features, e.g. way/12)"``.
    """

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], list] = {}

    def add(self, severity: str, message: str, example: object = None) -> None:
        entry = self._items.setdefault((severity, message), [0, example])
        entry[0] += 1

    def issues(self, artifact: str) -> list[Issue]:
        out = []
        for (severity, message), (count, example) in self._items.items():
            tail = f" ({count} feature{'s' if count != 1 else ''}"
            tail += f", e.g. {example})" if example is not None else ")"
            out.append(Issue(severity, artifact, message + tail))
        return out


# -- helpers ----------------------------------------------------------------
def _iter_positions(coords: object):
    """Yield every ``[x, y, ...]`` position inside arbitrarily nested coordinates.

    Iterative (explicit stack) so a deeply nested MultiPolygon cannot hit
    Python's recursion limit.
    """
    stack = [coords]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            if item and isinstance(item[0], (int, float)):
                yield item
            else:
                stack.extend(item)


def _looks_like_lonlat(geometry: dict) -> bool:
    """False if any sampled position is outside lon/lat range or not finite.

    UTM coordinates (e.g. 323166, 3109588) fail instantly, which catches the
    classic "wrote projected metres into a GeoJSON" bug WITHOUT geopandas.
    """
    for pos in _iter_positions(geometry.get("coordinates")):
        if len(pos) < 2:
            return False
        x, y = pos[0], pos[1]
        if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > 180 or abs(y) > 90:
            return False
    return True


def _label(props: dict, index: int) -> str:
    for key in ("id", "object_id", "settlement_id", "name"):
        if props.get(key) is not None:
            return str(props[key])
    return f"#{index}"


def _manifest_event_date(run: "RunPaths"):
    try:
        return parse_date(run.manifest().get("event_date"), "event date")
    except (InputValidationError, OSError, ValueError):
        return None


# -- value rules for specific GeoJSON files -----------------------------------
def _rule_affected(object_type: str) -> Callable:
    def rule(props: dict, hist: _Histogram, label: str) -> None:
        status = props.get("impact_status")
        if status is not None and status not in ALLOWED_IMPACT_STATUS:
            hist.add(ERROR, f"impact_status {status!r} not allowed; use one of "
                            f"{list(ALLOWED_IMPACT_STATUS)} (never claim 'destroyed')", label)
        for key in ("impact_fraction", "confidence"):
            if key not in props:
                continue                       # missing key is reported elsewhere
            value = props[key]
            # bool is a subclass of int, so exclude it explicitly.
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 <= value <= 1):
                hist.add(ERROR, f"{key} must be a number in [0, 1]", label)
        if props.get("object_type") not in (None, object_type):
            hist.add(ERROR, f"object_type must be {object_type!r}", label)
    return rule


def _rule_cutoff(props: dict, hist: _Histogram, label: str) -> None:
    if "blocking_segment_ids" in props and not isinstance(props["blocking_segment_ids"], list):
        hist.add(ERROR, "blocking_segment_ids must be a list of segment ids", label)
    if "reason" in props and not (isinstance(props["reason"], str) and props["reason"].strip()):
        hist.add(ERROR, "reason must be a non-empty string", label)


def _rule_destination(props: dict, hist: _Histogram, label: str) -> None:
    if props.get("destination_type") not in (None,) + ALLOWED_DESTINATION_TYPES:
        hist.add(ERROR, f"destination_type must be one of {list(ALLOWED_DESTINATION_TYPES)}", label)


_VALUE_RULES: dict[Artifact, Callable] = {
    Artifact.AFFECTED_ROADS: _rule_affected("road"),
    Artifact.AFFECTED_BRIDGES: _rule_affected("bridge"),
    Artifact.AFFECTED_BUILDINGS: _rule_affected("building"),
    Artifact.CUTOFF_SETTLEMENTS: _rule_cutoff,
    Artifact.OSM_HOSPITALS: _rule_destination,
}

_WARN_IF_EMPTY = {
    Artifact.OSM_ROADS, Artifact.OSM_SETTLEMENTS, Artifact.OSM_HOSPITALS,
}


# -- per-kind checkers ----------------------------------------------------------
def _check_json(artifact: Artifact, spec: ArtifactSpec, run: "RunPaths") -> list[Issue]:
    name = artifact.value
    try:
        data = read_json(run.path(artifact))
    except (OSError, ValueError) as exc:
        return [Issue(ERROR, name, f"not valid UTF-8 JSON: {exc}")]
    if not isinstance(data, dict):
        return [Issue(ERROR, name, "top level must be a JSON object")]
    missing = [k for k in spec.required_keys if k not in data]
    return [Issue(ERROR, name, f"missing key '{k}'") for k in missing]


def _check_geojson(artifact: Artifact, spec: ArtifactSpec, run: "RunPaths") -> list[Issue]:
    name = artifact.value
    try:
        data = read_json(run.path(artifact))
    except (OSError, ValueError) as exc:
        return [Issue(ERROR, name, f"not valid UTF-8 JSON: {exc}")]
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        return [Issue(ERROR, name, "must be a GeoJSON FeatureCollection")]
    features = data.get("features")
    if not isinstance(features, list):
        return [Issue(ERROR, name, "'features' must be a list")]

    hist = _Histogram()
    rule = _VALUE_RULES.get(artifact)
    step = max(1, len(features) // COORD_SAMPLE_TARGET)   # evenly spaced sample
    for i, feat in enumerate(features):
        props = feat.get("properties") if isinstance(feat, dict) else None
        geometry = feat.get("geometry") if isinstance(feat, dict) else None
        if not isinstance(props, dict):
            props = {}
        label = _label(props, i)
        if not isinstance(feat, dict) or feat.get("type") != "Feature":
            hist.add(ERROR, "entry is not a GeoJSON Feature", label)
            continue
        gtype = geometry.get("type") if isinstance(geometry, dict) else None
        if gtype not in spec.geometry_types:
            hist.add(ERROR, f"geometry type {gtype!r} not allowed; expected one of "
                            f"{list(spec.geometry_types)}", label)
        elif i % step == 0 and not _looks_like_lonlat(geometry):
            hist.add(ERROR, "coordinates are not lon/lat degrees (looks like projected "
                            "metres; GeoJSON must be EPSG:4326)", label)
        for key in spec.required_props:
            if key not in props:
                hist.add(ERROR, f"missing property '{key}'", label)
        if rule:
            rule(props, hist, label)

    issues = hist.issues(name)
    if not features and artifact in _WARN_IF_EMPTY:
        issues.append(Issue(WARNING, name, "contains no features; downstream analysis will be empty"))
    return issues


def _grid_signature(path) -> Optional[tuple]:
    import rasterio
    with rasterio.open(path) as src:
        return (src.crs, src.height, src.width, tuple(src.transform)[:6])


def _same_grid(a: tuple, b: tuple) -> bool:
    """CRS, shape and affine transform equal (transform within float tolerance).

    A tolerance is essential: two tools that compute the same grid can differ
    in the 12th decimal, and ``==`` would call that a mismatch.
    """
    if a[0] != b[0] or a[1:3] != b[1:3]:
        return False
    return all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6) for x, y in zip(a[3], b[3]))


def _check_raster(artifact: Artifact, spec: ArtifactSpec, run: "RunPaths") -> list[Issue]:
    name = artifact.value
    if not rasterio_available():
        return [Issue(WARNING, name, "rasterio not installed; raster checks skipped")]
    import numpy as np
    import rasterio

    path = run.path(artifact)
    issues: list[Issue] = []
    try:
        with rasterio.open(path) as src:
            crs = src.crs
            if crs is None:
                issues.append(Issue(ERROR, name, "has no CRS"))
            elif not crs.is_projected:
                issues.append(Issue(ERROR, name, f"CRS {crs} is geographic; rasters must use a "
                                                 "metric UTM CRS so areas/lengths are correct"))
            else:
                try:
                    expected = AOI.from_dict(run.manifest()["aoi"]).utm_epsg()
                    if crs.to_epsg() != expected:
                        issues.append(Issue(WARNING, name, f"CRS is EPSG:{crs.to_epsg()}, the AOI's "
                                                           f"UTM zone is EPSG:{expected}"))
                except (KeyError, InputValidationError, OSError, ValueError):
                    pass
            if artifact in (Artifact.FLOOD_MASK, Artifact.FLOOD_CONFIDENCE):
                band = src.read(1)
                if artifact is Artifact.FLOOD_MASK:
                    if not np.isin(band, (0, 1)).all():
                        found = np.unique(band)[:6].tolist()
                        issues.append(Issue(ERROR, name, f"values must be only 0 or 1 (found {found})"))
                else:
                    if np.isnan(band).any():
                        issues.append(Issue(ERROR, name, "contains NaN"))
                    elif band.min() < 0 or band.max() > 1:
                        issues.append(Issue(ERROR, name, f"values must be within [0, 1] "
                                                         f"(found {band.min():.3g}..{band.max():.3g})"))
    except (OSError, rasterio.errors.RasterioError) as exc:
        return [Issue(ERROR, name, f"cannot be opened as a raster: {exc}")]

    # Grid alignment against the reference grid (after.tif; before.tif for after.tif).
    reference = {
        Artifact.FLOOD_MASK: Artifact.AFTER,
        Artifact.FLOOD_CONFIDENCE: Artifact.AFTER,
        Artifact.AFTER: Artifact.BEFORE,
    }.get(artifact)
    if reference is not None:
        if not run.path(reference).exists():
            issues.append(Issue(WARNING, name, f"{reference.value} missing; grid alignment not verified"))
        else:
            try:
                if not _same_grid(_grid_signature(path), _grid_signature(run.path(reference))):
                    issues.append(Issue(ERROR, name, f"is not on the same grid as {reference.value} "
                                                     "(CRS / shape / transform differ)"))
            except (OSError, rasterio.errors.RasterioError):
                pass
    return issues


def _check_simple(artifact: Artifact, spec: ArtifactSpec, run: "RunPaths") -> list[Issue]:
    name, path = artifact.value, run.path(artifact)
    head = path.read_bytes()[:8]
    if spec.kind is Kind.PNG and head != b"\x89PNG\r\n\x1a\n":
        return [Issue(ERROR, name, "is not a PNG file (bad signature)")]
    if spec.kind is Kind.PDF and not head.startswith(b"%PDF"):
        return [Issue(ERROR, name, "is not a PDF file (bad header)")]
    if spec.kind is Kind.TEXT:
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return [Issue(ERROR, name, "is not valid UTF-8 text")]
    return []


_KIND_CHECKERS: dict[Kind, Callable] = {
    Kind.JSON: _check_json,
    Kind.GEOJSON: _check_geojson,
    Kind.RASTER: _check_raster,
    Kind.PNG: _check_simple,
    Kind.PDF: _check_simple,
    Kind.TEXT: _check_simple,
}


# -- cross-field checks for specific files ----------------------------------------
def _extra_metadata(run: "RunPaths") -> list[Issue]:
    name = Artifact.METADATA.value
    try:
        meta = read_json(run.path(Artifact.METADATA))
        before = parse_date(meta.get("before_date"), "before_date")
        after = parse_date(meta.get("after_date"), "after_date")
    except (OSError, ValueError, AttributeError, InputValidationError):
        return [Issue(ERROR, name, "before_date / after_date missing or unreadable (YYYY-MM-DD)")]
    issues: list[Issue] = []
    if before >= after:
        issues.append(Issue(ERROR, name, "before_date must be earlier than after_date"))
    event = _manifest_event_date(run)
    if event is not None and not (before <= event <= after):
        issues.append(Issue(ERROR, name, f"scene dates must bracket the event: "
                                         f"before {before} <= event {event} <= after {after}"))
    ro_before, ro_after = meta.get("relative_orbit_before"), meta.get("relative_orbit_after")
    if ro_before is not None and ro_after is not None and ro_before != ro_after:
        issues.append(Issue(ERROR, name, f"relative orbit differs ({ro_before} vs {ro_after}); "
                                         "only same-track pairs may be compared pixel by pixel"))
    try:
        multiple = int(run.manifest().get("config", {}).get("sentinel1", {}).get("gap_multiple_days", 6))
    except (OSError, ValueError, TypeError):
        multiple = 6
    gap = (after - before).days
    if multiple > 0 and gap % multiple != 0:
        issues.append(Issue(WARNING, name, f"gap of {gap} days is not a multiple of {multiple}; "
                                           "a same-orbit pair should be (check the orbit number)"))
    return issues


def _extra_snapshot(run: "RunPaths") -> list[Issue]:
    name = Artifact.OSM_SNAPSHOT.value
    event = _manifest_event_date(run)
    if event is None:
        return [Issue(WARNING, name, "event date unavailable; pre-event rule not verified")]
    try:
        assert_osm_snapshot_pre_event(read_json(run.path(Artifact.OSM_SNAPSHOT)).get("snapshot_date"), event)
    except DataRuleViolation as exc:
        return [Issue(ERROR, name, exc.detail)]
    except (InputValidationError, OSError, ValueError, AttributeError):
        return [Issue(ERROR, name, "snapshot_date missing or unreadable (YYYY-MM-DD)")]
    return []


def _extra_manifest(run: "RunPaths") -> list[Issue]:
    try:
        version = run.manifest().get("contract_version")
    except (OSError, ValueError):
        return []
    if version is not None and version != CONTRACT_VERSION:
        return [Issue(WARNING, Artifact.RUN_MANIFEST.value,
                      f"written for contract {version}, code is {CONTRACT_VERSION}")]
    return []


_EXTRA_CHECKS: dict[Artifact, Callable] = {
    Artifact.METADATA: _extra_metadata,
    Artifact.OSM_SNAPSHOT: _extra_snapshot,
    Artifact.RUN_MANIFEST: _extra_manifest,
}


def _check_provenance(artifact: Artifact, spec: ArtifactSpec, run: "RunPaths",
                      forbid_mock: bool) -> list[Issue]:
    if not spec.provenance:
        return []
    if spec.kind is Kind.RASTER and not rasterio_available():
        return []                                   # cannot read tags; already warned
    prov = read_provenance(run.path(artifact))
    name = artifact.value
    if prov is None:
        return [Issue(WARNING, name, "has no provenance stamp (producer/source/CRS/dates); "
                                     "write it with pipeline.fileio")]
    issues = []
    missing = [k for k in ("producer", "source", "processing_date") if not prov.get(k)]
    if missing:
        issues.append(Issue(WARNING, name, f"provenance lacks {missing}"))
    if prov.get("is_mock") and forbid_mock:
        issues.append(Issue(ERROR, name, "contains MOCK data, which must not be submitted"))
    return issues


# ==========================================================================
# Public API
# ==========================================================================
def check_artifact(run: "RunPaths", artifact: Artifact, *, forbid_mock: bool = False) -> list[Issue]:
    """All problems found in one file (empty list = valid)."""
    artifact = Artifact(artifact)
    spec = SPECS[artifact]
    path = run.path(artifact)
    if not path.exists():
        return [Issue(ERROR, artifact.value, "file is missing")]
    if path.stat().st_size == 0:
        return [Issue(ERROR, artifact.value, "file is empty")]
    issues = _KIND_CHECKERS[spec.kind](artifact, spec, run)
    if not any(i.severity == ERROR for i in issues):
        extra = _EXTRA_CHECKS.get(artifact)
        if extra:
            issues += extra(run)
    issues += _check_provenance(artifact, spec, run, forbid_mock)
    return issues


def check_stage(run: "RunPaths", stage: str, *, forbid_mock: bool = False) -> list[Issue]:
    """Validate every output of a stage (plus optional outputs that exist)."""
    contract = CONTRACTS[stage]
    issues: list[Issue] = []
    for artifact in contract.produces:
        issues += check_artifact(run, artifact, forbid_mock=forbid_mock)
    for artifact in contract.optional:
        if run.path(artifact).exists():
            issues += check_artifact(run, artifact, forbid_mock=forbid_mock)
    return issues


def check_run(run: "RunPaths", stages: Optional[Iterable[str]] = None, *,
              forbid_mock: bool = False, only_existing: bool = True) -> dict[str, list[Issue]]:
    """Validate several stages. With ``only_existing`` skip stages that have no output yet."""
    result: dict[str, list[Issue]] = {}
    for stage in (stages or STAGE_ORDER):
        contract = CONTRACTS[stage]
        if only_existing and not any(run.path(a).exists() for a in contract.produces):
            continue
        result[stage] = check_stage(run, stage, forbid_mock=forbid_mock)
    return result


def raise_for_issues(issues: list[Issue], stage: str, *, strict: bool = False) -> None:
    """Raise :class:`ContractError` if any error (or, with ``strict``, any issue) exists."""
    blocking = [i for i in issues if i.severity == ERROR or strict]
    if not blocking:
        return
    raise ContractError(
        f"Stage '{stage}' produced output that breaks the data contract "
        f"({len(blocking)} problem{'s' if len(blocking) != 1 else ''}): {blocking[0].message}",
        "\n".join(str(i) for i in blocking),
    )


def stage_outputs_exist(run: "RunPaths", stage: str) -> bool:
    return all(run.path(a).exists() for a in CONTRACTS[stage].produces)


def stage_is_fresh(run: "RunPaths", stage: str) -> bool:
    """``make``-style cache test: outputs exist and are not older than any input.

    If T1 re-runs ``acquire`` (new before.tif), the mtime of that file becomes
    newer than ``flood_mask.tif``, so ``infer`` -- and, transitively, every
    later stage -- is automatically considered stale and re-run.  No cache
    keys to maintain.
    """
    contract = CONTRACTS[stage]
    outputs = [run.path(a) for a in contract.produces]
    inputs = [run.path(a) for a in contract.needs]
    if not outputs or not all(p.exists() for p in outputs + inputs):
        return False
    newest_input = max((p.stat().st_mtime_ns for p in inputs), default=0)
    oldest_output = min(p.stat().st_mtime_ns for p in outputs)
    return oldest_output >= newest_input


def contracts_markdown() -> str:
    """Render the contract as Markdown (written to docs/contracts.md by bootstrap)."""
    lines = [
        "# Data contract (auto-generated - do not edit by hand)",
        "",
        f"Contract version: **{CONTRACT_VERSION}**. Source: `pipeline/contracts.py`.",
        "",
        "## Stages",
        "",
        "| Stage | Owner | Entry point | Reads | Writes |",
        "|---|---|---|---|---|",
    ]
    for c in CONTRACTS.values():
        reads = "<br>".join(f"`{a.value}`" for a in c.needs) or "-"
        writes = "<br>".join(f"`{a.value}`" for a in c.produces)
        if c.optional:
            writes += "<br>" + "<br>".join(f"`{a.value}` (optional)" for a in c.optional)
        entry = f"`{c.entrypoint}`" if c.entrypoint else "-"
        flag = " (required)" if c.required else ""
        lines.append(f"| {c.name}{flag} | {c.owner} | {entry} | {reads} | {writes} |")
    lines += ["", "Entry-point signature (all stages):", "",
              "```python", "def run_xxx(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None", "```",
              "", "## Files", "", "| File | Kind | Must contain |", "|---|---|---|"]
    for artifact, spec in SPECS.items():
        must = spec.description
        if spec.required_keys:
            must += " Keys: " + ", ".join(f"`{k}`" for k in spec.required_keys) + "."
        if spec.required_props:
            must += " Properties: " + ", ".join(f"`{k}`" for k in spec.required_props) + "."
        if spec.geometry_types:
            must += " Geometry: " + "/".join(spec.geometry_types) + "."
        lines.append(f"| `{artifact.value}` | {spec.kind.value} | {must} |")
    lines += ["", "Allowed `impact_status` values: "
              + ", ".join(f"`{s}`" for s in ALLOWED_IMPACT_STATUS) + ".", ""]
    return "\n".join(lines)
