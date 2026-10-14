"""Everything the dashboard needs to KNOW, with no Streamlit and no Folium.

Keeping the logic here (instead of inside UI code) means it can be unit-tested
without a browser, and the UI modules stay thin.  Contents:

1. ``LAYERS``                 the map layer registry (colours, popups, defaults)
2. feature preparation        thin geometry, trim + escape properties, cap counts
3. ``prepare_layers``         load everything the map needs, once (cacheable)
4. KPI cards, banners, captions
5. tables + CSV export
6. AOI presets/inputs, finding old runs, the demo run
7. progress text and the built-in limitations list
"""

from __future__ import annotations

import csv
import html
import io
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from pipeline.config import REPO_ROOT
from pipeline.contracts import Artifact
from pipeline.errors import InputValidationError
from pipeline.fileio import read_json
from pipeline.run import RunPaths
from pipeline.schemas import AOI
from reports.format import NA, fmt

# ==========================================================================
# 1. Layer registry
# ==========================================================================
RASTER_MASK, RASTER_PROB = "raster-mask", "raster-prob"
LINE, POLYGON, POINT_CIRCLE, POINT_ICON = "line", "polygon", "point-circle", "point-icon"

_AFFECTED_POPUP = (("object_id", "ID"), ("impact_status", "Status"),
                   ("impact_fraction", "Overlap with flood"), ("confidence", "Model confidence"))


@dataclass(frozen=True)
class LayerSpec:
    """One toggleable map layer.  Order in ``LAYERS`` = draw order, bottom to top."""

    key: str
    label: str
    artifact: Artifact
    kind: str
    color: str                       # hex; also used by the legend
    default_on: bool
    popup: tuple = ()                # ((property, label), ...)
    legend: str = ""                 # one line shown in the legend
    capped: bool = False             # context layers may be thinned to max_features


LAYERS: tuple = (
    LayerSpec("confidence", "Model confidence", Artifact.FLOOD_CONFIDENCE, RASTER_PROB,
              "#1f78b4", False, legend="Darker = model more sure"),
    LayerSpec("flood_mask", "Flood / debris extent", Artifact.FLOOD_MASK, RASTER_MASK,
              "#1f78b4", True, legend="Detected flood / debris"),
    LayerSpec("osm_roads", "All roads (pre-event OSM)", Artifact.OSM_ROADS, LINE,
              "#9a9a9a", False, popup=(("name", "Name"), ("highway_type", "Type"), ("bridge", "Bridge")),
              legend="Road network before the event", capped=True),
    LayerSpec("osm_buildings", "All buildings (pre-event OSM)", Artifact.OSM_BUILDINGS, POLYGON,
              "#c5b0d5", False, popup=(("id", "ID"), ("type", "Type")),
              legend="Buildings before the event", capped=True),
    LayerSpec("affected_buildings", "Affected buildings", Artifact.AFFECTED_BUILDINGS, POLYGON,
              "#9467bd", True, popup=_AFFECTED_POPUP, legend="Building overlapping the flood"),
    LayerSpec("affected_roads", "Affected roads", Artifact.AFFECTED_ROADS, LINE,
              "#d62728", True, popup=_AFFECTED_POPUP, legend="Road segment overlapping the flood"),
    LayerSpec("affected_bridges", "Affected bridges", Artifact.AFFECTED_BRIDGES, LINE,
              "#ff7f0e", True, popup=_AFFECTED_POPUP, legend="Bridge overlapping the flood"),
    LayerSpec("settlements", "Settlements", Artifact.OSM_SETTLEMENTS, POINT_CIRCLE,
              "#7f7f7f", True, popup=(("name", "Name"), ("name_ne", "Name (Nepali)"), ("id", "ID")),
              legend="Settlement"),
    LayerSpec("destinations", "Hospitals / towns", Artifact.OSM_HOSPITALS, POINT_ICON,
              "#2ca02c", True, popup=(("name", "Name"), ("destination_type", "Type")),
              legend="Hospital or town (destination)"),
    LayerSpec("cutoff", "Cut-off settlements", Artifact.CUTOFF_SETTLEMENTS, POINT_CIRCLE,
              "#e31a1c", True,
              popup=(("name", "Settlement"), ("reason", "Why cut off"),
                     ("nearest_destination", "Nearest destination"),
                     ("blocking_segment_ids", "Blocking segments")),
              legend="No road path to any destination"),
)
LAYER_BY_KEY = {spec.key: spec for spec in LAYERS}

#: Layers used to frame the map (the interesting ones, not the context layers).
FRAMING_LAYERS = ("affected_roads", "affected_bridges", "affected_buildings",
                  "settlements", "destinations", "cutoff")
MAX_LIST_ITEMS = 5


# ==========================================================================
# 2. Feature preparation
# ==========================================================================
def format_prop(key: str, value: Any) -> str:
    """Display text for one property.  Always HTML-escaped.

    OSM names and reasons are untrusted text, and the map inserts them into
    popups; escaping here means a name like ``<img onerror=...>`` is shown
    literally instead of being executed.
    """
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if key == "impact_fraction" and isinstance(value, (int, float)):
        return f"{100 * value:.0f}%"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
        text = ", ".join(items[:MAX_LIST_ITEMS])
        if len(items) > MAX_LIST_ITEMS:
            text += f" (+{len(items) - MAX_LIST_ITEMS} more)"
        return html.escape(text or "-")
    return html.escape(str(value))


def _thin_line(coords: list, tol: float) -> list:
    """Drop vertices closer than ``tol`` degrees to the last kept one; keep the last."""
    if len(coords) <= 2:
        return coords
    kept = [coords[0]]
    for p in coords[1:-1]:
        if abs(p[0] - kept[-1][0]) >= tol or abs(p[1] - kept[-1][1]) >= tol:
            kept.append(p)
    kept.append(coords[-1])
    return kept


def thin_geometry(geometry: dict, tol: float = 0.00003, decimals: int = 5) -> dict:
    """Round coordinates (~1 m at 5 decimals) and drop redundant vertices.

    Smaller GeoJSON = faster browser.  Polygon rings keep their closing
    vertex and at least 4 positions, otherwise they would be invalid.
    """
    def rnd(p):
        return [round(p[0], decimals), round(p[1], decimals)]

    def ring(r):
        thinned = _thin_line([rnd(p) for p in r], tol)
        return thinned if len(thinned) >= 4 else [rnd(p) for p in r]

    kind, coords = geometry.get("type"), geometry.get("coordinates")
    if kind == "Point":
        return {"type": kind, "coordinates": rnd(coords)}
    if kind == "LineString":
        return {"type": kind, "coordinates": _thin_line([rnd(p) for p in coords], tol)}
    if kind == "MultiLineString":
        return {"type": kind, "coordinates": [_thin_line([rnd(p) for p in line], tol) for line in coords]}
    if kind == "Polygon":
        return {"type": kind, "coordinates": [ring(r) for r in coords]}
    if kind == "MultiPolygon":
        return {"type": kind, "coordinates": [[ring(r) for r in poly] for poly in coords]}
    return geometry


def prepare_features(spec: LayerSpec, features: list, max_features: int) -> tuple:
    """Return ``(prepared_features, total, shown)``.

    * Capped (context) layers are thinned with an evenly spaced stride, so the
      shown subset covers the whole area instead of only the first rows.
    * Properties are trimmed to the popup fields and replaced by display text
      (this alone shrinks the GeoJSON a lot); every feature gets every key,
      because the map library requires identical keys on all features.
    """
    total = len(features)
    if spec.capped and max_features and total > max_features:
        step = math.ceil(total / max_features)
        features = features[::step]
    keys = [k for k, _ in spec.popup]
    out = []
    for feat in features:
        props = feat.get("properties") or {}
        out.append({
            "type": "Feature",
            "geometry": thin_geometry(feat.get("geometry") or {}),
            "properties": {k: format_prop(k, props.get(k)) for k in keys},
        })
    return out, total, len(out)


def iter_positions(geometry: dict):
    """Yield every ``(lon, lat)`` of a geometry (iterative, no recursion)."""
    stack = [geometry.get("coordinates")]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            if item and isinstance(item[0], (int, float)):
                yield item[0], item[1]
            else:
                stack.extend(item)


def representative_point(geometry: dict) -> Optional[tuple]:
    """``(lon, lat)`` to put a marker at: the point itself, or the mean of all vertices.

    Settlements may be mapped as areas in OSM; the mean of the vertices is a
    cheap, always-inside-the-neighbourhood anchor (not an exact centroid).
    """
    points = list(iter_positions(geometry))
    if not points:
        return None
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def popup_html(spec: "LayerSpec", props: dict) -> str:
    """Popup body: a small table of label/value rows.

    ``props`` already holds escaped display text (see :func:`prepare_features`);
    labels come from our own constants, so the result is safe HTML.
    """
    rows = "".join(f"<tr><td style='padding-right:8px;color:#555'>{html.escape(label)}</td>"
                   f"<td><b>{props.get(key, '-')}</b></td></tr>" for key, label in spec.popup)
    return f"<table style='font:12px sans-serif'>{rows}</table>"


def features_bounds(features: Iterable[dict]) -> Optional[tuple]:
    """``(west, south, east, north)`` of the features, or None if empty."""
    west = south = east = north = None
    for feat in features:
        for lon, lat in iter_positions(feat.get("geometry") or {}):
            west = lon if west is None else min(west, lon)
            east = lon if east is None else max(east, lon)
            south = lat if south is None else min(south, lat)
            north = lat if north is None else max(north, lat)
    return None if west is None else (west, south, east, north)


# ==========================================================================
# 3. Preparing the layers for a run
# ==========================================================================
@dataclass
class PreparedLayer:
    spec: LayerSpec
    features: list = field(default_factory=list)
    total: int = 0
    shown: int = 0
    png: Optional[Path] = None                  # raster layers
    bounds: Optional[tuple] = None              # (west, south, east, north) of the PNG
    note: str = ""                              # shown to the user when something is missing


def prepare_layers(run: RunPaths, cfg: dict, enabled: Iterable[str]) -> dict:
    """Load every enabled layer; problems become notes, never exceptions.

    This is the expensive step (file reads, raster resampling).  The app wraps
    it in ``st.cache_data`` so toggling a checkbox does not redo the work.
    """
    from dashboard import raster_overlay

    max_features = int(cfg.get("dashboard", {}).get("max_buildings_drawn", 5000))
    wanted = set(enabled)
    prepared: dict = {}
    for spec in LAYERS:
        if spec.key not in wanted:
            continue
        layer = PreparedLayer(spec)
        path = run.path(spec.artifact)
        if not path.exists():
            layer.note = f"'{spec.label}' is not available for this run."
        elif spec.kind in (RASTER_MASK, RASTER_PROB):
            try:
                kind = "mask" if spec.kind == RASTER_MASK else "probability"
                layer.png, layer.bounds = raster_overlay.render_overlay_file(run, spec.artifact, kind)
            except Exception as exc:                  # rasterio missing, bad file, ...
                layer.note = f"'{spec.label}' could not be drawn: {exc}"
        else:
            try:
                data = read_json(path)
                layer.features, layer.total, layer.shown = prepare_features(
                    spec, list(data.get("features", [])), max_features)
                if layer.shown < layer.total:
                    layer.note = f"'{spec.label}': showing {layer.shown:,} of {layer.total:,} (thinned for speed)."
            except (OSError, ValueError, AttributeError) as exc:
                layer.note = f"'{spec.label}' could not be read: {exc}"
        prepared[spec.key] = layer
    return prepared


def map_bounds(prepared: dict, aoi: Optional[AOI]) -> list:
    """``[[south, west], [north, east]]`` for ``fit_bounds``.

    Framed on the interesting data (affected objects, settlements); falls back
    to the raster footprint, then to the AOI.  Framing on data (not the AOI)
    keeps a mock run, whose data sits in the mock valley, correctly in view.
    """
    boxes = []
    for key in FRAMING_LAYERS:
        layer = prepared.get(key)
        if layer and layer.features:
            box = features_bounds(layer.features)
            if box:
                boxes.append(box)
    if not boxes:
        boxes = [layer.bounds for layer in prepared.values() if layer.bounds]
    if not boxes and aoi is not None:
        boxes = [aoi.bbox]
    if not boxes:
        boxes = [(85.0, 27.0, 86.0, 28.0)]            # harmless default (never reached in practice)
    west, south = min(b[0] for b in boxes), min(b[1] for b in boxes)
    east, north = max(b[2] for b in boxes), max(b[3] for b in boxes)
    pad_x, pad_y = max((east - west) * 0.08, 0.002), max((north - south) * 0.08, 0.002)
    return [[south - pad_y, west - pad_x], [north + pad_y, east + pad_x]]


def legend_items(enabled: Iterable[str]) -> list:
    """``[(color, text), ...]`` for the enabled layers, in draw order (top layer first)."""
    on = set(enabled)
    return [(s.color, s.legend or s.label) for s in reversed(LAYERS) if s.key in on]


def legend_html(items: list) -> str:
    """Small HTML box for the map corner.  Text is escaped."""
    rows = "".join(
        f'<div style="margin:4px 0;display:flex;align-items:center"><span style="display:inline-block;width:14px;'
        f'height:14px;border-radius:4px;background:{html.escape(color)};margin-right:8px;'
        f'box-shadow:inset 0 0 0 1px rgba(0,0,0,.12)"></span>{html.escape(text)}</div>' for color, text in items)
    return ('<div style="position:fixed;bottom:26px;left:12px;z-index:9999;background:rgba(255,255,255,.96);'
            'padding:10px 12px;border-radius:12px;font:12px/1.35 Inter,system-ui,sans-serif;color:#0F172A;'
            'box-shadow:0 6px 18px rgba(15,23,42,.18);border:1px solid #E2E8F0;max-width:260px">'
            '<div style="font-weight:700;font-size:11px;letter-spacing:.06em;text-transform:uppercase;'
            f'color:#64748B;margin-bottom:4px">Legend</div>{rows}</div>')


# ==========================================================================
# 4. KPI cards, banners, captions
# ==========================================================================
@dataclass(frozen=True)
class Card:
    key: str
    label: str
    value: str
    sub: str = ""
    available: bool = True


def kpi_cards(summary: Any) -> list:
    """The six headline cards.  Every string comes from the shared ``fmt``."""
    f = fmt(summary)
    names = (summary.cutoff_names if hasattr(summary, "cutoff_names") else summary.get("cutoff_names")) or []
    sub_cut = ", ".join(names[:3]) + (f" (+{len(names) - 3})" if len(names) > 3 else "")
    return [
        Card("flood", "Flooded / debris area", f["flood_area_km2"], "km\u00b2" if f["flood_area_km2"] != NA else "",
             f["flood_area_km2"] != NA),
        Card("roads", "Roads affected", f["affected_roads"],
             f"{f['affected_road_length_km']} km" if f["affected_road_length_km"] != NA else "",
             f["affected_roads"] != NA),
        Card("bridges", "Bridges affected", f["affected_bridges"], "", f["affected_bridges"] != NA),
        Card("buildings", "Buildings affected", f["affected_buildings"], "",
             f["affected_buildings"] != NA),
        Card("cutoff", "Settlements cut off", f["cutoff_settlements"], sub_cut, f["cutoff_settlements"] != NA),
        Card("confidence", "Mean model confidence", f["mean_confidence"], "on flagged pixels",
             f["mean_confidence"] != NA),
    ]


@dataclass(frozen=True)
class Banner:
    level: str          # "warning" | "info" | "error"
    text: str


DISCLAIMER = ("Educational prototype built for a hackathon. Satellite-derived estimates, not verified "
              "damage assessments, and not an operational tool.")


def banners(summary: Any, result: Any = None) -> list:
    """Messages shown above the map: mock warning, degraded stages, data-quality notes."""
    out = []
    s = summary.to_dict() if hasattr(summary, "to_dict") else (summary or {})
    if s.get("is_mock"):
        out.append(Banner("warning", "MOCK DATA: this is a synthetic demonstration scenario, not a real result."))
    for text in s.get("warnings", []) or []:
        if not text.startswith("MOCK DATA"):          # already shown above
            out.append(Banner("warning", text))
    if result is not None:
        for o in getattr(result, "outcomes", []):
            if o.name in ("report", "figures") and o.status == "skipped":
                continue                              # not built yet; not worth a banner
            if o.status == "failed":
                out.append(Banner("error", f"Stage '{o.name}' failed: {o.message}"))
    return out


def imagery_caption(summary: Any) -> str:
    """One line of provenance shown under the map."""
    f = fmt(summary)
    parts = []
    if f["before_date"] or f["after_date"]:
        parts.append(f"Sentinel-1 radar, before {f['before_date'] or '?'} / after {f['after_date'] or '?'}"
                     + (f", orbit {f['sentinel1_orbit']}" if f["sentinel1_orbit"] else ""))
    if f["osm_snapshot_date"]:
        parts.append(f"OpenStreetMap snapshot {f['osm_snapshot_date']}")
    if f["model_version"]:
        parts.append(f"model {f['model_version']}")
    return " | ".join(parts)


# ==========================================================================
# 5. Tables and CSV
# ==========================================================================
@dataclass
class Table:
    title: str
    columns: list
    rows: list
    total: int
    note: str = ""


def _affected_table(run: RunPaths, artifact: Artifact, title: str, max_rows: int) -> Optional[Table]:
    path = run.path(artifact)
    if not path.exists():
        return None
    try:
        feats = read_json(path).get("features", [])
    except (OSError, ValueError, AttributeError):
        return None
    rows = []
    for feat in feats:
        p = feat.get("properties") or {}
        rows.append({"ID": p.get("object_id", ""), "Status": p.get("impact_status", ""),
                     "Overlap": p.get("impact_fraction"), "Confidence": p.get("confidence")})
    rows.sort(key=lambda r: -(r["Overlap"] if isinstance(r["Overlap"], (int, float)) else 0))
    note = f"Showing the first {max_rows:,} of {len(rows):,}." if len(rows) > max_rows else ""
    return Table(title, ["ID", "Status", "Overlap", "Confidence"], rows[:max_rows], len(rows), note)


def build_tables(run: RunPaths, max_rows: int = 2000) -> dict:
    """``{key: Table}`` for the Tables tab; tables with no file are omitted."""
    tables = {}
    for key, artifact, title in (("roads", Artifact.AFFECTED_ROADS, "Affected roads"),
                                 ("bridges", Artifact.AFFECTED_BRIDGES, "Affected bridges"),
                                 ("buildings", Artifact.AFFECTED_BUILDINGS, "Affected buildings")):
        table = _affected_table(run, artifact, title, max_rows)
        if table:
            tables[key] = table
    path = run.path(Artifact.CUTOFF_SETTLEMENTS)
    if path.exists():
        try:
            feats = read_json(path).get("features", [])
        except (OSError, ValueError, AttributeError):
            feats = []
        rows = []
        for feat in feats:
            p = feat.get("properties") or {}
            blockers = p.get("blocking_segment_ids") or []
            rows.append({"Settlement": p.get("name", ""), "Why cut off": p.get("reason", ""),
                         "Nearest destination": p.get("nearest_destination", ""),
                         "Blocking segments": len(blockers) if isinstance(blockers, list) else ""})
        tables["cutoff"] = Table("Settlements cut off", ["Settlement", "Why cut off", "Nearest destination",
                                                         "Blocking segments"], rows, len(rows))
    return tables


def table_to_csv(table: Table) -> bytes:
    """CSV bytes with a UTF-8 BOM.

    The BOM ("utf-8-sig") makes Excel on Windows read Nepali settlement names
    correctly; without it Excel guesses cp1252 and shows garbage.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=table.columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(table.rows)
    return buffer.getvalue().encode("utf-8-sig")


# ==========================================================================
# 6. AOI inputs, previous runs, demo run
# ==========================================================================
@dataclass(frozen=True)
class EventPreset:
    """A real past flood to try: area + the date that shows the change (config/events/*.json)."""
    label: str
    aoi: AOI
    date: str
    note: str = ""


def load_events(repo_root: Path = REPO_ROOT) -> list:
    """Real flood events with known dates, for the sidebar and the landing page."""
    events = []
    for path in sorted((repo_root / "config" / "events").glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            aoi = AOI.from_dict(raw)
            events.append(EventPreset(aoi.name, aoi, str(raw.get("event_date", "")), str(raw.get("note", ""))))
        except Exception:
            continue
    return events


def load_presets(repo_root: Path = REPO_ROOT) -> dict:
    """``{label: AOI}`` for the area selector: the case-study box, real past events, a small mock valley."""
    presets = {}
    example = repo_root / "config" / "example_aoi.json"
    if example.exists():
        try:
            aoi = AOI.from_json_file(example)
            presets[aoi.name] = aoi
        except Exception:
            pass
    for event in load_events(repo_root):
        presets[event.label] = event.aoi
    presets["Mock valley (demo)"] = AOI(85.20, 28.05, 85.40, 28.20, "mock-valley")
    return presets


def preset_dates(repo_root: Path = REPO_ROOT) -> dict:
    """``{preset label: event date 'YYYY-MM-DD'}`` for presets that come with a date."""
    return {e.label: e.date for e in load_events(repo_root) if e.date}


def aoi_from_inputs(west: Any, south: Any, east: Any, north: Any, name: str = "custom",
                    max_km2: float = 2500.0) -> tuple:
    """``(AOI or None, message)``: validates live as the user types.  Never raises."""
    try:
        aoi = AOI(west, south, east, north, name or "custom").validate(max_km2=max_km2)
    except InputValidationError as exc:
        return None, exc.user_message
    return aoi, f"{aoi.area_km2():,.0f} km\u00b2 (limit {max_km2:,.0f} km\u00b2)"


@dataclass(frozen=True)
class RunInfo:
    path: Path
    label: str
    is_mock: bool
    mtime: float


def list_runs(out_root: Path) -> list:
    """Finished or partial runs under ``out_root`` (and ``out_root/_mock``), newest first."""
    runs = []
    for base in (out_root, out_root / "_mock"):
        if not base.is_dir():
            continue
        for manifest in base.glob("*/run.json"):
            folder = manifest.parent
            if folder.name.startswith("_"):             # e.g. _mock_template
                continue
            try:
                meta = read_json(manifest)
                name = meta.get("aoi", {}).get("name", folder.name)
                event = meta.get("event_date", "?")
            except (OSError, ValueError, AttributeError):
                continue
            mock = bool(meta.get("is_mock"))
            when = datetime.fromtimestamp(manifest.stat().st_mtime).strftime("%d %b %H:%M")
            runs.append(RunInfo(folder, f"{name} | {event} | {'MOCK | ' if mock else ''}{when}",
                                mock, manifest.stat().st_mtime))
    return sorted(runs, key=lambda r: -r.mtime)


def find_demo_run(repo_root: Path = REPO_ROOT) -> Optional[Path]:
    """The cached demo: ``data/sample/trishuli_run`` if present, else the mock sample."""
    for name in ("trishuli_run", "mock_run"):
        candidate = repo_root / "data" / "sample" / name
        if (candidate / "run.json").exists():
            return candidate
    return None


def run_version(run: RunPaths) -> float:
    """Token that changes when the run's results change (used as a cache key)."""
    stamps = []
    for artifact in (Artifact.SUMMARY, Artifact.RUN_LOG):
        p = run.path(artifact)
        stamps.append(p.stat().st_mtime if p.exists() else 0.0)
    return max(stamps)


# ==========================================================================
# 7. Progress text, limitations
# ==========================================================================
STATE_ICONS = {"pending": "\u23f3", "running": "\u25b6\ufe0f", "ok": "\u2705", "cached": "\u267b\ufe0f",
               "failed": "\u274c", "skipped": "\u23ed\ufe0f"}
STAGE_LABELS = {
    "init": "Create run folder", "acquire": "Satellite images (Sentinel-1)", "osm": "Map data (OpenStreetMap)",
    "infer": "Flood detection (AI model)", "damage": "Damage assessment", "connectivity": "Cut-off analysis",
    "summarize": "Summary numbers", "report": "Situation report", "figures": "Figures",
}


def stage_progress_markdown(states: dict) -> str:
    """Markdown list shown while the pipeline runs: one line per stage with an icon."""
    lines = []
    for name, label in STAGE_LABELS.items():
        state = states.get(name, "pending")
        lines.append(f"{STATE_ICONS.get(state, '')} **{label}** \u2014 {state}")
    return "\n\n".join(lines)


LIMITATIONS = (
    "Sentinel satellites revisit an area only every few days: this tool cannot warn of a sudden event "
    "and does not show the situation right now.",
    "Change detection needs a before/after pair from the same orbit track; without one, no result is produced.",
    "Radar shadow and layover in steep Himalayan terrain cause false positives and false negatives.",
    "Radar cannot cleanly separate water from debris; optical images need cloud-free scenes.",
    "10 m pixels: small buildings and narrow roads are poorly resolved.",
    "'Affected' means overlap with the detected flood, not verified structural damage.",
    "OpenStreetMap coverage varies; cut-off results use the pre-event road network only and ignore "
    "footpaths, unmapped roads and air access.",
    "Training data are mostly from other regions; accuracy on new scenes is shown only by the held-out evaluation.",
)
