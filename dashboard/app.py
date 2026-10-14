"""FloodResQ dashboard.   Start with:   streamlit run dashboard/app.py

Page flow:  sidebar (area, date, run, layers)  ->  header band  ->  notices  ->  KPI cards  ->
tabs [Map | Before/After | Tables | Report | Flood path | Case study | About]  ->  footer.
Without a result the page shows a landing view with real past floods to try.

Rules this file follows
-----------------------
* No analysis here.  Every number comes from ``summary.json`` via
  ``dashboard.data`` (the same formatter the report uses).
* Heavy work (reading layers, resampling rasters) is cached with
  ``st.cache_data`` keyed by run folder + a version token, so ticking a layer
  checkbox does not redo it.
* A failed or partial run shows a friendly message, never a stack trace.
* All styling lives in ``dashboard/style.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

# Streamlit puts the SCRIPT's folder on sys.path, not the repo root; this makes
# `import pipeline` work no matter where `streamlit run` was started from.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st                                          # noqa: E402

from dashboard import case_study, components, data, style       # noqa: E402
from pipeline.analyze import analyze_flood, load_result          # noqa: E402
from pipeline.config import load_config                          # noqa: E402
from pipeline.errors import PipelineError                        # noqa: E402
from pipeline.run import RunPaths                                # noqa: E402
from pipeline.schemas import AOI                                 # noqa: E402
from pipeline.stages import MOCKABLE_STAGES                      # noqa: E402

DEFAULT_EVENT_DATE = date(2026, 8, 26)
EARLIEST_EVENT_DATE = date(2014, 10, 3)        # start of routine Sentinel-1 data
AREA_KEY = "fr_area"

#: sidebar grouping of the map layers (keys from dashboard.data.LAYERS)
LAYER_GROUPS = (("Hazard", ("flood_mask", "confidence")),
                ("Impact", ("affected_roads", "affected_bridges", "affected_buildings")),
                ("Access", ("cutoff", "settlements", "destinations")),
                ("Context", ("osm_roads", "osm_buildings")))


@dataclass
class SidebarChoice:
    aoi: Optional[AOI] = None
    event_date: date = DEFAULT_EVENT_DATE
    mock_stages: list = field(default_factory=list)
    use_cache: bool = True
    run_clicked: bool = False
    demo_clicked: bool = False
    open_path: Optional[Path] = None
    enabled_layers: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Preparing map layers...")
def cached_layers(run_dir: str, version: float, enabled: tuple) -> dict:
    """``version`` is part of the cache key: new results invalidate the cache."""
    return data.prepare_layers(RunPaths(Path(run_dir)), load_config(), enabled)


@st.cache_data(show_spinner=False)
def cached_tables(run_dir: str, version: float) -> dict:
    return data.build_tables(RunPaths(Path(run_dir)))


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _parse_date(text: str, fallback: date) -> date:
    try:
        return date.fromisoformat(text)
    except (TypeError, ValueError):
        return fallback


def _select_event(label: str, when: str) -> None:
    """Button callback (runs before the next rerun, so widget state may be set here).
    Only the area is set: its date widget is keyed per area and defaults to the event date."""
    st.session_state[AREA_KEY] = label
    st.session_state.pop(f"fr_date_{label}", None)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar(cfg: dict) -> SidebarChoice:
    choice = SidebarChoice()
    with st.sidebar:
        _html(f'<div class="fr-side-brand">{style.LOGO_SVG_DARK}<b>FloodResQ</b></div>')

        _html('<div class="fr-side-h">1 &middot; Area</div>')
        presets = data.load_presets()
        dates = data.preset_dates()
        limit = float(cfg.get("limits", {}).get("max_aoi_km2", 2500))
        area = st.selectbox("Area", list(presets) + ["Custom"], key=AREA_KEY)
        if area == "Custom":
            name = st.text_input("Name", "custom")
            c1, c2 = st.columns(2)
            with c1:
                west = st.number_input("West (lon)", value=85.10, format="%.4f")
                south = st.number_input("South (lat)", value=27.85, format="%.4f")
            with c2:
                east = st.number_input("East (lon)", value=85.45, format="%.4f")
                north = st.number_input("North (lat)", value=28.30, format="%.4f")
            aoi, message = data.aoi_from_inputs(west, south, east, north, name, limit)
        else:
            aoi = presets[area]
            message = f"{aoi.area_km2():,.0f} km² (limit {limit:,.0f} km²)"
        (st.caption if aoi else st.error)(message)
        choice.aoi = aoi

        _html('<div class="fr-side-h">2 &middot; Flood date</div>')
        today = date.today()
        preset_day = _parse_date(dates.get(area, ""), DEFAULT_EVENT_DATE)
        choice.event_date = st.date_input("Flood date", value=min(preset_day, today), key=f"fr_date_{area}",
                                          min_value=EARLIEST_EVENT_DATE, max_value=today)
        if area in dates:
            st.caption(f"Date of this event: {preset_day:%d %b %Y}")
        choice.run_clicked = components.call_compat(st.button, "Run analysis", type="primary",
                                                    disabled=aoi is None)

        with st.expander("Advanced"):
            choice.mock_stages = list(st.multiselect(
                "Use mock data for these steps", list(MOCKABLE_STAGES), default=[],
                help="Development only. Mock results are clearly marked and are never mixed with real ones."))
            choice.use_cache = bool(st.checkbox("Reuse earlier results (cache)", value=True))

        _html('<div class="fr-side-h">3 &middot; Demo and earlier runs</div>')
        demo = data.find_demo_run()
        choice.demo_clicked = components.call_compat(
            st.button, "Load cached demo", disabled=demo is None,
            help="Opens a saved run instantly, with no processing.")
        runs = data.list_runs(REPO_ROOT / cfg.get("output", {}).get("root", "outputs"))
        if runs:
            labels = [r.label for r in runs]
            picked = st.selectbox("Open an earlier run", ["-"] + labels)
            if picked != "-" and components.call_compat(st.button, "Open selected run"):
                choice.open_path = runs[labels.index(picked)].path

        _html('<div class="fr-side-h">Map layers</div>')
        by_key = {spec.key: spec for spec in data.LAYERS}
        with st.expander("Choose layers", expanded=False):
            for group, keys in LAYER_GROUPS:
                st.caption(group)
                for key in keys:
                    spec = by_key.get(key)
                    if spec and st.checkbox(spec.label, value=spec.default_on, key=f"layer_{spec.key}"):
                        choice.enabled_layers.append(spec.key)
        choice.enabled_layers = [spec.key for spec in data.LAYERS if spec.key in choice.enabled_layers]
    return choice


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------
def run_analysis(choice: SidebarChoice) -> None:
    """Run the pipeline with a live progress list, then keep the result in session state."""
    states: dict = {}
    with st.status("Running analysis...", expanded=True) as status:
        box = st.empty()

        def on_progress(stage: str, state: str) -> None:
            states[stage] = state
            box.markdown(data.stage_progress_markdown(states))

        result = analyze_flood(choice.aoi, choice.event_date,
                               mock=set(choice.mock_stages) or False, use_cache=choice.use_cache,
                               on_progress=on_progress)
        status.update(label="Done" if result.ok else "Could not finish",
                      state="complete" if result.ok else "error", expanded=False)
    st.session_state["result"] = result


def open_run(path: Path) -> None:
    try:
        st.session_state["result"] = load_result(path)
    except PipelineError as exc:
        st.error(exc.user_message)


def open_from_link(cfg: dict) -> None:
    """``?run=<folder name>`` opens a finished run (a shareable link to a result).
    Only folder names inside the output root (or ``demo``) are accepted, never arbitrary paths."""
    params = getattr(st, "query_params", None)
    name = params.get("run") if params is not None else None
    if not name:
        return
    if name == "demo":
        demo = data.find_demo_run()
        if demo is not None:
            open_run(demo)
        return
    root = (REPO_ROOT / cfg.get("output", {}).get("root", "outputs")).resolve()
    for base in (root, root / "_mock"):
        candidate = (base / name).resolve()
        if candidate.parent == base and (candidate / "run.json").exists():
            open_run(candidate)
            return
    st.warning(f"No run named '{name}' was found in the outputs folder.")


# --------------------------------------------------------------------------
# Header chips, landing view
# --------------------------------------------------------------------------
def header_chips(result) -> list:
    summary = getattr(result, "summary", None) if result is not None else None
    if summary is None:
        return [("Sentinel-1 radar &middot; Copernicus DEM &middot; pre-event OpenStreetMap", "")]
    s = summary
    chips = [(f"<b>{style.esc(s.aoi_name or 'Area')}</b>", ""),
             (f"Event <b>{style.esc(s.event_date)}</b>", "")]
    if s.before_date or s.after_date:
        chips.append((f"Images {style.esc(s.before_date or '?')} &rarr; {style.esc(s.after_date or '?')}", ""))
    model = s.model_version or ""
    if model:
        is_ai = model.startswith("unet")
        chips.append((("AI model" if is_ai else "Non-AI baseline") + f": {style.esc(model)}",
                      "ai" if is_ai else "base"))
    if s.is_mock:
        chips.append(("MOCK DATA", "mock"))
    return chips


def render_landing() -> None:
    _html(style.section_html("Map a flood in three steps"))
    _html(style.steps_html())
    events = data.load_events()
    _html(style.section_html("Try a real flood event",
                             "Past floods with a clear before/after change in the radar images. Pick one, then "
                             "press Run analysis in the sidebar (first run: about 2 minutes)."))
    if events:
        for column, event in zip(st.columns(len(events)), events):
            with column:
                when = _parse_date(event.date, DEFAULT_EVENT_DATE)
                _html(style.event_card_html(event.label, f"{when:%d %b %Y}", event.note))
                components.call_compat(st.button, "Use this event", key=f"use_{event.label}",
                                       on_click=_select_event, args=(event.label, event.date))
    _html(style.callout_html("info", "Choose an area and a flood date in the sidebar, then press Run analysis, "
                                     "or load the cached demo."))


# --------------------------------------------------------------------------
# Result view
# --------------------------------------------------------------------------
def render_notices(summary, result) -> None:
    banners = data.banners(summary, result)
    top = [b for b in banners if b.level == "error" or b.text.startswith("MOCK DATA")]
    rest = [b for b in banners if b not in top]
    for b in top:
        _html(style.callout_html(b.level, b.text))
    if rest:
        with st.expander(f"Data notes and caveats ({len(rest)})", expanded=False):
            for b in rest:
                _html(style.callout_html(b.level, b.text))


def render_map_tab(result, summary, enabled: list) -> None:
    run = result.run
    prepared = cached_layers(str(run.root), data.run_version(run), tuple(enabled))
    try:
        aoi = run.aoi
    except Exception:
        aoi = None
    bounds = data.map_bounds(prepared, aoi)
    for layer in prepared.values():
        if layer.note:
            st.caption(layer.note)
    try:
        from dashboard.map_layers import build_map              # imported late: needs folium
        from streamlit_folium import st_folium
    except ImportError as exc:
        st.error(f"The map libraries are not installed ({exc}). Run: pip install folium streamlit-folium")
        return
    left, right = st.columns([3, 1])
    with left:
        fmap = build_map(prepared, bounds, enabled)
        # returned_objects=[] stops a rerun on every pan/zoom (the usual st_folium performance trap)
        components.call_compat(st_folium, fmap, height=640, returned_objects=[])
    with right:
        components.render_map_side_panel(run, summary)
    caption = data.imagery_caption(summary)
    if caption:
        st.caption(caption)


def render_result(result, choice: SidebarChoice) -> None:
    summary = result.summary
    render_notices(summary, result)
    _html(style.kpi_html(data.kpi_cards(summary)))
    run = result.run
    tabs = st.tabs(["Map", "Before / After", "Tables", "Report", "Flood path (bonus)", "Case study", "About"])
    with tabs[0]:
        render_map_tab(result, summary, choice.enabled_layers)
    with tabs[1]:
        components.render_before_after(run, summary)
    with tabs[2]:
        components.render_tables(cached_tables(str(run.root), data.run_version(run)))
    with tabs[3]:
        components.render_report_tab(run)
    with tabs[4]:
        components.render_flood_path(run)
    with tabs[5]:
        case_study.render()
    with tabs[6]:
        components.render_about(result)


def main() -> None:
    st.set_page_config(page_title="FloodResQ", page_icon="\U0001F30A", layout="wide",
                       initial_sidebar_state="expanded")
    _html(style.CSS)
    cfg = load_config()

    choice = render_sidebar(cfg)
    if choice.run_clicked and choice.aoi is not None:
        run_analysis(choice)
    elif choice.demo_clicked:
        demo = data.find_demo_run()
        if demo is not None:
            open_run(demo)
    elif choice.open_path is not None:
        open_run(choice.open_path)
    elif "result" not in st.session_state:
        open_from_link(cfg)

    result = st.session_state.get("result")
    _html(style.hero_html(header_chips(result)))
    _html(f'<div class="fr-disclaimer">ℹ️ {style.esc(data.DISCLAIMER)}</div>')
    if result is None:
        render_landing()
    elif not result.ok or result.summary is None:
        st.error(result.error or "This run has no summary yet.")
        if result.outcomes:
            with st.expander("What happened, step by step", expanded=True):
                components.render_stage_status(result.outcomes)
    else:
        render_result(result, choice)
    components.render_footer()


if __name__ == "__main__":
    main()
