"""Streamlit building blocks: small functions that draw one piece of the page.

No business logic lives here; everything it shows is prepared by
``dashboard.data`` (and is therefore unit-tested).  Every number printed comes
from ``summary.json`` through the shared formatter.  Styling: ``dashboard.style``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import streamlit as st

from dashboard import data, style
from pipeline.contracts import Artifact
from pipeline.fileio import read_json
from pipeline.run import RunPaths
from reports.format import ATTRIBUTION_LINES


def call_compat(function: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call a Streamlit function that is "stretch to full width" on every version.

    Streamlit renamed ``use_container_width=True`` to ``width="stretch"`` and
    plans to remove the old spelling.  We try the new form first, then the old
    one, then plain, so the dashboard keeps working whichever version
    ``pip install streamlit`` fetched.
    """
    last: Optional[Exception] = None
    for extra in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            return function(*args, **kwargs, **extra)
        except Exception as exc:        # version probing: an unknown argument raises on old/new versions
            last = exc
    assert last is not None
    raise last


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def render_banners(banners: list) -> None:
    for banner in banners:
        _html(style.callout_html(banner.level, banner.text))


def render_kpis(cards: list) -> None:
    """The six headline cards (unavailable values say so instead of showing a fake 0)."""
    _html(style.kpi_html(cards))


def render_stage_status(outcomes: list) -> None:
    """Which stages ran, were cached, failed or were skipped, with their messages."""
    lines = []
    for o in outcomes:
        icon = data.STATE_ICONS.get(o.status, "")
        label = data.STAGE_LABELS.get(o.name, o.name)
        text = f"{icon} **{label}** — {o.status}"
        if o.seconds >= 0.05:
            text += f" ({o.seconds:.1f} s)"
        if o.message:
            text += f": {o.message}"
        lines.append(text)
    st.markdown("\n\n".join(lines))


# --------------------------------------------------------------------------
# Map side panel
# --------------------------------------------------------------------------
def render_map_side_panel(run: RunPaths, summary: Any) -> None:
    """Who is cut off and why, next to the map."""
    _html(style.section_html("Cut-off settlements"))
    try:
        feats = read_json(run.path(Artifact.CUTOFF_SETTLEMENTS)).get("features", [])
    except (OSError, ValueError, AttributeError):
        feats = None
    if feats is None:
        st.caption("Not available for this run.")
    elif not feats:
        st.caption("None: every analysed settlement still has a road route to a hospital or town.")
    else:
        for feat in feats[:12]:
            p = feat.get("properties") or {}
            with st.expander(f"\U0001F534 {p.get('name', '?')}", expanded=False):
                st.caption(f"Nearest destination before the flood: {p.get('nearest_destination', '-')}")
                st.write(p.get("reason", ""))
        if len(feats) > 12:
            st.caption(f"... and {len(feats) - 12} more (see the Tables tab).")
    try:
        conn = read_json(run.path(Artifact.CONNECTIVITY_SUMMARY))
        dest = conn.get("destinations_used") or []
        st.caption(f"Settlements analysed: {conn.get('settlements_total', '-')} · destinations: "
                   f"{len(dest)} hospital(s)/town(s)")
    except (OSError, ValueError, AttributeError):
        pass


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def _column_config() -> Optional[dict]:
    cc = getattr(st, "column_config", None)
    if cc is None:
        return None
    try:
        return {"Overlap": cc.ProgressColumn("Overlap with flood", min_value=0.0, max_value=1.0, format="%.2f"),
                "Confidence": cc.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f")}
    except Exception:
        return None


def render_tables(tables: dict) -> None:
    if not tables:
        st.info("No result tables are available for this run.")
        return
    config = _column_config()
    for key, table in tables.items():
        _html(style.section_html(f"{table.title} ({table.total:,})"))
        if not table.rows:
            st.caption("Nothing found.")
            continue
        kwargs = {"hide_index": True}
        if config and key != "cutoff":
            kwargs["column_config"] = config
        call_compat(st.dataframe, table.rows, **kwargs)
        if table.note:
            st.caption(table.note)
        st.download_button(f"Download {table.title.lower()} (CSV)", data=data.table_to_csv(table),
                           file_name=f"{key}.csv", mime="text/csv", key=f"dl_{key}")


# --------------------------------------------------------------------------
# Before / after
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Preparing the before/after images...", max_entries=6)
def _pair(run_dir: str, version: float) -> dict:
    from dashboard import raster_overlay
    return raster_overlay.load_pair(RunPaths(Path(run_dir)))


def _version(run: RunPaths) -> float:
    p = run.path(Artifact.AFTER)
    return p.stat().st_mtime if p.exists() else 0.0


def render_before_after(run: RunPaths, summary: Any) -> None:
    """Before, after and change on ONE contrast scale, plus a before<->after swipe."""
    from dashboard import raster_overlay

    before_date = getattr(summary, "before_date", "") or ""
    after_date = getattr(summary, "after_date", "") or ""
    _html(style.section_html(
        "Before and after the flood",
        "Sentinel-1 radar (VV). Both images use the same contrast, so a change in brightness is a real change "
        "on the ground. Dark = smooth surface such as water; bright = rough surface such as debris or rubble."))
    pair = None
    if run.path(Artifact.BEFORE).exists() and run.path(Artifact.AFTER).exists():
        try:
            pair = _pair(str(run.root), _version(run))
        except Exception as exc:
            st.info(f"Preview unavailable: {exc}")
    else:
        st.info("Image not available for this run.")

    if pair is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            _html(style.image_label_html("Before", before_date))
            call_compat(st.image, raster_overlay.png_bytes("before", pair))
        with c2:
            _html(style.image_label_html("After", after_date))
            call_compat(st.image, raster_overlay.png_bytes("after", pair))
        with c3:
            _html(style.image_label_html("Change (after − before)", ""))
            call_compat(st.image, raster_overlay.png_bytes("change", pair))
            _html(style.change_scale_html())

        _html(style.section_html("Swipe between the two dates",
                                 "Move the slider to fade from the before image to the after image."))
        slider = getattr(st, "slider", None)
        mix = 50 if slider is None else slider(
            f"Before ({before_date or '?'})  ↔  After ({after_date or '?'})", 0, 100, 50, 5)
        left, mid, right = st.columns([1, 2, 1])
        with mid:
            call_compat(st.image, raster_overlay.png_bytes("blend", pair, mix / 100.0))

    if run.path(Artifact.FIG_FLOOD).exists():
        _html(style.section_html("Detected flood / debris", "Yellow = pixels the detector marked as new water "
                                                            "or debris, drawn on the after image."))
        left, mid, right = st.columns([1, 2, 1])
        with mid:
            call_compat(st.image, str(run.path(Artifact.FIG_FLOOD)))
    optical = run.root / "extras" / "optical"
    looks = [(t, optical / f"quicklook_{k}.png") for t, k in (("Sentinel-2 before", "before"),
                                                              ("Sentinel-2 after", "after"))]
    looks = [(t, p) for t, p in looks if p.exists()]
    if looks:
        _html(style.section_html("Optical view (Sentinel-2, clear-sky pixels only)"))
        for column, (title, path) in zip(st.columns(len(looks)), looks):
            with column:
                st.caption(title)
                call_compat(st.image, str(path))


# --------------------------------------------------------------------------
# Flood path (bonus)
# --------------------------------------------------------------------------
def render_flood_path(run: RunPaths) -> None:
    """BONUS tab: trace a flood path downhill from an upstream point (gis/flood_path.py)."""
    _html(style.section_html(
        "Trace a flood path downhill",
        "Pick a point upstream (for example a glacial lake or the landslide source). The path follows the "
        "steepest descent on the Copernicus DEM; settlements within the corridor are listed in downstream "
        "order. A screening tool, not a hydraulic model."))
    if not (run.root / "extras" / "dem.tif").exists():
        st.info("This run has no elevation model (extras/dem.tif). It is written by the 'acquire' stage of a "
                "real run; mock/demo runs do not include it.")
        return
    try:
        aoi = run.aoi
        default_lon, default_lat = aoi.center
    except Exception:
        default_lon, default_lat = 85.3, 28.1
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    lon = c1.number_input("Start longitude", value=float(round(default_lon, 4)), format="%.4f")
    lat = c2.number_input("Start latitude", value=float(round(default_lat, 4)), format="%.4f")
    corridor = c3.number_input("Corridor half-width (m)", value=500, min_value=50, max_value=5000, step=50)
    with c4:
        go = call_compat(st.button, "Trace flood path", type="primary")
    if not go:
        return
    from gis.flood_path import trace_flood_path
    try:
        fp = trace_flood_path(lon, lat, run_dir=run.root, corridor_m=float(corridor))
    except Exception as exc:
        st.error(f"Could not trace a path from this point: {exc}")
        return
    st.success(f"Path length {fp.length_km} km, elevation drop {fp.drop_m} m ({fp.stop_reason}). "
               f"{len(fp.settlements)} settlement(s) within {int(corridor)} m of the path.")
    import pandas as pd
    st.map(pd.DataFrame({"lon": [p[0] for p in fp.lonlat], "lat": [p[1] for p in fp.lonlat]}))
    if fp.settlements:
        call_compat(st.dataframe, pd.DataFrame([{k: v for k, v in s.items() if k != "lonlat"}
                                                for s in fp.settlements]), hide_index=True)
    st.download_button("Download path (GeoJSON)", data=json.dumps(fp.to_geojson(), ensure_ascii=False),
                       file_name="flood_path.geojson", mime="application/geo+json")


# --------------------------------------------------------------------------
# Report, about, footer
# --------------------------------------------------------------------------
def render_report_tab(run: RunPaths) -> None:
    """Show the situation report (English, plus Nepali when it exists) with download buttons."""
    variants = [("English", Artifact.REPORT_TXT, Artifact.REPORT_PDF, Artifact.REPORT_HTML, "en")]
    if run.path(Artifact.REPORT_NE_TXT).exists():
        variants.append(("नेपाली", Artifact.REPORT_NE_TXT, None,
                         Artifact.REPORT_NE_HTML, "ne"))
    if not run.path(Artifact.REPORT_TXT).exists():
        st.info("The situation report has not been generated for this run yet.")
        return
    _html(style.section_html("One-page situation report",
                             "Every number comes from the same summary file as the dashboard cards."))
    left, right = st.columns([3, 1])
    with right:
        chosen = variants[0]
        if len(variants) > 1:
            label = st.radio("Language", [v[0] for v in variants], horizontal=True)
            chosen = next(v for v in variants if v[0] == label)
        _, txt, pdf, html_file, code = chosen
        text = run.path(txt).read_text(encoding="utf-8")
        call_compat(st.download_button, "Download report (text)" if code == "en" else
                    "Download report, Nepali (text)", data=text.encode("utf-8"), file_name=run.path(txt).name,
                    mime="text/plain", key=f"dl_report_txt_{code}")
        if pdf is not None and run.path(pdf).exists():
            call_compat(st.download_button, "Download report (PDF)", data=run.path(pdf).read_bytes(),
                        file_name=run.path(pdf).name, mime="application/pdf", key=f"dl_report_pdf_{code}")
        if html_file is not None and run.path(html_file).exists():
            call_compat(st.download_button, "Download printable version (HTML)" if code == "en" else
                        "Download printable version, Nepali (HTML)", data=run.path(html_file).read_bytes(),
                        file_name=run.path(html_file).name, mime="text/html", key=f"dl_report_html_{code}")
            if code == "ne":
                st.caption("Open the HTML file in a browser and use Print > Save as PDF to get a Nepali PDF.")
    with left:
        st.code(text, language=None)


def render_about(result: Any) -> None:
    left, right = st.columns(2)
    with left:
        _html(style.section_html("What this does"))
        st.markdown(
            "It compares a radar image from before and after a flood (Sentinel-1, same orbit track), detects "
            "flooded and debris-covered ground, overlays pre-event OpenStreetMap buildings, roads and bridges, "
            "and finds settlements with no remaining road connection to a hospital or town.")
        _html(style.section_html("Data"))
        st.markdown("Sentinel-1/2 and the Copernicus DEM (Copernicus programme), OpenStreetMap (pre-event "
                    "snapshot via ohsome), Kuro Siwo for model training.")
        if result is not None and getattr(result, "outcomes", None):
            _html(style.section_html("Processing steps in this run"))
            render_stage_status(result.outcomes)
    with right:
        _html(style.section_html("What this system cannot do"))
        for item in data.LIMITATIONS:
            st.markdown(f"- {item}")
        _html(style.section_html("Responsible use"))
        st.markdown("The August 2026 disaster is recent and many families are still affected. "
                    "This is an educational prototype; it shows no images of people and must not be used for "
                    "operational decisions.")


def render_footer() -> None:
    st.divider()
    for line in ATTRIBUTION_LINES:
        st.caption(line)
