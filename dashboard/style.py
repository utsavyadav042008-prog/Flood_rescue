"""Look and feel of the dashboard: one CSS block and small HTML building blocks.

Everything visual lives here so the rest of the UI code stays about content.
All text inserted into HTML is escaped (OSM names are untrusted input).
"""

from __future__ import annotations

import html
from typing import Iterable, Optional

# --------------------------------------------------------------------------
# Palette (also used by the map legend and the comparison images)
# --------------------------------------------------------------------------
NAVY = "#0B1F3A"
NAVY_2 = "#12355B"
WATER = "#0EA5E9"
ALERT = "#E5484D"
AMBER = "#F59E0B"
GOOD = "#16A34A"
MUTED = "#64748B"
BORDER = "#E2E8F0"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, p, li, label, input, textarea, button, h1, h2, h3, h4 {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}}
/* keep Streamlit's icon font (expander arrows etc.), otherwise the ligature name shows as text */
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {{
  font-family: 'Material Symbols Rounded' !important;
}}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}

/* ---------- header band ---------- */
.fr-hero {{
  display: flex; align-items: center; justify-content: space-between; gap: 18px; flex-wrap: wrap;
  background: linear-gradient(120deg, {NAVY} 0%, {NAVY_2} 60%, #1B4F7A 100%);
  color: #fff; border-radius: 16px; padding: 18px 24px; margin-bottom: 10px;
  box-shadow: 0 8px 24px rgba(11, 31, 58, .18);
}}
.fr-brand {{ display: flex; align-items: center; gap: 14px; }}
.fr-logo {{ width: 46px; height: 46px; border-radius: 12px; background: rgba(255,255,255,.12);
  display: flex; align-items: center; justify-content: center; }}
.fr-title {{ font-size: 1.55rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1; }}
.fr-sub {{ font-size: .88rem; opacity: .78; margin-top: 2px; }}
.fr-chips {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
.fr-chip {{ background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18);
  padding: 5px 11px; border-radius: 999px; font-size: .78rem; white-space: nowrap; }}
.fr-chip b {{ font-weight: 700; }}
.fr-chip.mock {{ background: {AMBER}; border-color: {AMBER}; color: #111; font-weight: 700; }}
.fr-chip.ai {{ background: rgba(22,163,74,.25); border-color: rgba(22,163,74,.6); }}
.fr-chip.base {{ background: rgba(245,158,11,.22); border-color: rgba(245,158,11,.55); }}
.fr-disclaimer {{ color: {MUTED}; font-size: .8rem; margin: 2px 4px 14px; }}

/* ---------- KPI cards ---------- */
.fr-kpis {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin: 6px 0 16px; }}
@media (max-width: 1200px) {{ .fr-kpis {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
@media (max-width: 700px) {{ .fr-kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.fr-kpi {{ background: #fff; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(15,23,42,.04); position: relative; overflow: hidden; }}
.fr-kpi:before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); }}
.fr-kpi-head {{ display: flex; align-items: center; gap: 8px; color: {MUTED}; font-size: .8rem; font-weight: 600;
  line-height: 1.25; min-height: 2.6em; }}
.fr-kpi-icon {{ width: 26px; height: 26px; border-radius: 8px; display: flex; align-items: center;
  justify-content: center; background: var(--soft); color: var(--accent); font-size: 14px; }}
.fr-kpi-value {{ font-size: 1.85rem; font-weight: 800; color: #0F172A; margin-top: 8px; letter-spacing: -.02em; }}
.fr-kpi-value small {{ font-size: .85rem; font-weight: 600; color: {MUTED}; margin-left: 4px; }}
.fr-kpi-sub {{ font-size: .78rem; color: {MUTED}; margin-top: 2px; min-height: 1.1em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.fr-kpi.na .fr-kpi-value {{ color: #94A3B8; font-size: 1.3rem; }}

/* ---------- callouts ---------- */
.fr-callout {{ border-radius: 12px; padding: 11px 14px; margin: 6px 0; font-size: .88rem;
  border: 1px solid; display: flex; gap: 10px; align-items: flex-start; }}
.fr-callout.warn {{ background: #FFFBEB; border-color: #FDE68A; color: #78350F; }}
.fr-callout.error {{ background: #FEF2F2; border-color: #FECACA; color: #7F1D1D; }}
.fr-callout.info {{ background: #F0F9FF; border-color: #BAE6FD; color: #0C4A6E; }}

/* ---------- tabs ---------- */
[data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
[data-baseweb="tab"] {{ padding: 10px 16px !important; border-radius: 10px 10px 0 0; font-weight: 600; }}
[data-baseweb="tab"][aria-selected="true"] {{ color: {NAVY_2} !important; background: #EAF2FB; }}
[data-baseweb="tab-highlight"] {{ background-color: {NAVY_2} !important; }}

/* ---------- section cards ---------- */
.fr-section-title {{ font-size: 1.05rem; font-weight: 700; color: #0F172A; margin: 4px 0 2px; }}
.fr-section-sub {{ font-size: .84rem; color: {MUTED}; margin-bottom: 10px; }}
.fr-img-label {{ display: flex; justify-content: space-between; align-items: baseline; margin: 2px 0 6px; }}
.fr-img-label b {{ font-size: .95rem; }}
.fr-date {{ font-size: .78rem; color: {NAVY_2}; background: #EAF2FB; padding: 2px 9px; border-radius: 999px; font-weight: 600; }}
.fr-scale {{ display: flex; align-items: center; gap: 8px; font-size: .75rem; color: {MUTED}; margin-top: 4px; }}
.fr-scale .bar {{ flex: 1; height: 8px; border-radius: 4px;
  background: linear-gradient(90deg, #1D4ED8, #93C5FD, #F8FAFC, #FDBA74, #C2410C); }}

/* ---------- landing ---------- */
.fr-steps {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 8px 0 18px; }}
@media (max-width: 900px) {{ .fr-steps {{ grid-template-columns: 1fr; }} }}
.fr-step {{ background: #fff; border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 18px; }}
.fr-step .n {{ width: 28px; height: 28px; border-radius: 50%; background: {NAVY_2}; color: #fff; font-weight: 700;
  display: flex; align-items: center; justify-content: center; font-size: .85rem; margin-bottom: 8px; }}
.fr-step h4 {{ margin: 0 0 4px; font-size: 1rem; }}
.fr-step p {{ margin: 0; color: {MUTED}; font-size: .86rem; }}
.fr-event {{ background: #fff; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 16px; min-height: 128px; }}
.fr-event h4 {{ margin: 0 0 4px; font-size: .98rem; }}
.fr-event p {{ margin: 6px 0 0; color: {MUTED}; font-size: .82rem; }}

/* ---------- sidebar ---------- */
.fr-side-brand {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
.fr-side-brand b {{ font-size: 1.15rem; color: {NAVY}; }}
.fr-side-h {{ font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
  color: {MUTED}; margin: 14px 0 4px; }}

/* ---------- misc ---------- */
.fr-footer {{ color: #94A3B8; font-size: .74rem; border-top: 1px solid {BORDER}; margin-top: 26px; padding-top: 10px; }}
div[data-testid="stExpander"] details {{ border-radius: 12px; border-color: {BORDER}; background: #fff; }}
iframe {{ border-radius: 12px; }}
.stButton > button[kind="primary"] {{ background: {NAVY_2}; border-color: {NAVY_2}; font-weight: 600; }}
.stButton > button[kind="primary"]:hover {{ background: {NAVY}; border-color: {NAVY}; }}
.stButton > button {{ border-radius: 10px; }}
.stDownloadButton > button {{ border-radius: 10px; }}
</style>
"""

LOGO_SVG = (
    '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M2 12c2 0 2-2 4-2s2 2 4 2 2-2 4-2 2 2 4 2 2-2 4-2"/>'
    '<path d="M2 18c2 0 2-2 4-2s2 2 4 2 2-2 4-2 2 2 4 2 2-2 4-2"/><path d="M12 2l3 5H9z" fill="white"/></svg>')
LOGO_SVG_DARK = LOGO_SVG.replace('stroke="white"', f'stroke="{NAVY_2}"').replace('fill="white"', f'fill="{NAVY_2}"')

#: (icon, accent colour, soft background) per KPI key
KPI_STYLE = {
    "flood": ("\U0001F30A", WATER, "#E0F2FE"),
    "roads": ("\U0001F6E3️", ALERT, "#FEE2E2"),
    "bridges": ("\U0001F309", AMBER, "#FEF3C7"),
    "buildings": ("\U0001F3E0", "#8B5CF6", "#EDE9FE"),
    "cutoff": ("⚠️", ALERT, "#FEE2E2"),
    "confidence": ("\U0001F3AF", GOOD, "#DCFCE7"),
}


def esc(text: object) -> str:
    return html.escape("" if text is None else str(text))


def hero_html(chips: Iterable[tuple] = ()) -> str:
    """Header band.  ``chips`` = [(text, css_class), ...]; text may contain <b> from our own code only."""
    chip_html = "".join(f'<span class="fr-chip {cls}">{text}</span>' for text, cls in chips)
    return (f'<div class="fr-hero"><div class="fr-brand"><div class="fr-logo">{LOGO_SVG}</div><div>'
            '<div class="fr-title">FloodResQ</div>'
            '<div class="fr-sub">Where did the flood hit, what was damaged, and who is cut off &mdash; '
            'estimated from satellite data</div></div></div>'
            f'<div class="fr-chips">{chip_html}</div></div>')


def kpi_html(cards: list) -> str:
    """HTML grid for the six KPI cards (``dashboard.data.Card`` objects)."""
    out = []
    for c in cards:
        icon, accent, soft = KPI_STYLE.get(c.key, ("•", NAVY_2, "#EAF2FB"))
        if c.key == "cutoff" and c.available and c.value not in ("0", "n/a"):
            accent, soft = ALERT, "#FEE2E2"
        value = esc(c.value) if c.available else "not available"
        unit = f"<small>{esc(c.sub)}</small>" if c.available and c.sub and c.key in ("flood",) else ""
        sub = "" if c.key == "flood" else (esc(c.sub) if c.available and c.sub else
                                           ("" if c.available else "stage did not run"))
        out.append(f'<div class="fr-kpi{"" if c.available else " na"}" style="--accent:{accent};--soft:{soft}">'
                   f'<div class="fr-kpi-head"><span class="fr-kpi-icon">{icon}</span>{esc(c.label)}</div>'
                   f'<div class="fr-kpi-value">{value}{unit}</div><div class="fr-kpi-sub">{sub}</div></div>')
    return f'<div class="fr-kpis">{"".join(out)}</div>'


def callout_html(level: str, text: str) -> str:
    icon = {"warning": "⚠️", "error": "⛔", "info": "ℹ️"}.get(level, "ℹ️")
    css = {"warning": "warn", "error": "error"}.get(level, "info")
    return f'<div class="fr-callout {css}"><span>{icon}</span><span>{esc(text)}</span></div>'


def section_html(title: str, sub: str = "") -> str:
    return (f'<div class="fr-section-title">{esc(title)}</div>'
            + (f'<div class="fr-section-sub">{esc(sub)}</div>' if sub else ""))


def image_label_html(title: str, date: Optional[str] = "") -> str:
    return (f'<div class="fr-img-label"><b>{esc(title)}</b>'
            + (f'<span class="fr-date">{esc(date)}</span>' if date else "") + "</div>")


def change_scale_html() -> str:
    return ('<div class="fr-scale"><span>darker after (water)</span><div class="bar"></div>'
            '<span>brighter after (debris, rubble)</span></div>')


def steps_html() -> str:
    steps = (("Choose an area and a flood date", "Pick a preset or a real past event in the sidebar, or type a "
              "custom box (up to 2,500 km²)."),
             ("Run the analysis", "Sentinel-1 radar before and after the date, pre-event OpenStreetMap, flood "
              "detection, damage overlay and road-network analysis run automatically."),
             ("Read the answers", "Flood extent, affected roads, bridges and buildings, cut-off settlements, a "
              "before/after comparison and a one-page report in English and Nepali."))
    cards = "".join(f'<div class="fr-step"><div class="n">{i}</div><h4>{esc(t)}</h4><p>{esc(d)}</p></div>'
                    for i, (t, d) in enumerate(steps, start=1))
    return f'<div class="fr-steps">{cards}</div>'


def event_card_html(name: str, date: str, note: str) -> str:
    return (f'<div class="fr-event"><h4>{esc(name)}</h4><span class="fr-date">{esc(date)}</span>'
            f'<p>{esc(note)}</p></div>')


def footer_html(lines: Iterable[str]) -> str:
    return '<div class="fr-footer">' + "<br>".join(esc(line) for line in lines) + "</div>"
