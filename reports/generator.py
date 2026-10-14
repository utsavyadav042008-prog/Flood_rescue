"""Situation-report generator: ``summary.json`` -> text, HTML and PDF.

Pipeline of this module::

    summary dict -> fmt() -> build_model() -> ReportModel -+-> render_text()
                                                           +-> render_html()
                                                           +-> render_pdf()   (English only)

Why a ``ReportModel`` in the middle?  All wording decisions (what to show when
a value is missing, which warnings to list, how many names fit) are made ONCE
in ``build_model``.  The three renderers only draw, so the text, the HTML and
the PDF can never disagree with each other.

Principles
----------
* Numbers come only from ``summary.json`` through ``reports.format.fmt``.
* ``None`` is printed as "not available", never as 0.
* Wording is hedged ("affected", "no road connection found"), never
  "destroyed" or "confirmed".
* One page: lists are capped, and the PDF shrinks to fit (``KeepInFrame``).
"""

from __future__ import annotations

import base64
import html
import logging
import textwrap
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional
from xml.sax.saxutils import escape as xml_escape

from pipeline.contracts import Artifact
from pipeline.fileio import atomic_path, read_json
from pipeline.run import RunPaths
from pipeline.schemas import AOI, Summary
from reports.format import (ATTRIBUTION_LINES, MAX_LISTED_SETTLEMENTS, NA, fmt, to_devanagari_digits)
from reports.strings import STRINGS

log = logging.getLogger("floodresq")

LANGUAGES = ("en", "ne")
MAX_WARNINGS = 6              # warnings listed on the one-page report
MAX_WARNING_CHARS = 240
MAX_NAME_CHARS = 60           # settlement names are shortened beyond this
TEXT_WIDTH = 88


# ==========================================================================
# 1. The model
# ==========================================================================
@dataclass
class Section:
    title: str
    paragraphs: list = field(default_factory=list)
    bullets: list = field(default_factory=list)


@dataclass
class ReportModel:
    lang: str
    title: str
    subtitle: str
    banner: str                                    # MOCK warning or ""
    meta: list                                     # [(label, value)]
    kpi_title: str
    kpis: list                                     # [(label, value)]
    cutoff_title: str
    cutoff_items: list                             # settlement names (max 10)
    cutoff_more: str                               # "... and N more" or ""
    cutoff_message: str                            # shown when there is no list
    cutoff_note: str                               # definition of "cut off"
    sections: list                                 # [Section]
    attribution_title: str
    attribution: list
    image: Optional[Path] = None
    image_caption: str = ""
    footnote: str = ""                             # e.g. "translation not reviewed"

    def all_strings(self) -> list:
        """Every string that will be printed (used to detect characters the PDF font lacks)."""
        out = [self.title, self.subtitle, self.banner, self.kpi_title, self.cutoff_title, self.cutoff_more,
               self.cutoff_message, self.cutoff_note, self.attribution_title, self.image_caption, self.footnote]
        out += [x for pair in self.meta + self.kpis for x in pair] + list(self.cutoff_items) + self.attribution
        for s in self.sections:
            out += [s.title] + list(s.paragraphs) + list(s.bullets)
        return [x for x in out if x]


def _pretty_timestamp(ts: str) -> str:
    """``2026-10-07T10:00:00+00:00`` -> ``2026-10-07 10:00 UTC`` (other formats are left unchanged)."""
    if len(ts) >= 25 and ts[10] == "T" and ts.endswith("+00:00"):
        return f"{ts[:10]} {ts[11:16]} UTC"
    return ts


def _shorten(text: str, limit: int) -> str:
    text = " ".join(str(text).split())               # also collapses newlines in odd OSM names
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "\u2026"


def build_model(summary: Any, lang: str = "en", *, nepali_reviewed: bool = False,
                image: Optional[Path] = None) -> ReportModel:
    """Decide everything that will be said.  No drawing happens here."""
    if lang not in STRINGS:
        raise ValueError(f"unsupported language {lang!r}; choose from {LANGUAGES}")
    s = summary.to_dict() if hasattr(summary, "to_dict") else dict(summary)
    f, T = fmt(s), STRINGS[lang]
    loc = to_devanagari_digits if lang == "ne" else (lambda text: text)
    na = T["na"]

    def value(raw: str) -> str:
        return na if raw == NA else raw

    def with_unit(raw: str, unit: str) -> str:
        return na if raw == NA else f"{raw} {unit}"

    # -- header ---------------------------------------------------------------------------------
    before, after, orbit = f["before_date"], f["after_date"], f["sentinel1_orbit"]
    if before or after:
        imagery = T["imagery_value"].format(before=before or "?", after=after or "?")
        if orbit:
            imagery += T["imagery_orbit"].format(orbit=orbit)
        imagery = loc(imagery)
    else:
        imagery = na
    meta = [
        (T["m_area"], f["aoi_name"] or na),
        (T["m_event"], loc(f["event_date"]) if f["event_date"] else na),
        (T["m_imagery"], imagery),
        (T["m_osm"], T["osm_value"].format(date=loc(f["osm_snapshot_date"])) if f["osm_snapshot_date"] else na),
        (T["m_model"], f["model_version"] or na),
        (T["m_generated"], loc(_pretty_timestamp(f["processing_date"])) if f["processing_date"] else na),
    ]

    # -- key figures ---------------------------------------------------------------------------------
    roads = value(f["affected_roads"])
    if roads != na and f["affected_road_length_km"] != NA:
        roads += f" ({f['affected_road_length_km']} {T['unit_km']})"
    kpis = [
        (T["k_flood"], with_unit(f["flood_area_km2"], T["unit_km2"])),
        (T["k_roads"], roads),
        (T["k_bridges"], value(f["affected_bridges"])),
        (T["k_buildings"], value(f["affected_buildings"])),
        (T["k_cutoff"], value(f["cutoff_settlements"])),
    ]
    kpis = [(label, loc(v)) for label, v in kpis]

    # -- cut-off list -----------------------------------------------------------------------------------
    names = [_shorten(n, MAX_NAME_CHARS) for n in (s.get("cutoff_names") or [])]
    total = s.get("cutoff_settlements")
    shown = names[:MAX_LISTED_SETTLEMENTS]
    hidden = max(len(names), total if isinstance(total, int) else 0) - len(shown)
    message = ""
    if total is None:
        message = T["cutoff_na"]
    elif total == 0:
        message = T["cutoff_none"]

    # -- text sections --------------------------------------------------------------------------------------
    quality = Section(T["quality_title"])
    quality.paragraphs.append(T["quality_conf"].format(conf=loc(f["mean_confidence"]))
                              if f["mean_confidence"] != NA else T["quality_conf_na"])
    notes = [w for w in (s.get("warnings") or []) if not str(w).startswith("MOCK DATA")]
    if notes:
        quality.paragraphs.append(T["quality_warnings"])
        quality.bullets = [_shorten(w, MAX_WARNING_CHARS) for w in notes[:MAX_WARNINGS]]
        if len(notes) > MAX_WARNINGS:
            quality.bullets.append(T["warnings_more"].format(n=loc(str(len(notes) - MAX_WARNINGS))))

    footnote = T["unreviewed"] if lang == "ne" and not nepali_reviewed else ""
    return ReportModel(
        lang=lang, title=T["title"], subtitle=T["subtitle"],
        banner=T["mock_banner"] if s.get("is_mock") else "",
        meta=meta, kpi_title=T["kpi_title"], kpis=kpis,
        cutoff_title=T["cutoff_title"], cutoff_items=shown,
        cutoff_more=T["cutoff_more"].format(n=loc(str(hidden))) if hidden > 0 else "",
        cutoff_message=message, cutoff_note=T["cutoff_note"] if total is not None else "",
        sections=[Section(T["how_title"], [loc(T["how_text"])]), quality,
                  Section(T["limits_title"], [loc(T["limits_text"])])],
        attribution_title=T["attribution_title"], attribution=list(ATTRIBUTION_LINES),
        image=image, image_caption=T["map_caption"] if image else "", footnote=footnote)


# ==========================================================================
# 2. Plain text
# ==========================================================================
def render_text(model: ReportModel, width: int = TEXT_WIDTH) -> str:
    def wrap(text: str, first: str = "", rest: str = "") -> str:
        return textwrap.fill(text, width, initial_indent=first, subsequent_indent=rest,
                             break_long_words=False, break_on_hyphens=False)

    lines = [model.title, "=" * min(len(model.title), width), wrap(model.subtitle)]
    if model.banner:
        lines += ["", wrap("*** " + model.banner + " ***")]
    lines.append("")
    lines += [wrap(f"{label}: {val}", rest="    ") for label, val in model.meta]
    lines += ["", model.kpi_title, "-" * len(model.kpi_title)]
    lines += [wrap(f"- {label}: {val}", rest="  ") for label, val in model.kpis]
    lines += ["", model.cutoff_title, "-" * len(model.cutoff_title)]
    if model.cutoff_message:
        lines.append(wrap(model.cutoff_message))
    lines += [wrap(f"- {name}", rest="  ") for name in model.cutoff_items]
    if model.cutoff_more:
        lines.append(f"  {model.cutoff_more}")
    if model.cutoff_note:
        lines.append(wrap(model.cutoff_note))
    for section in model.sections:
        lines += ["", section.title, "-" * len(section.title)]
        lines += [wrap(p) for p in section.paragraphs]
        lines += [wrap(f"- {b}", rest="  ") for b in section.bullets]
    lines += ["", model.attribution_title, "-" * len(model.attribution_title)]
    lines += [wrap(a) for a in model.attribution]
    if model.footnote:
        lines += ["", wrap(model.footnote)]
    return "\n".join(lines) + "\n"


# ==========================================================================
# 3. HTML (printable; also the way to get a Nepali PDF)
# ==========================================================================
_CSS = """
@page { size: A4; margin: 14mm; }
body { font: 13px/1.45 "Segoe UI","Nirmala UI","Noto Sans Devanagari","Mangal",Arial,sans-serif;
       color: #1b1b1b; max-width: 820px; margin: 1.2em auto; padding: 0 1em; }
h1 { font-size: 20px; margin: 0 0 .2em; } h2 { font-size: 14px; margin: 1.1em 0 .3em; color: #1f4e79; }
.sub { color: #555; margin: 0 0 .8em; }
.banner { background: #b00020; color: #fff; padding: .5em .8em; font-weight: bold; margin: .6em 0; }
table { border-collapse: collapse; width: 100%; } td { padding: 3px 8px; vertical-align: top; }
.kpi td { border-bottom: 1px solid #ddd; } .kpi td:last-child { text-align: right; font-weight: bold; }
.meta td:first-child { color: #555; white-space: nowrap; width: 1%; }
.small { font-size: 11px; color: #444; } img { max-width: 100%; border: 1px solid #ccc; }
@media print { body { margin: 0; max-width: none; } }
"""


def render_html(model: ReportModel) -> str:
    e = html.escape
    out = [f'<!DOCTYPE html><html lang="{STRINGS[model.lang]["html_lang"]}"><head><meta charset="utf-8">',
           f"<title>{e(model.title)}</title><style>{_CSS}</style></head><body>",
           f"<h1>{e(model.title)}</h1><p class='sub'>{e(model.subtitle)}</p>"]
    if model.banner:
        out.append(f"<div class='banner'>{e(model.banner)}</div>")
    out.append("<table class='meta'>" + "".join(
        f"<tr><td>{e(a)}</td><td>{e(b)}</td></tr>" for a, b in model.meta) + "</table>")
    out.append(f"<h2>{e(model.kpi_title)}</h2><table class='kpi'>" + "".join(
        f"<tr><td>{e(a)}</td><td>{e(b)}</td></tr>" for a, b in model.kpis) + "</table>")
    if model.image and Path(model.image).exists():
        data = base64.b64encode(Path(model.image).read_bytes()).decode("ascii")
        out.append(f"<p><img alt='' src='data:image/png;base64,{data}'><br>"
                   f"<span class='small'>{e(model.image_caption)}</span></p>")
    out.append(f"<h2>{e(model.cutoff_title)}</h2>")
    if model.cutoff_message:
        out.append(f"<p>{e(model.cutoff_message)}</p>")
    if model.cutoff_items or model.cutoff_more:
        out.append("<ul>" + "".join(f"<li>{e(n)}</li>" for n in model.cutoff_items)
                   + (f"<li>{e(model.cutoff_more)}</li>" if model.cutoff_more else "") + "</ul>")
    if model.cutoff_note:
        out.append(f"<p class='small'>{e(model.cutoff_note)}</p>")
    for section in model.sections:
        out.append(f"<h2>{e(section.title)}</h2>" + "".join(f"<p>{e(p)}</p>" for p in section.paragraphs))
        if section.bullets:
            out.append("<ul>" + "".join(f"<li>{e(b)}</li>" for b in section.bullets) + "</ul>")
    out.append(f"<h2>{e(model.attribution_title)}</h2>" + "".join(
        f"<p class='small'>{e(a)}</p>" for a in model.attribution))
    if model.footnote:
        out.append(f"<p class='small'><i>{e(model.footnote)}</i></p>")
    out.append("</body></html>")
    return "\n".join(out) + "\n"


# ==========================================================================
# 4. PDF (English; reportlab's built-in Helvetica cannot draw Devanagari)
# ==========================================================================
def pdf_safe(text: str) -> str:
    """Replace characters the built-in PDF font cannot print with '?'.

    The base-14 fonts cover Windows-1252 only; anything else (for example a
    settlement name in Devanagari) would otherwise print as a black box.
    """
    return text.encode("cp1252", "replace").decode("cp1252")


def render_pdf(model: ReportModel, path: Path) -> None:
    """Write a one-page A4 PDF.  Content that is too long is SHRUNK to fit, never spilled to page 2."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Frame, Image, KeepInFrame, Paragraph, Spacer, Table, TableStyle

    def esc(text: str) -> str:                       # Paragraph parses mini-XML: escape & < >
        return xml_escape(pdf_safe(text))

    width, height = A4
    margin = 14 * mm
    fw, fh = width - 2 * margin, height - 2 * margin

    base = ParagraphStyle("base", fontName="Helvetica", fontSize=8.8, leading=11.2)
    title = ParagraphStyle("title", parent=base, fontName="Helvetica-Bold", fontSize=14, leading=17)
    sub = ParagraphStyle("sub", parent=base, textColor=colors.HexColor("#555555"), fontSize=8.5)
    h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=9.6, leading=12,
                        textColor=colors.HexColor("#1f4e79"), spaceBefore=7, spaceAfter=2)
    small = ParagraphStyle("small", parent=base, fontSize=7, leading=8.6, textColor=colors.HexColor("#444444"))
    bullet = ParagraphStyle("bullet", parent=base, leftIndent=11, bulletIndent=2)
    value_style = ParagraphStyle("value", parent=base, fontName="Helvetica-Bold", alignment=2)  # right

    story: list = [Paragraph(esc(model.title), title), Paragraph(esc(model.subtitle), sub), Spacer(1, 3)]
    if model.banner:
        banner = Table([[Paragraph(esc(model.banner), ParagraphStyle("b", parent=base, fontName="Helvetica-Bold",
                                                                     textColor=colors.white))]], colWidths=[fw])
        banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#b00020")),
                                    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story += [banner, Spacer(1, 3)]

    meta = Table([[Paragraph(esc(a), small), Paragraph(esc(b), base)] for a, b in model.meta],
                 colWidths=[fw * 0.16, fw * 0.84])
    meta.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 1), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [meta, Paragraph(esc(model.kpi_title), h2)]

    kpi = Table([[Paragraph(esc(a), base), Paragraph(esc(b), value_style)] for a, b in model.kpis],
                colWidths=[fw * 0.68, fw * 0.32])
    kpi.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                             ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    story.append(kpi)

    if model.image and Path(model.image).exists():
        iw, ih = ImageReader(str(model.image)).getSize()
        draw_w = min(fw * 0.7, 120 * mm)
        draw_h = min(draw_w * ih / iw, 62 * mm)
        draw_w = draw_h * iw / ih                      # keep the aspect ratio after capping the height
        story += [Spacer(1, 4), Image(str(model.image), width=draw_w, height=draw_h),
                  Paragraph(esc(model.image_caption), small)]

    story.append(Paragraph(esc(model.cutoff_title), h2))
    if model.cutoff_message:
        story.append(Paragraph(esc(model.cutoff_message), base))
    for name in model.cutoff_items:
        story.append(Paragraph(esc(name), bullet, bulletText="\u2022"))
    if model.cutoff_more:
        story.append(Paragraph(esc(model.cutoff_more), bullet))
    if model.cutoff_note:
        story.append(Paragraph(esc(model.cutoff_note), small))
    for section in model.sections:
        story.append(Paragraph(esc(section.title), h2))
        story += [Paragraph(esc(p), base) for p in section.paragraphs]
        story += [Paragraph(esc(b), bullet, bulletText="\u2022") for b in section.bullets]
    story.append(Paragraph(esc(model.attribution_title), h2))
    story += [Paragraph(esc(a), small) for a in model.attribution]
    if any(pdf_safe(t) != t for t in model.all_strings()):
        story.append(Paragraph(esc(STRINGS["en"]["pdf_note"]), small))

    with atomic_path(path) as tmp:
        pdf = canvas.Canvas(str(tmp), pagesize=A4)
        pdf.setTitle(pdf_safe(model.title))
        pdf.setAuthor("FloodResQ")
        pdf.setSubject("Satellite-derived flood impact (educational prototype)")
        frame = Frame(margin, margin, fw, fh, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        # mode="shrink" scales everything down until it fits; vAlign="TOP" keeps it at the top of the page
        frame.addFromList([KeepInFrame(fw, fh, story, mode="shrink", vAlign="TOP")], pdf)
        pdf.save()


# ==========================================================================
# 5. Putting it together
# ==========================================================================
def generate_report(summary: Any, lang: str = "en", *, nepali_reviewed: bool = False) -> str:
    """The plain-text report for ``summary`` (a ``Summary`` or its dict)."""
    return render_text(build_model(summary, lang, nepali_reviewed=nepali_reviewed))


def _write_text(path: Path, text: str) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(text, encoding="utf-8", newline="\n")


def write_reports(run: RunPaths, summary: Any, cfg: dict) -> list:
    """Write every enabled report file into the run folder; return the paths written."""
    report_cfg = cfg.get("report", {})
    languages = [l for l in report_cfg.get("languages", ["en"]) if l in LANGUAGES]
    if "en" not in languages:
        languages.insert(0, "en")                     # the contract requires the English text
    reviewed = bool(report_cfg.get("nepali_reviewed", False))
    image = run.path(Artifact.FIG_FLOOD) if run.path(Artifact.FIG_FLOOD).exists() else None
    written: list = []

    en = build_model(summary, "en", image=image)
    _write_text(run.path(Artifact.REPORT_TXT), render_text(en))
    _write_text(run.path(Artifact.REPORT_HTML), render_html(en))
    written += [run.path(Artifact.REPORT_TXT), run.path(Artifact.REPORT_HTML)]
    try:
        render_pdf(en, run.path(Artifact.REPORT_PDF))
        written.append(run.path(Artifact.REPORT_PDF))
    except ImportError:
        log.warning("reportlab is not installed: skipping situation_report.pdf (pip install reportlab)")

    if "ne" in languages:
        ne = build_model(summary, "ne", nepali_reviewed=reviewed, image=image)
        _write_text(run.path(Artifact.REPORT_NE_TXT), render_text(ne))
        _write_text(run.path(Artifact.REPORT_NE_HTML), render_html(ne))
        written += [run.path(Artifact.REPORT_NE_TXT), run.path(Artifact.REPORT_NE_HTML)]
    return written


def run_report(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None:
    """Stage entry point (signature fixed by the contract): reads summary.json, writes the reports."""
    data = read_json(run.path(Artifact.SUMMARY))
    summary = Summary.from_dict(data)                  # drops the 'provenance' key and unknown keys
    write_reports(run, summary, cfg)
