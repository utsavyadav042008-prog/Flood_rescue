"""Case-study tab: our Trishuli result compared with the official reference maps.

This is the ONE dashboard file allowed to name the reference activation (the
data-rule test whitelists it).  It never reads the reference maps themselves:
it only displays ``evaluation/case_study_results.json``, a small results file
produced offline by T2/T3 (check-only comparison).  The official maps were
NOT used as an input to any pipeline stage.

Expected JSON (every key optional; missing values are shown as "not available")::

    {
      "reference": "EMSR927",
      "flood_extent": {"iou": 0.62, "precision": 0.71, "recall": 0.80, "f1": 0.75},
      "infrastructure": {
        "roads":     {"reference_flagged": 40, "also_flagged": 31, "extra_flagged": 9},
        "bridges":   {"reference_flagged": 5,  "also_flagged": 4,  "extra_flagged": 1},
        "buildings": {"reference_flagged": 120, "also_flagged": 70, "extra_flagged": 15}
      },
      "notes": ["Disagreement near steep slopes: radar shadow.", "..."],
      "generated_utc": "2026-10-20T10:00:00+00:00"
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pipeline.config import REPO_ROOT
from reports.format import NA

RESULTS_PATH = REPO_ROOT / "evaluation" / "case_study_results.json"
CREDIT = "European Union, Copernicus Emergency Management Service data"
CHECK_ONLY_NOTE = ("Check-only comparison. The official reference maps were NOT used as input to any "
                   "step of our pipeline.")

EXTENT_LABELS = (("iou", "IoU (overlap of flood extent)"), ("precision", "Precision"),
                 ("recall", "Recall"), ("f1", "F1 score"))
INFRA_LABELS = (("roads", "Roads"), ("bridges", "Bridges"), ("buildings", "Buildings"))


@dataclass
class CaseStudy:
    reference: str = ""
    extent: dict = field(default_factory=dict)
    infrastructure: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    generated: str = ""


def _number(value: Any) -> Optional[float]:
    return None if isinstance(value, bool) or not isinstance(value, (int, float)) else float(value)


def parse_case_study(data: Any) -> CaseStudy:
    """Tolerant parser: wrong types become ``None``, never exceptions."""
    if not isinstance(data, dict):
        return CaseStudy()
    extent = data.get("flood_extent") if isinstance(data.get("flood_extent"), dict) else {}
    infra_in = data.get("infrastructure") if isinstance(data.get("infrastructure"), dict) else {}
    infra = {}
    for key, _ in INFRA_LABELS:
        item = infra_in.get(key) if isinstance(infra_in.get(key), dict) else {}
        ref, also, extra = (_number(item.get(k)) for k in ("reference_flagged", "also_flagged", "extra_flagged"))
        agreement = also / ref if ref and also is not None else None     # share of reference objects we found
        infra[key] = {"reference": ref, "also": also, "extra": extra, "agreement": agreement}
    notes = data.get("notes") if isinstance(data.get("notes"), list) else []
    return CaseStudy(str(data.get("reference", "")), {k: _number(extent.get(k)) for k, _ in EXTENT_LABELS},
                     infra, [str(n) for n in notes], str(data.get("generated_utc", "")))


def load_case_study(path: Path = RESULTS_PATH) -> Optional[CaseStudy]:
    """Parsed results, or ``None`` if the file does not exist / is unreadable."""
    try:
        return parse_case_study(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def _pct(value: Optional[float]) -> str:
    return NA if value is None else f"{100 * value:.0f}%"


def _int(value: Optional[float]) -> str:
    return NA if value is None else f"{int(value):,}"


def extent_rows(cs: CaseStudy) -> list:
    return [{"Metric": label, "Value": f"{cs.extent.get(key):.2f}" if cs.extent.get(key) is not None else NA}
            for key, label in EXTENT_LABELS]


def infrastructure_rows(cs: CaseStudy) -> list:
    rows = []
    for key, label in INFRA_LABELS:
        v = cs.infrastructure.get(key, {})
        rows.append({"Object": label, "Flagged in reference": _int(v.get("reference")),
                     "Also flagged by us": _int(v.get("also")), "Only flagged by us": _int(v.get("extra")),
                     "Reference objects found": _pct(v.get("agreement"))})
    return rows


def render() -> None:
    """Streamlit rendering (imported lazily so this module is testable without Streamlit)."""
    import streamlit as st

    st.subheader("Case study: Trishuli flood, August 2026")
    st.caption(CHECK_ONLY_NOTE)
    cs = load_case_study()
    if cs is None:
        st.info("Comparison results are not available yet (evaluation/case_study_results.json is missing).")
    else:
        st.markdown("**Flood extent agreement**")
        st.dataframe(extent_rows(cs), hide_index=True)
        st.markdown("**Infrastructure flagged**")
        st.dataframe(infrastructure_rows(cs), hide_index=True)
        if cs.notes:
            st.markdown("**Where and why we disagree**")
            for note in cs.notes:
                st.markdown(f"- {note}")
        if cs.generated:
            st.caption(f"Results generated {cs.generated}")
    st.caption(f"Reference credit: {CREDIT}.")
