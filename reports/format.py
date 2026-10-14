"""Number and text formatting shared by the dashboard AND the situation report.

Why a shared module?  The dashboard cards and the report must print the same
numbers with the same rounding.  Both call :func:`fmt`, so they cannot
disagree, and neither recomputes anything: every value comes from
``summary.json`` (see ``pipeline/summary.py``).

``None`` means "stage did not run" and is printed as "not available",
never as ``0``.
"""

from __future__ import annotations

from typing import Any, Mapping

NA = "not available"

#: Required attribution, word for word from the challenge brief.
ATTRIBUTION_LINES = (
    "Contains modified Copernicus Sentinel data 2026.",
    "Produced using Copernicus WorldDEM-30 \u00a9 DLR e.V. 2010\u20132014 and \u00a9 Airbus Defence "
    "and Space GmbH 2014\u20132018 provided under COPERNICUS by the European Union and ESA; "
    "all rights reserved.",
    "\u00a9 OpenStreetMap contributors.",
)

MAX_LISTED_SETTLEMENTS = 10


def num(value: Any, decimals: int = 1) -> str:
    """``1234.5678`` -> ``'1,234.6'``; ``None`` -> ``'not available'``."""
    return NA if value is None else f"{float(value):,.{decimals}f}"


def count(value: Any) -> str:
    """``1234`` -> ``'1,234'``; ``None`` -> ``'not available'``."""
    return NA if value is None else f"{int(value):,}"


def cutoff_list(names: list, cutoff_total: Any) -> str:
    """Bulleted list of cut-off settlements, at most 10, then '... and N more'."""
    names = list(names or [])
    shown = names[:MAX_LISTED_SETTLEMENTS]
    text = "\n".join(f"- {n}" for n in shown)
    if len(names) > MAX_LISTED_SETTLEMENTS:
        text += f"\n- ... and {len(names) - MAX_LISTED_SETTLEMENTS} more"
    if text:
        return text
    return "None found" if cutoff_total == 0 else NA


def fmt(summary: Any) -> dict[str, str]:
    """All display strings for a summary (a ``Summary`` or its dict)."""
    s: Mapping[str, Any] = summary.to_dict() if hasattr(summary, "to_dict") else summary
    return {
        "aoi_name": s.get("aoi_name", ""),
        "event_date": s.get("event_date", ""),
        "before_date": s.get("before_date", ""),
        "after_date": s.get("after_date", ""),
        "sentinel1_orbit": s.get("sentinel1_orbit", ""),
        "processing_date": s.get("processing_date", ""),
        "model_version": s.get("model_version", ""),
        "osm_snapshot_date": s.get("osm_snapshot_date", ""),
        "flood_area_km2": num(s.get("flood_area_km2")),
        "affected_roads": count(s.get("affected_roads")),
        "affected_road_length_km": num(s.get("affected_road_length_km")),
        "affected_bridges": count(s.get("affected_bridges")),
        "affected_buildings": count(s.get("affected_buildings")),
        "cutoff_settlements": count(s.get("cutoff_settlements")),
        "mean_confidence": num(s.get("mean_confidence"), 2),
        "warnings": " ".join(s.get("warnings", []) or []),
        "cutoff_list": cutoff_list(s.get("cutoff_names"), s.get("cutoff_settlements")),
    }


# --------------------------------------------------------------------------
# Nepali digits
# --------------------------------------------------------------------------
_TO_DEVANAGARI = str.maketrans("0123456789", "\u0966\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f")
_FROM_DEVANAGARI = str.maketrans("\u0966\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f", "0123456789")


def to_devanagari_digits(text: str) -> str:
    """``'2026-08-26'`` -> ``'\u0968\u0966\u0968\u096c-\u0966\u096e-\u0968\u096c'`` (digits only; separators unchanged)."""
    return text.translate(_TO_DEVANAGARI)


def to_ascii_digits(text: str) -> str:
    """Inverse of :func:`to_devanagari_digits`; used by tests and the number guard."""
    return text.translate(_FROM_DEVANAGARI)
