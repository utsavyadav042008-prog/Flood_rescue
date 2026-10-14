"""Hackathon data rules turned into code.

The brief says: Copernicus EMS / UNOSAT damage maps and OSM edits made after
the event may only be used to CHECK results; using them as input leads to
disqualification.  Two cheap guards live here:

1. ``assert_osm_snapshot_pre_event``  -> runtime guard used by the validators
2. ``scan_for_forbidden_inputs``      -> static guard used by the test suite

Neither guard is airtight (a determined person can always hide a path), but
they catch the realistic accident: someone copy-pastes a reference file path
into pipeline code at 2 a.m.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Iterable

from pipeline.errors import DataRuleViolation
from pipeline.schemas import parse_date

#: Text that must never appear in pipeline code (check-only data sources).
FORBIDDEN_TEXT = re.compile(
    r"data[/\\]reference|EMSR\s?\d{3}|UNOSAT|copernicus[_ -]?ems", re.IGNORECASE
)

#: Folders that must never mention check-only data.
GUARDED_FOLDERS = (
    "pipeline", "acquisition", "preprocessing", "models", "gis", "dashboard", "reports",
)


def effective_osm_snapshot(configured: object, event_date: date) -> date:
    """The OSM snapshot actually used for an event: the configured (latest allowed)
    snapshot, but never later than the day BEFORE the event.

    This is what lets the system run "for any date the judges choose": a 2021
    flood automatically uses the map of 2021-06-14, the Aug 2026 case study the
    latest snapshot allowed by the brief (2026-07-27).
    """
    from datetime import timedelta
    return min(parse_date(configured, "OSM snapshot date"), event_date - timedelta(days=1))


def assert_osm_snapshot_pre_event(snapshot: object, event_date: date) -> date:
    """Raise unless the OSM snapshot date is strictly before the event date."""
    snap = parse_date(snapshot, "OSM snapshot date")
    if snap >= event_date:
        raise DataRuleViolation(
            "OpenStreetMap snapshot must be older than the flood event.",
            f"OSM snapshot {snap} is not before event date {event_date}; "
            "post-event OSM edits may only be used to check results.",
        )
    return snap


def scan_for_forbidden_inputs(
    repo_root: Path,
    folders: Iterable[str] = GUARDED_FOLDERS,
    allow: Iterable[str] = (),
) -> list[tuple[str, int, str]]:
    """Return ``[(relative_path, line_number, matched_text), ...]``.

    ``allow`` lists relative paths (forward slashes) that may legitimately
    mention check-only data, e.g. ``dashboard/case_study.py``.
    This file is always skipped because it contains the patterns themselves.
    """
    allowed = {a.replace("\\", "/") for a in allow}
    hits: list[tuple[str, int, str]] = []
    for folder in folders:
        base = repo_root / folder
        if not base.exists():
            continue
        for py in sorted(base.rglob("*.py")):
            rel = py.relative_to(repo_root).as_posix()
            if rel in allowed or rel == "pipeline/rules.py":
                continue
            text = py.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                match = FORBIDDEN_TEXT.search(line)
                if match:
                    hits.append((rel, lineno, match.group(0)))
    return hits
