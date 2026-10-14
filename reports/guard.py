"""Number guard: prove that no number in a text was invented.

The brief's second AI option demands that "every number must come from your
system's maps, never invented by the language model".  Our report is built
from templates, so it cannot invent numbers, but this guard turns that promise
into a CHECK that tests (and any future language-model rewrite) can run:

    unexpected_numbers(text, summary)   ->  []      # all numbers are explained

A number is "explained" when it appears in the summary (any value, date or
name), in the fixed wording (e.g. "10 m" pixels, attribution years), or is the
"... and N more" count derived from the summary.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from reports.format import (ATTRIBUTION_LINES, MAX_LISTED_SETTLEMENTS, fmt, to_ascii_digits)
from reports.strings import STRINGS

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def extract_numbers(text: str) -> set[str]:
    """All numbers in ``text`` as normalised strings (ASCII digits, no thousands commas)."""
    return {m.group(0).rstrip(",").replace(",", "") for m in _NUMBER.finditer(to_ascii_digits(text))}


def _summary_dict(summary: Any) -> dict:
    return summary.to_dict() if hasattr(summary, "to_dict") else dict(summary)


def allowed_numbers(summary: Any) -> set[str]:
    """Numbers that may legitimately appear in a report about ``summary``."""
    s = _summary_dict(summary)
    texts: list[str] = [str(v) for v in fmt(s).values()]
    texts += [str(n) for n in (s.get("cutoff_names") or [])]
    texts += [str(w) for w in (s.get("warnings") or [])]
    allowed = set()
    for text in texts:
        allowed |= extract_numbers(text)
    names = s.get("cutoff_names") or []
    total = s.get("cutoff_settlements") or 0
    hidden = max(len(names), total if isinstance(total, int) else 0) - MAX_LISTED_SETTLEMENTS
    if hidden > 0:
        allowed.add(str(hidden))                      # the "... and N more" count
    return allowed


def static_numbers() -> set[str]:
    """Numbers that are part of the fixed wording of every language."""
    numbers: set[str] = set()
    for table in STRINGS.values():
        for text in table.values():
            numbers |= extract_numbers(text)
    for line in ATTRIBUTION_LINES:
        numbers |= extract_numbers(line)
    return numbers


def unexpected_numbers(text: str, summary: Any, extra_allowed: Iterable[str] = ()) -> list[str]:
    """Numbers in ``text`` that neither the summary nor the fixed wording explains."""
    allowed = allowed_numbers(summary) | static_numbers() | set(extra_allowed)
    return sorted(extract_numbers(text) - allowed)
