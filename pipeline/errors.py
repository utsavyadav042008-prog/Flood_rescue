"""Exception hierarchy for the whole pipeline.

Why a hierarchy instead of plain ``Exception``?
-----------------------------------------------
The dashboard must show a *friendly* sentence to the user while the full
technical detail goes to ``run_log.json``.  Every ``PipelineError`` therefore
carries two strings:

``user_message``  short, safe to show in the UI ("No same-orbit pair found...")
``detail``        technical text for logs (defaults to ``user_message``)

Anything that is NOT a ``PipelineError`` is a bug and is reported generically
by the stage runner (Part 2).
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class. ``str(error)`` returns the detailed text."""

    def __init__(self, user_message: str, detail: str = "") -> None:
        self.user_message = user_message
        self.detail = detail or user_message
        super().__init__(self.detail)


class InputValidationError(PipelineError):
    """The user's AOI or event date is invalid."""


class ConfigError(PipelineError):
    """config/default.yaml is missing, unreadable or inconsistent."""


class AcquisitionError(PipelineError):
    """No usable satellite data (no scenes, no same-orbit pair, download failed)."""


class ModelError(PipelineError):
    """Flood-segmentation inference failed."""


class GISError(PipelineError):
    """OSM extraction, damage overlay or connectivity analysis failed."""


class ContractError(PipelineError):
    """A stage wrote files that break the data contract (see contracts.py)."""


class DataRuleViolation(PipelineError):
    """A hackathon data rule was broken (e.g. OSM snapshot newer than the event).

    These are never "retry" situations: the run must stop, because breaking the
    rule can disqualify the team.
    """
