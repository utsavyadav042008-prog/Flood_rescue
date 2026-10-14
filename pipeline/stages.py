"""Stage registry: turns a stage NAME into a callable.

Every stage is a function with the same signature (see ``contracts.py``)::

    fn(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None

Where do the functions come from?  In priority order:

1. ``overrides``      explicit functions passed by tests / notebooks
2. mock stages        ``pipeline/mocks.py`` (only for the stages you list)
3. real entry points  ``"module:function"`` strings from ``CONTRACTS``,
                      imported lazily, only when the stage actually runs

Lazy import means T2's heavy ``import torch`` cannot break a run that mocks
or never reaches the ``infer`` stage, and a teammate's syntax error only
fails *their* stage instead of the whole program.
"""

from __future__ import annotations

import importlib
from datetime import date
from typing import Callable, Iterable, Mapping, Optional, Union

from pipeline.contracts import CONTRACTS
from pipeline.errors import InputValidationError
from pipeline.run import RunPaths
from pipeline.schemas import AOI

StageFn = Callable[[RunPaths, AOI, date, dict], None]
MockSpec = Union[bool, Iterable[str], None]

#: Stages that have a mock implementation (the ones other teammates own).
MOCKABLE_STAGES = ("acquire", "osm", "infer", "damage", "connectivity")


class LazyEntrypoint:
    """Callable wrapper around ``"package.module:function"``; imports on first call."""

    def __init__(self, stage: str, entrypoint: str, owner: str) -> None:
        self.stage, self.entrypoint, self.owner = stage, entrypoint, owner

    def __repr__(self) -> str:
        return f"LazyEntrypoint({self.stage!r} -> {self.entrypoint!r})"

    def __call__(self, run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None:
        module_name, func_name = self.entrypoint.split(":")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # exc.name is the module that could not be found.  If it is the stage
            # module itself (or one of its parent packages) the teammate simply has
            # not written it yet -> "not implemented".  If it is something else
            # (e.g. 'torch') a DEPENDENCY is missing: that is a real error and must
            # surface with its original message, so re-raise.
            missing = exc.name or ""
            if module_name == missing or module_name.startswith(missing + "."):
                raise NotImplementedError(
                    f"stage '{self.stage}' is not implemented yet: module '{module_name}' "
                    f"not found (owner: {self.owner})") from exc
            raise
        function = getattr(module, func_name, None)
        if function is None:
            raise NotImplementedError(
                f"stage '{self.stage}' is not implemented yet: '{module_name}' has no "
                f"function '{func_name}' (owner: {self.owner})")
        return function(run, aoi, event_date, cfg)


def normalise_mock(mock: MockSpec) -> frozenset[str]:
    """``True`` -> all mockable stages, ``False``/``None`` -> none, iterable -> validated set."""
    if mock is True:
        return frozenset(MOCKABLE_STAGES)
    if not mock:
        return frozenset()
    names = frozenset(mock)
    unknown = names - set(MOCKABLE_STAGES)
    if unknown:
        raise InputValidationError(
            f"Cannot mock {sorted(unknown)}; mockable stages are {list(MOCKABLE_STAGES)}.")
    return names


def get_stage_functions(
    mock: MockSpec = False,
    overrides: Optional[Mapping[str, StageFn]] = None,
) -> dict[str, StageFn]:
    """Return ``{stage_name: callable}`` for every stage except ``init``."""
    from pipeline import mocks      # local import keeps this module light

    mocked = normalise_mock(mock)
    overrides = dict(overrides or {})
    functions: dict[str, StageFn] = {}
    for name, contract in CONTRACTS.items():
        if name == "init":
            continue                       # created by RunPaths.create, not a callable stage
        if name in overrides:
            functions[name] = overrides[name]
        elif name in mocked:
            functions[name] = mocks.mock_stage(name)
        else:
            functions[name] = LazyEntrypoint(name, contract.entrypoint, contract.owner)
    return functions
