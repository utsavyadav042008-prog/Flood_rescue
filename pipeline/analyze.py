"""``analyze_flood`` - the one function the dashboard and the CLI call.

Read this file top to bottom; it has five parts:

1. Result objects            ``StageOutcome``, ``PipelineResult``
2. The stage graph           derived from the contract (no hand-written order)
3. One stage, safely         ``_run_one``  (cache -> clean -> run -> validate)
4. The scheduler             ``analyze_flood``  (parallel, degrades, aborts)
5. CLI                       ``python -m pipeline.analyze ...``

Behaviour in one paragraph
--------------------------
Stages that do not depend on each other run in parallel (OSM download while
the satellite pair is fetched).  A failure in a *required* stage (acquire,
infer, summarize) stops the run.  A failure in a *degradable* stage (osm,
damage, connectivity) only skips the stages that need its output; the summary
then shows "not available" instead of a fake 0.  ``analyze_flood`` never
raises for expected problems: it returns ``PipelineResult(ok=False, error=...)``
with a sentence that is safe to show in the UI.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import platform
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import date
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Callable, Mapping, Optional

from pipeline import stages
from pipeline.config import REPO_ROOT, load_config
from pipeline.contracts import (
    CONTRACT_VERSION, CONTRACTS, ERROR, STAGE_ORDER, WARNING, Artifact,
    check_stage, raise_for_issues, stage_is_fresh,
)
from pipeline.errors import ContractError, PipelineError
from pipeline.fileio import read_json, read_provenance, write_json
from pipeline.rules import assert_osm_snapshot_pre_event, effective_osm_snapshot
from pipeline.run import RunPaths
from pipeline.schemas import AOI, Summary, parse_event_date, utc_now_iso

log = logging.getLogger("floodresq")

# ==========================================================================
# 1. Result objects
# ==========================================================================
OK, CACHED, FAILED, SKIPPED = "ok", "cached", "failed", "skipped"
GOOD = (OK, CACHED)
ProgressFn = Callable[[str, str], None]          # (stage_name, state)


@dataclass
class StageOutcome:
    """What happened to one stage in one run."""

    name: str
    status: str                       # ok | cached | failed | skipped
    seconds: float = 0.0
    message: str = ""                 # short, safe to show in the UI
    detail: str = ""                  # technical text / traceback (log only)
    warnings: list = field(default_factory=list)   # non-blocking contract warnings

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seconds"] = round(self.seconds, 2)
        return d


@dataclass
class PipelineResult:
    run_dir: Optional[Path]           # None when the inputs were rejected before a run existed
    summary: Optional[Summary]
    outcomes: list
    ok: bool
    error: str = ""                   # friendly sentence when ok is False
    failed_stage: str = ""
    mock_stages: tuple = ()

    @property
    def run(self) -> Optional[RunPaths]:
        return RunPaths(self.run_dir) if self.run_dir else None

    def outcome(self, name: str) -> Optional[StageOutcome]:
        return next((o for o in self.outcomes if o.name == name), None)

    @property
    def degraded(self) -> list[str]:
        """Names of stages that failed or were skipped although the run itself is ok."""
        return [o.name for o in self.outcomes if o.status in (FAILED, SKIPPED)]


# ==========================================================================
# 2. The stage graph (derived from the contract)
# ==========================================================================
RUNNABLE = tuple(n for n in STAGE_ORDER if n != "init")
REQUIRED = frozenset(n for n in RUNNABLE if CONTRACTS[n].required)
#: Stages that consume "whatever exists": they still run when upstream stages failed.
TOLERANT = frozenset({"summarize", "report", "figures"})
#: Cheap stages that are never served from the cache: the report depends on settings
#: that are not file inputs (report languages, translation-reviewed flag), so a cached
#: copy could silently be out of date.
ALWAYS_RERUN = frozenset({"report"})


def stage_graph(names: tuple = RUNNABLE) -> dict[str, set[str]]:
    """``{stage: {stages it must wait for}}``, computed from ``needs``/``produces``.

    Stage B waits for stage A when A produces a file B needs.  Nobody writes
    the order by hand, so it cannot go out of date: add an Artifact to a
    contract and the schedule (and the parallelism) follows.  ``osm`` ends up
    with no dependencies, which is exactly why it runs next to ``acquire``.
    """
    producer = {a: n for n in names for a in CONTRACTS[n].produces + CONTRACTS[n].optional}
    return {
        n: {producer[a] for a in CONTRACTS[n].needs if a in producer and producer[a] != n}
        for n in names
    }


# ==========================================================================
# 3. One stage, safely
# ==========================================================================
def _clean_outputs(run: RunPaths, name: str) -> None:
    """Delete the stage's old outputs before it runs.

    Without this, a stage that crashes half-way would leave LAST run's files in
    place, and the contract validator (which only looks at files) would happily
    accept them.  Stale-but-valid-looking output is the worst failure mode.
    """
    contract = CONTRACTS[name]
    for artifact in contract.produces + contract.optional:
        try:
            run.path(artifact).unlink(missing_ok=True)
        except OSError:
            pass          # e.g. Windows file locked by the dashboard; the stage will overwrite it


def _cache_matches_mode(run: RunPaths, name: str, is_mocked: bool) -> bool:
    """A cached output must come from the same kind of stage (mock vs real).

    Cache freshness is based on file times only, so without this check a mock
    result left in a folder would be reused by a later REAL run.
    """
    if name not in stages.MOCKABLE_STAGES:
        return True       # T4's own stages are pure functions of their inputs; never mocked
    for artifact in CONTRACTS[name].produces:
        path = run.path(artifact)
        if path.suffix in (".json", ".geojson", ".tif"):
            prov = read_provenance(path)
            if prov is not None and bool(prov.get("is_mock")) != is_mocked:
                return False
    return True


def _run_one(name: str, fn, run: RunPaths, aoi: AOI, event: date, cfg: dict, *,
             use_cache: bool, is_mocked: bool, strict: bool, forbid_mock: bool) -> StageOutcome:
    """Execute ONE stage in a worker thread.  Never raises: always returns an outcome."""
    t0 = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - t0

    try:
        # -- 1. cache ------------------------------------------------------------------
        if (use_cache and name not in ALWAYS_RERUN and stage_is_fresh(run, name)
                and _cache_matches_mode(run, name, is_mocked)):
            issues = check_stage(run, name, forbid_mock=forbid_mock)
            blocking = [i for i in issues if i.severity == ERROR or strict]
            if not blocking:           # a corrupted cache heals itself by re-running below
                return StageOutcome(name, CACHED, elapsed(),
                                    warnings=[str(i) for i in issues if i.severity == WARNING])
        # -- 2. run ----------------------------------------------------------------------
        _clean_outputs(run, name)
        fn(run, aoi, event, copy.deepcopy(cfg))      # a stage can never modify the shared config
        # -- 3. validate against the contract ---------------------------------------------
        issues = check_stage(run, name, forbid_mock=forbid_mock)
        raise_for_issues(issues, name, strict=strict)
        warnings = [str(i) for i in issues if i.severity == WARNING]
        slow = float(cfg.get("pipeline", {}).get("slow_stage_seconds", 0) or 0)
        if slow and elapsed() > slow:
            warnings.append(f"slow: took {elapsed():.0f}s (budget {slow:.0f}s)")
        return StageOutcome(name, OK, elapsed(), warnings=warnings)

    except NotImplementedError as exc:
        # Teammate's stub / missing module.  Not a crash: the stage is simply not there yet.
        text = str(exc) or f"stage '{name}' is not implemented yet"
        return StageOutcome(name, SKIPPED, elapsed(), text[:1].upper() + text[1:])
    except PipelineError as exc:              # includes ContractError from validation
        return StageOutcome(name, FAILED, elapsed(), exc.user_message, exc.detail)
    except ImportError as exc:
        return StageOutcome(name, FAILED, elapsed(),
                            f"Stage '{name}' could not be loaded (missing package?): {exc}",
                            traceback.format_exc())
    except Exception:                          # a bug: log it, show a generic sentence
        log.exception("Unexpected error in stage %s", name)
        return StageOutcome(name, FAILED, elapsed(),
                            f"Unexpected error in stage '{name}'. See run_log.json.",
                            traceback.format_exc())


# ==========================================================================
# run_log.json
# ==========================================================================
def _write_run_log(run: RunPaths, outcomes: Mapping[str, StageOutcome], *, started: str,
                   mock_stages: frozenset, use_cache: bool, finished: str = "",
                   ok: Optional[bool] = None, error: str = "") -> None:
    """Atomically (re)write run_log.json.  Called after EVERY stage, so a crash or a
    killed process still leaves a readable log, and ``summarize`` can read the
    statuses of upstream stages from it."""
    ordered = [outcomes[n].to_dict() for n in STAGE_ORDER if n in outcomes]
    write_json(run.path(Artifact.RUN_LOG), {
        "contract_version": CONTRACT_VERSION,
        "run_id": run.manifest().get("run_id", ""),
        "started_utc": started, "finished_utc": finished,
        "ok": ok, "error": error,
        "use_cache": use_cache, "mock_stages": sorted(mock_stages),
        "python": platform.python_version(), "platform": platform.platform(),
        "stages": ordered,
    })


def _emit(callback: Optional[ProgressFn], stage: str, state: str) -> None:
    """Call the UI callback; a bug in the UI must never kill the pipeline."""
    if callback is None:
        return
    try:
        callback(stage, state)
    except Exception:
        log.exception("progress callback failed")


# ==========================================================================
# 4. The scheduler
# ==========================================================================
def analyze_flood(
    aoi: AOI,
    event_date: object,
    *,
    cfg: Optional[dict] = None,
    out_root: Optional[str | Path] = None,
    use_cache: bool = True,
    mock: stages.MockSpec = False,
    stage_overrides: Optional[Mapping[str, stages.StageFn]] = None,
    on_progress: Optional[ProgressFn] = None,
    strict: bool = False,
    forbid_mock: bool = False,
) -> PipelineResult:
    """Run the whole analysis for ``aoi`` and ``event_date``.

    ``mock``            ``True`` = fake every stage T1-T3 own; or a set such as
                        ``{"osm", "damage"}`` to fake only those.
    ``use_cache``       skip stages whose valid outputs are newer than their inputs.
    ``on_progress``     ``fn(stage, state)`` with state in running/ok/cached/failed/
                        skipped.  ALWAYS called from the thread that called
                        ``analyze_flood`` (safe for Streamlit).
    ``strict``          contract warnings count as errors (final audit).
    ``forbid_mock``     files marked ``is_mock`` count as errors (final audit).
    ``stage_overrides`` ``{stage: function}``, mainly for tests.
    """
    started = utc_now_iso()
    cfg = cfg or load_config()
    mock_stages: frozenset = frozenset()
    run: Optional[RunPaths] = None
    outcomes: dict[str, StageOutcome] = {}

    try:
        mock_stages = stages.normalise_mock(mock)
        event = parse_event_date(event_date)
        # Fail fast on the disqualification rule, BEFORE downloading anything.
        assert_osm_snapshot_pre_event(effective_osm_snapshot(cfg["osm"]["snapshot_date"], event), event)
        base = Path(out_root) if out_root else Path(cfg["output"]["root"])
        if not base.is_absolute():
            base = REPO_ROOT / base
        if mock_stages:
            base = base / "_mock"        # mock and real results never share a folder
        run = RunPaths.create(aoi, event, cfg, out_root=base, is_mock=bool(mock_stages))
        outcomes["init"] = StageOutcome("init", OK)

        functions = stages.get_stage_functions(mock_stages, stage_overrides)
        graph = stage_graph()
        sorter = TopologicalSorter(graph)
        sorter.prepare()
        running: dict[Future, str] = {}
        aborted_by = ""
        max_workers = max(1, int(cfg.get("pipeline", {}).get("max_workers", 3)))

        def record(outcome: StageOutcome) -> None:
            outcomes[outcome.name] = outcome
            _emit(on_progress, outcome.name, outcome.status)
            _write_run_log(run, outcomes, started=started, mock_stages=mock_stages, use_cache=use_cache)

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="stage") as pool:
            while sorter.is_active():
                if not aborted_by:
                    for name in sorter.get_ready():
                        blockers = sorted(d for d in graph[name] if outcomes[d].status not in GOOD)
                        if blockers and name not in TOLERANT:
                            record(StageOutcome(name, SKIPPED, 0.0,
                                                f"Needs output from '{blockers[0]}', which did not finish."))
                            sorter.done(name)
                        else:
                            _emit(on_progress, name, "running")
                            future = pool.submit(
                                _run_one, name, functions[name], run, aoi, event, cfg,
                                use_cache=use_cache, is_mocked=name in mock_stages,
                                strict=strict, forbid_mock=forbid_mock)
                            running[future] = name
                if not running:
                    if aborted_by:
                        break            # drained; the remaining stages are never started
                    continue             # inline skips unlocked new stages
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    name = running.pop(future)
                    outcome = future.result()          # _run_one never raises
                    record(outcome)
                    sorter.done(name)
                    if name in REQUIRED and outcome.status not in GOOD and not aborted_by:
                        aborted_by = name

        for name in RUNNABLE:                          # stages that never started
            if name not in outcomes:
                record(StageOutcome(name, SKIPPED, 0.0, f"Run stopped because '{aborted_by}' failed."))

        failed_required = next((outcomes[n] for n in RUNNABLE
                                if n in REQUIRED and outcomes[n].status not in GOOD), None)
        ok = failed_required is None
        error = ""
        if failed_required is not None:
            error = failed_required.message or f"Stage '{failed_required.name}' did not finish."
        _write_run_log(run, outcomes, started=started, mock_stages=mock_stages, use_cache=use_cache,
                       finished=utc_now_iso(), ok=ok, error=error)

        summary = None
        # Only trust summary.json if THIS run's summarize stage produced it; after a
        # failure an old file from an earlier run may still be lying in the folder.
        if outcomes["summarize"].status in GOOD:
            try:
                summary = Summary.from_dict(read_json(run.path(Artifact.SUMMARY)))
            except (OSError, ValueError):
                summary = None
        return PipelineResult(run.root, summary, [outcomes[n] for n in STAGE_ORDER if n in outcomes],
                              ok, error, failed_required.name if failed_required else "",
                              tuple(sorted(mock_stages)))

    except PipelineError as exc:          # bad input, data-rule violation, config problem ...
        return PipelineResult(run.root if run else None, None,
                              [outcomes[n] for n in STAGE_ORDER if n in outcomes],
                              False, exc.user_message, "", tuple(sorted(mock_stages)))
    except Exception:                      # a bug in the scheduler itself
        log.exception("Unexpected error in analyze_flood")
        return PipelineResult(run.root if run else None, None,
                              [outcomes[n] for n in STAGE_ORDER if n in outcomes],
                              False, "Unexpected internal error. See the log.", "",
                              tuple(sorted(mock_stages)))


def load_result(run_dir: str | Path) -> PipelineResult:
    """Rebuild a PipelineResult from a finished run folder (instant "demo mode").

    Uses ``run_log.json`` when present; otherwise infers ``cached`` for every
    stage whose output files exist.
    """
    run = RunPaths.open(run_dir)
    log_data = None
    try:
        log_data = read_json(run.path(Artifact.RUN_LOG))
    except (OSError, ValueError):
        pass
    outcomes: list[StageOutcome] = []
    if isinstance(log_data, dict):
        for item in log_data.get("stages", []):
            try:
                outcomes.append(StageOutcome(
                    item["name"], item["status"], float(item.get("seconds", 0)),
                    item.get("message", ""), item.get("detail", ""), list(item.get("warnings", []))))
            except (KeyError, TypeError, ValueError):
                continue
        mock_stages = tuple(log_data.get("mock_stages", []))
    else:
        mock_stages = ()
        for name in STAGE_ORDER:
            exists = all(run.path(a).exists() for a in CONTRACTS[name].produces)
            outcomes.append(StageOutcome(name, CACHED if exists else SKIPPED))
    summary = None
    if run.path(Artifact.SUMMARY).exists():
        summary = Summary.from_dict(read_json(run.path(Artifact.SUMMARY)))
    failed = next((o for o in outcomes if o.name in REQUIRED and o.status not in GOOD), None)
    ok = summary is not None and failed is None
    error = "" if ok else ((failed.message or f"Stage '{failed.name}' has no results in this run.")
                           if failed else "This run has no summary.json yet.")
    return PipelineResult(run.root, summary, outcomes, ok, error, failed.name if failed else "", mock_stages)


# ==========================================================================
# 5. CLI
# ==========================================================================
def _format_outcomes(result: PipelineResult) -> str:
    # ASCII only: the Windows console (cp1252) raises UnicodeEncodeError on fancy symbols.
    lines = []
    for o in result.outcomes:
        lines.append(f"  [{o.status:^7}] {o.name:<13} {o.seconds:6.1f}s  {o.message}")
        for w in o.warnings:
            lines.append(f"            ! {w}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None,
         stage_overrides: Optional[Mapping[str, stages.StageFn]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.analyze",
                                 description="Run the FloodResQ analysis for an AOI and a flood date.")
    ap.add_argument("--aoi", required=True, help="path to an AOI json (see config/example_aoi.json)")
    ap.add_argument("--date", required=True, help="flood date, YYYY-MM-DD")
    ap.add_argument("--mock", action="store_true", help="fake every stage owned by T1-T3")
    ap.add_argument("--mock-stages", nargs="+", choices=stages.MOCKABLE_STAGES, metavar="STAGE",
                    help="fake only these stages (replace them one by one as teammates deliver)")
    ap.add_argument("--no-cache", action="store_true", help="re-run every stage")
    ap.add_argument("--strict", action="store_true", help="treat contract warnings as errors")
    ap.add_argument("--forbid-mock", action="store_true", help="fail if any output is marked mock")
    ap.add_argument("--out", help="output root (default: config output.root)")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)

    try:
        aoi = AOI.from_json_file(args.aoi)
    except (OSError, ValueError, PipelineError) as exc:
        print(f"ERROR: cannot read AOI file: {getattr(exc, 'user_message', exc)}")
        return 1

    mock: stages.MockSpec = args.mock_stages if args.mock_stages else args.mock
    result = analyze_flood(
        aoi, args.date, out_root=args.out, use_cache=not args.no_cache, mock=mock,
        stage_overrides=stage_overrides, strict=args.strict, forbid_mock=args.forbid_mock,
        on_progress=None if args.json else (lambda s, st: print(f"  .. {s:<13} {st}", flush=True)))

    if args.json:
        print(json.dumps({"ok": result.ok, "error": result.error, "run_dir": str(result.run_dir),
                          "outcomes": [o.to_dict() for o in result.outcomes],
                          "summary": result.summary.to_dict() if result.summary else None},
                         indent=2, ensure_ascii=False))
    else:
        print("\nStages:\n" + _format_outcomes(result))
        if result.summary:
            s = result.summary
            print(f"\nFlood area km2: {s.flood_area_km2}   roads: {s.affected_roads}   "
                  f"bridges: {s.affected_bridges}   buildings: {s.affected_buildings}   "
                  f"cut off: {s.cutoff_settlements}")
            for w in s.warnings:
                print(f"  ! {w}")
        print(f"\n{'OK' if result.ok else 'FAILED: ' + result.error}"
              + (f"  ->  {result.run_dir}" if result.run_dir else ""))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
