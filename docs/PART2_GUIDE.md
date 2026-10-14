# Part 2 Guide — `analyze_flood()` Pipeline, Stage Runner, Caching, Mock Mode, Validation

Builds on Part 1 (`contracts.py`, `RunPaths`, `fileio`). Read `PART1_GUIDE.md` first.

---

## 1. What Part 2 gives the team

| Need | Where it lives |
|---|---|
| One call that runs everything | `pipeline/analyze.py` → `analyze_flood(aoi, date, ...)` |
| Turn a stage name into a function | `pipeline/stages.py` |
| Fake T1/T2/T3 outputs so we can build today | `pipeline/mocks.py` (+ `--mock`, `--mock-stages`) |
| Merge all numbers into `summary.json` | `pipeline/summary.py` |
| Re-open a finished run instantly (demo mode) | `analyze.load_result(run_dir)` |
| Command line | `python -m pipeline.analyze --aoi ... --date ...` |

## 2. Reading order (about 30 minutes)

1. `pipeline/analyze.py` sections 1–2: result objects and `stage_graph`.
2. `analyze.py` section 3: `_run_one`, the life of ONE stage.
3. `analyze.py` section 4: the scheduler loop in `analyze_flood`.
4. `pipeline/stages.py` (the registry) → `pipeline/mocks.py` → `pipeline/summary.py`.
5. `tests/test_analyze.py` (the test names read like a specification).

## 3. The picture

```
           analyze_flood(aoi, date)
                    |
   validate inputs + OSM-snapshot-before-event rule   (fail fast, nothing created)
                    |
          RunPaths.create  -> outputs/<name>_<date>_<id>/run.json
                    |
        stage graph, derived from the contract
                    |
   +----------------+----------------+
   |                                 |
 acquire (T1)  ---> infer (T2)      osm (T3)         <- run in PARALLEL
   |                  |              |
   +-------+----------+--------------+
           v
      damage (T3) --> connectivity (T3)
           |                |
           +-------+--------+
                   v
   summarize (T4) -> report (T4) -> figures (T4)      <- run even if damage etc. failed

 every stage = [cache?] -> clean old outputs -> run -> validate against the contract
```

## 4. How to use it

**Python (dashboard / notebook)**
```python
from pipeline.analyze import analyze_flood, load_result
from pipeline.schemas import AOI

result = analyze_flood(AOI.from_json_file("config/example_aoi.json"), "2026-08-26",
                       mock={"osm", "damage", "connectivity"},      # or True, or False
                       on_progress=lambda stage, state: print(stage, state))
if not result.ok:
    print(result.error)                 # friendly sentence, safe for the UI
else:
    print(result.summary.cutoff_names, result.degraded)

demo = load_result("data/sample/trishuli_run")      # instant, no recomputation
```

**Command line (Git Bash, repo root)**
```bash
python -m pipeline.analyze --aoi config/example_aoi.json --date 2026-08-26 --mock
python -m pipeline.analyze --aoi config/example_aoi.json --date 2026-08-26 --mock-stages osm damage connectivity
python -m pipeline.analyze --aoi config/example_aoi.json --date 2026-08-26 --no-cache
python -m pipeline.analyze --aoi config/example_aoi.json --date 2026-08-26 --strict --forbid-mock   # Day-13 audit
```
Exit code 0 = ok, 1 = failed.

**Replace mocks one stage at a time**

| Day | Command |
|---|---|
| 3 | `--mock` |
| 4–5 | `--mock-stages osm damage connectivity` (acquire + infer are real) |
| 6–7 | `--mock-stages connectivity` |
| 8 | no flag: everything real |

`--mock` needs `rasterio` (the mock includes GeoTIFFs); without it you get a clear message instead of a crash.

## 5. What happens when something goes wrong

| Situation | Result | Stage statuses |
|---|---|---|
| Bad bbox, too large, bad/future date | `ok=False`, friendly `error`, no folder created | none ran |
| OSM snapshot date not before the event | `ok=False` ("snapshot must be older than the flood event") | none ran |
| No same-orbit pair (`AcquisitionError`) | `ok=False`, T1's message | acquire failed; infer…figures skipped |
| Model fails | `ok=False` | infer failed; later skipped |
| OSM fails (timeout, empty) | `ok=True`, KPIs "not available" + warning | osm failed; damage, connectivity skipped; summarize runs |
| Damage fails | `ok=True` | damage failed; connectivity skipped |
| No hospital/town found | `ok=True`, cut-off = "not available" | connectivity failed |
| Teammate's stage not written yet | stage `skipped` with owner name; if it is required the run stops with that message | skipped |
| Stage writes an invalid file | stage `failed`: "breaks the data contract (N problems): first one" | failed (full list in `detail`) |
| Unexpected bug in a stage | generic sentence in UI, traceback in `run_log.json` | failed |
| `report` / `figures` not written yet (Parts 3–4) | skipped, run stays `ok` | skipped |

Required stages: `acquire`, `infer`, `summarize`. Everything else degrades.

## 6. Function reference

### `pipeline/analyze.py`
| Name | What it does |
|---|---|
| `analyze_flood(aoi, event_date, *, mock, use_cache, on_progress, strict, forbid_mock, cfg, out_root, stage_overrides)` | the whole pipeline; returns `PipelineResult`, never raises for expected problems |
| `PipelineResult` | `ok`, `error`, `failed_stage`, `summary`, `outcomes`, `run_dir`, `.run`, `.outcome(name)`, `.degraded` |
| `StageOutcome` | `name`, `status` (ok / cached / failed / skipped), `seconds`, `message` (UI), `detail` (log), `warnings` |
| `stage_graph()` | `{stage: {stages it waits for}}`, derived from the contract |
| `_run_one(...)` | one stage in a worker thread: cache → clean → run → validate; never raises |
| `load_result(run_dir)` | rebuild a result from `run_log.json` + `summary.json` |
| `main(argv)` | the CLI |

### `pipeline/stages.py`
| Name | What it does |
|---|---|
| `get_stage_functions(mock, overrides)` | `{stage: callable}`; priority overrides > mock > real |
| `LazyEntrypoint` | imports `"module:function"` only when the stage runs |
| `normalise_mock(spec)` | `True` / set / `False` → frozenset of stage names |

### `pipeline/summary.py`
| Name | What it does |
|---|---|
| `build_summary(run)` | read-only aggregation → `Summary` (None where data is missing) |
| `run_summarize(...)` | the stage entry point; writes `summary.json` |

### `pipeline/mocks.py`
| Name | What it does |
|---|---|
| `mock_stage(name)` | copy that stage's contracted files from the template mock run |
| `ensure_template()` | builds the template once, thread-safe |

## 7. Clever tricks (not obvious at first glance)

1. **The schedule is computed, not written.** `stage_graph` says "B waits for A if A produces
   a file B needs". `osm` needs nothing, so it automatically runs beside `acquire`. Add an
   Artifact to a contract and the order follows. Uses the standard library's
   `graphlib.TopologicalSorter`.
2. **Parallel stages, but the UI callback stays single-threaded.** Workers only execute
   stages; the scheduler loop (in the caller's thread) submits work, collects results and
   calls `on_progress`. So Streamlit never sees a callback from a foreign thread, and no
   queue is needed. A test asserts this.
3. **Three kinds of stage.** *Required* (acquire, infer, summarize) abort the run;
   *degradable* (osm, damage, connectivity) only skip what depends on them; *tolerant*
   (summarize, report, figures) still run on whatever exists. That is why a dead OSM
   server yields a partial result instead of a blank screen.
4. **`None` flows end to end.** A skipped stage leaves its summary fields `None`, which
   the report prints as "not available", never as a reassuring `0`.
5. **Old outputs are deleted before a stage runs.** The validator only looks at files, so
   a stage that crashes half-way would otherwise leave *last run's* valid-looking files
   behind. Test: `test_stale_outputs_are_removed_before_a_stage_reruns`.
6. **Three-layer cache.** (a) make-style: outputs must not be older than inputs; (b) the
   cached files are re-validated, so a corrupted cache silently heals by re-running;
   (c) mock output is never reused by a real stage (provenance `is_mock` is compared), and
   mock runs live in a separate `outputs/_mock/` folder.
7. **"Not written yet" vs "dependency missing".** `ModuleNotFoundError.name` tells whether
   the stage's own module is absent (→ skipped, "owner: T3") or something it imports, such as
   `torch`, is absent (→ a real failure with the original message).
8. **`run_log.json` is written after every stage**, so even a killed process leaves a log.
   It is also an *input* of `summarize`, which turns failed stages into summary warnings.
   Information travels through files, not through shared memory.
9. **Two-message errors.** `PipelineError.user_message` goes to the UI; `detail` (or the full
   traceback for unexpected errors) goes only to `run_log.json`. Users never see a stack trace.
10. **Fail fast on the disqualification rule.** The OSM-snapshot-before-event check runs
    before any download, not after five minutes of work.
11. **Every stage gets its own deep copy of the config**, so a buggy stage cannot change
    settings for the stages after it.
12. **Mocks keep the real validators honest.** Dates inside `metadata.json` and
    `osm/snapshot.json` are regenerated from the user's event date (not copied), and
    `summarize` is the *real* code running on mock inputs. The mock flag reaches the summary
    through provenance, and `--forbid-mock` refuses to pass a run that contains mock data.
13. **Thread-safe template.** `acquire` and `osm` both ask for the mock template at the same
    moment; a lock makes sure it is built once.
14. **ASCII-only console output.** Windows consoles (cp1252) raise `UnicodeEncodeError` on
    symbols like check marks, which would crash a successful run at the last line.
15. **A broken UI callback cannot kill a run** (exceptions inside `on_progress` are logged).
16. **Parallelism is proven by a test.** Two stages wait on a `threading.Barrier(2)`; a
    sequential scheduler would time out. The test name is `test_osm_and_acquire_really_run_at_the_same_time`.

## 8. What each teammate must do to plug in

Implement the stage function (the stub `bootstrap_repo.py` already wrote lists its inputs and outputs):
```python
def run_damage(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None:
    mask = run.path(Artifact.FLOOD_MASK)                 # read
    ...
    write_geojson(run.path(Artifact.AFFECTED_ROADS), features, provenance)   # write
```
Raise `AcquisitionError` / `ModelError` / `GISError` with a short `user_message` for expected
problems. Check your own output with `python scripts/validate_run.py <run_dir> --stage damage`
before telling T4 it is done.

## 9. Tested and untested

- **Run here (107 tests pass, 6 skipped):** graph, parallelism, caching (fresh / stale / corrupted / mock-vs-real), every failure row in section 5, the OSM rule, bad inputs, summary edge cases, `load_result`, CLI exit codes, the real-stub default state.
- **Not run here:** everything needing `rasterio`: the raster statistics (`_raster_stats`, which tests replace with fixed numbers), mock rasters and the 6 skipped tests, including `FullMockWithRastersTests` (full `--mock` run). Install `requirements.txt`, run `python -m unittest discover -s tests -t .`, and expect 0 skipped. If one fails, send me the traceback.
- Stages cannot be force-stopped (a Python thread cannot be killed). `slow_stage_seconds` only adds a warning; the real protection is the AOI size limit.

## 10. Notes on changes to Part 1

- `config/default.yaml` gained a `pipeline:` section (`max_workers`, `slow_stage_seconds`). It is not part of the cache fingerprint, so existing runs stay valid.
- A stray empty folder named `{pipeline,scripts,tests,docs,config}` (from a failed shell command) was in the Part 1 zip; it is removed. Delete it if you see it.
