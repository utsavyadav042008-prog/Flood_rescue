# Part 1 Guide — Repo, Environment, Git Rules & Data Contracts

Read this once, then open the code in the order given in section 2.
Every module also has its own docstring; this guide is the map.

---

## 1. What Part 1 gives the team

| Need | Where it lives |
|---|---|
| "Which file does each stage read/write?" | `pipeline/contracts.py` (`CONTRACTS`, `Artifact`) |
| "Is this output valid?" | `contracts.check_stage()` / `python scripts/validate_run.py <run_dir>` |
| Shared AOI / date / summary objects | `pipeline/schemas.py` |
| One folder per run, with a manifest | `pipeline/run.py` (`RunPaths`) |
| Safe file writing | `pipeline/fileio.py` |
| Shared settings | `pipeline/config.py` + `config/default.yaml` |
| Hackathon data rules as code | `pipeline/rules.py` |
| Day-1 repo skeleton | `scripts/bootstrap_repo.py` |
| Git safety net | `scripts/check_repo_hygiene.py`, `scripts/install_git_hooks.py` |
| Fake data to build against | `scripts/make_mock_outputs.py` |

## 2. Reading order (about 40 minutes)

1. `pipeline/errors.py` (1 min) — two-message exceptions: friendly + detailed.
2. `pipeline/schemas.py` (5) — `AOI`, `Summary`, `Provenance`, `run_id`.
3. `pipeline/config.py` (3) — YAML template, merge, fingerprint.
4. `pipeline/contracts.py` (15) — **the core.** Read sections 1, 2, 3 in the file.
5. `pipeline/run.py` + `pipeline/fileio.py` (7).
6. `scripts/make_mock_outputs.py` (5) — shows the contract being satisfied.
7. `scripts/bootstrap_repo.py`, `check_repo_hygiene.py`, `install_git_hooks.py` (4).

Dependency order inside `pipeline/` (no circular imports):
`errors → geo_math → schemas → config → rules → fileio → contracts → run`

## 3. Day-1 commands (Windows 10, Git Bash)

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt -r requirements-dev.txt

python scripts/bootstrap_repo.py            # skeleton (safe to re-run)
python scripts/install_git_hooks.py         # every teammate runs this once
python scripts/make_mock_outputs.py         # fake run -> data/sample/mock_run
python scripts/validate_run.py data/sample/mock_run
python -m unittest discover -s tests -t .   # or: pytest -q
```

Expected on a machine WITH rasterio: all tests pass, **0 skipped**.
(See section 7: the raster tests could not run where this code was written.)

## 4. Function reference

### `pipeline/schemas.py`
| Name | What it does |
|---|---|
| `parse_date(value)` | `date` / `datetime` / `"2026-08-26"` / `"2026-08-26T05:30:00Z"` → `date` |
| `parse_event_date(value)` | same + rejects future dates and dates before Sentinel-1 (2014-10-03) |
| `AOI` | frozen bbox. `.bbox`, `.center`, `.area_km2()`, `.utm_epsg()`, `.validate()`, `.slug()`, `.from_dict()` (accepts `bbox:[w,s,e,n]` too), `.from_json_file()` |
| `Summary` | the only object the dashboard and report read. `None` = not computed, `0` = nothing found. `.unavailable()` lists missing KPIs |
| `Provenance` | producer / source / CRS / event date / model version / timestamp / `is_mock` |
| `run_id(...)` | 8-char deterministic id used as cache key and in the folder name |

### `pipeline/config.py`
| Name | What it does |
|---|---|
| `DEFAULT_YAML`, `DEFAULT_CONFIG` | the YAML text (with comments) and the dict parsed from it |
| `load_config(path, overrides)` | defaults ← `config/default.yaml` ← overrides |
| `deep_merge(base, override)` | nested merge, so old config files keep working |
| `config_fingerprint(cfg)` | hash of only the result-relevant sections |

### `pipeline/contracts.py`
| Name | What it does |
|---|---|
| `Artifact` | enum of every file in a run folder (`Artifact.FLOOD_MASK.value == "flood_mask.tif"`) |
| `SPECS` | per-file rules: kind, required keys/properties, allowed geometry types |
| `CONTRACTS` | per-stage: owner, entry point, `needs`, `produces`, `required` |
| `check_artifact(run, artifact)` | all problems in one file (list of `Issue`) |
| `check_stage(run, stage)` | all problems in a stage's outputs |
| `check_run(run)` | every stage that has output |
| `raise_for_issues(issues, stage, strict)` | raises `ContractError` (warnings count only with `strict=True`) |
| `stage_is_fresh(run, stage)` | cache test: outputs exist and are newer than inputs |
| `contracts_markdown()` | renders `docs/contracts.md` from the code |

### `pipeline/run.py`
| Name | What it does |
|---|---|
| `RunPaths.create(aoi, date, cfg)` | validates, makes `outputs/<name>_<date>_<id>/`, writes `run.json` |
| `RunPaths.open(path)` | opens an existing run (needs `run.json`) |
| `run.path(Artifact.X)` / `run[Artifact.X]` | absolute path of a file |
| `run.ensure_parent(Artifact.X)` | creates `osm/` etc. and returns the path |
| `run.aoi`, `run.event_date`, `run.manifest()` | read back what the run was created with |

### `pipeline/fileio.py`
| Name | What it does |
|---|---|
| `write_json`, `write_geojson`, `write_geojson_from_gdf` | atomic UTF-8 writers with optional provenance |
| `feature(geometry, **props)` | build one GeoJSON feature |
| `write_geotiff`, `north_up_transform` | compressed GeoTIFF with provenance tag |
| `read_json`, `read_provenance` | readers (provenance works on json/geojson/tif) |
| `mask_area_km2(path)` | flooded km² from the mask; refuses non-metric CRS |

### Stage entry points (what T1–T3 implement)
```python
def run_xxx(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None:
    ...   # read run.path(Artifact.A), write run.path(Artifact.B)
```
`bootstrap_repo.py` writes these stubs from the contract, so each teammate opens
a file that already lists exactly what to read and write.

## 5. Clever tricks (not obvious at first glance)

1. **`Artifact` is a `str` enum.** `run.path("flod_mask.tif")` (typo) raises
   `ValueError` immediately; a hard-coded string would silently create a wrong file.
2. **`DEFAULT_YAML` is the single source of truth.** The dict is *parsed from* the
   YAML text, so `config/default.yaml` (comments included) and the code defaults can
   never drift. A test also checks the file written by bootstrap equals the dict.
3. **Cache fingerprint covers only result-relevant config sections.** Changing the
   output folder or a dashboard setting does not throw away cached runs. `default=str`
   keeps it stable when YAML turns an unquoted `2026-07-27` into a `date`.
4. **`run_id` survives float noise.** The bbox is rounded to 4 decimals (~11 m), JSON
   is canonical (`sort_keys`), and the AOI *name* is not hashed. `RunPaths.create` also
   reuses an existing folder with the same id even if the name changed.
5. **Make-style freshness (`stage_is_fresh`).** Compare file modification times: if T1
   regenerates `before.tif`, `flood_mask.tif` becomes older than its input, so `infer`
   (and everything after it) reruns. There is no cache key to maintain.
6. **Atomic writes, Windows-proof.** Write to a temp file *in the same folder*, then
   `os.replace` (atomic only on one filesystem). On Windows the replace can fail while
   Streamlit has the file open, so it retries for about a second.
7. **Always UTF-8.** Windows' default encoding (cp1252) crashes on Nepali names such
   as `गाउँ`. Every read/write states `encoding="utf-8"`; `ensure_ascii=False` keeps
   the files human-readable. A test round-trips a Devanagari name.
8. **JSON hardening.** numpy scalars (`np.int64`, a classic crash) are converted
   automatically, and `allow_nan=False` turns a silent invalid `NaN` into a clear error.
9. **Catching projected coordinates without geopandas.** A valid lon/lat is within
   ±180/±90; UTM numbers like `323166, 3109588` fail that range test instantly. This
   catches "wrote UTM metres into a GeoJSON" using only the standard library.
10. **Problem histogram + stride sampling.** One bad property in 50 000 buildings gives
    ONE line ("… (50000 features, e.g. way/12)"), not 50 000. Coordinates are checked
    on an evenly spaced sample of ≤2000 features so validation stays fast.
11. **Grid comparison with tolerance.** Two tools can compute the same raster grid
    differing in the 12th decimal; `==` would call that a mismatch, so shape/CRS are
    compared exactly and the affine transform with `math.isclose`.
12. **Small Python gotchas handled on purpose.** `datetime` is a subclass of `date`, so
    it is tested first in `parse_date`. A frozen dataclass is coerced with
    `object.__setattr__`. `-180 <= w < e <= 180` is a chained comparison that is
    `False` for NaN, so NaN boxes are rejected for free.
13. **Pure-Python UTM (`geo_math.py`).** On Windows, `rasterio`/`pyproj` are the usual
    Day-1 install failures; the mock generator and tests must still run. The
    Krüger-series maths is verified against an *independent* numerical meridian-arc
    integral to 1 mm in `tests/test_geo_math.py`.
14. **Shared git hooks.** `.git/hooks` is not versioned; `git config core.hooksPath
    .githooks` points git at a tracked folder instead. The hook checks the **staged**
    blob (`git cat-file -s :path`), not the working-tree file, and `--diff-filter=ACMR`
    skips deletions so removing a bad file is never blocked. The hook file is written
    with LF line endings (CRLF breaks the shebang in Git Bash).
15. **Bootstrap never clobbers.** Existing files are skipped unless `--force`, so
    re-running it after teammates started is harmless. Generated files
    (`docs/contracts.md`) are refreshed every time because they are derived from code.
16. **Self-documenting stubs.** The stage stubs and `docs/contracts.md` are generated
    from `CONTRACTS`; documentation cannot disagree with the validators.
17. **The mock is geometrically consistent.** Flood mask, roads, buildings and numbers
    all come from the same two functions (`river_y`, `half_width`), so they agree like
    real outputs should. Every mock file carries `is_mock: true`, and
    `validate_run.py --forbid-mock` fails a submission that still contains mock data.
18. **Two severity levels.** Day-to-day, warnings (e.g. missing provenance) don't stop
    anyone; on Day 13 run with `--strict` and they become errors.
19. **Provenance travels inside the file.** GeoJSON: a top-level foreign member (allowed
    by RFC 7946). GeoTIFF: ONE JSON-string tag (tags are strings only; JSON keeps types).
20. **The data-rule guard is already live.** `scan_for_forbidden_inputs` fails a test if
    pipeline code mentions reference data; the OSM-snapshot check fails a run whose OSM
    date is not strictly before the event.

## 6. Decisions to confirm with the team on Day 1

- **CRS:** rasters in the AOI's UTM zone (`AOI.utm_epsg()`, 32645 for Trishuli), GeoJSON in EPSG:4326.
- **`before.tif` / `after.tif`:** band 1 = VV backscatter in dB, same grid, same relative orbit.
- **`flood_mask.tif`:** only 0/1, no nodata. **`flood_confidence.tif`:** probability of the flood class, 0–1.
- **`osm/hospitals.geojson`** holds hospitals *and* towns, distinguished by `destination_type`.
- **T3, please note:** the connectivity stage also needs `affected_bridges.geojson`; a flooded bridge must remove the road edge that crosses it (the mock's Village D shows this).
- **OSM source:** use the ohsome API with a snapshot date before 26 Aug 2026 (the brief's rule). `osm/snapshot.json` records the date and the validator enforces it.
- **`config/example_aoi.json` is a PLACEHOLDER box** (85.10–85.45 E, 27.85–28.30 N) for the corridor. Refine it from the brief's description of the Bhote Koshi–Trishuli corridor and ordinary maps, **not** from published damage-map extents (data rule).
- **Final model weights** go to `models/release/` (allowed by the hook, ≤ 50 MB); experimental checkpoints never get committed.

## 7. What has and has not been tested

- Run here: 74 tests pass (schemas, config, UTM maths, contract validators, broken-output cases, freshness, hygiene hook with a real git repo, bootstrap, rules).
- **Not run here:** 5 tests plus every code path that imports `rasterio` (`write_geotiff`, `mask_area_km2`, raster validation, mock rasters). The sandbox had no `rasterio` and no network. They are written to skip cleanly without it. After installing `requirements.txt`, run the test suite once: expect 0 skipped. If a raster test fails, send me the traceback.

## 8. Pitfalls

- Do not name a `unittest.TestCase` attribute `run` (it shadows `TestCase.run`); tests use `rp`.
- Even a comment containing the check-only data names trips the data-rule scan. That is deliberate; whitelist the single case-study file if needed.
- `scripts/make_mock_outputs.py` without rasterio produces vectors/JSON only (summary says so); rerun after installing it.
