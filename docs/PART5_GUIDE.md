# Part 5 Guide — Case Study, Documentation, Acceptance Tests & Submission

Builds on Parts 1–4. Nothing here runs inside the pipeline; it proves, documents and packages the result.

```bash
python scripts/audit_submission.py --skip-tests      # what is still missing (today: honest FAILs)
python -m unittest discover -s tests -t .            # everything incl. acceptance tests for T1-T3
```

---

## 1. What Part 5 gives the team

| Need | Where |
|---|---|
| Compare our flood with the official reference, **check-only** | `evaluation/case_study.py` (+ `metrics.py`, `reference.py`) |
| An independent answer to "who is cut off?" for T3 to test against | `evaluation/oracle.py` |
| Tests that T1–T3's real code must pass (skipped until they exist) | `tests/test_stage_acceptance.py` |
| Real documents: README, limitations, attribution + citations, data sources | `README.md`, `docs/*.md` |
| 6-page report plan, 3-minute video script, rehearsal + Q&A, final checklist | `docs/REPORT_TEMPLATE.md`, `DEMO_SCRIPT.md`, `JUDGING_REHEARSAL.md`, `FINAL_AUDIT.md` |
| Put a finished run into the repo for instant demo | `scripts/export_demo_run.py` |
| Numbers for the report copied from the system, not retyped | `scripts/report_facts.py` |
| Automatic Day-13 checklist | `scripts/audit_submission.py` |

## 2. The case-study workflow (Day 11)

1. T1/T2 produce a real Trishuli run; `python scripts/export_demo_run.py outputs/<run>`.
2. **Download the official reference yourself** from the Copernicus EMS portal (activation EMSR927: the flood-extent
   polygons and, if available, the area-of-interest polygon) into `data/reference/` (git-ignored). Convert to GeoJSON if needed.
   I cannot fetch these for you, and I have not seen their real format; check that the file holds polygons.
3. `python -m evaluation.case_study --run data/sample/trishuli_run --reference data/reference/<extent>.geojson --reference-aoi data/reference/<aoi>.geojson --note "<your explanation of a disagreement>"`
4. Open the dashboard: the **Case study** tab fills in. Use `docs/figures/case_study_agreement.png` (green agree, red only we,
   blue only the reference) in the report.
5. Write 3–5 human `--note`s explaining *why* we disagree (radar shadow, resolution, different dates, OSM gaps) and **where we are worse**.

What it computes: IoU, precision, recall, F1 inside the reference's own area; areas in km²; and for roads, bridges and buildings
"in reference / also found by us / only by us", using **our own damage method applied to the reference extent**
(so any difference comes from the flood extent, not from a different method).

## 3. Function reference

| Name | Purpose |
|---|---|
| `metrics.confusion / scores / extent_metrics` | pixel counts and IoU, precision, recall, F1 (`None` when undefined) |
| `metrics.agreement_codes / agreement_png` | the green/red/blue agreement map, resampled to lon/lat |
| `metrics.compare_ids(ours, reference)` | set comparison of flagged object ids |
| `reference.load_reference_polygons(path)` | GeoJSON polygons (shapefile/GeoPackage via geopandas if installed) |
| `reference.rasterize_polygons(polygons, transform, shape, epsg)` | burn polygons onto OUR pixel grid |
| `oracle.analyse(roads, settlements, destinations, removed_ids)` | `connected` / `cut_off` / `unconnected_before` per settlement, with blocking segment ids |
| `oracle.expand_removed(affected_roads, affected_bridges, roads)` | adds all segments of an affected bridge |
| `case_study.run_case_study(...)` | the whole comparison; writes `evaluation/case_study_results.json` and the PNG |
| `audit_submission.run_audit(root)` | list of PASS / FAIL / WARN / MANUAL checks |
| `export_demo_run.export_demo_run(run, dest)` | copy a real run into `data/sample/trishuli_run` |
| `report_facts.facts_markdown(run)` | Markdown block of key results + case-study table + exact attribution |

## 4. Clever tricks (not obvious at first glance)

1. **The oracle found a real bug in my own mock.** Village D's side road did not share a vertex with the main road, so in
   the road graph D was unreachable even before the flood; any correct implementation by T3 would have "failed" my expectations.
   Fixed: the side road now starts at an actual main-road vertex, like two OSM ways sharing a node.
2. **Three statuses, not two.** `unconnected_before` (an OpenStreetMap gap) is separated from `cut_off` (caused by the flood).
   Calling already-isolated villages "cut off by the flood" would be a quiet lie.
3. **"Same method, different mask."** Reference-flagged roads/bridges/buildings are produced by running OUR damage stage on the
   reference extent in a scratch folder. Comparing against numbers from a different method would measure the methods, not the flood.
4. **Fair scoring region.** Outside the reference's own analysis area, "not flooded" only means "not examined". With
   `--reference-aoi` those pixels are excluded; without it the results say the scores may be pessimistic.
5. **No GDAL needed to rasterise.** Pixel centres are tested for point-in-polygon (matplotlib's path code) only inside each polygon's
   pixel window and in chunks, so a few polygons on a huge raster stay cheap; holes are subtracted.
6. **Undefined is not zero.** If neither map has flooded pixels, IoU is `None` ("not available"), not a misleading 0 or 1.
7. **Check-only is enforced three ways.** (a) the reference path may not be inside the run folder; (b) a test fails if any pipeline,
   model, GIS, dashboard or report file imports `evaluation`; (c) the text scan fails if they mention the reference. Results leave
   `evaluation/` only as plain numbers in one JSON file; the dashboard's case-study tab is the single allow-listed file.
8. **Acceptance tests skip instead of fail.** Each calls a teammate's real entry point; `NotImplementedError` becomes a skip with the
   owner's name. The suite is green today and starts to bite the day a stage is delivered. Expectations come from the oracle.
9. **The audit script is honest about the present.** Empty folders, stub documents, a MOCK demo run, a missing or too long report, a
   missing video, leftover `TODO`, unpinned requirements and a failing/skipped test suite are all reported; `MANUAL` items list what only a human can judge.
10. **`export_demo_run` refuses what would embarrass you:** mock runs, files over the 5 MB git limit, and dashboard cache files.
11. **Numbers are copied, not typed.** `report_facts.py` prints the report's table from `summary.json` and the case-study JSON, so the report
    cannot disagree with the dashboard.
12. **Citations were checked** against the NeurIPS 2024 proceedings page and the CVPR 2020 workshop page. For Sen1Floods11 I give the DOI
    and omit page numbers because the sources I found disagree.

## 5. Agreements and to-dos by person

- **T3:** road feature ids must look like `<way id>#<k>` and an affected bridge's `object_id` is the way id (the oracle and the mock use this).
  Report `unconnected_before` settlements separately, not as cut off. Your connectivity stage must reproduce the mock result (B, C, D cut off; A connected).
- **T2:** held-out metrics on **unseen Himalayan scenes** go into report page 3 and the README; cite the training data; put final weights in `models/release/`.
- **T1:** the dates/orbit of the Trishuli pair for report page 2; export the demo run.
- **T4 (you):** reference download and case-study run, the report, the video, `pip freeze` into `requirements.txt`, Nepali review, final audit.

## 6. Tested and NOT tested

- **Tested here (271 tests pass, 10 skipped):** all metrics; polygon rasterisation (squares, holes, clipping, partial overlap); the oracle on
  the mock scenario (A connected, B and C cut by the main road, D cut by the bridge, blocking segments named); the case-study orchestrator with
  injected loader/writer/damage method; the data-rule structure tests; the audit checks on constructed trees; demo export; report facts.
- **Not tested here:** (1) the default rasterio loader/writer used by `case_study` and the real T3 damage stage on a scratch folder
  (no rasterio, and T3's code does not exist yet); (2) reading real EMSR927 files (I do not know their exact format);
  (3) the 4 acceptance tests and 6 other tests skipped for lack of rasterio; (4) the audit's `git`, hygiene and test-suite subprocess checks in a real clone.
  After installing the requirements, run the suite once: expect 0 skipped (the acceptance tests will then skip individually until each stage exists).
- **Not done by me, by nature:** the written report, the video, the Nepali review, and the real case-study numbers. The documents give structure and checks, not results.

## 7. Changes outside Part 5

- `scripts/make_mock_outputs.py`: side road to village D starts at a main-road vertex (see trick 1). Regenerate any saved mock run.
- `.gitignore` and the bootstrap: submission videos are ignored; `docs/report/`, `docs/figures/`, `submission/` exist.
- `README.md`, `docs/limitations.md`, `docs/attribution.md`, `docs/data_sources.md` replace the Day-1 stubs.
