# Part 3 Guide — Map Dashboard (Streamlit + Folium)

Builds on Parts 1–2 (`RunPaths`, contract, `analyze_flood`, `load_result`).

```bash
pip install -r requirements.txt
python scripts/make_mock_outputs.py          # a fake run in data/sample/mock_run (needs rasterio for rasters)
streamlit run dashboard/app.py               # then press "Load cached demo"
```

---

## 1. Design rule: logic is testable, glue is thin

| Layer | Files | Imports Streamlit/Folium? | Tested how |
|---|---|---|---|
| Pure logic | `dashboard/data.py`, `raster_overlay.py`, `case_study.py` (parser), `reports/format.py` | no | real unit tests |
| Folium glue | `dashboard/map_layers.py` | folium | fake Folium |
| Streamlit glue | `dashboard/components.py`, `dashboard/app.py` | streamlit | fake Streamlit |

The dashboard **computes nothing**. Every number is read from `summary.json` and printed
through `reports/format.py:fmt`, the same function the situation report will use (Part 4).

## 2. What the screen contains

```
sidebar                         main area
+------------------------+      +-------------------------------------------------------+
| Area (preset / Custom) |      | title, one-line purpose, "educational prototype" bar  |
| live area in km2       |      | banners: MOCK DATA, failed stages, data-quality notes |
| Flood date             |      | 6 KPI cards (n/a when a stage did not run)            |
| [Run analysis]         |      | tabs: Map | Before/After | Tables | Report |          |
| Advanced: mock stages, |      |       Case study | About                              |
|   cache on/off         |      | footer: the 3 required attribution lines              |
| [Load cached demo]     |      +-------------------------------------------------------+
| Open an earlier run    |
| Map layers (10 boxes)  |
+------------------------+
```

**Map layers** (draw order bottom → top; the cut-off layer is always on top)

| Layer | Source file | Look | Default |
|---|---|---|---|
| Model confidence | `flood_confidence.tif` | blue, opacity = probability | off |
| Flood / debris extent | `flood_mask.tif` | translucent blue | on |
| All roads (pre-event) | `osm/roads.geojson` | thin grey lines | off |
| All buildings (pre-event) | `osm/buildings.geojson` | light purple | off |
| Affected buildings | `affected_buildings.geojson` | purple fill | on |
| Affected roads | `affected_roads.geojson` | red lines | on |
| Affected bridges | `affected_bridges.geojson` | thick orange lines | on |
| Settlements | `osm/settlements.geojson` | small grey dots | on |
| Hospitals / towns | `osm/hospitals.geojson` | green plus icons | on |
| **Cut-off settlements** | `cutoff_settlements.geojson` | large red dot, white ring | on |

Colour is never the only cue: lines, fills, dots and icons also differ in shape.

## 3. Reading order (about 30 minutes)

1. `dashboard/data.py` — sections 1 (layer registry), 4 (KPI cards), 6 (inputs/runs).
2. `dashboard/raster_overlay.py` — how a UTM raster becomes a map picture.
3. `dashboard/map_layers.py` — 90 lines, Folium assembly.
4. `dashboard/app.py` — `main()` top to bottom, then `components.py`.
5. `tests/test_dashboard_data.py`, `tests/test_dashboard_ui.py` (test names are the spec).

## 4. Function reference

### `dashboard/data.py` (pure)
| Name | Purpose |
|---|---|
| `LAYERS`, `LAYER_BY_KEY`, `LayerSpec` | layer registry: key, label, source file, kind, colour, default, popup fields |
| `format_prop(key, value)` | display text for a property; **HTML-escaped** |
| `thin_geometry(geom)` | round coordinates (~1 m) and drop redundant vertices |
| `prepare_features(spec, features, max)` | trim properties to popup fields, thin geometry, cap context layers |
| `prepare_layers(run, cfg, enabled)` | load everything the map needs (the expensive, cacheable step) |
| `map_bounds(prepared, aoi)` | where to point the map: data → raster footprint → AOI |
| `legend_items`, `legend_html` | map legend (escaped) |
| `representative_point`, `popup_html` | marker anchor and popup table for point layers |
| `kpi_cards(summary)` | the six headline cards |
| `banners(summary, result)` | mock warning, failed-stage errors, quality notes |
| `imagery_caption(summary)` | "Sentinel-1 before … / after …, orbit …, OSM snapshot …, model …" |
| `build_tables(run)`, `table_to_csv(table)` | Tables tab + Excel-safe CSV |
| `load_presets`, `aoi_from_inputs` | area selector and live validation |
| `list_runs`, `find_demo_run`, `run_version` | earlier runs, demo run, cache key |
| `stage_progress_markdown(states)` | live progress list while the pipeline runs |
| `LIMITATIONS`, `DISCLAIMER` | built-in text for the About tab and banner |

### `dashboard/raster_overlay.py`
| Name | Purpose |
|---|---|
| `encode_png_rgba(rgba)` | PNG writer using only `zlib` + `struct` |
| `warp_to_lonlat(array, transform, epsg)` | UTM raster → lon/lat grid by inverse mapping |
| `mask_rgba`, `probability_rgba`, `gray_rgba` | colouring |
| `overlay_png(kind, array, transform, epsg)` | pure end-to-end (testable without rasterio) |
| `render_overlay_file`, `render_gray_file` | read with rasterio, cache PNGs in `figures/_overlay_*.png` |

### `dashboard/components.py`, `app.py`, `map_layers.py`, `case_study.py`
`call_compat` (version-proof Streamlit calls), `render_*` (one per page area),
`build_map`, `render_sidebar`, `run_analysis`, `open_run`, `main`, `parse_case_study`.

## 5. Clever tricks (not obvious at first glance)

1. **Inverse-mapping resample instead of drawing the UTM bounding box.** UTM "north" is rotated
   against true north (about 0.8° at the edge of the Trishuli zone), so a raster placed by its
   bounding box is sheared by hundreds of metres. For every pixel of an output lon/lat grid we
   compute where it lies in the raster (`lonlat_to_utm_array`, vectorised) and copy that pixel.
   No GDAL warp, and the image is capped at 1200 px so the browser never gets 20 megapixels. A test
   puts a blob at known UTM coordinates and checks it lands at the right lon/lat.
2. **PNG encoder in 20 lines of standard library** (zlib + struct + CRC32), so a translucent
   overlay does not need Pillow. The test decodes the file again and compares every pixel.
3. **The overlay is a picture, never a number.** Nearest-neighbour resampling can drop thin
   features, so no KPI is ever computed from it.
4. **Overlay passed to Folium as a file path, not a data URI.** Folium treats a string that is not a
   known URL scheme as a file and embeds it itself; a data URI is mistaken for a file name (a known
   Folium issue). The PNG is cached in the run folder and regenerated only if the raster is newer.
5. **Untrusted text is HTML-escaped.** OSM names and reasons end up in popups; `format_prop` escapes
   them, so a place called `<img onerror=...>` is shown literally. A test covers it.
6. **Properties trimmed to the popup fields and every feature gets every key.** It cuts the GeoJSON
   size a lot, and the map library requires identical keys on all features.
7. **Context layers are thinned with an even stride**, not "first N rows", so the shown subset covers
   the whole area; the affected-object layers are never capped.
8. **Map framed on the data, not the AOI.** A mock run's data sits in the mock valley, so framing on
   the AOI would show an empty map.
9. **Heavy work cached, light work redone.** `prepare_layers` (file reads, resampling) is wrapped in
   `st.cache_data` with `run_version` as part of the key; ticking a checkbox only rebuilds the cheap
   Folium objects. `returned_objects=[]` stops a rerun on every pan or zoom.
10. **`call_compat` tries `width="stretch"`, then `use_container_width=True`, then nothing.** Streamlit
    is replacing the old argument and plans to remove it; this keeps the app working on any version
    `pip install streamlit` fetches.
11. **CSV with a UTF-8 BOM** so Excel on Windows shows Nepali names correctly.
12. **Point layers are individual markers; lines/polygons are GeoJSON.** Tens of settlements as markers
    with hand-built popups avoid relying on how a library styles GeoJSON points.
13. **The page never shows a stack trace.** Failed runs show `result.error` plus a step-by-step list;
    missing layers become one-line notes; missing map libraries give an install hint.
14. **Honesty by default:** disclaimer bar, MOCK banner, unavailable KPIs shown as "n/a" (never 0),
    imagery dates and orbit under the map, limitations in the About tab, no images of people.
15. **Case-study tab reads only a results file.** `dashboard/case_study.py` is the single dashboard file
    allowed to name the reference activation (the data-rule test whitelists exactly that file) and it never
    loads the reference maps themselves.
16. **Test harness lesson:** `from dashboard import components` first looks at an attribute of the
    already-imported package, so deleting `sys.modules` entries is not enough; `tests/fakes.py` also
    deletes those attributes, otherwise later tests silently use the previous fake.

## 6. Agreements needed with teammates

- **T3:** GeoJSON properties named in the contract; cut-off `reason` must be a readable sentence
  (it is shown in the popup and table); `blocking_segment_ids` is a list.
- **T2/T3 (Part 5):** write `evaluation/case_study_results.json` in the schema documented at the top of
  `dashboard/case_study.py`. Until it exists the tab says so.
- **Part 4:** reuse `reports/format.py:fmt` and `ATTRIBUTION_LINES`. The Report tab shows
  `situation_report.txt` (and the PDF if present) as soon as the report stage writes them; the
  Before/After tab uses `figures/before.png` / `after.png` if present, otherwise draws a grey preview itself.
- Put a real run in `data/sample/trishuli_run/` (small files only) before the demo: "Load cached demo" opens it instantly.

## 7. Tested and NOT tested

- **Tested here (178 tests pass in the whole repo, 6 skipped):** all pure logic, raster resampling and PNG
  encoding, layer registry vs the contract, KPI/banner/table/CSV behaviour, and the whole UI flow
  (first screen, demo, run success/failure, invalid area, layer toggles, partial results, footer text)
  against **fake** Streamlit/Folium.
- **Not tested here:** the real Streamlit, Folium and streamlit-folium (not installable in my sandbox) and
  everything using `rasterio` (reading real GeoTIFFs for overlays/previews). The fakes only prove that my
  code runs and wires correctly, not that the libraries accept every argument. Two points I checked in
  the docs: Streamlit's `use_container_width` → `width="stretch"` change, and Folium's file-path handling.
- **Manual check after installing (5 minutes):**
  1. `streamlit run dashboard/app.py` opens without an exception.
  2. "Load cached demo": 6 cards, map shows red/orange/purple objects and 3 large red dots; clicking a red
     dot shows the reason.
  3. Untick/tick a layer: the map updates quickly (no full reprocessing).
  4. With rasters present: the blue flood overlay sits on the river, not shifted.
  5. Tables tab: 4 tables and CSV downloads; open one CSV in Excel and check the Nepali names.
  6. Run with `Advanced > mock these steps` and watch the live step list.
  If a library call fails, send me the traceback: the likeliest spots are `st.status`, `folium.GeoJsonPopup`
  and the `st_folium` size arguments.

## 8. Changes outside Part 3

- `reports/format.py` is new and is the single home of number formatting and the exact attribution text
  (with the real © symbols). `scripts/bootstrap_repo.py` now takes the attribution from it, so README and
  `docs/attribution.md` use the brief's wording.
- `pipeline/geo_math.py` gained `lonlat_to_utm_array` (vectorised UTM).
- **Bug fix in Part 2:** `analyze_flood` could return the `summary.json` of an EARLIER run after a failure
  (the old file was still in the folder). It now only returns a summary the current run produced. And
  `load_result` now gives a readable error when a required stage has no results.
