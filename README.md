# FloodResQ

Educational prototype for **Multimodal AI Hackathon 2026, Track B: Mapping Flood Damage from Space**.
Given an **area and a flood date**, it uses Sentinel-1 radar (before/after, same orbit track), optional Sentinel-2,
the Copernicus DEM and **pre-event** OpenStreetMap to answer three questions for rescue teams:

1. **Where did the flood hit?** Flood/debris segmentation on the radar pair (U-Net, or a labelled non-AI fallback).
2. **What was damaged?** Roads, bridges and buildings overlapping the detected flood.
3. **Who is cut off?** Settlements with no remaining road path to a hospital or town.

Plus the **bonus**: trace a flood path downhill from any upstream point and list the settlements along it.
It is **not an operational tool**. See [Limitations](#limitations).

> **How to install, configure, run and deploy: see [`project_instruction.md`](project_instruction.md).**

## Quick start
```bash
python -m venv .venv && source .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python scripts/make_mock_outputs.py                            # contract-valid demo run (synthetic, marked MOCK)
streamlit run dashboard/app.py                                 # then: "Load cached demo"
python -m pipeline.analyze --aoi config/example_aoi.json --date 2026-08-26        # real run
```

## How it works

```
AOI + flood date
   |-- acquire  (T1)  Sentinel-1 same-track pair -> 10 m UTM grid, DEM, slope, layover/shadow + permanent-water masks
   |-- osm      (T1+T3) pre-event OSM (ohsome; Overpass-attic fallback) -> roads split into graph segments
   v
 infer       (T2)  U-Net from a verified model package, else change detection (water darkening + debris brightening)
 damage      (T3)  flood polygons x roads / bridges / buildings
 connectivity(T3)  remove flooded segments + bridges -> Dijkstra from all hospitals/towns -> cut-off settlements
 summarize, report, figures (T4) -> summary.json (single source of numbers), EN + Nepali one-page report, PNGs
 dashboard   (T4)  Streamlit map, KPI cards, tables, report, flood-path bonus, case-study tab
```

Every stage reads and writes fixed files in one run folder (`outputs/<area>_<date>_<id>/`) and is validated
against the data contract in [`pipeline/contracts.py`](pipeline/contracts.py) (rendered in `docs/contracts.md`).
Independent stages run in parallel; a failed optional stage shows "not available", never a fake zero.

| Folder | Owner | What is in it |
|---|---|---|
| `acquisition/` | T1 (+T3) | Sentinel-1/2 search + download (`sentinel.py`, `sentinel2.py`), DEM tiles (`dem.py`), OSM ohsome client (`ohsome.py`), Overpass fallback + tag normalisers (`overpass.py`), stage glue (`s1_stage.py`, `osm.py`) |
| `preprocessing/` | T1 | SNAP chain (`sentinel1.py`), Planetary Computer RTC backend (`s1_rtc.py`), common grid + DEM masks (`alignment.py`, `utm.py`), Sentinel-2 layers (`sentinel2.py`) |
| `models/` | T2 | data (`dataset.py`), U-Net training (`train.py`), evaluation (`evaluate.py`, `eval_metrics.py`), tiled inference + packaging (`inference.py`, `tiling.py`, `pack_model.py`), fallback (`change_detection.py`), stage glue (`stage.py`) |
| `gis/` | T3 | road graph (`road_graph.py`), damage overlay (`damage_overlay.py`, stage `damage.py`), cut-off engine (`cutoff.py`, stage `connectivity.py`), validation toolkit (`validation.py`), flood-path bonus (`flood_path.py`) |
| `pipeline/`, `dashboard/`, `reports/`, `scripts/` | T4 | orchestration, contracts, summary, figures, Streamlit app, EN/NE report, audit tools |
| `evaluation/` | T2+T4 | check-only comparison with the reference maps (`case_study.py`, `model_case_study.py`, `compare.py`, `ems_reference.py`), oracle |

## Model and training data
Flood segmentation: U-Net (ResNet-34 encoder, 6 input channels: before/after/change VV+VH) trained on Kuro Siwo
(Bountos et al., 2024). Training, threshold selection on validation only, Himalayan hold-out evaluation and packaging:
`models/train.py`, `models/evaluate.py`, `models/pack_model.py` (details in `docs/teams/T2_ai_README.md`).
The packaged model goes to `models/release/model_package/`. Without it the pipeline uses the non-AI fallback and
says so on every output.

## Case study: August 2026 Trishuli flood
```bash
python -m pipeline.analyze --aoi config/example_aoi.json --date 2026-08-26
python -m evaluation.case_study --run outputs/<run_folder> \
    --reference data/reference/<flood_extent>.geojson --reference-aoi data/reference/<aoi>.geojson
```
The official Copernicus EMS maps (activation EMSR927) are used **only to check** results; they are never an input
(`data/reference/` is git-ignored and read only by `evaluation/`).

## Limitations
Satellites revisit every few days (no early warning; the map shows the situation at image time); radar shadow and
layover in steep terrain cause errors; 10 m pixels miss small buildings; "affected" is overlap with the flood, not
verified damage; cut-off results depend on OpenStreetMap completeness; without a trained model the map comes from a
labelled non-AI fallback. Full list: [`docs/limitations.md`](docs/limitations.md).

## Tests
```bash
python -m unittest discover -s tests -t .      # or: pytest -q
python scripts/audit_submission.py             # final checklist
```

## Data rules (hackathon)
Allowed inputs: Sentinel-1/2, Copernicus DEM, OpenStreetMap as it was before the event, the listed training
datasets. Published damage maps and post-event OSM edits are check-only. The OSM snapshot date is forced to be before
the event (and every feature's last-edit date is audited); a test and the audit script verify that pipeline code
never references check-only data.

## Attribution
Contains modified Copernicus Sentinel data 2026.

Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

© OpenStreetMap contributors.

Training data and credits: `docs/attribution.md` (Kuro Siwo: Bountos et al., 2024).

## Team
Four-person team: T1 satellite data, T2 AI model, T3 GIS analysis, T4 integration/dashboard. Each teammate's
original notes are in `docs/teams/` and `docs/PART*_GUIDE.md`.
