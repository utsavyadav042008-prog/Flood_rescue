# FloodResQ — Project instructions (install, run, deploy)

This is the practical guide for the team: how to install the integrated system, run it from the command line and the
dashboard, train/plug in the AI model, and deploy the dashboard as a server for the live judging.

Contents
1. [What you are running](#1-what-you-are-running)
2. [Requirements](#2-requirements)
3. [Install](#3-install)
4. [Credentials and optional tools](#4-credentials-and-optional-tools)
5. [Check the installation](#5-check-the-installation)
6. [Run the analysis](#6-run-the-analysis)
7. [Configuration you are likely to change](#7-configuration-you-are-likely-to-change)
8. [The AI model](#8-the-ai-model)
9. [Case study (EMSR927) and evaluation](#9-case-study-emsr927-and-evaluation)
10. [Deploy the dashboard server](#10-deploy-the-dashboard-server)
11. [Judging-day checklist](#11-judging-day-checklist)
12. [Troubleshooting](#12-troubleshooting)
13. [How the four parts were integrated](#13-how-the-four-parts-were-integrated)

---

## 1. What you are running

```
AOI + flood date
  acquire       Sentinel-1 same-orbit before/after pair -> 10 m UTM grid + DEM, slope,
                layover/shadow mask, permanent-water mask (+ optional Sentinel-2)
  osm           OpenStreetMap AS IT WAS before the event (ohsome API; Overpass-attic fallback)
  infer         flood/debris mask: U-Net if a model package exists, else labelled change detection
  damage        flooded roads / bridges / buildings
  connectivity  settlements with no road route left to a hospital or town
  summarize -> report (EN + Nepali) -> figures       dashboard reads the same summary.json
```

* One run = one folder in `outputs/<area>_<date>_<id>/`. Re-running the same area/date reuses finished stages
  (cache); `--no-cache` forces everything again.
* Every stage's files are checked against the data contract (`docs/contracts.md`). If an optional stage fails
  (OSM, damage, connectivity) the result shows "not available" instead of a wrong zero.

## 2. Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | **3.11 or 3.12** (best wheel coverage for rasterio/geopandas/torch) |
| RAM | 8 GB | 16 GB (32 GB if you use ESA SNAP on full scenes) |
| Disk | 5 GB | 20 GB with the SNAP backend (each Sentinel-1 GRD scene is 1–2 GB) |
| Internet | needed for a live run (Sentinel-1, DEM, OSM downloads) | a cached demo run works offline |
| OS | Windows 10/11, Linux, macOS | Linux for servers |

## 3. Install

### 3a. pip + virtual environment (simplest)

Windows (PowerShell):
```powershell
git clone https://github.com/utsavyadav042008-prog/Flood_rescue.git
cd Flood_rescue
git switch integration                # the integrated branch (merge into main when the team agrees)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

Windows (Git Bash) — same, but activate with `source .venv/Scripts/activate`.

Linux / macOS:
```bash
git clone https://github.com/utsavyadav042008-prog/Flood_rescue.git && cd Flood_rescue
git switch integration
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

### 3b. conda / mamba (use this if `rasterio`, `geopandas` or `pyproj` fail to install with pip on Windows)

```bash
conda create -n floodresq -c conda-forge python=3.11 gdal rasterio geopandas pyproj shapely pyogrio osmnx
conda activate floodresq
pip install -r requirements.txt -r requirements-dev.txt
```

### 3c. AI model packages (only for training or running the U-Net)

```bash
# CPU only:
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
# GPU: pick the command for your CUDA version at https://pytorch.org/get-started/locally/
pip install -r requirements-ml.txt
```
Without these, everything still runs and the flood map comes from the non-AI fallback (clearly labelled).

## 4. Credentials and optional tools

Copy the template and fill it in (the file is git-ignored — never commit it):
```bash
cp .env.example .env
```

| Setting | Needed for | How to get it |
|---|---|---|
| `CDSE_USERNAME`, `CDSE_PASSWORD` | Sentinel-1 backend `cdse_snap` (GRD downloads) and Sentinel-2 bands | free account at https://dataspace.copernicus.eu |
| `SNAP_GPT` | backend `cdse_snap` | install ESA SNAP (free, https://step.esa.int), then point this to `.../esa-snap/bin/gpt` (`gpt.exe` on Windows) if it is not on PATH |

Nothing else needs an account: the Planetary Computer backend, the Copernicus DEM tiles and the ohsome / Overpass
OSM APIs are open.

### Which Sentinel-1 backend will run?

`config/default.yaml → sentinel1.backend` (default `auto`):

| Backend | What it does | Needs | Speed |
|---|---|---|---|
| `auto` | `cdse_snap` if SNAP **and** CDSE credentials are present, otherwise `planetary_computer` | – | – |
| `planetary_computer` | streams the AOI window of terrain-corrected Sentinel-1 RTC (same acquisitions) | internet | **minutes** — best for live demos |
| `cdse_snap` | reference chain: download full GRD scenes, ESA SNAP orbit/noise/calibration/Refined Lee/terrain correction | SNAP + CDSE account, 2–4 GB download | 20–60 min first time, cached afterwards |
| `local` | your own preprocessed GeoTIFFs (UTM, bands VV,VH in dB) + dates/orbit in `sentinel1.local` | files | instant |

Both the downloaded scenes (`data/raw/`, `data/interim/`, `data/processed/`) and the search results
(`data/cache/`) are cached, so the second run of the same area is fast.

## 5. Check the installation

```bash
python -m unittest discover -s tests -t .        # ~510 offline tests (a few skip without rasterio/reportlab/torch)
python scripts/make_mock_outputs.py              # writes data/sample/mock_run (synthetic, marked MOCK)
python -m pipeline.analyze --aoi config/example_aoi.json --date 2026-08-26 --mock-stages acquire osm
```
The last command runs the REAL detection, damage, cut-off, report and figure stages on the synthetic valley and must
end with `OK -> outputs/_mock/...` and "cut off: 3".

Tests that download live data are skipped by default; to include them:
```bash
FLOODRESQ_NETWORK_TESTS=1 python -m unittest discover -s tests -t .      # PowerShell: $env:FLOODRESQ_NETWORK_TESTS=1
```

## 6. Run the analysis

### 6a. Dashboard (what the judges see)
```bash
streamlit run dashboard/app.py
```
Open http://localhost:8501, then either
* **Load cached demo** (instant, offline), or
* choose an **Area** (preset or *Custom* west/south/east/north, max 2,500 km²) and a **Flood date**, press **Run analysis**.
  Progress of every stage is shown live.
* or click **Use this event** on the start page: real past floods whose radar change is clearly visible
  (presets in `config/events/`; the date is filled in automatically):

  | Event | Area preset | Flood date to enter | Radar images used | What you see |
  |---|---|---|---|---|
  | Sindh floods, Pakistan | Sindh floods (Dadu, Pakistan) | **2022-08-25** | 2022-08-15 → 2022-08-27 | large areas turn dark (new water), ~375 km² detected |
  | Melamchi debris flood, Nepal | Melamchi debris flood (Nepal) | **2021-06-16** | 2021-06-08 → 2021-06-20 | change along the river through Melamchi Bazaar |

  Use the day AFTER an evening flood (Melamchi struck on the evening of 15 June 2021): a radar pass on the event
  day itself may have been taken before the flood.

Tabs: **Map** (flood extent, model confidence, affected roads/bridges/buildings, settlements, hospitals/towns,
cut-off settlements, satellite/terrain basemaps), **Before/After** (same-contrast before and after images, a
change map — blue = darker/new water, orange = brighter/debris — and a swipe slider between the two dates),
**Tables** (CSV export), **Report** (English + Nepali, TXT/PDF/HTML), **Flood path (bonus)**, **Case study**,
**About** (limitations, attributions).

Share a finished run with a link: `http://localhost:8501/?run=<run folder name in outputs/>` (or `?run=demo`).

### 6b. Command line
```bash
python -m pipeline.analyze --aoi config/example_aoi.json --date 2026-08-26
```
Useful flags: `--no-cache` (re-run every stage), `--json` (machine-readable result), `--out DIR`,
`--mock-stages acquire osm ...` (replace chosen stages with synthetic data), `--strict` and `--forbid-mock` (final audit).

An AOI file is a small JSON (copy `config/example_aoi.json`):
```json
{"name": "Trishuli corridor", "west": 85.1, "south": 27.85, "east": 85.45, "north": 28.3}
```

### 6c. What a run folder contains
| File | Meaning |
|---|---|
| `before.tif`, `after.tif` | Sentinel-1 VV/VH (dB) on the shared UTM grid |
| `metadata.json` | scene dates, orbit, platform, backend, warnings, attributions |
| `osm/*.geojson`, `osm/snapshot.json` | pre-event roads (graph segments), bridges, buildings, settlements, hospitals/towns + snapshot evidence |
| `flood_mask.tif`, `flood_confidence.tif` | flood/debris mask (0/1) and confidence (0–1) |
| `affected_roads/bridges/buildings.geojson`, `damage_summary.json` | what was hit |
| `cutoff_settlements.geojson`, `connectivity_summary.json` | who is cut off, why, nearest destination, blocking segments, repair priorities |
| `summary.json` | the ONE set of numbers used by the dashboard and the report |
| `situation_report.txt/.pdf/.html`, `situation_report_ne.txt/.html` | one-page situation report (EN, Nepali) |
| `figures/*.png` | before/after, flood, damage and cut-off maps for slides and the video |
| `extras/` | DEM, slope, terrain and permanent-water masks, Sentinel-2 layers, scene pair, model notes |
| `run_log.json` | status and timing of every stage |

### 6d. Other commands
```bash
python scripts/validate_run.py outputs/<run>                 # check a run against the data contract
python scripts/make_report.py outputs/<run> --print          # regenerate the situation report
python -m gis.flood_path --run outputs/<run> --lon 85.36 --lat 28.27     # bonus: flood path + settlements on it
python scripts/export_demo_run.py outputs/<run>              # make it the "Load cached demo" run (data/sample/trishuli_run)
python scripts/audit_submission.py                           # final submission checklist
```

## 7. Configuration you are likely to change

All settings live in `config/default.yaml` (the code defaults are identical; a missing key falls back to them).

| Key | Default | Why you would change it |
|---|---|---|
| `sentinel1.backend` | `auto` | force `planetary_computer` for fast demos or `cdse_snap` for the reference chain |
| `sentinel1.search_window_days` | 40 | widen if no same-orbit pair is found |
| `sentinel2.enabled` | false | turn on optical evidence (needs CDSE credentials; useful only when the sky is clear) |
| `osm.snapshot_date` | 2026-07-27 | latest pre-event OSM snapshot allowed; the code always keeps it before the event |
| `osm.network_buffer_km` | 5 | roads/hospitals are fetched this far outside the box so routes leaving the AOI are seen |
| `model.package` | `models/release/model_package` | where the trained U-Net package lives |
| `infer.*` | see file | thresholds of the non-AI fallback; terrain/permanent-water masking |
| `damage.road_min_block_m` | 20 | wet length that blocks a road segment |
| `connectivity.block_flooded_bridges` | true | debris floods usually destroy bridges; set false to treat them as passable |
| `connectivity.include_footpaths` | false | include trails when deciding "cut off" |
| `limits.max_aoi_km2` | 2500 | largest box allowed |

Settings that change results are part of the run id, so a changed setting creates a new run folder instead of
silently reusing old results.

## 8. The AI model

The brief's AI component is the **flood segmentation model** (T2). The full workflow is in
`docs/teams/T2_ai_README.md`; in short:

```bash
# 1. data: Kuro Siwo patches as .npz (adapt read_patch_npz in models/dataset.py to the real layout first)
python models/dataset.py scan --root /data/KuroSiwo_npz --out config --force-test <himalayan_event> ...
# 2. train (GPU recommended; Colab/Kaggle is enough)
python models/train.py forward-check --config config/train.yaml
python models/train.py train --config config/train.yaml
# 3. evaluate: threshold chosen on validation only, Himalayan events held out
python models/evaluate.py run --checkpoint models/weights/baseline/best.pt --splits config/splits.json \
    --data-root /data/KuroSiwo_npz --himalayan-events <event> --out outputs/eval
# 4. package (weights + frozen threshold + model card + SHA-256 manifest)
python models/pack_model.py build --checkpoint models/weights/baseline/best.pt \
    --protocol outputs/eval/eval_protocol.json --results outputs/eval/results.json \
    --out models/release/model_package
python models/pack_model.py verify models/release/model_package
```

When `models/release/model_package/` exists and PyTorch is installed, the `infer` stage uses the U-Net
automatically; otherwise it uses the change-detection fallback and every output says so.
The package (~100 MB) is git-ignored: share it with **Git LFS** (`git lfs track "models/release/model_package/*.pt"`)
or as a GitHub release asset, and copy it into `models/release/model_package/` on each machine.

Self-tests that need no data or GPU: `python models/evaluate.py demo`, `python models/inference.py demo`,
`python evaluation/model_case_study.py demo`.

## 9. Case study (EMSR927) and evaluation

The Copernicus EMS maps are **check-only** (using them as an input disqualifies the team). Put them in
`data/reference/` (git-ignored; only `evaluation/` may read it) and compare after the run:

```bash
python -m pipeline.analyze --aoi config/example_aoi.json --date 2026-08-26
python -m evaluation.case_study --run outputs/<run> \
    --reference data/reference/<EMSR927_flood_extent>.geojson --reference-aoi data/reference/<EMSR927_aoi>.geojson
```
Results go to `evaluation/case_study_results.json` and appear in the dashboard's *Case study* tab. The pixel-level
comparison with bootstrap confidence intervals and the misregistration check is
`python evaluation/model_case_study.py run ...` (see `docs/teams/T2_ai_README.md`).

## 10. Deploy the dashboard server

The dashboard is a Streamlit app; "the server" is `streamlit run dashboard/app.py`. Pick one option.

### 10a. On your laptop, reachable from the local network (simplest for judging)
```bash
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```
Others on the same Wi-Fi open `http://<your-laptop-IP>:8501` (Windows: allow Python through the firewall when asked;
find the IP with `ipconfig`).

### 10b. Docker (any machine with Docker)
```bash
docker build -t floodresq .                          # add --build-arg WITH_ML=1 for the U-Net (CPU PyTorch)
docker run -d --name floodresq -p 8501:8501 --env-file .env \
    -v floodresq-data:/app/data -v floodresq-out:/app/outputs floodresq
```
Then open http://localhost:8501. The volumes keep downloads and results between restarts. The image contains no
ESA SNAP, so it uses the Planetary Computer backend. To use your trained model, copy it in:
`docker cp models/release/model_package floodresq:/app/models/release/`.

### 10c. Linux VM (e.g. Ubuntu 22.04/24.04 on any cloud), always on, with HTTPS
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git nginx
git clone https://github.com/utsavyadav042008-prog/Flood_rescue.git /opt/floodresq && cd /opt/floodresq
git switch integration
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env && nano .env              # credentials if you use cdse_snap / Sentinel-2
.venv/bin/python scripts/make_mock_outputs.py
```
Run it as a service — `/etc/systemd/system/floodresq.service`:
```ini
[Unit]
Description=FloodResQ dashboard
After=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/opt/floodresq
ExecStart=/opt/floodresq/.venv/bin/streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload && sudo systemctl enable --now floodresq
```
Put nginx in front (Streamlit needs WebSockets) — `/etc/nginx/sites-available/floodresq`:
```nginx
server {
    listen 80;
    server_name your.domain.example;
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```
```bash
sudo ln -s /etc/nginx/sites-available/floodresq /etc/nginx/sites-enabled/ && sudo nginx -t && sudo systemctl reload nginx
sudo apt install -y certbot python3-certbot-nginx && sudo certbot --nginx -d your.domain.example   # HTTPS
```
Size the VM with at least 2 vCPU / 8 GB RAM; a live run on a 2,500 km² box holds several 5,000 × 5,000 rasters in memory.

### 10d. Hosted notebooks / Streamlit Community Cloud
Possible for showing a cached demo, but free tiers (~1 GB RAM) are too small for live runs on large areas. Prefer
10a–10c for judging.

## 11. Judging-day checklist

1. **Day before:** run the Trishuli case study (`python -m pipeline.analyze --aoi config/example_aoi.json --date 2026-08-26`),
   then `python scripts/export_demo_run.py outputs/<run>` so *Load cached demo* shows real results offline.
2. Pre-warm caches for any area you expect (results land in `data/` and `outputs/`).
3. Keep `sentinel1.backend: planetary_computer` for live runs (minutes, no SNAP); start with boxes of a few hundred km².
4. Check the internet connection to `planetarycomputer.microsoft.com`, `api.ohsome.org` and
   `copernicus-dem-30m.s3.amazonaws.com` (DEM tiles).
5. `python scripts/audit_submission.py` must pass; `python scripts/validate_run.py outputs/<run> --forbid-mock` on the demo run.
6. If a live run fails, open the "What happened, step by step" panel: each stage shows a plain-language reason.

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `pip install rasterio` / `geopandas` fails on Windows | use the conda route (3b) or Python 3.11/3.12 where wheels exist |
| "No same-track before/after Sentinel-1 pair found" | the message lists every orbit track and what is missing; widen `sentinel1.search_window_days`, check the date, or the post-event scene is not published yet |
| "SNAP 'gpt' not found" | install ESA SNAP and set `SNAP_GPT`, or use `sentinel1.backend: planetary_computer` |
| "Downloading needs a free CDSE account" | fill `CDSE_USERNAME` / `CDSE_PASSWORD` in `.env` |
| log says `ohsome refused elements/geometry (HTTP 403)` | handled automatically: the client switches to ohsome's full-history endpoint for the same pre-event snapshot. If ohsome is down, the Overpass history (attic) API is used |
| "No hospital or town is mapped near this area" | the cut-off analysis falls back to "road exits" to the wider network and says so in the report |
| Flood map warning "AI model not used" | no `models/release/model_package` or no PyTorch: install `requirements-ml.txt` and copy the package |
| Area too large | max 2,500 km² (`limits.max_aoi_km2`); split the area |
| Report PDF missing | `pip install reportlab` (TXT and HTML reports are always written) |
| Map tab empty | `pip install folium streamlit-folium` |
| Stale results after changing code | `--no-cache`, or tick off "Reuse earlier results" in the dashboard's *Advanced* panel |

## 13. How the four parts were integrated

The repository follows T4's architecture (`pipeline/` contracts and scheduler). Each teammate's code was kept as a
library and connected through a small stage module:

| Teammate | Original | Now | Stage glue |
|---|---|---|---|
| T1 satellite data | `Part 1–5` (sentinel, sentinel1, alignment/dem/utm, sentinel2, osm) | `acquisition/sentinel.py`, `acquisition/dem.py`, `acquisition/sentinel2.py`, `acquisition/ohsome.py`, `preprocessing/*` | `acquisition/s1_stage.py` (acquire), `acquisition/osm.py` (osm) |
| T2 AI model | flat `*.py` files | `models/*` (core), `evaluation/compare.py`, `ems_reference.py`, `report_tools.py`, `model_case_study.py`, `model_audit.py`, `model_figures.py` | `models/stage.py` (infer) |
| T3 GIS | `part01–05_*.py` | `acquisition/overpass.py` (OSM normalisers + fallback), `gis/road_graph.py`, `gis/damage_overlay.py`, `gis/cutoff.py`, `gis/validation.py` | `gis/damage.py`, `gis/connectivity.py` |
| T4 integration | `part 1–5` zips | `pipeline/`, `dashboard/`, `reports/`, `scripts/`, `evaluation/`, `tests/` | — |

Changes made during integration (so nobody is surprised):
* **`models/eval_metrics.py` was missing** from T2's branch (imported by evaluation, inference and the case study).
  It was rebuilt from its call sites and `tests/test_eval_metrics.py`; all 33 of those tests pass. If the original
  file still exists on someone's laptop, diff the two.
* T3's Overpass downloader returned the *current* map (a data-rule breach for a past event). The OSM stage now uses
  T1's ohsome client (historical snapshot) and keeps Overpass only as a fallback with an **attic `[date:...]` query**.
* The public ohsome instance currently answers HTTP 403 for `elements/geometry`; T1's client now falls back to
  `elementsFullHistory/geometry` and keeps only the versions valid at the snapshot instant.
* New: `preprocessing/s1_rtc.py` (Planetary Computer backend, no SNAP), `models/change_detection.py` (non-AI fallback
  with debris detection), `pipeline/figures.py` (T4's figure stage was not written yet), `gis/flood_path.py` (bonus),
  a *Flood path* dashboard tab, `.env` loading, Dockerfile and Streamlit server config.
* Road segments use the contract's `<way id>#<k>` ids, produced by T3's splitter, so damage, connectivity and T4's
  oracle all speak about the same graph edges. The connectivity stage agrees with T4's independent oracle on the
  mock scenario (tested).
