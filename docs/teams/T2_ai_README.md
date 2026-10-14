# FloodResQ: AI / Computer Vision module (Teammate 2)

Multimodal AI Hackathon 2026, **Track B: Mapping Flood Damage from Space** (case study: the August 2026 Trishuli flood, Nepal).

This folder is the AI part of the team's system: it learns to find flooded pixels from Sentinel-1 radar (before and after the event), proves how well it works, and wraps the model so the rest of the team can call it with one function.

> Educational prototype, **not an operational tool**. Do not use it for rescue decisions.

---

## What it does

```
before.tif + after.tif (Sentinel-1 VV,VH in dB, same grid, same orbit)
        │
        ▼
   U-Net (6 input channels → no water / permanent water / flood)
        │
        ▼
 flood_mask.tif  +  flood_confidence.tif  +  inference_meta.json
```

| Part | Folder | What it covers |
|---|---|---|
| 1. Data & inputs | `part1_code/` | reads Kuro Siwo patches, builds the 6-channel input, splits by flood event, normalization stats |
| 2. Model & training | `part2_code/` | U-Net (ResNet34), loss, training loop, checkpoints, experiment log |
| 3. Evaluation & metrics | `part3_code/` | IoU / F1 / precision / recall, confidence intervals, non-AI baseline, error causes, Himalayan test |
| 4. Inference & packaging | `part4_code/` | tiled prediction on big images, GeoTIFF outputs, model package with frozen threshold |
| 5. Case study & report | `part5_code/` | EMSR927 comparison, limitations, attribution, Q&A, final audit |

Each part has a matching guide: `01_data_and_inputs.md` … `05_case_study_limitations_and_report.md`. They explain every function and the non-obvious tricks. Read the guide first, then the code.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib pillow pyyaml
pip install torch segmentation-models-pytorch           # training and real inference
pip install rasterio pyproj                             # real GeoTIFFs and reprojecting EMSR927
pip install pypdf geopandas                             # optional: PDF page check, shapefile reference
```

Python 3.10+ is recommended. A GPU (Colab or Kaggle is enough) is needed for training; inference falls back to CPU.

---

## Folder layout

This handout keeps one folder per part. The parts import each other by path, so keep the folders **side by side**. In the final team repo, copy all `.py` files into `models/` (flat) and the tests into `tests/`; the imports work in both layouts.

```
T2_AI_workflow/
  README.md
  01_… 05_*.md           guides for each part
  part1_code/            dataset.py, test_dataset.py
  part2_code/            train.py, train_utils.py, config_train.yaml, test_train_utils.py
  part3_code/            evaluate.py, eval_metrics.py, test_eval_metrics.py
  part4_code/            inference.py, tiling.py, geo_io.py, pack_model.py, test_inference.py
  part5_code/            case_study.py, compare.py, ems_reference.py, figures.py,
                         report_tools.py, audit.py, test_part5.py
```

---

## Quick check (no data, no GPU, a few minutes)

Every part has a self-test on synthetic data. Run these first to confirm your setup:

```bash
cd part1_code && python test_dataset.py && python dataset.py demo --out demo_out && cd ..
cd part2_code && python test_train_utils.py && python train.py selftest && cd ..      # needs PyTorch + smp
cd part3_code && python test_eval_metrics.py && python evaluate.py demo --out demo_eval && cd ..
cd part4_code && python test_inference.py && python inference.py demo --out demo_infer && cd ..
cd part5_code && python test_part5.py && python case_study.py demo --out demo_case && cd ..
```

Each demo ends with `PART n OK`. Test counts: 11, 20, 33, 26, 24.

---

## Run the real pipeline

**1. Data (Day 1–2).** Download a few Kuro Siwo events, then adapt `read_patch_npz` and `DEFAULT_LABEL_LUT` in `part1_code/dataset.py` to the real file layout (see `01_data_and_inputs.md`, section 3). Keep Himalayan / mountain events out of training:

```bash
python part1_code/dataset.py scan --root /data/KuroSiwo_npz --out config --force-test <himalayan_event_folder> ...
```

**2. Train (Day 3–6).**

```bash
cp part2_code/config_train.yaml config/train.yaml        # set data_root
python part2_code/train.py forward-check --config config/train.yaml
python part2_code/train.py overfit       --config config/train.yaml    # loss must collapse
python part2_code/train.py train         --config config/train.yaml
python part2_code/train.py train --config config/train.yaml --set lr=3e-4 run_name=lr_3e-4
python part2_code/train.py table                                       # experiment table
```

**3. Evaluate (Day 7–10).** Chooses the threshold on validation only, then freezes it:

```bash
python part3_code/evaluate.py run --checkpoint models/weights/baseline/best.pt \
    --splits config/splits.json --data-root /data/KuroSiwo_npz \
    --himalayan-events <himalayan_event_folder> --out outputs/eval
```

**4. Package and predict (Day 8–10).**

```bash
python part4_code/pack_model.py build --checkpoint models/weights/baseline/best.pt \
    --protocol outputs/eval/eval_protocol.json --results outputs/eval/results.json --out model_package
python part4_code/pack_model.py verify model_package
python part4_code/inference.py predict --before before.tif --after after.tif --out outputs/ --package model_package
```

From Python (what Teammates 3 and 4 call):

```python
from inference import predict_flood
res = predict_flood("before.tif", "after.tif", "outputs/", package="model_package")
res["mask"], res["confidence"], res["summary"], res["warnings"]
```

**5. Case study and audit (Day 11–13).** EMSR927 is used **only to check** results:

```bash
python part5_code/case_study.py run --before before.tif --after after.tif --package model_package \
    --reference emsr927_flood.geojson --footprint emsr927_aoi.geojson \
    --eval-results outputs/eval/results.json --splits config/splits.json --out outputs/case
python part5_code/audit.py --package model_package --results outputs/eval/results.json \
    --case outputs/case/comparison.json --readme README.md --report report.md \
    --limitations outputs/case/docs/limitations.md --attribution outputs/case/docs/attribution.md \
    --figures outputs/case/figures --splits config/splits.json --out audit_report.md
```

---

## Outputs (contract with the rest of the team)

| File | Format | Meaning |
|---|---|---|
| `flood_mask.tif` | uint8 | `1` flood, `0` not flood, `255` no data |
| `flood_confidence.tif` | float32, 0–1 | raw model probability of flood |
| `inference_meta.json` | JSON | model hash, threshold and its source, tiling, statistics, warnings |
| `model_package/` | folder | weights, frozen threshold, generated model card, SHA-256 manifest |

Both rasters have exactly the input's size, CRS and transform. Use the mask for polygons and statistics, the confidence for shading.

---

## Data rules (read this)

- **Allowed inputs:** Sentinel-1, Sentinel-2, Copernicus DEM, OpenStreetMap as it was before the event, and the listed training datasets.
- **Not allowed as inputs:** Copernicus EMS (EMSR927), UNOSAT or any published damage map, and OpenStreetMap edits made after the event. They may only be used to **check** results. Breaking this rule leads to disqualification.
- The code enforces this: the case study refuses inputs that look like or are byte-identical to reference files, locks the prediction before the reference is opened, and refuses a hand-picked threshold. These controls make accidents visible; they are not proof against deliberate cheating.
- Splits are by flood **event**, never by patch. The decision threshold is chosen on validation data only.

---

## Known limitations (short version)

The full, auto-generated list is in `limitations.md` after the case study.

- Training labels mark flood **water**, not debris; debris detection is indirect and weaker.
- Training data is mostly non-Himalayan; radar shadow in steep valleys can look like water.
- Needs a same-orbit before/after pair. Sentinel satellites revisit only every few days, so the system maps damage after an event and **cannot warn ahead**.
- Narrow corridors near the 10 m pixel size can be missed; wet snow and ice can look like water.
- Results on the Himalayan test set may rest on very few events; read the confidence intervals.

---

## Testing status (be honest in Q&A)

- All five parts pass their synthetic-data tests and demos.
- **Not yet verified on real data:** the PyTorch code (model, training loop, checkpoint loading), real GeoTIFF reading/writing with `rasterio`, reprojection with `pyproj`, shapefile reading, and anything involving real Kuro Siwo or EMSR927 files. The first real runs are the true tests.
- Dataset file layout, label codes and the EMSR927 format are **assumptions** written before seeing the real files.
- The dB cut-offs in the error-cause table (−18 dB "dark before", 3 dB "weak change") are heuristics.

---

## Attribution

Contains modified Copernicus Sentinel data 2026.

Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

© OpenStreetMap contributors.

Training data: Kuro Siwo (Bountos et al., 2024, NeurIPS 2024 Datasets and Benchmarks). Sen1Floods11 (Bonafilia et al., CVPR Workshops 2020) if used. *Check full references against each dataset's repository README before submitting.*

Reference data, used for checking only: European Union, Copernicus Emergency Management Service data (EMSR927).

Software: PyTorch, segmentation-models-pytorch, NumPy, SciPy, Matplotlib, Pillow, rasterio. Exact versions are written to `docs/attribution.md` by the case-study run.

---

## Team

Role: Teammate 2, AI / Computer Vision Lead. Team: FloodResQ. Author: _your name here_.
