# Data sources and how they are used

| Source | Where it enters | Notes |
|---|---|---|
| Sentinel-1 GRD (VV, VH), Copernicus Data Space | `acquisition/sentinel.py` (search, same-track pair, download) -> `preprocessing/sentinel1.py` (ESA SNAP) | backend `cdse_snap`; same relative orbit and direction enforced; recorded in `metadata.json` |
| Sentinel-1 RTC (VV, VH), Microsoft Planetary Computer `sentinel-1-rtc` | `preprocessing/s1_rtc.py` | backend `planetary_computer` (default without SNAP); the same Sentinel-1 acquisitions, already terrain-corrected; same pair-selection code as above |
| Sentinel-2 L2A | `acquisition/sentinel2.py`, `preprocessing/sentinel2.py` | optional (`sentinel2.enabled`), clear-sky pixels only; adds new-water / vegetation-loss evidence |
| Copernicus DEM GLO-30 | `acquisition/dem.py`, `preprocessing/alignment.py`, `gis/flood_path.py` | slope, radar layover/shadow mask, permanent-water guard, flood-path bonus |
| OpenStreetMap via ohsome API | `acquisition/ohsome.py` -> `acquisition/osm.py` | historical snapshot strictly **before the event** (config `osm.snapshot_date`), edit dates audited, stored in `osm/snapshot.json` and checked by the validator |
| OpenStreetMap via Overpass *attic* query | `acquisition/overpass.py` | automatic fallback if ohsome fails; asks for the map AS IT WAS on the same snapshot date (`[date:...]`) |
| Kuro Siwo (+ Sen1Floods11 if used) | `models/dataset.py`, `models/train.py` | model training only; cite Bountos et al., 2024 |
| Reference flood maps (EMS activation) | `evaluation/` only, check-only | never read by pipeline code (tested by `tests/test_hygiene_bootstrap_rules.py` and `scripts/audit_submission.py`) |

Rules in force: published damage maps and post-event OSM edits are never inputs. See `docs/attribution.md` for
licences and exact citations, and `docs/contracts.md` for the file formats between stages.
