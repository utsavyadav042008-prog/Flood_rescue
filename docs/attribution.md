# Attribution, licences and citations

## Required attribution (appears in the dashboard footer, README, situation report and report)

Contains modified Copernicus Sentinel data 2026.

Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

© OpenStreetMap contributors.

## Datasets

| Data | Use | Licence / terms |
|---|---|---|
| Sentinel-1, Sentinel-2 (Copernicus Data Space) | before/after images | Copernicus free, full and open data policy |
| Copernicus DEM (WorldDEM-30) | terrain / flood path | see the required line above |
| OpenStreetMap, pre-event snapshot via ohsome API (`https://api.ohsome.org/v1`) | buildings, roads, bridges, settlements | ODbL; a published modified OSM extract must stay under ODbL |
| Kuro Siwo | model training | MIT licence per the NeurIPS 2024 paper (repository README also states CC BY); free use with credit |
| Sen1Floods11 (only if used) | model training | repository has no licence file; mirrors list CC BY 4.0; non-commercial educational use |
| Copernicus Emergency Management Service, activation EMSR927 | **checking results only** | free of charge; credit below |

## Citations

- **Kuro Siwo:** Bountos, N. I., Sdraka, M., Zavras, A., Karavias, A., Karasante, I., Herekakis, T., Thanasou, A.,
  Michail, D., Papoutsis, I. (2024). *Kuro Siwo: 33 billion m² under the water. A global multi-temporal satellite
  dataset for rapid flood mapping.* Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and
  Benchmarks Track. https://github.com/Orion-AI-Lab/KuroSiwo
- **Sen1Floods11** (cite only if it was used for training): Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E.
  (2020). *Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.*
  IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020.
  doi:10.1109/CVPRW50498.2020.00113. https://github.com/cloudtostreet/Sen1Floods11

## Credit for the reference maps (check-only comparison)

European Union, Copernicus Emergency Management Service data

These maps were used only to compare with our results in the case study. They were never an input to any step of the pipeline.
