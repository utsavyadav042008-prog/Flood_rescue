# What this system cannot do

FloodResQ is an **educational prototype**, not an operational tool. Read this before relying on any number.

## Time
- Sentinel-1 revisits an area only every few days (6 or 12 days on the same orbit track). The system shows the
  situation **at image time**, not now, and **cannot warn of a sudden event** such as the glacier collapse that
  started this flood; it works after the fact.
- Processing takes minutes, so even a fresh image is not real-time.

## Data
- Change detection needs a before/after pair from the **same orbit track**. Without such a pair the system returns
  no result instead of guessing.
- **Radar shadow and layover** in steep Himalayan valleys create false positives and false negatives; dark shadows can
  look like water.
- Radar cannot cleanly separate **water from mud and debris**, and wet snow can mimic water. Optical images (Sentinel-2)
  help only when the sky is clear, which is rare in monsoon season.
- **10 m pixels**: small buildings, footpaths and narrow roads are poorly resolved or invisible.
- **OpenStreetMap** coverage varies. Missing roads or buildings are missing from every result. We use only the
  snapshot from before the event, so roads built or changed since are unknown.

## Method
- "Affected" means **overlap with the detected flood or debris**. It does **not** confirm structural damage, and we
  report no casualties, population or economic impact (we have no data for them).
- "Cut off" means no road path to any hospital or town in the pre-event OSM network after removing the segments the
  satellite shows as affected. Footpaths, unmapped roads, air access and temporary repairs are ignored. A settlement
  that was already unreachable before the flood is reported separately, not as cut off by the flood.
- Impact fractions come from geometric overlap with a mask, not from field inspection.

## Model
- The segmentation model is trained on datasets that contain **few Himalayan scenes**; accuracy on new valleys is
  shown only by the held-out evaluation in the written report and can be much lower than on training regions.
- If no trained model package (or PyTorch) is available, the flood map comes from a **non-AI change-detection
  fallback** (VV darkening for water, VV brightening near water for debris). The run says so in its warnings and in
  `model_version` (`change-detection-baseline`); its "confidence" is a heuristic score, not a calibrated probability.
- Debris detection relies on radar **brightening** on gentle slopes near water. Wet vegetation, new construction or
  harvested fields can also brighten, and debris that is as rough as the ground it covered is missed.
- The Planetary Computer backend uses terrain-corrected RTC backscatter without the Refined Lee speckle filter of the
  SNAP chain, so results from the two backends differ slightly for the same pair.
- Model confidence is a probability from the network, not a guarantee; it can be confidently wrong.
- Results change with the chosen before/after pair and the model version (both are printed on every report).

## System
- Maximum area per run is limited (2,500 km2) to keep live demos fast; results for very large areas are not provided.
- Mock mode and mock data exist for development; anything marked MOCK is synthetic and must not be read as a result.
- A stage that fails is reported as "not available"; it is never shown as zero.

## Responsible use
The August 2026 disaster is recent and many families are still affected. The tool shows no images of people. Outputs
must not be used for rescue decisions without verification by qualified responders.
