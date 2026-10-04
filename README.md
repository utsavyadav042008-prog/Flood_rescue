# FloodResQ — GIS, Damage & Connectivity Module

**Multimodal AI Hackathon 2026 · Track B: Mapping Flood Damage from Space**
**Event:** August 2026 Trishuli River flood, Nepal
**Owner:** Utsav (Teammate 3 — GIS / Damage / Connectivity Lead)

---

## 1. What this module does

Satellite-derived flood extents tell us *where the water is*. This module answers the two questions that matter to responders:

1. **What was hit?** Which roads, bridges, buildings and facilities fall inside the flood footprint.
2. **Who is cut off?** Which settlements can no longer reach a hospital, market or district headquarters once flooded roads and bridges are removed from the network.

It takes OpenStreetMap (OSM) data plus a flood-extent layer from the upstream teammates and produces damage overlays, a before/after road network, and a ranked list of isolated settlements.

---

## 2. Position in the pipeline

```
 Satellite imagery ──► Flood extent (from Teammates 1 & 2)
                              │
                              ▼
        ┌─────────────── THIS MODULE ───────────────┐
        │ 01 OSM Acquisition  →  02 Road Graph      │
        │          ↓                  ↓             │
        │ 03 Damage Overlay   →  04 Connectivity    │
        │                         ↓                 │
        │ 05 Validation, Case Study & Hand-off      │
        └───────────────────────────────────────────┘
                              │
                              ▼
          Maps, tables and layers for the final demo
```

**Input from upstream:** flood-extent polygons (GeoJSON / GeoPackage / raster-to-vector output), in a known CRS.
**Output to downstream:** damage layers, cut-off settlement list, summary statistics and map-ready files.

---

## 3. The five parts

| Part | Title | Purpose | Key output |
|------|-------|---------|-----------|
| 01 | OSM Data Acquisition & Normalisation | Download OSM data for the study area and clean it into consistent schemas | Normalised roads, bridges, buildings, settlements, facilities |
| 02 | Road Graph Construction | Turn road lines into a routable graph with travel-time weights | Pre-flood network graph |
| 03 | Damage Overlay (What Was Hit?) | Intersect flood extent with infrastructure | Flagged features and damage statistics |
| 04 | Connectivity & Cut-off Settlements (Who Is Cut Off?) | Remove damaged edges and test reachability | List of isolated settlements, detours and unreachable cases |
| 05 | Validation, Testing, Case Study & Hand-off | Sanity-check results, test the code, run the Trishuli case, package outputs | Validation report, case-study maps, hand-off bundle |

Each part is a standalone, readable file with full Python code, per-function documentation, and notes on any non-obvious tricks, so it can be studied and run independently.

---

## 4. Repository layout

> Adjust file names to match what is actually in the repo.

```
gis_module/
├── README.md
├── requirements.txt
├── part01_osm_acquisition.py
├── part02_road_graph.py
├── part03_damage_overlay.py
├── part04_connectivity.py
├── part05_validation_handoff.py
├── data/
│   ├── raw/          # untouched OSM downloads and flood extent
│   ├── processed/    # normalised layers
│   └── outputs/      # final damage + connectivity results
└── figures/          # maps for the demo and report
```

---

## 5. Setup

**Requirements:** Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install geopandas shapely pyproj osmnx networkx pandas numpy matplotlib
```

---

## 6. How to run

Run the parts in order. Each part reads the previous part's output.

```bash
python part01_osm_acquisition.py
python part02_road_graph.py
python part03_damage_overlay.py
python part04_connectivity.py
python part05_validation_handoff.py
```

Before the first run, set the study-area boundary and the path to the flood-extent file at the top of Part 01 and Part 03.

---

## 7. Inputs and outputs

**Inputs**
- Study-area boundary (Trishuli corridor, Nepal)
- OSM data (downloaded in Part 01)
- Flood-extent polygons from the satellite analysis team

**Outputs**
- Normalised infrastructure layers
- Routable road graph (pre-flood and post-flood)
- Damage overlay: flooded roads, bridges, buildings, facilities
- Cut-off settlement list with the reason each is isolated
- Summary tables (counts, lengths, affected population proxies where available)
- Map figures for the demo

---

## 8. Design notes and caveats

- **CRS discipline.** All geometry is reprojected to a metric CRS before measuring lengths or buffering; results are exported back to EPSG:4326 for web maps.
- **OSM completeness varies.** Rural Nepal has uneven road and building coverage, so "not cut off" means "not cut off *according to mapped roads*". Unmapped trails are not counted.
- **Flood extent is an input, not ground truth.** Damage results are only as good as the upstream flood mask, including its cloud and shadow errors.
- **Road status is binary.** A road is treated as passable or impassable. Partial damage and depth-based passability are out of scope.
- **Bridges matter most.** A single flooded bridge can isolate a whole valley, so bridges are handled explicitly in Parts 03 and 04.

---

## 9. Validation

Part 05 covers:
- Unit tests for the core functions (geometry cleaning, graph build, overlay logic)
- Sanity checks (feature counts before and after normalisation, graph connectivity before the flood)
- Spot-checks of flagged features against imagery or known reports
- The Trishuli case study with maps

---

## 10. Hand-off

At the end of Part 05, the following are handed to the team for the final demo and report:
- `data/outputs/` (damage and connectivity layers)
- `figures/` (maps)
- Summary statistics table
- A short list of known limitations (Section 8)

---

## 11. Team

| Role | Owner |
|------|-------|
| GIS / Damage / Connectivity Lead (Teammate 3) | Utsav |
| Other teammates | See the main project README / team plan |