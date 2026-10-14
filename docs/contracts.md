# Data contract (auto-generated - do not edit by hand)

Contract version: **1.0**. Source: `pipeline/contracts.py`.

## Stages

| Stage | Owner | Entry point | Reads | Writes |
|---|---|---|---|---|
| init (required) | T4 | - | - | `run.json` |
| acquire (required) | T1 | `acquisition.s1_stage:run_acquire` | `run.json` | `before.tif`<br>`after.tif`<br>`metadata.json` |
| osm | T3 | `acquisition.osm:run_osm` | `run.json` | `osm/snapshot.json`<br>`osm/roads.geojson`<br>`osm/bridges.geojson`<br>`osm/buildings.geojson`<br>`osm/settlements.geojson`<br>`osm/hospitals.geojson` |
| infer (required) | T2 | `models.stage:run_infer` | `before.tif`<br>`after.tif` | `flood_mask.tif`<br>`flood_confidence.tif` |
| damage | T3 | `gis.damage:run_damage` | `flood_mask.tif`<br>`osm/roads.geojson`<br>`osm/bridges.geojson`<br>`osm/buildings.geojson` | `affected_roads.geojson`<br>`affected_bridges.geojson`<br>`affected_buildings.geojson`<br>`damage_summary.json` |
| connectivity | T3 | `gis.connectivity:run_connectivity` | `osm/roads.geojson`<br>`osm/settlements.geojson`<br>`osm/hospitals.geojson`<br>`affected_roads.geojson`<br>`affected_bridges.geojson` | `cutoff_settlements.geojson`<br>`connectivity_summary.json` |
| summarize (required) | T4 | `pipeline.summary:run_summarize` | `metadata.json`<br>`osm/snapshot.json`<br>`flood_mask.tif`<br>`flood_confidence.tif`<br>`damage_summary.json`<br>`connectivity_summary.json` | `summary.json` |
| report | T4 | `reports.generator:run_report` | `summary.json` | `situation_report.txt`<br>`situation_report.pdf` (optional)<br>`situation_report.html` (optional)<br>`situation_report_ne.txt` (optional)<br>`situation_report_ne.html` (optional) |
| figures | T4 | `pipeline.figures:run_figures` | `before.tif`<br>`after.tif`<br>`flood_mask.tif`<br>`affected_roads.geojson`<br>`cutoff_settlements.geojson` | `figures/before.png`<br>`figures/after.png`<br>`figures/flood_map.png`<br>`figures/damage_map.png`<br>`figures/connectivity_map.png` |

Entry-point signature (all stages):

```python
def run_xxx(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None
```

## Files

| File | Kind | Must contain |
|---|---|---|
| `run.json` | json | Run identity: AOI, event date, config used, contract version. Keys: `contract_version`, `run_id`, `aoi`, `event_date`, `config_fingerprint`, `model_version`, `created_utc`, `config`. |
| `before.tif` | raster | Pre-event Sentinel-1 backscatter (band 1 = VV in dB), metric CRS. |
| `after.tif` | raster | Post-event Sentinel-1, SAME grid and orbit track as before.tif. |
| `metadata.json` | json | Scene facts for the report and the same-orbit check. Keys: `source`, `before_date`, `after_date`, `relative_orbit_before`, `relative_orbit_after`, `orbit_direction`, `crs`. |
| `osm/snapshot.json` | json | Which pre-event OSM snapshot was used (data-rule evidence). Keys: `snapshot_date`, `provider`. |
| `osm/roads.geojson` | geojson | Road segments (graph edges) before the event. Properties: `id`, `highway_type`, `bridge`, `name`. Geometry: LineString/MultiLineString. |
| `osm/bridges.geojson` | geojson | Bridge ways before the event. Properties: `id`, `name`. Geometry: LineString/MultiLineString. |
| `osm/buildings.geojson` | geojson | Building footprints before the event. Properties: `id`, `type`. Geometry: Polygon/MultiPolygon. |
| `osm/settlements.geojson` | geojson | Villages / towns whose connectivity is analysed. Properties: `id`, `name`. Geometry: Point/Polygon/MultiPolygon. |
| `osm/hospitals.geojson` | geojson | Destinations: hospitals and towns. Properties: `id`, `name`, `destination_type`. Geometry: Point/Polygon/MultiPolygon. |
| `flood_mask.tif` | raster | Binary flood/debris mask (0/1, no nodata), same grid as after.tif. |
| `flood_confidence.tif` | raster | Model probability of the flood class, float 0..1, same grid. |
| `affected_roads.geojson` | geojson | Road segments overlapping the flood mask. Properties: `object_id`, `object_type`, `impact_status`, `impact_fraction`, `confidence`. Geometry: LineString/MultiLineString. |
| `affected_bridges.geojson` | geojson | Bridges overlapping the flood mask. Properties: `object_id`, `object_type`, `impact_status`, `impact_fraction`, `confidence`. Geometry: LineString/MultiLineString. |
| `affected_buildings.geojson` | geojson | Buildings overlapping the flood mask. Properties: `object_id`, `object_type`, `impact_status`, `impact_fraction`, `confidence`. Geometry: Polygon/MultiPolygon. |
| `damage_summary.json` | json | Counts and lengths of affected objects. Keys: `affected_roads`, `affected_road_length_km`, `affected_bridges`, `affected_buildings`. |
| `cutoff_settlements.geojson` | geojson | Settlements with no road path to any destination. Properties: `settlement_id`, `name`, `reason`, `nearest_destination`, `blocking_segment_ids`. Geometry: Point/Polygon/MultiPolygon. |
| `connectivity_summary.json` | json | Cut-off counts and names. Keys: `cutoff_settlements`, `cutoff_names`, `settlements_total`, `destinations_used`. |
| `summary.json` | json | The single object read by dashboard AND report. Keys: `flood_area_km2`, `affected_roads`, `affected_road_length_km`, `affected_bridges`, `affected_buildings`, `cutoff_settlements`, `cutoff_names`, `mean_confidence`, `aoi_name`, `event_date`, `before_date`, `after_date`, `sentinel1_orbit`, `crs`, `model_version`, `osm_snapshot_date`, `processing_date`, `is_mock`, `warnings`. |
| `situation_report.txt` | text | One-page situation report. |
| `situation_report.pdf` | pdf | PDF of the report (English). |
| `situation_report.html` | text | Printable HTML of the report (English). |
| `situation_report_ne.txt` | text | Nepali report, plain text. |
| `situation_report_ne.html` | text | Nepali report, printable HTML. |
| `figures/before.png` | png | Presentation figure. |
| `figures/after.png` | png | Presentation figure. |
| `figures/flood_map.png` | png | Presentation figure. |
| `figures/damage_map.png` | png | Presentation figure. |
| `figures/connectivity_map.png` | png | Presentation figure. |
| `run_log.json` | json | Stage statuses and timings. |

Allowed `impact_status` values: `affected`, `impacted`, `potentially_damaged`.
