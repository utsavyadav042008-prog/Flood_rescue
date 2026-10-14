"""Case study: compare OUR Trishuli result with the official reference (check-only).

    python -m evaluation.case_study --run outputs/<run> \
        --reference data/reference/<extent>.geojson [--reference-aoi data/reference/<aoi>.geojson] \
        --note "Disagreement on steep slopes: radar shadow."

What it does
------------
1. loads OUR flood mask and puts the reference polygons on the SAME pixel grid;
2. computes IoU / precision / recall / F1 (inside the reference's area of interest);
3. runs the SAME damage method on the reference mask in a scratch folder, so
   "reference-flagged" and "ours" differ only by the flood extent, not by method;
4. writes ``evaluation/case_study_results.json`` (read by the dashboard's Case-study
   tab) and a colour agreement map ``docs/figures/case_study_agreement.png``.

DATA RULE: reference files are CHECK-ONLY.  They are read here, never by the
pipeline; only plain numbers leave this module.  The reference path must lie
outside the run folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from evaluation import metrics, reference
from pipeline.config import REPO_ROOT
from pipeline.contracts import Artifact
from pipeline.errors import InputValidationError, PipelineError
from pipeline.fileio import read_json, write_json
from pipeline.run import RunPaths
from pipeline.schemas import utc_now_iso

RESULTS_PATH = REPO_ROOT / "evaluation" / "case_study_results.json"
FIGURE_PATH = REPO_ROOT / "docs" / "figures" / "case_study_agreement.png"
CREDIT = "European Union, Copernicus Emergency Management Service data"
METHOD_NOTE = ("Reference-flagged objects were obtained by applying OUR damage overlay to the reference "
               "flood extent, so both columns use the same method and the same pre-event OpenStreetMap data.")

GridLoader = Callable[[RunPaths, Artifact], tuple]       # -> (array, transform6, epsg)
DamageFn = Callable[[RunPaths], None]                    # fills a scratch run with affected_*.geojson
MaskWriter = Callable[[Path, np.ndarray, tuple, int], None]


def default_loader(run: RunPaths, artifact: Artifact) -> tuple:
    from dashboard.raster_overlay import _read_raster
    return _read_raster(run.path(artifact))


def default_mask_writer(path: Path, array: np.ndarray, transform: tuple, epsg: int) -> None:
    from affine import Affine
    from pipeline.fileio import write_geotiff
    write_geotiff(path, array.astype("uint8"), transform=Affine(*transform), epsg=epsg)


def default_damage(scratch: RunPaths) -> None:
    """Run T3's real damage stage on the scratch run (needs their implementation)."""
    from pipeline.stages import LazyEntrypoint
    from pipeline.contracts import CONTRACTS
    contract = CONTRACTS["damage"]
    LazyEntrypoint("damage", contract.entrypoint, contract.owner)(
        scratch, scratch.aoi, scratch.event_date, scratch.manifest().get("config") or {})


def assert_check_only(reference_path: Path, run: RunPaths) -> None:
    """Refuse reference files stored inside the run folder (they could leak into a pipeline stage)."""
    ref, root = Path(reference_path).resolve(), run.root.resolve()
    if root == ref or root in ref.parents:
        raise InputValidationError("Keep reference data outside the run folder (use data/reference/).")


def object_ids(run: RunPaths, artifact: Artifact) -> set:
    try:
        feats = read_json(run.path(artifact)).get("features", [])
    except (OSError, ValueError):
        return set()
    return {str((f.get("properties") or {}).get("object_id")) for f in feats}


def scratch_damage_ids(run: RunPaths, ref_mask: np.ndarray, transform: tuple, epsg: int, *,
                       write_mask: MaskWriter = default_mask_writer, damage_fn: DamageFn = default_damage) -> dict:
    """Apply the damage method to the REFERENCE mask in a temporary run folder.

    The scratch folder holds copies of the pre-event OSM layers and run.json plus the
    reference mask; nothing is written into the real run folder.
    """
    with tempfile.TemporaryDirectory(prefix="floodresq_ref_") as tmp:
        scratch = RunPaths(Path(tmp) / "scratch")
        scratch.root.mkdir()
        shutil.copy(run.path(Artifact.RUN_MANIFEST), scratch.path(Artifact.RUN_MANIFEST))
        (scratch.root / "osm").mkdir()
        for art in (Artifact.OSM_SNAPSHOT, Artifact.OSM_ROADS, Artifact.OSM_BRIDGES, Artifact.OSM_BUILDINGS,
                    Artifact.OSM_SETTLEMENTS, Artifact.OSM_HOSPITALS):
            if run.path(art).exists():
                shutil.copy(run.path(art), scratch.path(art))
        write_mask(scratch.path(Artifact.FLOOD_MASK), ref_mask, transform, epsg)
        damage_fn(scratch)
        return {key: object_ids(scratch, art) for key, art in (
            ("roads", Artifact.AFFECTED_ROADS), ("bridges", Artifact.AFFECTED_BRIDGES),
            ("buildings", Artifact.AFFECTED_BUILDINGS))}


def run_case_study(run: RunPaths, reference_extent: Path, *, reference_aoi: Optional[Path] = None,
                   name: str = "EMSR927", notes: tuple = (), infrastructure: bool = True,
                   out_json: Path = RESULTS_PATH, out_png: Optional[Path] = FIGURE_PATH,
                   loader: GridLoader = default_loader, write_mask: MaskWriter = default_mask_writer,
                   damage_fn: DamageFn = default_damage) -> dict:
    """Compute everything and write the results; returns the results dict."""
    assert_check_only(reference_extent, run)
    ours, transform, epsg = loader(run, Artifact.FLOOD_MASK)
    ours = ours > 0
    ref_mask = reference.rasterize_polygons(reference.load_reference_polygons(reference_extent),
                                            transform, ours.shape, epsg)
    valid = None
    if reference_aoi is not None:
        valid = reference.rasterize_polygons(reference.load_reference_polygons(reference_aoi),
                                             transform, ours.shape, epsg)
    pixel_km2 = abs(transform[0] * transform[4]) / 1e6
    extent = metrics.extent_metrics(ours, ref_mask, pixel_km2, valid)

    auto_notes = []
    a, r = extent["areas_km2"]["ours"], extent["areas_km2"]["reference"]
    if r:
        auto_notes.append(f"Inside the compared area our flood extent is {a:,.1f} km2 versus {r:,.1f} km2 in the "
                          f"reference ({100 * (a - r) / r:+.0f}%).")
    if valid is None:
        auto_notes.append("No reference area of interest was given, so pixels outside the reference's own "
                          "analysis area count as 'not flooded'; scores may be pessimistic.")

    infra = {}
    if infrastructure:
        ref_ids = scratch_damage_ids(run, ref_mask, transform, epsg, write_mask=write_mask, damage_fn=damage_fn)
        for key, art in (("roads", Artifact.AFFECTED_ROADS), ("bridges", Artifact.AFFECTED_BRIDGES),
                         ("buildings", Artifact.AFFECTED_BUILDINGS)):
            c = metrics.compare_ids(object_ids(run, art), ref_ids[key])
            infra[key] = {k: c[k] for k in ("reference_flagged", "also_flagged", "extra_flagged", "missed")}

    results = {
        "reference": name, "check_only": True, "credit": CREDIT, "method": METHOD_NOTE,
        "flood_extent": {k: (None if extent[k] is None else round(extent[k], 4))
                         for k in ("iou", "precision", "recall", "f1")},
        "pixel_counts": extent["counts"], "areas_km2": extent["areas_km2"],
        "infrastructure": infra, "notes": list(notes) + auto_notes, "generated_utc": utc_now_iso(),
        "run_id": run.manifest().get("run_id", ""),
    }
    write_json(out_json, results)
    if out_png is not None:
        png, bounds = metrics.agreement_png(metrics.agreement_codes(ours, ref_mask, valid), transform, epsg)
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        Path(out_png).write_bytes(png)
        Path(out_png).with_suffix(".json").write_text(json.dumps({"bounds": list(bounds)}), encoding="utf-8")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare our flood extent with the reference (check-only).")
    ap.add_argument("--run", required=True, help="our run folder (with flood_mask.tif)")
    ap.add_argument("--reference", required=True, help="reference flood-extent polygons (GeoJSON)")
    ap.add_argument("--reference-aoi", help="reference area-of-interest polygons (GeoJSON), recommended")
    ap.add_argument("--name", default="EMSR927")
    ap.add_argument("--note", action="append", default=[], help="a human-written observation (repeatable)")
    ap.add_argument("--no-infra", action="store_true", help="skip the road/bridge/building comparison")
    ap.add_argument("--out", default=str(RESULTS_PATH))
    args = ap.parse_args(argv)
    try:
        results = run_case_study(RunPaths.open(args.run), Path(args.reference),
                                 reference_aoi=Path(args.reference_aoi) if args.reference_aoi else None,
                                 name=args.name, notes=tuple(args.note), infrastructure=not args.no_infra,
                                 out_json=Path(args.out))
    except PipelineError as exc:
        print(f"ERROR: {exc.user_message}")
        return 1
    fe = results["flood_extent"]
    print(f"IoU {fe['iou']}  precision {fe['precision']}  recall {fe['recall']}  F1 {fe['f1']}")
    print(f"results -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
