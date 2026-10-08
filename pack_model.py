"""
pack_model.py  --  Part 4: turning a trained checkpoint into a shareable MODEL PACKAGE
=====================================================================================

Repo location: models/pack_model.py          (pure python: no PyTorch needed to package/verify)

THE GOAL
--------
One folder anybody on the team (or a judge) can download and use:

    model_package/
        best.pt                  the weights (Part 2 checkpoint: also holds the normalization stats)
        eval_protocol.json       the probability threshold chosen on VALIDATION (Part 3)
        model_card.md            one-page honest summary, GENERATED from real results
        package_manifest.json    SHA-256 fingerprint of every file above
        USAGE.txt                three lines showing how to load it

CONTENTS
    1. sha256_file / manifest         fingerprints
    2. package_model                  build the folder
    3. verify_package                 check it is complete and uncorrupted
    4. render_model_card              the generated model_card.md
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

MANIFEST_NAME = "package_manifest.json"
REQUIRED_FILES = ("best.pt", "eval_protocol.json", "model_card.md")

ATTRIBUTION = [
    "Contains modified Copernicus Sentinel data 2026.",
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH "
    "2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.",
    "© OpenStreetMap contributors.",
]


class PackageError(RuntimeError):
    """The model package is missing files or its files do not match the manifest."""


# =============================================================================
# 1. FINGERPRINTS
# =============================================================================
def sha256_file(path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in 1 MB chunks (checkpoints can be 100+ MB)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def build_manifest(folder) -> Dict:
    """{"files": {name: {"sha256", "bytes"}}, "created": ...} for every file
    in `folder` except the manifest itself."""
    folder = Path(folder)
    files = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.name != MANIFEST_NAME:
            files[p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return {"files": files, "created": time.strftime("%Y-%m-%d %H:%M:%S")}


# =============================================================================
# 2. BUILD THE PACKAGE
# =============================================================================
def load_checkpoint_meta(ckpt_path) -> Dict:
    """Read only the JSON metadata of a Part 2 checkpoint (imports PyTorch lazily)."""
    import torch
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover  (old PyTorch)
        ck = torch.load(ckpt_path, map_location="cpu")
    return json.loads(ck["meta_json"])


def package_model(ckpt_path, protocol_path, out_dir, results_json=None,
                  meta: Optional[Dict] = None) -> Path:
    """Create the package folder and return its path.

    ckpt_path     Part 2 best.pt
    protocol_path Part 3 eval_protocol.json (frozen threshold)
    results_json  Part 3 results.json; if given, the model card quotes its
                  numbers (pass the real one; never type metrics by hand)
    meta          pre-loaded checkpoint metadata (only for tests; normally
                  read from the checkpoint, which needs PyTorch)

    TRICK T6 (the model card cannot disagree with the evaluation):
        Every number in model_card.md is copied by code from results.json and
        eval_protocol.json.  Nobody retypes metrics, so a late retraining that
        changes the results cannot leave a stale number in the card, and the
        report, the card and the dashboard all quote the same figures.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_path, out / "best.pt")
    shutil.copy2(protocol_path, out / "eval_protocol.json")
    meta = meta if meta is not None else load_checkpoint_meta(ckpt_path)
    protocol = json.loads(Path(protocol_path).read_text())
    results = json.loads(Path(results_json).read_text()) if results_json else None
    (out / "model_card.md").write_text(
        render_model_card(meta, protocol, results, sha256_file(out / "best.pt")))
    (out / "USAGE.txt").write_text(
        "from inference import FloodPredictor\n"
        f"p = FloodPredictor.from_package('{out.name}')\n"
        "p.predict_files('before.tif', 'after.tif', 'outputs/')   # -> flood_mask.tif, flood_confidence.tif\n")
    (out / MANIFEST_NAME).write_text(json.dumps(build_manifest(out), indent=2))
    return out


# =============================================================================
# 3. VERIFY
# =============================================================================
def verify_package(folder) -> List[str]:
    """Return a list of problems (empty = package is complete and intact).

    TRICK T7 (detect the wrong or damaged weights early):
        Checkpoints travel through Drive, chat and USB sticks.  A half-downloaded
        file or an older best.pt silently gives different (worse) maps.  The
        manifest hash catches that BEFORE inference, with a clear message,
        instead of a mysterious accuracy drop during judging.
    """
    folder = Path(folder)
    problems: List[str] = []
    if not (folder / MANIFEST_NAME).exists():
        return [f"{MANIFEST_NAME} missing"]
    manifest = json.loads((folder / MANIFEST_NAME).read_text())
    for name in REQUIRED_FILES:
        if name not in manifest["files"]:
            problems.append(f"{name} not listed in the manifest")
    for name, info in manifest["files"].items():
        p = folder / name
        if not p.exists():
            problems.append(f"{name} is missing")
        elif sha256_file(p) != info["sha256"]:
            problems.append(f"{name} does not match its recorded SHA-256 (corrupted or replaced)")
    return problems


def read_package(folder) -> Dict:
    """Verify, then return {"checkpoint": path, "threshold": float, "protocol": dict}."""
    problems = verify_package(folder)
    if problems:
        raise PackageError("model package problem(s): " + "; ".join(problems))
    folder = Path(folder)
    protocol = json.loads((folder / "eval_protocol.json").read_text())
    return {"checkpoint": str(folder / "best.pt"), "threshold": float(protocol["threshold"]),
            "protocol": protocol}


# =============================================================================
# 4. MODEL CARD
# =============================================================================
def _f(v, nd: int = 3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):.{nd}f}"


def _results_table(results: Sequence[Dict]) -> str:
    lines = ["| Group | Method | Precision | Recall | IoU [95% CI] | F1 | Events |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        c, ci = r["counts"], r["ci_iou"]
        lines.append(f"| {r['name']} | {r['kind']} | {_f(c.get('precision'))} | {_f(c.get('recall'))} | "
                     f"{_f(c.get('iou'))} [{_f(ci.get('low'))}, {_f(ci.get('high'))}] | {_f(c.get('f1'))} | "
                     f"{r['n_events']} |")
    return "\n".join(lines)


def _top_failure_lines(results: Sequence[Dict]) -> List[str]:
    out = []
    for r in results:
        if r["kind"] != "model":
            continue
        for key, label in (("fp", "false alarms"), ("fn", "missed flood")):
            d = r["errors"][key]
            tot = sum(d.values())
            if tot:
                cause, n = max(d.items(), key=lambda kv: kv[1])
                out.append(f"- **{r['name']}**, {label}: mostly `{cause}` ({100 * n / tot:.0f}% of {tot} pixels)")
    return out


def render_model_card(meta: Dict, protocol: Dict, results: Optional[Sequence[Dict]],
                      ckpt_sha: str) -> str:
    """Build model_card.md as text.  Sections that need human judgement are the
    fixed 'Limitations' bullets; all numbers come from the inputs."""
    cfg = meta.get("config", {})
    norm = meta.get("norm", {}).get("meta", {})
    L: List[str] = []
    L += ["# Model card: FloodResQ flood segmentation (U-Net)", "",
          "> Educational prototype built for the Multimodal AI Hackathon 2026, Track B. "
          "**Not an operational tool.**", ""]
    L += ["## What it does", "",
          "Takes a BEFORE and an AFTER Sentinel-1 radar image (VV+VH, dB, same orbit track, same grid) and "
          "predicts, for every pixel, the probability that it was flooded by the event. Output: a flood "
          "mask (0/1, 255 = no data) and a confidence map.", ""]
    L += ["## Model", "",
          f"- Architecture: U-Net, encoder `{cfg.get('encoder', '?')}`, "
          f"{cfg.get('in_channels', '?')} input channels (before VV/VH, after VV/VH, change VV/VH), "
          f"{cfg.get('num_classes', '?')} classes (no water / permanent water / flood)",
          f"- Checkpoint epoch {meta.get('epoch', '?')}, trained {meta.get('created', '?')}",
          f"- Checkpoint SHA-256: `{ckpt_sha[:16]}…`",
          f"- Decision threshold: **{protocol['threshold']:.3f}** on P(flood), chosen on validation events "
          f"by `{protocol.get('selection_metric', '?')}` (frozen {protocol.get('frozen_at', '?')})", ""]
    L += ["## Training data", "",
          "- Kuro Siwo (Bountos et al., 2024, NeurIPS 2024 Datasets & Benchmarks). Take the full reference "
          "from the dataset repository README.",
          f"- Training files: {norm.get('n_files', '?')} patches; file-list fingerprint "
          f"`{norm.get('files_fingerprint', '?')}`",
          "- The splits are by flood EVENT, never by patch.", ""]
    L += ["## Evaluation (copied by code from Part 3 results.json)", ""]
    if results:
        L += [_results_table(results), "",
              "Validation rows are NOT honest scores (the threshold was chosen there). A confidence "
              "interval from fewer than 5 events is unreliable.", "", "### Most frequent failure causes", ""]
        L += _top_failure_lines(results) or ["- (no errors recorded)"]
        L.append("")
    else:
        L += ["_No results.json was supplied when this package was built._", ""]
    L += ["## Known limitations", "",
          "- Trained on flood WATER labels; debris/mud is not labelled, so debris detection is indirect and weaker.",
          "- Training data is mostly non-Himalayan; radar shadow in steep valleys can look like water.",
          "- Narrow corridors close to the 10 m pixel size can be missed or merged.",
          "- Wet snow, ice and other smooth surfaces can resemble water.",
          "- Needs a same-orbit BEFORE/AFTER pair; Sentinel revisit (days) means it cannot warn ahead of an event.",
          "- Output is 'satellite-derived change detected', not 'confirmed destroyed'.", ""]
    L += ["## Required attribution", ""] + [f"- {a}" for a in ATTRIBUTION] + [""]
    return "\n".join(L)


# =============================================================================
# COMMAND LINE
# =============================================================================
def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Part 4: model packaging")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="create a model package folder")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--protocol", required=True, help="Part 3 eval_protocol.json")
    p.add_argument("--results", default=None, help="Part 3 results.json")
    p.add_argument("--out", default="model_package")
    v = sub.add_parser("verify", help="check a package folder against its manifest")
    v.add_argument("folder")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        out = package_model(a.checkpoint, a.protocol, a.out, a.results)
        print(f"package written to {out}; problems: {verify_package(out) or 'none'}")
    else:
        probs = verify_package(a.folder)
        print("OK" if not probs else "PROBLEMS:\n- " + "\n- ".join(probs))
        raise SystemExit(1 if probs else 0)


if __name__ == "__main__":
    main()
