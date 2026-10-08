"""
inference.py  --  Part 4 of Teammate 2's workflow: INFERENCE & PACKAGING
========================================================================

Repo location: models/inference.py
Needs: geo_io.py, tiling.py, pack_model.py (this folder), Part 1 dataset.py,
       Part 3 eval_metrics.py; Part 2 train.py + PyTorch ONLY for real checkpoints.

WHAT THIS FILE DOES
-------------------
Turns the trained model into ONE function the rest of the team can call without
knowing anything about PyTorch:

    before.tif + after.tif  ──►  predict_flood()  ──►  flood_mask.tif
                                                       flood_confidence.tif
                                                       inference_meta.json

THE PIPELINE (inside FloodPredictor.predict_files)
--------------------------------------------------
    read_raster x2            geo_io.py      georeferenced arrays
    check_same_grid           geo_io.py      refuse mismatched grids
    make_raw_patch            Part 1         SAME cleaning as training
    tile + predict + stitch   tiling.py      overlap + smooth blending
    postprocess               below          threshold, river veto, speckle removal
    write_raster x2           geo_io.py      same georeference as the input
    inference_meta.json       below          provenance, stats, warnings

OUTPUT CONTRACT (agree with Teammates 3 and 4)
    flood_mask        uint8:   1 = flood, 0 = not flood, 255 = no data
    flood_confidence  float32: model P(flood) in [0,1] (raw, NOT post-processed)
    Both have exactly the input's size, CRS and transform.
    Use the MASK for polygons/statistics, the CONFIDENCE for shading/uncertainty.

HOW TO READ THIS FILE: sections 1..7 in order. "TRICK Tn" marks non-obvious ideas;
the same list is in 04_inference_and_packaging.md.

COMMANDS
    python inference.py demo --out demo_out
    python inference.py predict --before b.tif --after a.tif --out outputs/ --package model_package
    python pack_model.py build --checkpoint best.pt --protocol eval_protocol.json --results results.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "part1_code", _HERE.parent / "part2_code", _HERE.parent / "part3_code"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dataset import Normalizer, build_input, make_raw_patch  # noqa: E402
from eval_metrics import counts_from_arrays, remove_small_blobs  # noqa: E402
from geo_io import (RasterMeta, check_same_grid, make_npz_raster, read_raster,  # noqa: E402
                    validate_outputs, write_raster)
from geo_io import _is_tif  # noqa: E402
from pack_model import ATTRIBUTION, PackageError, package_model, read_package, sha256_file, verify_package  # noqa: E402
from tiling import ModelFn, pad_to_min, predict_tiled, safe_call, with_tta  # noqa: E402

NO_WATER, PERM_IDX, FLOOD_IDX = 0, 1, 2
N_CLASSES = 3
MASK_NODATA = 255
SCHEMA_VERSION = 1


# =============================================================================
# 1. SMALL CHECKS
# =============================================================================
def detect_unit(arr: np.ndarray) -> str:
    """Guess whether radar values are 'db' or 'linear'.

    TRICK T15: dB backscatter of land/water is almost always NEGATIVE (about -8
    to -25), linear backscatter is always POSITIVE (0..1).  So the sign of the
    median tells them apart.  A linear image fed to a model trained on dB gives
    plausible-looking nonsense, so we detect it instead of trusting a flag.
    Pass unit="db"/"linear" explicitly to override.
    """
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        raise ValueError("input raster has no valid pixels")
    if v.size > 1_000_000:
        v = v[:: v.size // 1_000_000]
    return "linear" if float(np.median(v)) > 0 else "db"


def sanity_warnings(invalid_fraction: float, median_shift_db: Optional[float],
                    flood_fraction: Optional[float] = None) -> List[str]:
    """Human-readable warnings for results that look suspicious.

    TRICK T12 (cheap sanity checks that catch the commonest input mistakes):
      * Most of a scene does not change, so the MEDIAN change in VV should be
        close to 0 dB.  A median shift above 3 dB means the two images differ
        globally (different orbit track, different acquisition mode, or
        different calibration), and then the 'change' is not a flood signal.
      * A real flood rarely covers more than half of the valid area; if the map
        says so, suspect misaligned or mismatched inputs before believing it.
    These are WARNINGS (written to the meta file and printed), never silent.
    """
    msgs: List[str] = []
    if invalid_fraction > 0.5:
        msgs.append(f"{100 * invalid_fraction:.0f}% of pixels have no data")
    if median_shift_db is not None and abs(median_shift_db) > 3.0:
        msgs.append(f"median VV change is {median_shift_db:+.1f} dB (expected about 0): the pair may not be "
                    f"from the same orbit track or acquisition mode")
    if flood_fraction is not None:
        if flood_fraction > 0.5:
            msgs.append(f"{100 * flood_fraction:.0f}% of valid pixels are flagged as flood: suspiciously large; "
                        f"check alignment and orbit")
        elif flood_fraction == 0:
            msgs.append("no flood pixels found")
    return msgs


# =============================================================================
# 2. POST-PROCESSING
# =============================================================================
def fill_small_holes(mask: np.ndarray, max_pixels: int) -> np.ndarray:
    """Fill tiny 'not flood' holes completely enclosed by flood (max_pixels or
    fewer pixels).  Holes that touch the image border are never filled (they may
    be the edge of a big dry area)."""
    from scipy import ndimage
    if max_pixels <= 0:
        return mask.copy()
    labels, n = ndimage.label(~mask)
    if n == 0:
        return mask.copy()
    sizes = np.bincount(labels.ravel())
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    fill = sizes <= max_pixels
    fill[border] = False
    fill[0] = False
    return mask | fill[labels]


def postprocess(prob: np.ndarray, invalid: np.ndarray, threshold: float,
                min_blob_pixels: int = 8, fill_hole_pixels: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Class probabilities [3,H,W] -> (mask uint8 [H,W], confidence float32 [H,W]).

    Steps:  P(flood) >= threshold  AND  not predicted permanent water
            -> delete specks smaller than min_blob_pixels
            -> fill holes up to fill_hole_pixels
            -> pixels with no data are forced to 255 in the mask.

    TRICK T8 (two safeguards):
      (a) "Permanent-water veto": a pixel whose most likely class is permanent
          water is never flood, even if P(flood) passes a LOW threshold.  With a
          threshold >= 0.5 this is automatic; the veto keeps the behaviour safe
          if the validation threshold ends up below 0.5.
      (b) The confidence map is the RAW P(flood), not post-processed.  The mask
          may remove a speck that the confidence still shows faintly: that is
          deliberate, so uncertainty stays visible instead of being cleaned away.
    """
    pf = prob[FLOOD_IDX]
    flood = (pf >= threshold) & (prob.argmax(axis=0) != PERM_IDX)
    flood = remove_small_blobs(flood, min_blob_pixels)
    flood = fill_small_holes(flood, fill_hole_pixels)
    flood &= ~invalid
    mask = flood.astype(np.uint8)
    mask[invalid] = MASK_NODATA
    conf = np.clip(pf, 0.0, 1.0).astype(np.float32)
    conf[invalid] = 0.0
    return mask, conf


# =============================================================================
# 3. THE PREDICTOR
# =============================================================================
@dataclass
class Prediction:
    prob: np.ndarray            # [3, H, W] float32: no water / permanent water / flood
    invalid: np.ndarray         # [H, W] bool: no usable radar data
    unit: str
    invalid_fraction: float = 0.0
    median_shift_db: Optional[float] = None     # median (after - before) VV over valid pixels
    seconds: float = 0.0


class FloodPredictor:
    """Loads the model ONCE and predicts any number of scenes.

    Create it with one of:
        FloodPredictor.from_package("model_package")      <- normal use
        FloodPredictor.from_checkpoint("best.pt")         <- while developing
        FloodPredictor(model_fn, normalizer)              <- tests (any numpy model)

    `model_fn(batch)` takes float32 [N,6,T,T] and returns probabilities
    [N,3,T,T].  Keeping PyTorch behind that one function is what lets the whole
    pipeline be tested without it.
    """

    def __init__(self, model_fn: ModelFn, normalizer: Normalizer, *, tile: int = 224,
                 overlap: int = 32, batch_size: int = 8, tta: bool = False,
                 default_threshold: float = 0.5, threshold_source: str = "default (0.5)",
                 model_info: Optional[Dict] = None):
        self.norm = normalizer
        self.tile, self.overlap, self.batch_size, self.tta = tile, overlap, batch_size, tta
        fn = with_tta(model_fn) if tta else model_fn
        self._fn: ModelFn = lambda batch: safe_call(fn, batch)        # TRICK T4: survives GPU out-of-memory
        self.default_threshold = default_threshold
        self.threshold_source = threshold_source
        self.model_info = model_info or {}

    # ---- constructors needing PyTorch ----
    @classmethod
    def from_checkpoint(cls, ckpt_path, device: Optional[str] = None, **kw) -> "FloodPredictor":
        """Load a Part 2 checkpoint.  Normalization stats come from INSIDE it."""
        from train import load_checkpoint          # Part 2
        dev = pick_device(device)
        model, meta = load_checkpoint(ckpt_path, dev)
        nm = meta["norm"]
        norm = Normalizer(np.array(nm["mean"], np.float32), np.array(nm["std"], np.float32), nm.get("meta", {}))
        cfg = meta.get("config", {})
        info = {"checkpoint": str(ckpt_path), "sha256": sha256_file(ckpt_path), "device": dev,
                "epoch": meta.get("epoch"), "encoder": cfg.get("encoder"),
                "in_channels": cfg.get("in_channels"), "created": meta.get("created")}
        return cls(TorchModelFn(model, dev), norm, model_info=info, **kw)

    @classmethod
    def from_package(cls, folder, device: Optional[str] = None, **kw) -> "FloodPredictor":
        """Verify the package (hashes), load its weights, and use ITS threshold.

        TRICK T14 (the threshold travels with the model): the number chosen on
        validation in Part 3 sits in the package next to the weights, so nobody
        can run the model with a threshold from a different checkpoint, and the
        case study uses exactly the frozen value.
        """
        info = read_package(folder)
        kw.setdefault("default_threshold", info["threshold"])
        kw.setdefault("threshold_source", "package eval_protocol.json (frozen on validation)")
        return cls.from_checkpoint(info["checkpoint"], device, **kw)

    # ---- prediction on arrays ----
    def predict_arrays(self, before: np.ndarray, after: np.ndarray, unit: str = "auto") -> Prediction:
        """before/after: float [2,H,W] (VV, VH).  Returns class probabilities.

        TRICK T11 (one code path for training and inference): the cleaning and
        input building call Part 1's `make_raw_patch` and `build_input`, the SAME
        functions the training data went through (units, no-data handling,
        clipping, change channels, normalization). There is no second
        implementation that could drift out of sync.
        """
        t0 = time.time()
        before, after = np.asarray(before, np.float32), np.asarray(after, np.float32)
        if before.ndim != 3 or before.shape[0] != 2 or before.shape != after.shape:
            raise ValueError(f"before/after must both be [2,H,W]; got {before.shape} and {after.shape}")
        h, w = before.shape[1:]
        if unit == "auto":
            unit = detect_unit(before)
            if detect_unit(after) != unit:
                raise ValueError("before and after seem to use different units (dB vs linear)")
        raw = make_raw_patch(before, after, np.zeros((h, w), np.uint8), unit=unit)

        T = self.tile
        pb, _ = pad_to_min(raw.before, T)
        pa, _ = pad_to_min(raw.after, T)
        pinv, _ = pad_to_min(raw.invalid, T)
        H, W = pb.shape[1:]

        def build_tile(y: int, x: int) -> np.ndarray:                  # TRICK T3 (tiling.py)
            return build_input(pb[:, y:y + T, x:x + T], pa[:, y:y + T, x:x + T],
                               pinv[y:y + T, x:x + T], self.norm)

        prob = predict_tiled(build_tile, self._fn, H, W, N_CLASSES, T, self.overlap, self.batch_size)
        prob = np.ascontiguousarray(prob[:, :h, :w])
        valid = ~raw.invalid
        shift = float(np.median((raw.after[0] - raw.before[0])[valid])) if valid.any() else None
        return Prediction(prob, raw.invalid, unit, float(raw.invalid.mean()), shift, time.time() - t0)

    def as_scene_predictor(self) -> Callable:
        """Adapter for Part 3: Scene -> P(flood) map, through THIS full pipeline
        (tiling and blending included), so evaluation numbers describe what
        inference really does.  Post-processing is not applied (Part 3
        evaluates raw probabilities against a threshold)."""
        def predict(scene) -> np.ndarray:
            p = self.predict_arrays(scene.before, scene.after, unit="db")
            out = p.prob[FLOOD_IDX].copy()
            out[scene.invalid] = 0.0
            return out
        return predict

    def resolve_threshold(self, threshold: Optional[float]) -> Tuple[float, str]:
        if threshold is None:
            return float(self.default_threshold), self.threshold_source
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1 (exclusive)")
        return float(threshold), "argument"

    # ---- prediction on files ----
    def predict_files(self, before_path, after_path, out_dir, threshold: Optional[float] = None,
                      unit: str = "auto", min_blob_pixels: int = 8, fill_hole_pixels: int = 8) -> Dict:
        """The full contract: two rasters in, two rasters + a meta file out.
        Returns {"mask": path, "confidence": path, "meta": path, "summary": dict}."""
        timings: Dict[str, float] = {}
        t = time.time()
        rb, ra = read_raster(before_path), read_raster(after_path)
        check_same_grid(rb.meta, ra.meta)
        for name, r in (("before", rb), ("after", ra)):
            if r.data.shape[0] != 2:
                raise ValueError(f"{name} raster must have 2 bands (VV, VH); it has {r.data.shape[0]}")
        timings["read_s"] = time.time() - t

        pred = self.predict_arrays(rb.data, ra.data, unit)
        timings["predict_s"] = pred.seconds
        thr, thr_src = self.resolve_threshold(threshold)

        t = time.time()
        mask, conf = postprocess(pred.prob, pred.invalid, thr, min_blob_pixels, fill_hole_pixels)
        valid_n = int((~pred.invalid).sum())
        flood_n = int((mask == 1).sum())
        flood_frac = flood_n / valid_n if valid_n else float("nan")
        warns = sanity_warnings(pred.invalid_fraction, pred.median_shift_db,
                                flood_frac if valid_n else None)
        timings["postprocess_s"] = time.time() - t

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ext = ".tif" if _is_tif(before_path) else ".npz"
        mask_path, conf_path = out / f"flood_mask{ext}", out / f"flood_confidence{ext}"
        tags = {"producer": "FloodResQ Part 4", "threshold": f"{thr:.4f}",
                "model_sha256": self.model_info.get("sha256", "unknown"),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mask_values": "1=flood 0=not flood 255=no data"}
        t = time.time()
        write_raster(mask_path, mask, rb.meta, np.uint8, nodata=MASK_NODATA, tags=tags)
        write_raster(conf_path, conf, rb.meta, np.float32, nodata=None, tags=tags)
        timings["write_s"] = time.time() - t

        px = rb.meta.pixel_size_m
        summary = {
            "valid_pixels": valid_n, "flood_pixels": flood_n, "flood_fraction": flood_frac,
            "flood_area_km2": (flood_n * px * px / 1e6) if px else None,
            "mean_confidence_in_flood": float(conf[mask == 1].mean()) if flood_n else None}
        meta = {
            "schema_version": SCHEMA_VERSION,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "inputs": {"before": Path(before_path).name, "after": Path(after_path).name,
                       "unit": pred.unit,
                       "grid": {"width": rb.meta.width, "height": rb.meta.height, "crs": rb.meta.crs,
                                "pixel_size_m": rb.meta.pixel_size_m}},
            "model": self.model_info,
            "threshold": thr, "threshold_source": thr_src,
            "postprocess": {"min_blob_pixels": min_blob_pixels, "fill_hole_pixels": fill_hole_pixels,
                            "permanent_water_veto": True},
            "tiling": {"tile": self.tile, "overlap": self.overlap, "batch_size": self.batch_size,
                       "tta": self.tta},
            "summary": summary, "timings_s": {k: round(v, 3) for k, v in timings.items()},
            "warnings": warns,
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
            "attribution": ATTRIBUTION,
        }
        meta_path = out / "inference_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        for msg in warns:
            warnings.warn(msg)
        return {"mask": str(mask_path), "confidence": str(conf_path), "meta": str(meta_path),
                "summary": summary, "warnings": warns}


# =============================================================================
# 4. PYTORCH GLUE  (the only code here that needs PyTorch)
# =============================================================================
def pick_device(requested: Optional[str] = None) -> str:
    """'cuda' if a GPU exists, else 'cpu'.  An explicit 'cuda' request on a
    machine without a GPU falls back to CPU with a warning (judges' laptops!)."""
    import torch
    if requested in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available; falling back to CPU")
        return "cpu"
    return requested


class TorchModelFn:
    """numpy [N,6,T,T] in -> numpy probabilities [N,3,T,T] out."""

    def __init__(self, model, device: str):
        self.model, self.device = model, device

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
            return self.model(x).softmax(dim=1).float().cpu().numpy()


# =============================================================================
# 5. THE ONE-LINE FUNCTION FOR TEAMMATES
# =============================================================================
_CACHE: Dict[Tuple, FloodPredictor] = {}


def predict_flood(before_path, after_path, out_dir, package: str = "model_package",
                  threshold: Optional[float] = None, device: Optional[str] = None,
                  tta: bool = False, **kw) -> Dict:
    """Teammates call THIS.  `package` is a model_package folder (preferred) or a
    bare .pt checkpoint.  Returns {"mask", "confidence", "meta", "summary", "warnings"}.

    TRICK T13 (load once): building the model takes seconds (reading weights,
    moving to the GPU).  The dashboard calls this function for every new area, so
    predictors are cached by (package, device, tta, manifest mtime) and reused;
    replacing the package on disk automatically invalidates the cache.
    """
    p = Path(package)
    stamp = (p / "package_manifest.json").stat().st_mtime if p.is_dir() else p.stat().st_mtime
    key = (str(p.resolve()), device, tta, stamp)
    if key not in _CACHE:
        _CACHE[key] = (FloodPredictor.from_package(p, device, tta=tta) if p.is_dir()
                       else FloodPredictor.from_checkpoint(p, device, tta=tta))
    return _CACHE[key].predict_files(before_path, after_path, out_dir, threshold, **kw)


# =============================================================================
# 6. DEMO (synthetic scene, fake per-pixel model, no PyTorch)
# =============================================================================
def make_synthetic_scene(h: int = 300, w: int = 450, seed: int = 0):
    """A river (permanent water) with a flood band after the event, a no-data
    strip, and speckle noise.  Returns (before, after, truth[H,W] with 0/1/2)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    centre = w * 0.45 + 25 * np.sin(yy / 40.0)
    d = np.abs(xx - centre)
    river, flood = d < 4, (d < 22) & (d >= 4)

    def scene(with_flood: bool):
        vv = -11 + rng.normal(0, 1.2, (h, w))
        vv[river] = -23
        if with_flood:
            vv[flood] = -22
        vh = vv - 6 + rng.normal(0, 1.0, (h, w))
        return np.stack([vv, vh]).astype(np.float32)

    before, after = scene(False), scene(True)
    before[:, :, -3:] = np.nan
    after[:, :, -3:] = np.nan
    truth = np.zeros((h, w), np.uint8)
    truth[river], truth[flood] = 1, 2
    truth[:, -3:] = 255
    return before, after, truth


def fake_model_fn(batch: np.ndarray) -> np.ndarray:
    """A PER-PIXEL stand-in model (works on the un-normalized demo input):
    flood where VV dropped a lot; permanent water where it was dark before."""
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    pf = sig(-(batch[:, 4] + 6.0) / 1.5)
    pp = sig(-(batch[:, 0] + 17.0) / 1.5) * (1 - pf)
    return np.stack([np.clip(1 - pf - pp, 0, 1), pp, pf], axis=1).astype(np.float32)


def run_demo(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    before, after, truth = make_synthetic_scene()
    h, w = truth.shape
    meta = RasterMeta(w, h, "EPSG:32645", (10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0), 10.0)
    make_npz_raster(out / "before.npz", before, meta)
    make_npz_raster(out / "after.npz", after, meta)
    ident = Normalizer(np.zeros(6, np.float32), np.ones(6, np.float32), {})
    pred = FloodPredictor(fake_model_fn, ident, default_threshold=0.5,
                          threshold_source="demo", model_info={"checkpoint": "FAKE per-pixel demo model"})

    res = pred.predict_files(out / "before.npz", out / "after.npz", out / "result")
    problems = validate_outputs(res["mask"], res["confidence"], out / "before.npz")
    assert not problems, problems
    mask = read_raster(res["mask"]).data[0]
    c = counts_from_arrays(mask == 1, truth)
    print(f"image {h}x{w} (not a multiple of the 224 tile): IoU vs truth = {c.iou:.3f}, "
          f"precision {c.precision:.3f}, recall {c.recall:.3f}")
    print("summary:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["summary"].items()})
    assert c.iou > 0.9 and (mask == 255).sum() == 3 * h

    # tiled result must equal the plain whole-image per-pixel result (no seams)
    whole = FloodPredictor(fake_model_fn, ident, tile=224, overlap=32, batch_size=3).predict_arrays(before, after)
    solo = FloodPredictor(fake_model_fn, ident, tile=224, overlap=0, batch_size=1).predict_arrays(before, after)
    assert np.allclose(whole.prob, solo.prob, atol=1e-5), "tiling changed the result"
    print("tiled == untiled per-pixel prediction (no seam artefacts)")

    # packaging with a fake checkpoint file + metadata
    ckpt = out / "fake_best.pt"
    ckpt.write_bytes(b"not a real checkpoint, demo only")
    protocol = out / "eval_protocol.json"
    protocol.write_text(json.dumps({"threshold": 0.5, "selection_metric": "f1", "frozen_at": "demo"}))
    pkg = package_model(ckpt, protocol, out / "model_package",
                        meta={"config": {"encoder": "resnet34", "in_channels": 6, "num_classes": 3},
                              "epoch": 0, "created": "demo", "norm": {"meta": {}}})
    assert not verify_package(pkg)
    (pkg / "best.pt").write_bytes(b"tampered")
    assert verify_package(pkg), "tampering must be detected"
    print("package built, verified, and tampering detected")
    print("\nPART 4 OK")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Part 4: inference & packaging")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="synthetic end-to-end run, no PyTorch needed")
    d.add_argument("--out", default="demo_infer")
    p = sub.add_parser("predict", help="run the model on a before/after pair")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--package", default="model_package", help="package folder or .pt file")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--unit", default="auto", choices=["auto", "db", "linear"])
    p.add_argument("--device", default=None)
    p.add_argument("--tta", action="store_true", help="flip-averaging, 4x slower")
    a = ap.parse_args(argv)
    if a.cmd == "demo":
        run_demo(a.out)
    else:
        res = predict_flood(a.before, a.after, a.out, a.package, a.threshold, a.device, a.tta, unit=a.unit)
        print(json.dumps({k: res[k] for k in ("mask", "confidence", "meta", "summary", "warnings")}, indent=2))


if __name__ == "__main__":
    main()
