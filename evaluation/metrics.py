"""Agreement metrics between OUR result and a reference (check-only).

Pure numpy; no rasterio.  Vocabulary used everywhere:

* TP  both say "flooded"          (agreement)
* FP  only WE say flooded          (we over-detect, or the reference missed it)
* FN  only the REFERENCE says flooded  (we missed it)
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from dashboard.raster_overlay import encode_png_rgba, warp_to_lonlat

#: Colours of the agreement map (RGBA): green = agree, red = only we, blue = only reference.
AGREEMENT_COLOURS = {1: (44, 160, 44, 210), 2: (214, 39, 40, 210), 3: (31, 119, 180, 210)}


def confusion(pred: np.ndarray, ref: np.ndarray, valid: Optional[np.ndarray] = None) -> dict:
    """Pixel counts ``{tp, fp, fn, tn}``.  ``valid`` restricts the comparison to pixels the
    reference actually covered (outside the reference's own area of interest, "not flooded"
    would just mean "not examined" and would unfairly count against us)."""
    pred, ref = np.asarray(pred, dtype=bool), np.asarray(ref, dtype=bool)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: ours {pred.shape} vs reference {ref.shape}")
    mask = np.ones(pred.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    return {"tp": int((pred & ref & mask).sum()), "fp": int((pred & ~ref & mask).sum()),
            "fn": int((~pred & ref & mask).sum()), "tn": int((~pred & ~ref & mask).sum())}


def _ratio(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def scores(c: dict) -> dict:
    """IoU, precision, recall, F1 from a confusion dict.  ``None`` when undefined
    (e.g. no flooded pixel in either map): an undefined score is not a score of 0."""
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return {"iou": _ratio(tp, tp + fp + fn), "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn), "f1": _ratio(2 * tp, 2 * tp + fp + fn)}


def extent_metrics(pred: np.ndarray, ref: np.ndarray, pixel_km2: float,
                   valid: Optional[np.ndarray] = None) -> dict:
    """Scores plus areas in km2 (``ours``, ``reference``, ``overlap``) inside the valid region."""
    c = confusion(pred, ref, valid)
    return {**scores(c), "counts": c,
            "areas_km2": {"ours": round((c["tp"] + c["fp"]) * pixel_km2, 3),
                          "reference": round((c["tp"] + c["fn"]) * pixel_km2, 3),
                          "overlap": round(c["tp"] * pixel_km2, 3)}}


def agreement_codes(pred: np.ndarray, ref: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """uint8 map: 0 nothing, 1 agree, 2 only ours, 3 only reference (outside ``valid`` -> 0)."""
    pred, ref = np.asarray(pred, dtype=bool), np.asarray(ref, dtype=bool)
    codes = np.zeros(pred.shape, dtype=np.uint8)
    codes[pred & ref] = 1
    codes[pred & ~ref] = 2
    codes[~pred & ref] = 3
    if valid is not None:
        codes[~np.asarray(valid, dtype=bool)] = 0
    return codes


def agreement_png(codes: np.ndarray, transform: tuple, epsg: int, max_px: int = 1400):
    """Colour the agreement map and resample it to lon/lat; returns ``(png_bytes, bounds)``."""
    values, inside, bounds = warp_to_lonlat(codes, transform, epsg, max_px=max_px)
    rgba = np.zeros(values.shape + (4,), dtype=np.uint8)
    for code, colour in AGREEMENT_COLOURS.items():
        rgba[inside & (values == code)] = colour
    return encode_png_rgba(rgba), bounds


def compare_ids(ours: Iterable[str], reference: Iterable[str]) -> dict:
    """Compare the sets of object ids flagged by us and by the reference method.

    ``reference_flagged``  how many objects the reference flags
    ``also_flagged``       of those, how many we flag too
    ``extra_flagged``      objects only we flag
    ``agreement``          also / reference (share of reference objects we found), or None
    """
    ours, ref = set(ours), set(reference)
    also = len(ours & ref)
    return {"reference_flagged": len(ref), "also_flagged": also, "extra_flagged": len(ours - ref),
            "missed": len(ref - ours), "agreement": _ratio(also, len(ref))}
