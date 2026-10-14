"""
tiling.py  --  Part 4: running a patch-sized model over a BIG image (pure numpy)
===============================================================================

Repo location: models/tiling.py

THE PROBLEM
-----------
The U-Net was trained on 224x224 patches.  A real area of interest (AOI) can be
thousands of pixels wide and will not fit in GPU memory at once.  So we cut it
into overlapping tiles, predict each tile, and stitch the predictions back
together.  Everything here is plain numpy (the model is passed in as a function),
so it is fully unit-tested without PyTorch.

    big image ──► tile_starts() ──► windows (y, x) with overlap
                                         │ batches of tiles
                                         ▼
                                 model_fn(batch) -> class probabilities
                                         │ weight with blend_window()
                                         ▼
                              accumulate / divide by weight sum ──► full-size probabilities

CONTENTS
    1. tile_starts             where tiles start along one axis
    2. blend_window            smooth weights so seams vanish
    3. pad_to_min              make small images at least one tile big
    4. predict_tiled           the stitching loop
    5. safe_call               survive GPU out-of-memory by splitting the batch
    6. with_tta                flip-averaging wrapper (test-time augmentation)
"""
from __future__ import annotations

from typing import Callable, Iterator, List, Tuple

import numpy as np

# A model function: float32 [N, C_in, T, T]  ->  float32 probabilities [N, C_out, T, T]
ModelFn = Callable[[np.ndarray], np.ndarray]


# =============================================================================
# 1. TILE POSITIONS
# =============================================================================
def tile_starts(length: int, tile: int, stride: int) -> List[int]:
    """Start coordinates of tiles along ONE axis so the whole axis is covered.

    TRICK T1 (last tile snaps to the edge):
        With a fixed stride the last tile usually hangs over the edge or leaves
        a thin uncovered strip.  We always add one extra tile that STARTS AT
        `length - tile`, so it ends exactly at the image edge and is a full-size
        tile.  It overlaps its neighbour more than the others, which is
        harmless because overlaps are blended (section 2).
    """
    if tile <= 0 or stride <= 0:
        raise ValueError("tile and stride must be positive")
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return sorted(set(starts))


# =============================================================================
# 2. BLEND WEIGHTS
# =============================================================================
def blend_window(tile: int, overlap: int) -> np.ndarray:
    """2-D weight map [tile, tile]: 1.0 in the middle, smoothly falling toward
    the tile edges across `overlap` pixels, never exactly 0.

    TRICK T2 (why not just average overlapping tiles?):
        A CNN is least reliable at the edge of its input: it lacks context there
        (the other side of the edge is missing / zero-padded).  A plain average
        gives edge predictions the same vote as centre predictions, which leaves
        visible seams (a faint grid) in the flood map.  Instead each tile's
        prediction is multiplied by this window before accumulation and the sum
        is divided by the summed weights, so centre pixels dominate wherever two
        tiles overlap.
        The ramp is  0.5*(1 - cos(pi * (i+1)/(overlap+1)))  for i = 0..overlap-1:
        strictly between 0 and 1, so no division by zero is possible, and at the
        outer image border (covered by one tile only) the weight cancels out in
        the division, so border predictions are left untouched.
    """
    if overlap <= 0:
        return np.ones((tile, tile), np.float32)
    ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(overlap) + 1) / (overlap + 1)))
    w = np.ones(tile, np.float64)
    w[:overlap] = ramp
    w[-overlap:] = np.minimum(w[-overlap:], ramp[::-1])
    return np.outer(w, w).astype(np.float32)


# =============================================================================
# 3. SMALL IMAGES
# =============================================================================
def pad_to_min(arr: np.ndarray, min_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad the last two axes (bottom/right) so each is at least `min_size`.
    Returns (padded, (pad_h, pad_w)); crop with `arr[..., :H, :W]` afterwards.
    Reflection is used (not zeros) so the model never sees an artificial black
    border; very thin inputs fall back to edge replication."""
    h, w = arr.shape[-2:]
    ph, pw = max(0, min_size - h), max(0, min_size - w)
    if ph == 0 and pw == 0:
        return arr, (0, 0)
    widths = [(0, 0)] * (arr.ndim - 2) + [(0, ph), (0, pw)]
    mode = "reflect" if min(h, w) > 1 else "edge"
    return np.pad(arr, widths, mode=mode), (ph, pw)


# =============================================================================
# 4. THE STITCHING LOOP
# =============================================================================
def iter_windows(h: int, w: int, tile: int, overlap: int) -> Iterator[Tuple[int, int]]:
    stride = tile - overlap
    for y in tile_starts(h, tile, stride):
        for x in tile_starts(w, tile, stride):
            yield y, x


def predict_tiled(build_tile: Callable[[int, int], np.ndarray], model_fn: ModelFn,
                  h: int, w: int, n_classes: int, tile: int = 224, overlap: int = 32,
                  batch_size: int = 8) -> np.ndarray:
    """Predict a full [n_classes, H, W] probability map from overlapping tiles.

    build_tile(y, x) -> float32 [C_in, tile, tile] model input for the window at
                        (y, x).  The CALLER supplies it, so the full 6-channel
                        stack of a huge scene never has to exist in memory.
    h, w            : image size, already >= tile (use pad_to_min first).

    TRICK T3 (inputs are built tile by tile):
        Normalization is element-wise, so building each tile's input from the
        raw before/after crop gives exactly the same numbers as building the
        whole image first (this equality is a unit test).  Peak memory is the
        raw image plus ONE batch, not raw + a 6-channel copy.
    Accumulators are float32 [C,H,W] + [H,W]: for a 5000x5000 scene about 400 MB (3 x 25M x 4 B + 25M x 4 B).
    """
    if tile % 32 != 0:
        raise ValueError("tile must be a multiple of 32 for this U-Net")
    if not 0 <= overlap < tile:
        raise ValueError("need 0 <= overlap < tile")
    if h < tile or w < tile:
        raise ValueError(f"image {h}x{w} smaller than tile {tile}; pad it first")

    window = blend_window(tile, overlap)
    acc = np.zeros((n_classes, h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)

    pending: List[Tuple[int, int]] = []

    def flush() -> None:
        if not pending:
            return
        batch = np.stack([build_tile(y, x) for y, x in pending]).astype(np.float32)
        probs = model_fn(batch)
        if probs.shape != (len(pending), n_classes, tile, tile):
            raise ValueError(f"model returned {probs.shape}, expected "
                             f"{(len(pending), n_classes, tile, tile)}")
        for (y, x), p in zip(pending, probs):
            acc[:, y:y + tile, x:x + tile] += p * window
            wsum[y:y + tile, x:x + tile] += window
        pending.clear()

    for y, x in iter_windows(h, w, tile, overlap):
        pending.append((y, x))
        if len(pending) == batch_size:
            flush()
    flush()
    if (wsum <= 0).any():                                   # cannot happen; guards against bugs
        raise RuntimeError("some pixels were not covered by any tile")
    return acc / wsum


# =============================================================================
# 5. OUT-OF-MEMORY SAFETY
# =============================================================================
def safe_call(model_fn: ModelFn, batch: np.ndarray, min_batch: int = 1) -> np.ndarray:
    """Call model_fn(batch); if it fails with an out-of-memory error, split the
    batch in two halves and try each (recursively).

    TRICK T4: the batch size that fits on the team's Colab GPU may not fit on a
    judge's laptop GPU.  Instead of crashing during a LIVE evaluation, we halve
    the batch until it fits (down to 1 tile), and the result is identical because
    tiles are independent.  Any other error is re-raised untouched.  We match on
    the message text ("out of memory"), which works for both CUDA and CPU
    allocators and for old and new PyTorch versions.
    """
    try:
        return model_fn(batch)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower() or len(batch) <= min_batch:
            raise
        mid = len(batch) // 2
        return np.concatenate([safe_call(model_fn, batch[:mid], min_batch),
                               safe_call(model_fn, batch[mid:], min_batch)], axis=0)


# =============================================================================
# 6. TEST-TIME AUGMENTATION
# =============================================================================
def with_tta(model_fn: ModelFn) -> ModelFn:
    """Wrap a model so each tile is predicted 4 times (as is, flipped
    horizontally, flipped vertically, flipped both) and the un-flipped results
    are averaged.

    TRICK T5: averaging the model with its mirror images smooths out random
    mistakes for free (no retraining), typically a small but steady gain.
    The flips are undone BEFORE averaging, so pixels stay aligned.  Cost: 4x
    slower, so it is off by default.  For a model that is exactly flip-symmetric
    the result equals the plain prediction (a unit test), which proves the
    flip/un-flip bookkeeping is right.
    """
    def wrapped(batch: np.ndarray) -> np.ndarray:
        outs = []
        for axes in ((), (-1,), (-2,), (-1, -2)):
            x = np.flip(batch, axes) if axes else batch
            p = model_fn(np.ascontiguousarray(x))
            outs.append(np.flip(p, axes) if axes else p)
        return np.mean(outs, axis=0)
    return wrapped
