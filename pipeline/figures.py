"""Stage 'figures' (T4): presentation PNGs for the report, slides and demo video.

Everything is drawn in the run's UTM grid (metres) so the raster and the
vectors line up without reprojection libraries: GeoJSON lon/lat is converted
with ``pipeline.geo_math`` (vectorised).  A missing input never crashes the
stage: that panel says "not available" instead, because ``figures`` is a
TOLERANT stage that also runs after upstream failures.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pipeline.contracts import Artifact
from pipeline.fileio import atomic_path, read_json
from pipeline.run import RunPaths
from pipeline.schemas import AOI

MAX_PX = 1200          # rasters are block-averaged down to at most this many pixels per side
ATTRIBUTION = ("Contains modified Copernicus Sentinel data 2026. © OpenStreetMap contributors. "
               "Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018.")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _read(run: RunPaths, artifact: Artifact, band: int = 1):
    """(array, extent [xmin, xmax, ymin, ymax], epsg) downsampled, or None."""
    import numpy as np
    try:
        import rasterio
    except ImportError:
        return None
    path = run.path(artifact)
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        step = max(1, int(np.ceil(max(src.width, src.height) / MAX_PX)))
        arr = src.read(band, out_shape=(src.height // step or 1, src.width // step or 1)).astype("float32")
        b = src.bounds
        return arr, [b.left, b.right, b.bottom, b.top], src.crs.to_epsg()


def _lines_xy(fc: dict, epsg: int):
    import numpy as np
    from pipeline.geo_math import lonlat_to_utm_array
    out = []
    for f in fc.get("features", []):
        g = f.get("geometry") or {}
        parts = [g["coordinates"]] if g.get("type") == "LineString" else \
            (g["coordinates"] if g.get("type") == "MultiLineString" else [])
        for p in parts:
            a = np.asarray(p, float)
            if len(a) >= 2:
                out.append(np.column_stack(lonlat_to_utm_array(a[:, 0], a[:, 1], epsg)))
    return out


def _points_xy(fc: dict, epsg: int):
    import numpy as np
    from pipeline.geo_math import lonlat_to_utm_array
    pts, names = [], []
    for f in fc.get("features", []):
        g = f.get("geometry") or {}
        c = g.get("coordinates")
        if g.get("type") == "Point":
            xy = c
        elif g.get("type") == "Polygon":
            xy = np.asarray(c[0], float).mean(axis=0)
        else:
            continue
        pts.append(xy[:2])
        names.append(str((f.get("properties") or {}).get("name", "")))
    if not pts:
        return None, []
    a = np.asarray(pts, float)
    return np.column_stack(lonlat_to_utm_array(a[:, 0], a[:, 1], epsg)), names


def _fc(run: RunPaths, artifact: Artifact) -> dict:
    try:
        return read_json(run.path(artifact))
    except (OSError, ValueError):
        return {"features": []}


def _save(fig, run: RunPaths, artifact: Artifact) -> None:
    fig.text(0.01, 0.005, ATTRIBUTION, fontsize=5.5, color="#555555")
    with atomic_path(run.ensure_parent(artifact)) as tmp:
        fig.savefig(tmp, dpi=130, format="png")
    _plt().close(fig)


def _not_available(run: RunPaths, artifact: Artifact, title: str, why: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.42, f"Not available: {why}", ha="center", fontsize=10, color="#b00020", wrap=True)
    _save(fig, run, artifact)


def _base(ax, raster, cmap="gray", vmin=None, vmax=None, alpha=1.0):
    arr, ext, _ = raster
    ax.imshow(arr, extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, interpolation="nearest")
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([])
    ax.set_yticks([])


def _radar(run, artifact, title):
    import numpy as np
    r = _read(run, artifact)
    if r is None:
        return _not_available(run, {Artifact.BEFORE: Artifact.FIG_BEFORE, Artifact.AFTER: Artifact.FIG_AFTER}[artifact],
                              title, "radar image missing (or rasterio not installed)")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 6))
    finite = r[0][np.isfinite(r[0])]
    lo, hi = (np.percentile(finite, [2, 98]) if finite.size else (-25, 0))
    _base(ax, r, "gray", lo, hi)
    ax.set_title(f"{title} (Sentinel-1 VV, dB)")
    _save(fig, run, {Artifact.BEFORE: Artifact.FIG_BEFORE, Artifact.AFTER: Artifact.FIG_AFTER}[artifact])


def run_figures(run: RunPaths, aoi: AOI, event_date: date, cfg: dict) -> None:
    import numpy as np
    plt = _plt()
    meta = {}
    try:
        meta = read_json(run.path(Artifact.METADATA))
    except (OSError, ValueError):
        pass
    _radar(run, Artifact.BEFORE, f"Before {meta.get('before_date', '')}".strip())
    _radar(run, Artifact.AFTER, f"After {meta.get('after_date', '')}".strip())

    after = _read(run, Artifact.AFTER)
    mask = _read(run, Artifact.FLOOD_MASK)
    epsg = (mask or after or (None, None, aoi.utm_epsg()))[2]
    title_flood = f"Flood / debris extent, event {event_date}"
    if mask is None:
        _not_available(run, Artifact.FIG_FLOOD, title_flood, "flood mask missing")
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        if after is not None:
            finite = after[0][np.isfinite(after[0])]
            lo, hi = (np.percentile(finite, [2, 98]) if finite.size else (-25, 0))
            _base(ax, after, "gray", lo, hi)
        m = np.ma.masked_where(mask[0] < 0.5, mask[0])
        _base(ax, (m, mask[1], mask[2]), "autumn", 0, 1, 0.75)
        ax.set_title(title_flood)
        _save(fig, run, Artifact.FIG_FLOOD)

    def vector_map(artifact, title, draw):
        fig, ax = plt.subplots(figsize=(7, 6))
        if mask is not None:
            m = np.ma.masked_where(mask[0] < 0.5, mask[0])
            _base(ax, (m, mask[1], mask[2]), "Blues", 0, 1.4, 0.55)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        for seg in _lines_xy(_fc(run, Artifact.OSM_ROADS), epsg):
            ax.plot(seg[:, 0], seg[:, 1], color="#9a9a9a", lw=0.5, zorder=1)
        draw(ax)
        ax.set_title(title)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
        _save(fig, run, artifact)

    def draw_damage(ax):
        for i, seg in enumerate(_lines_xy(_fc(run, Artifact.AFFECTED_ROADS), epsg)):
            ax.plot(seg[:, 0], seg[:, 1], color="#d62728", lw=2.0, zorder=3, label="affected road" if i == 0 else None)
        for i, seg in enumerate(_lines_xy(_fc(run, Artifact.AFFECTED_BRIDGES), epsg)):
            ax.plot(seg[:, 0], seg[:, 1], color="#ff7f0e", lw=3.0, zorder=4, label="affected bridge" if i == 0 else None)
        xy, _ = _points_xy(_fc(run, Artifact.AFFECTED_BUILDINGS), epsg)
        if xy is not None:
            ax.scatter(xy[:, 0], xy[:, 1], s=6, color="#9467bd", zorder=3, label="affected building")

    def draw_connectivity(ax):
        for i, seg in enumerate(_lines_xy(_fc(run, Artifact.AFFECTED_ROADS), epsg)):
            ax.plot(seg[:, 0], seg[:, 1], color="#d62728", lw=2.0, zorder=3, label="flooded road" if i == 0 else None)
        for i, seg in enumerate(_lines_xy(_fc(run, Artifact.AFFECTED_BRIDGES), epsg)):
            ax.plot(seg[:, 0], seg[:, 1], color="#ff7f0e", lw=3.0, zorder=3, label="flooded bridge" if i == 0 else None)
        xy, _ = _points_xy(_fc(run, Artifact.OSM_SETTLEMENTS), epsg)
        if xy is not None:
            ax.scatter(xy[:, 0], xy[:, 1], s=22, color="#2ca02c", edgecolor="k", lw=0.4, zorder=4, label="settlement")
        xy, names = _points_xy(_fc(run, Artifact.CUTOFF_SETTLEMENTS), epsg)
        if xy is not None:
            ax.scatter(xy[:, 0], xy[:, 1], s=60, color="#e31a1c", edgecolor="k", lw=0.6, zorder=5, label="cut off")
            for (x, y), n in list(zip(xy, names))[:25]:
                ax.annotate(n, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        xy, _ = _points_xy(_fc(run, Artifact.OSM_HOSPITALS), epsg)
        if xy is not None:
            ax.scatter(xy[:, 0], xy[:, 1], s=90, marker="P", color="#1f77b4", edgecolor="k", lw=0.5, zorder=5,
                       label="hospital / town")

    vector_map(Artifact.FIG_DAMAGE, "What was hit? (roads, bridges, buildings)", draw_damage)
    vector_map(Artifact.FIG_CONNECTIVITY, "Who is cut off?", draw_connectivity)
