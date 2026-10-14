"""Safe readers/writers for every file type in the data contract.

Rules enforced here so that nobody has to remember them
-------------------------------------------------------
* Writes are ATOMIC (tmp file in the same folder + ``os.replace``): the
  dashboard can never read a half-written file.
* Text files are always UTF-8 (Windows defaults to cp1252, which crashes on
  Nepali place names).
* JSON never contains ``NaN``/``Infinity`` (invalid JSON; browsers choke).
* numpy scalars (``np.int64`` ...) are converted automatically.
* GeoJSON written from a GeoDataFrame is forced to EPSG:4326.
* ``rasterio`` is imported lazily, so everything except raster I/O works
  before the geo stack is installed.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from pipeline.errors import ContractError
from pipeline.schemas import Provenance

#: Name of the GeoTIFF tag that carries the provenance JSON string.
RASTER_PROVENANCE_TAG = "FLOODRESQ_PROVENANCE"


# --------------------------------------------------------------------------
# Atomic writing
# --------------------------------------------------------------------------
def _replace_with_retry(src: Path, dst: Path, tries: int = 8, delay: float = 0.15) -> None:
    """``os.replace`` with retries.

    On Windows, replacing a file that another process has open (Streamlit
    reading the map while the pipeline rewrites it) raises PermissionError.
    Retrying for ~1 s makes the race harmless.
    """
    for attempt in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(delay)


@contextmanager
def atomic_path(path: str | Path) -> Iterator[Path]:
    """Yield a temporary path; on clean exit it atomically replaces ``path``.

    The temp file is created in the SAME directory as the target because
    ``os.replace`` is only atomic within one filesystem.  If the ``with``
    block raises, the temp file is deleted and the old target is untouched.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        _replace_with_retry(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


# --------------------------------------------------------------------------
# JSON / GeoJSON
# --------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    """Teach ``json`` about dates, paths and numpy scalars/arrays."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return obj.as_posix()
    if hasattr(obj, "tolist"):          # numpy arrays AND numpy scalars
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _dumps(obj: Any, label: str) -> str:
    try:
        return json.dumps(
            obj, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default
        ) + "\n"
    except ValueError as exc:   # raised by allow_nan=False
        raise ContractError(
            f"{label} contains NaN or Infinity, which is not valid JSON.",
            f"{label}: {exc}",
        ) from exc


def _prov_dict(provenance: Optional[Provenance | dict]) -> Optional[dict]:
    if provenance is None:
        return None
    return provenance.to_dict() if hasattr(provenance, "to_dict") else dict(provenance)


def write_json(path: str | Path, obj: dict, provenance: Optional[Provenance | dict] = None) -> Path:
    """Write a dict as pretty UTF-8 JSON, adding a ``provenance`` key if given."""
    path = Path(path)
    prov = _prov_dict(provenance)
    if prov is not None:
        obj = {**obj, "provenance": prov}
    with atomic_path(path) as tmp:
        tmp.write_text(_dumps(obj, path.name), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def feature(geometry: dict, **properties: Any) -> dict:
    """Build one GeoJSON Feature: ``feature({"type": "Point", ...}, id="n1")``."""
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def write_geojson(
    path: str | Path,
    features: list[dict],
    provenance: Optional[Provenance | dict] = None,
) -> Path:
    """Write a FeatureCollection (WGS84 lon/lat coordinates expected).

    ``provenance`` is stored as a top-level *foreign member*, which RFC 7946
    explicitly allows, so every GeoJSON viewer still opens the file.
    """
    path = Path(path)
    collection: dict = {"type": "FeatureCollection", "features": features}
    prov = _prov_dict(provenance)
    if prov is not None:
        collection["provenance"] = prov
    with atomic_path(path) as tmp:
        tmp.write_text(_dumps(collection, path.name), encoding="utf-8")
    return path


def write_geojson_from_gdf(path: str | Path, gdf: Any, provenance: Optional[Provenance | dict] = None) -> Path:
    """Write a GeoDataFrame as GeoJSON, reprojecting to EPSG:4326 first.

    The forced ``to_crs(4326)`` means a teammate working in UTM cannot
    accidentally write metre coordinates into a GeoJSON file.
    """
    collection = json.loads(gdf.to_crs(4326).to_json())
    return write_geojson(path, collection["features"], provenance)


# --------------------------------------------------------------------------
# Rasters (rasterio is imported lazily)
# --------------------------------------------------------------------------
def _require_rasterio() -> Any:
    try:
        import rasterio  # noqa: WPS433 (lazy import is intentional)
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise ImportError(
            "rasterio is required for raster I/O. Install it with "
            "`pip install rasterio` (or use conda-forge on Windows)."
        ) from exc
    return rasterio


def rasterio_available() -> bool:
    try:
        import rasterio  # noqa: F401
    except ImportError:
        return False
    return True


def write_geotiff(
    path: str | Path,
    array: Any,
    *,
    transform: Any,
    epsg: int,
    provenance: Optional[Provenance | dict] = None,
    nodata: Optional[float] = None,
) -> Path:
    """Write a 2-D (single band) or 3-D (bands, rows, cols) array as GeoTIFF.

    Compression is deflate with the predictor that matches the dtype
    (3 = floating point, 2 = integer), which shrinks sample data a lot.
    Provenance is stored as ONE JSON string in a tag, because GeoTIFF tags
    can only hold strings and JSON keeps types intact.
    """
    import numpy as np

    rasterio = _require_rasterio()
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError("array must be 2-D or 3-D (bands, rows, cols)")
    predictor = 3 if np.issubdtype(arr.dtype, np.floating) else 2

    with atomic_path(path) as tmp:
        # driver is explicit because the temp file's extension is ".tmp".
        with rasterio.open(
            tmp, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
            count=arr.shape[0], dtype=arr.dtype, crs=f"EPSG:{epsg}",
            transform=transform, nodata=nodata, compress="deflate", predictor=predictor,
        ) as dst:
            dst.write(arr)
            prov = _prov_dict(provenance)
            if prov is not None:
                dst.update_tags(**{RASTER_PROVENANCE_TAG: json.dumps(prov)})
    return Path(path)


def north_up_transform(x_min: float, y_max: float, pixel_size: float) -> Any:
    """Affine transform for a north-up grid whose top-left corner is (x_min, y_max)."""
    from rasterio.transform import from_origin

    return from_origin(x_min, y_max, pixel_size, pixel_size)


def read_provenance(path: str | Path) -> Optional[dict]:
    """Return the provenance dict stored in a JSON, GeoJSON or GeoTIFF, or ``None``."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in (".json", ".geojson"):
            data = read_json(path)
            return data.get("provenance") if isinstance(data, dict) else None
        if suffix in (".tif", ".tiff"):
            rasterio = _require_rasterio()
            with rasterio.open(path) as src:
                raw = src.tags().get(RASTER_PROVENANCE_TAG)
            return json.loads(raw) if raw else None
    except (OSError, ValueError, ImportError):
        return None
    return None


def mask_area_km2(mask_path: str | Path) -> float:
    """Flooded area in km2 from a binary mask GeoTIFF.

    Refuses geographic CRS (degrees): pixel area in degrees^2 is meaningless,
    and silently returning a wrong number is the worst possible failure for a
    report that rescue teams might read.
    """
    import numpy as np

    rasterio = _require_rasterio()
    with rasterio.open(mask_path) as src:
        if src.crs is None or not src.crs.is_projected:
            raise ContractError(
                "flood_mask.tif must use a projected (metre) CRS such as UTM.",
                f"{mask_path}: CRS is {src.crs}",
            )
        pixel_m2 = abs(src.transform.a * src.transform.e)
        flooded = int(np.count_nonzero(src.read(1) > 0))
    return flooded * pixel_m2 / 1_000_000.0
