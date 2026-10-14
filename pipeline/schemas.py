"""Small, dependency-free data classes shared by every teammate.

Contents
--------
``parse_date`` / ``parse_event_date``  forgiving but strict date handling
``AOI``                                 bounding box + metric helpers
``Summary``                             the ONE object dashboard and report read
``Provenance``                          "where did this file come from" stamp
``run_id``                              deterministic folder id (cache key)
``utc_now_iso``                         timezone-aware timestamp helper

Only the standard library is used on purpose, so that this module imports
(and is testable) even before the geo stack is installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pipeline.errors import InputValidationError

#: First date with routine Sentinel-1 data; earlier event dates are rejected.
S1_FIRST_ROUTINE_DATA = date(2014, 10, 3)


def utc_now_iso() -> str:
    """Timezone-aware UTC timestamp, e.g. ``2026-10-04T09:30:12+00:00``.

    ``datetime.utcnow()`` is avoided: it returns a *naive* datetime and is
    deprecated, which causes silent timezone bugs when files are compared.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
def parse_date(value: Any, label: str = "date") -> date:
    """Convert ``date`` / ``datetime`` / ISO string to a plain ``date``.

    Accepts ``"2026-08-26"`` and also full timestamps such as
    ``"2026-08-26T05:30:00Z"`` (what Copernicus catalogues return) by looking
    only at the first 10 characters.
    """
    # NOTE: datetime is a SUBCLASS of date, so it must be tested first,
    # otherwise ``isinstance(x, date)`` would return a datetime unchanged.
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            pass
    raise InputValidationError(
        f"Could not read {label} {value!r}. Use the format YYYY-MM-DD."
    )


def parse_event_date(value: Any, *, today: Optional[date] = None) -> date:
    """Parse the flood date and reject impossible values.

    ``today`` is injectable so unit tests do not depend on the real clock.
    """
    event = parse_date(value, "event date")
    today = today or date.today()
    if event > today:
        raise InputValidationError(f"Event date {event} is in the future.")
    if event < S1_FIRST_ROUTINE_DATA:
        raise InputValidationError(
            f"Event date {event} is before routine Sentinel-1 coverage "
            f"({S1_FIRST_ROUTINE_DATA})."
        )
    return event


# --------------------------------------------------------------------------
# Area of interest
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AOI:
    """Axis-aligned bounding box in WGS84 degrees (EPSG:4326).

    The box must not cross the antimeridian (west < east always).
    ``frozen=True`` makes instances hashable and immutable, so one AOI object
    can be passed to every stage without anyone modifying it by accident.
    """

    west: float
    south: float
    east: float
    north: float
    name: str = "custom"

    def __post_init__(self) -> None:
        # A frozen dataclass forbids ``self.x = ...``; ``object.__setattr__``
        # is the sanctioned escape hatch.  We coerce here because JSON, numpy
        # and Streamlit widgets hand us ints, numpy floats or strings.
        for key in ("west", "south", "east", "north"):
            try:
                object.__setattr__(self, key, float(getattr(self, key)))
            except (TypeError, ValueError) as exc:
                raise InputValidationError(
                    f"AOI value '{key}' is not a number: {getattr(self, key)!r}"
                ) from exc

    # -- geometry helpers --------------------------------------------------
    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """``(west, south, east, north)``: the order GeoJSON and rasterio use."""
        return (self.west, self.south, self.east, self.north)

    @property
    def center(self) -> tuple[float, float]:
        """``(lon, lat)`` of the box centre."""
        return ((self.west + self.east) / 2.0, (self.south + self.north) / 2.0)

    def area_km2(self) -> float:
        """Approximate area (equirectangular, good to ~1% for boxes < 5 deg)."""
        lat_mid = math.radians(self.center[1])
        width_km = (self.east - self.west) * 111.320 * math.cos(lat_mid)
        height_km = (self.north - self.south) * 110.574
        return abs(width_km * height_km)

    def utm_epsg(self) -> int:
        """EPSG code of the UTM zone containing the box centre.

        All rasters of a run are stored in this CRS because UTM units are
        metres, which makes pixel areas and road lengths correct.
        Norway/Svalbard zone exceptions are ignored (irrelevant for the
        Himalaya).  A box that straddles two zones uses the centre's zone.
        """
        lon, lat = self.center
        # lon == 180 would give zone 61, hence the clamp to 60.
        zone = min(int((lon + 180.0) // 6.0) + 1, 60)
        return (32600 if lat >= 0 else 32700) + zone

    # -- validation ----------------------------------------------------------
    def validate(self, max_km2: float = 2500.0, min_km2: float = 0.0) -> "AOI":
        """Raise :class:`InputValidationError` if the box is unusable; return self."""
        # Written as a chained comparison on purpose: any NaN makes it False,
        # so NaN coordinates are rejected without a separate isnan() check.
        ok = (
            -180.0 <= self.west < self.east <= 180.0
            and -90.0 <= self.south < self.north <= 90.0
        )
        if not ok:
            raise InputValidationError(
                "Bounding box is invalid: need -180<=west<east<=180 and "
                "-90<=south<north<=90."
            )
        area = self.area_km2()
        if area > max_km2:
            raise InputValidationError(
                f"Area is {area:,.0f} km2; the limit is {max_km2:,.0f} km2. "
                "Choose a smaller box."
            )
        if area < min_km2:
            raise InputValidationError(
                f"Area is {area:.2f} km2; the minimum is {min_km2} km2."
            )
        return self

    # -- naming / (de)serialisation ------------------------------------------
    def slug(self) -> str:
        """Filesystem-safe lowercase name used in run-folder names."""
        s = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")
        return s[:40] or "aoi"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AOI":
        """Accept ``{"west":..,"south":..,"east":..,"north":..}`` or ``{"bbox":[w,s,e,n]}``."""
        name = data.get("name", "custom")
        if "bbox" in data:
            try:
                west, south, east, north = data["bbox"]
            except (TypeError, ValueError) as exc:
                raise InputValidationError("'bbox' must be [west, south, east, north].") from exc
            return cls(west, south, east, north, name)
        try:
            return cls(data["west"], data["south"], data["east"], data["north"], name)
        except KeyError as exc:
            raise InputValidationError(f"AOI is missing key {exc}.") from exc

    @classmethod
    def from_json_file(cls, path: str | Path) -> "AOI":
        # Explicit utf-8: Windows' default (cp1252) would break non-ASCII names.
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# Summary (single source of truth for dashboard AND report)
# --------------------------------------------------------------------------
#: Numeric KPI fields where ``None`` means "stage did not run / failed".
KPI_FIELDS = (
    "flood_area_km2",
    "affected_roads",
    "affected_road_length_km",
    "affected_bridges",
    "affected_buildings",
    "cutoff_settlements",
    "mean_confidence",
)


@dataclass
class Summary:
    """Everything the dashboard cards and the situation report may print.

    ``None`` means *not computed* and is deliberately different from ``0``
    (which means *computed, nothing found*).  The report prints "not
    available" for ``None`` so a failed stage can never look like good news.
    """

    flood_area_km2: Optional[float] = None
    affected_roads: Optional[int] = None
    affected_road_length_km: Optional[float] = None
    affected_bridges: Optional[int] = None
    affected_buildings: Optional[int] = None
    cutoff_settlements: Optional[int] = None
    cutoff_names: list = field(default_factory=list)
    mean_confidence: Optional[float] = None
    # context lines printed by the report
    aoi_name: str = ""
    event_date: str = ""
    before_date: str = ""
    after_date: str = ""
    sentinel1_orbit: str = ""
    crs: str = ""
    model_version: str = ""
    osm_snapshot_date: str = ""
    processing_date: str = ""
    is_mock: bool = False
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Summary":
        """Build from a dict, ignoring unknown keys.

        Ignoring unknown keys keeps old cached ``summary.json`` files loadable
        after someone adds a field (forward compatibility).
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def unavailable(self) -> list[str]:
        """Names of KPI fields that are still ``None``."""
        return [k for k in KPI_FIELDS if getattr(self, k) is None]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
@dataclass
class Provenance:
    """Stamp stored inside every output file (the plan's "every geospatial
    output should include CRS, source, processing date, event date, model
    version")."""

    producer: str                    # e.g. "gis.damage"
    source: str                      # e.g. "Sentinel-1 GRD + OSM snapshot 2026-07-27"
    crs: str = ""
    event_date: str = ""
    model_version: str = ""
    processing_date: str = field(default_factory=utc_now_iso)
    is_mock: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Run id
# --------------------------------------------------------------------------
def run_id(aoi: AOI, event_date: date, config_fingerprint: str, model_version: str) -> str:
    """8-character deterministic id: same inputs -> same id -> cache hit.

    * The bbox is rounded to 4 decimals (~11 m) so float noise from a UI
      widget (``85.10000000001``) does not create a "new" run.
    * The AOI *name* is not hashed: renaming an area must not invalidate it.
    * JSON is dumped with ``sort_keys`` and fixed separators so the byte
      string (and therefore the hash) is canonical.
    """
    payload = {
        "bbox": [round(v, 4) for v in aoi.bbox],
        "event_date": event_date.isoformat(),
        "config": config_fingerprint,
        "model": model_version,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]
