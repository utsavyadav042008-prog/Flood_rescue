"""
acquisition/ohsome.py  —  PART 5: Pre-event OpenStreetMap snapshot (+ cache, tests, hand-off)
==========================================================================================

WHAT THIS FILE DOES (one paragraph)
-----------------------------------
Downloads OpenStreetMap exactly AS IT WAS before the flood (never the current
map, which may contain post-event edits: that would break the data rules),
one raw GeoJSON file per layer (roads, bridges, buildings, hospitals,
settlements), caches them on disk, and hands them to the team: the files are
copied into the Part 3 analysis package under stable names and an ``osm``
entry (snapshot date, counts, warnings, attribution) is merged into that
package's ``metadata.json``. It does NO road-graph or attribute work: that is
Teammate 3's job (see "Who owns what" in the Part 5 notes).

PUBLIC ENTRY POINTS
-------------------
    snap = fetch_osm_snapshot((85.1, 27.9, 85.5, 28.3), "2026-08-26")
    snap.snapshot_date            ->  date(2026, 7, 27)   (always < event date)
    snap.files["roads"]           ->  data/raw/osm/<aoi_key>/roads_2026-07-27.geojson
    snap.warnings                 ->  caveats for the report (sparse roads...)

    # ---- the pipeline call (same shape as Part 4's build_optical_layers) ----
    pair = ScenePair.load_json("data/processed/<key>/scene_pair.json")   # Part 1
    res  = build_osm_layers(pair, "data/processed/<key>")                # Part 3 folder
    res.status        "used" | "partial" | "unavailable"
    res.reason        a sentence for the report / dashboard
    res.files         {"roads": ".../osm/roads.geojson", ...}

    Writes next to Part 3's files:
        osm/<layer>.geojson     raw pre-event OSM, EPSG:4326, stable names
        osm.json                status + snapshot + counts + params hash (like optical.json)
        and an "osm" entry (+ attribution) merged into Part 3's metadata.json

    python -m acquisition.ohsome --scene-pair data/processed/<key>/scene_pair.json
    python -m acquisition.ohsome --bbox 85.1 27.9 85.5 28.3 --date 2026-08-26
    python -m acquisition.ohsome --selftest          # offline unit tests, no network

HOW IT FITS THE OTHER PARTS
---------------------------
  Part 1  uses AOI, ScenePair, run_key, the error base class AcquisitionError;
          the package folder is data/processed/<run_key(aoi, event_date)>.
  Part 3  owns metadata.json (grid, attributions, warnings). We only ADD an "osm"
          entry, like Part 4 adds "optical". Part 3 rewrites metadata.json when it
          rebuilds a package; a cache hit here re-adds the entry automatically.
  Part 4  sets the convention followed here: optional data never stops the
          pipeline (status used/partial/unavailable, reason_code, never raises
          unless raise_errors=True), result sidecar json with a params hash, TTL
          re-check of non-final results.
  The ONE exception to "never raises": a snapshot date that is not before the event
  (SnapshotDateError) always raises. That is a rule breach, not a data gap.

DEPENDENCIES: ``requests`` and ``numpy`` (already needed by Parts 1-3).
No account or API key: the ohsome API is open.

MAP OF THE FILE
---------------
  1. Errors + constants
  2. Config
  3. Snapshot-date rule ....... THE rule that can disqualify the team
  4. ohsome client ............ POST, retries, automatic splitting of big areas
  5. Layer fetch + audit ...... merge cells, de-duplicate, check edit dates
  6. Statistics + warnings .... road density, empty layers (feeds "limitations")
  7. Cache + manifest ......... atomic files, exact-key reuse, offline mode
  8. fetch_osm_snapshot ....... raw fetch (layers, cache, fallback)
  9. Package hand-off ......... attach_to_package, build_osm_layers (pipeline entry)
 10. CLI
 11. Self-tests (offline, fake HTTP session)

NON-OBVIOUS TRICKS USED HERE (search for "TRICK n")
---------------------------------------------------
  TRICK 1  The snapshot date is COMPUTED from the event date, strictly before
           it, and capped at the latest snapshot the brief allows. A date on or
           after the event raises SnapshotDateError, so it cannot be used by
           accident. The final check runs again on the finished result.
  TRICK 2  ohsome is asked for ``time=<date>`` (a historical snapshot), never
           for "now". OSMnx/Overpass are NOT used: they return current data.
  TRICK 3  ``clipGeometry=false``: every way is returned WHOLE even when it
           crosses the AOI border. Roads therefore stay connected at the edge
           (good for routing) and pieces returned by two neighbouring cells can
           be de-duplicated by their OSM id without losing any geometry.
  TRICK 4  Big AOIs are split into cells; a cell that times out or is "too
           large" is split again into 4 (up to a depth limit). Small areas stay
           one request.
  TRICK 5  Edit-date audit: every feature carries its last-edit timestamp
           (``properties=tags,metadata``). If ANY feature was edited after the
           snapshot, we raise: independent evidence the rule was respected.
  TRICK 6  Cache key = (bbox, snapshot date, filter), checked INSIDE the file,
           not only from the file name. A stale or different file is refetched.
  TRICK 7  Files are written to ``.part`` then atomically renamed (a half file
           never looks complete), like the DEM and scene downloads.
  TRICK 8  Only ``roads`` is mandatory. A failure of another layer becomes a
           warning plus a "missing" entry, so the demo still runs.
  TRICK 9  Demo-day fallback: if the live fetch fails, a bundled sample folder
           (``fallback_dir``) is used, but ONLY if the snapshot date written
           in that file is before the event date.
  TRICK 10 Sparse-OSM warning: road length per km2 is measured so that "no road
           to this village" is reported as "maybe unmapped" (remote Nepal).
  TRICK 11 HTTP session and sleep are injectable => everything is tested offline.
  TRICK 12 A historical snapshot never changes, so a finished ("used") result is
           cached FOREVER; "partial"/"unavailable" results are re-tried after
           ``cache_ttl_h`` hours (Part 4 convention). Errors are never cached.
  TRICK 13 The params hash covers filters, snapshot date, AOI and rule settings,
           not timeouts or retries, so tuning the network does not invalidate data.

DATA RULES / ATTRIBUTION
------------------------
Licence ODbL: credit "(c) OpenStreetMap contributors". ``OSM_ATTRIBUTION`` is
written into every GeoJSON file and into metadata.json. No published damage-map product
or published damage map is ever used here.

NOT VERIFIED LIVE
-----------------
The ohsome parameter names and filter syntax below follow the public ohsome v1
documentation, but this code could not be run against the live API from the
offline build environment. First run:  ``python -m acquisition.ohsome --bbox ...
--date ... -v``  on a SMALL bbox and read the log. If ohsome rejects a filter,
the error message shows the server's text; adjust ``DEFAULT_LAYERS`` only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import requests

from acquisition.sentinel import (AOI, AcquisitionError, InvalidInputError, ScenePair,
                                  _coerce_date, run_key)

log = logging.getLogger("acquisition.ohsome")


# ============================================================================
# 1. ERRORS + CONSTANTS
# ============================================================================

class OsmError(AcquisitionError):
    """The OSM snapshot could not be obtained (network, server, bad answer)."""


class SnapshotDateError(OsmError, InvalidInputError):
    """The snapshot date is not strictly before the event date (data rule)."""


OSM_ATTRIBUTION = ("Contains data \u00a9 OpenStreetMap contributors, available under the "
                   "Open Database Licence (ODbL).")

# One ohsome filter per layer (ohsome filter syntax). Adjust here only.
DEFAULT_LAYERS = {
    "roads": "highway=* and type:way",
    "bridges": "bridge=* and bridge!=no and type:way",
    "buildings": "building=* and type:way",
    "hospitals": "amenity=hospital",
    "settlements": "place in (city, town, village, hamlet)",
}

KM_PER_DEG = 111.195     # mean length of one degree of latitude (spherical Earth)


# ============================================================================
# 2. CONFIG
# ============================================================================

@dataclass
class OsmConfig:
    base_url: str = "https://api.ohsome.org/v1"      # endpoint given in the brief
    endpoint: str = "elements/geometry"
    layers: dict = field(default_factory=lambda: dict(DEFAULT_LAYERS))
    required_layers: tuple = ("roads",)               # TRICK 8
    properties: str = "tags,metadata"                 # metadata => @lastEdit (TRICK 5)
    clip_geometry: bool = False                       # TRICK 3

    # --- snapshot rule (TRICK 1)
    latest_allowed_snapshot: Optional[date] = date(2026, 7, 27)   # from the brief
    min_days_before_event: int = 1                    # snapshot <= event - 1 day

    # --- splitting big areas (TRICK 4)
    split_deg: float = 0.2                            # cell edge, degrees (~22 km)
    max_split_depth: int = 2
    max_cells: int = 64

    # --- network behaviour
    max_retries: int = 4
    retry_base_delay_s: float = 2.0
    timeout_s: float = 180.0

    # --- cache + quality
    cache_root: Path = Path("data/raw/osm")
    min_road_km_per_km2: float = 0.15                 # below this: "OSM looks sparse"
    verify_edit_dates: bool = True
    cache_ttl_h: float = 24.0                         # re-check partial/unavailable (not hashed)

    def effective_params(self) -> dict:
        """Settings that change the DATA (hashed, TRICK 13). Network knobs are left out."""
        return {"layers": dict(self.layers), "required_layers": list(self.required_layers),
                "properties": self.properties, "clip_geometry": self.clip_geometry,
                "latest_allowed_snapshot": (self.latest_allowed_snapshot.isoformat()
                                            if self.latest_allowed_snapshot else None),
                "min_days_before_event": self.min_days_before_event,
                "split_deg": self.split_deg, "min_road_km_per_km2": self.min_road_km_per_km2,
                "verify_edit_dates": self.verify_edit_dates}


# ============================================================================
# 3. SNAPSHOT-DATE RULE  (TRICK 1)
# ============================================================================

def resolve_snapshot_date(event_date, cfg: Optional[OsmConfig] = None,
                          requested=None) -> date:
    """Pick the OSM snapshot date for an event.

    default    = latest date that is (a) at least ``min_days_before_event`` days
                 before the event and (b) not later than ``latest_allowed_snapshot``.
    requested  = an explicit date; accepted only if it is strictly before the event.
    """
    cfg = cfg or OsmConfig()
    event = _coerce_date(event_date)
    newest_ok = event - timedelta(days=max(1, cfg.min_days_before_event))
    if requested is not None:
        snap = _coerce_date(requested)
        assert_snapshot_before_event(snap, event)
        return snap
    if cfg.latest_allowed_snapshot is not None:
        newest_ok = min(newest_ok, cfg.latest_allowed_snapshot)
    return newest_ok


def assert_snapshot_before_event(snapshot, event_date) -> None:
    """Raise SnapshotDateError unless snapshot < event (strictly)."""
    snap, event = _coerce_date(snapshot), _coerce_date(event_date)
    if snap >= event:
        raise SnapshotDateError(
            f"OSM snapshot {snap} is not before the event date {event}. Post-event OSM "
            "edits are forbidden as inputs; choose an earlier snapshot.")


# ============================================================================
# 4. OHSOME CLIENT
# ============================================================================

class _Exhausted(Exception):
    """Internal: retries used up. ``splittable`` says if a smaller box may help."""

    def __init__(self, message: str, splittable: bool):
        super().__init__(message)
        self.splittable = splittable


def _quarters(b: tuple) -> list:
    x0, y0, x1, y1 = b
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    return [(x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)]


class OhsomeClient:
    """POSTs one filter + one box + one snapshot date to ohsome, returns features."""

    #: Used when the snapshot endpoint is refused (seen in Oct 2026: the public instance
    #: answers HTTP 403 for elements/geometry while elementsFullHistory/geometry works).
    FULL_HISTORY_ENDPOINT = "elementsFullHistory/geometry"

    def __init__(self, cfg: OsmConfig, session: Optional[requests.Session] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        self.session = session or requests.Session()
        if hasattr(self.session, "headers"):        # test fakes have no headers
            self.session.headers.setdefault("User-Agent", "FloodResQ/1.0 (educational flood-mapping prototype)")
        self.sleep = sleep
        self.requests_made = 0
        self.full_history = False         # switched on automatically after a 403 (see _post)

    @staticmethod
    def state_at(features: list, snapshot: date) -> list:
        """Full-history answer -> the map AS IT WAS at ``snapshot`` 00:00 UTC.

        With ``time=<snapshot-1d>,<snapshot>`` every returned version carries
        ``@validFrom``/``@validTo``; the versions still valid at the END of the range
        are exactly the elements that existed at the snapshot instant.  Nothing
        after the snapshot can appear, because the range ends there."""
        end = snapshot.isoformat()
        latest: dict = {}
        for f in features:
            p = f.get("properties") or {}
            if not str(p.get("@validTo", "")).startswith(end):
                continue
            oid = p.get("@osmId") or f.get("id")
            if oid not in latest or str(p.get("@validFrom", "")) >= str(
                    (latest[oid].get("properties") or {}).get("@validFrom", "")):
                latest[oid] = f
        return list(latest.values())

    def fetch(self, bbox: tuple, flt: str, snapshot: date, depth: int = 0) -> list:
        try:
            return self._post(bbox, flt, snapshot)
        except _Exhausted as exc:
            if exc.splittable and depth < self.cfg.max_split_depth:
                log.warning("ohsome: %s for box %s; splitting it in 4", exc,
                            tuple(round(v, 4) for v in bbox))
                out: list = []
                for q in _quarters(bbox):
                    out += self.fetch(q, flt, snapshot, depth + 1)
                return out
            raise OsmError(f"ohsome request failed for {flt!r}: {exc}") from exc

    def _post(self, bbox: tuple, flt: str, snapshot: date) -> list:
        cfg = self.cfg

        def request():
            if self.full_history:
                url = f"{cfg.base_url.rstrip('/')}/{self.FULL_HISTORY_ENDPOINT}"
                t = f"{(snapshot - timedelta(days=1)).isoformat()},{snapshot.isoformat()}"
            else:
                url = f"{cfg.base_url.rstrip('/')}/{cfg.endpoint}"
                t = snapshot.isoformat()                                # TRICK 2
            return url, {"bboxes": ",".join(f"{v:.6f}" for v in bbox), "time": t, "filter": flt,
                         "properties": cfg.properties,
                         "clipGeometry": "true" if cfg.clip_geometry else "false"}

        url, params = request()
        last, splittable = "no attempt", False
        for attempt in range(1, cfg.max_retries + 1):
            self.requests_made += 1
            try:
                resp = self.session.post(url, data=params, timeout=cfg.timeout_s)
            except requests.Timeout as exc:
                last, splittable = f"timeout ({exc})", True
            except requests.ConnectionError as exc:
                last, splittable = f"connection error ({exc})", False
            else:
                code = resp.status_code
                if code == 200:
                    try:
                        payload = resp.json()
                    except ValueError:
                        payload = None
                    if isinstance(payload, dict) and isinstance(payload.get("features"), list):
                        feats = payload["features"]
                        return self.state_at(feats, snapshot) if self.full_history else feats
                    last, splittable = "answer was not a GeoJSON feature collection", True
                elif code == 413:
                    raise _Exhausted("HTTP 413 (request too large)", True)
                elif code == 429 or code >= 500:
                    last, splittable = f"HTTP {code}", code in (502, 503, 504)
                elif code == 403 and not self.full_history:
                    log.warning("ohsome refused %s (HTTP 403); switching to %s for the same snapshot",
                                cfg.endpoint, self.FULL_HISTORY_ENDPOINT)
                    self.full_history = True
                    url, params = request()
                    continue                                            # retry at once, no back-off
                else:
                    snippet = (getattr(resp, "text", "") or "")[:300].replace("\n", " ")
                    raise OsmError(f"ohsome rejected the request (HTTP {code}): {snippet}")
            if attempt < cfg.max_retries:
                self.sleep(cfg.retry_base_delay_s * 2 ** (attempt - 1) * (1 + 0.25 * random.random()))
        raise _Exhausted(last, splittable)


# ============================================================================
# 5. LAYER FETCH + AUDIT
# ============================================================================

def split_bbox(aoi: AOI, step: float) -> list:
    """Cells of at most ``step`` degrees that tile the AOI exactly."""
    nx = max(1, math.ceil((aoi.max_lon - aoi.min_lon) / step - 1e-9))
    ny = max(1, math.ceil((aoi.max_lat - aoi.min_lat) / step - 1e-9))
    xs = np.linspace(aoi.min_lon, aoi.max_lon, nx + 1)
    ys = np.linspace(aoi.min_lat, aoi.max_lat, ny + 1)
    return [(float(xs[i]), float(ys[j]), float(xs[i + 1]), float(ys[j + 1]))
            for j in range(ny) for i in range(nx)]


def _dedupe(features: Sequence[dict]) -> list:
    """Drop empty geometries and repeats of the same OSM element (TRICK 3)."""
    seen, out = set(), []
    for f in features:
        if not f or f.get("geometry") is None:
            continue
        oid = (f.get("properties") or {}).get("@osmId") or f.get("id")
        if oid is not None:
            if oid in seen:
                continue
            seen.add(oid)
        out.append(f)
    return out


def audit_edit_dates(features: Sequence[dict], snapshot: date) -> Optional[date]:
    """Latest ``@lastEdit`` date found (None if the field is absent).

    Raises SnapshotDateError if any feature was edited after the snapshot
    (TRICK 5): that would mean post-event data got in."""
    latest: Optional[date] = None
    for f in features:
        raw = (f.get("properties") or {}).get("@lastEdit")
        if not raw:
            continue
        try:
            d = date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        if latest is None or d > latest:
            latest = d
    if latest is not None and latest > snapshot + timedelta(days=1):
        raise SnapshotDateError(f"OSM data contains an edit from {latest}, later than the "
                                f"snapshot {snapshot}. Refusing to use it.")
    return latest


def fetch_layer(client: OhsomeClient, aoi: AOI, flt: str, snapshot: date,
                cfg: OsmConfig) -> list:
    cells = split_bbox(aoi, cfg.split_deg)
    if len(cells) > cfg.max_cells:
        raise InvalidInputError(f"AOI needs {len(cells)} OSM requests (limit {cfg.max_cells}). "
                                "Use a smaller area.")
    feats: list = []
    for cell in cells:
        feats += client.fetch(cell, flt, snapshot)
    feats = _dedupe(feats)
    if cfg.verify_edit_dates:
        audit_edit_dates(feats, snapshot)
    return feats


# ============================================================================
# 6. STATISTICS + WARNINGS
# ============================================================================

def _lines(geom: dict) -> list:
    t, c = geom.get("type"), geom.get("coordinates")
    if t == "LineString":
        return [np.asarray(c, dtype=float)]
    if t == "MultiLineString":
        return [np.asarray(part, dtype=float) for part in c]
    return []


def road_length_km(features: Sequence[dict], aoi: AOI) -> float:
    """Length of road segments whose midpoint lies inside the AOI (equirectangular)."""
    total = 0.0
    for f in features:
        for pts in _lines(f["geometry"]):
            if len(pts) < 2:
                continue
            d = np.diff(pts[:, :2], axis=0)
            mid = (pts[:-1, :2] + pts[1:, :2]) / 2
            inside = ((mid[:, 0] >= aoi.min_lon) & (mid[:, 0] <= aoi.max_lon) &
                      (mid[:, 1] >= aoi.min_lat) & (mid[:, 1] <= aoi.max_lat))
            seg = np.hypot(d[:, 1] * KM_PER_DEG,
                           d[:, 0] * KM_PER_DEG * np.cos(np.radians(mid[:, 1])))
            total += float(seg[inside].sum())
    return total


def layer_stats(name: str, features: Sequence[dict], aoi: AOI) -> dict:
    types: dict = {}
    for f in features:
        t = f["geometry"].get("type", "?")
        types[t] = types.get(t, 0) + 1
    stats = {"count": len(features), "geometry_types": types}
    if name == "roads":
        km = road_length_km(features, aoi)
        stats["length_km"] = round(km, 2)
        stats["km_per_km2"] = round(km / max(aoi.area_km2(), 1e-9), 3)
    return stats


def quality_warnings(stats: dict, missing: dict, cfg: OsmConfig) -> list:
    """Plain-language caveats for the limitations section (TRICK 10)."""
    w = []
    r = stats.get("roads")
    if r is not None:
        if r["count"] == 0:
            w.append("OSM has NO mapped roads in this AOI at the snapshot date: routing "
                     "and access results are not meaningful here (the area may be unmapped).")
        elif r["km_per_km2"] < cfg.min_road_km_per_km2:
            w.append(f"OSM roads look sparse ({r['km_per_km2']:.2f} km per km2). A village "
                     "'without a road' may simply be unmapped; interpret access results with care.")
    for name, label in (("settlements", "settlement names"), ("hospitals", "hospitals"),
                        ("buildings", "buildings")):
        s = stats.get(name)
        if s is not None and s["count"] == 0:
            w.append(f"No {label} found in OSM for this AOI at the snapshot date "
                     "(may be missing from the map, not absent on the ground).")
    for name, why in missing.items():
        w.append(f"OSM layer '{name}' is unavailable: {why}")
    return w


# ============================================================================
# 7. CACHE + MANIFEST
# ============================================================================

def _atomic_write_json(path: Path, obj, indent=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")                 # TRICK 7
    part.write_text(json.dumps(obj, indent=indent, separators=None if indent else (",", ":")))
    os.replace(part, path)


def layer_path(out_dir: Path, name: str, snapshot: date) -> Path:
    return out_dir / f"{name}_{snapshot.isoformat()}.geojson"


def write_layer(path: Path, name: str, features: list, aoi: AOI, flt: str,
                snapshot: date) -> None:
    _atomic_write_json(path, {
        "type": "FeatureCollection", "name": name, "crs_note": "EPSG:4326 (lon, lat)",
        "osm_snapshot": snapshot.isoformat(), "filter": flt,
        "bbox": [round(v, 5) for v in aoi.as_tuple()], "attribution": OSM_ATTRIBUTION,
        "features": features})


def read_layer(path: Path, snapshot: Optional[date], flt: Optional[str],
               aoi: Optional[AOI]) -> Optional[dict]:
    """The file's content if it matches the requested key (TRICK 6), else None.
    Pass ``None`` for a field to skip checking it."""
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or not isinstance(d.get("features"), list):
        return None
    if snapshot is not None and d.get("osm_snapshot") != snapshot.isoformat():
        return None
    if flt is not None and d.get("filter") != flt:
        return None
    if aoi is not None and d.get("bbox") != [round(v, 5) for v in aoi.as_tuple()]:
        return None
    return d


# ============================================================================
# 8. fetch_osm_snapshot
# ============================================================================

@dataclass
class LayerResult:
    name: str
    path: Path
    stats: dict
    source: str            # "ohsome" | "cache" | "fallback"

    @property
    def from_cache(self) -> bool:
        return self.source == "cache"


@dataclass
class OsmSnapshot:
    """Result of Part 5. ``status`` / ``reason`` mirror Part 4's OpticalResult."""
    aoi: AOI
    event_date: date
    snapshot_date: date
    out_dir: Path
    layers: dict = field(default_factory=dict)       # name -> LayerResult
    missing: dict = field(default_factory=dict)      # name -> reason
    warnings: list = field(default_factory=list)
    requests_made: int = 0
    seconds: float = 0.0
    error: str = ""                                  # set only by build_osm_layers on failure
    from_cache: bool = False                         # whole result reused from osm.json
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc)
                             .isoformat(timespec="seconds"))

    @property
    def files(self) -> dict:
        return {k: v.path for k, v in self.layers.items()}

    @property
    def status(self) -> str:
        if self.error or not self.layers:
            return "unavailable"
        return "partial" if self.missing else "used"

    @property
    def reason_code(self) -> str:
        return "error" if self.error else ("missing_layers" if self.missing else "ok")

    @property
    def reason(self) -> str:
        if self.error:
            return self.error
        if self.missing:
            return f"OSM layers missing: {', '.join(self.missing)}"
        return f"{len(self.layers)} OSM layers at snapshot {self.snapshot_date}"

    def to_metadata(self) -> dict:
        return {"status": self.status, "reason_code": self.reason_code, "reason": self.reason,
                "snapshot_date": self.snapshot_date.isoformat(),
                "event_date": self.event_date.isoformat(),
                "source": "ohsome API (historical snapshot, not current OSM)",
                "attribution": OSM_ATTRIBUTION,
                "layers": {k: {"file": v.path.name, "source": v.source, **v.stats}
                           for k, v in self.layers.items()},
                "missing_layers": dict(self.missing), "warnings": list(self.warnings),
                "created_utc": self.created_utc}

    def summary(self) -> str:
        lines = [f"OSM: {self.status.upper()}{'  (cache)' if self.from_cache else ''}"
                 f"  -  {self.reason}  (snapshot {self.snapshot_date}, event {self.event_date})"]
        for k, v in self.layers.items():
            extra = f", {v.stats['length_km']} km" if "length_km" in v.stats else ""
            lines.append(f"  {k:<12} {v.stats['count']:>7} features{extra}  [{v.source}]")
        lines += [f"  MISSING {k}: {why}" for k, why in self.missing.items()]
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def _find_fallback(fallback_dir, name: str, snapshot: date, event: date) -> Optional[tuple]:
    """(path, content) of a bundled sample for this layer whose OWN snapshot date
    is before the event (TRICK 9)."""
    if not fallback_dir:
        return None
    base = Path(fallback_dir)
    for cand in sorted(base.glob(f"{name}*.geojson")):
        d = read_layer(cand, None, None, None)
        if d and d.get("osm_snapshot"):
            try:
                assert_snapshot_before_event(d["osm_snapshot"], event)
            except OsmError:
                log.warning("fallback %s ignored: its snapshot is not before the event", cand)
                continue
            return cand, d
    return None


def fetch_osm_snapshot(bbox: Sequence[float], event_date, *, snapshot_date=None,
                       layers: Optional[Sequence[str]] = None,
                       config: Optional[OsmConfig] = None,
                       session: Optional[requests.Session] = None,
                       progress: Optional[Callable[[str, float], None]] = None,
                       fallback_dir=None, offline: bool = False,
                       sleep: Callable[[float], None] = time.sleep) -> OsmSnapshot:
    """Part 5 entry point:  AOI + event date  ->  raw pre-event OSM GeoJSON files.

    snapshot_date  explicit snapshot (must be < event); default computed (TRICK 1)
    layers         subset of config.layers (default all)
    progress       callback ``progress(message, fraction 0..1)`` for the dashboard
    fallback_dir   folder with bundled sample layers used if the live fetch fails
    offline        never touch the network: cache (or fallback) only
    Raises: InvalidInputError, SnapshotDateError, OsmError.
    """
    t0 = time.time()
    cfg = config or OsmConfig()
    event = _coerce_date(event_date)
    aoi = AOI.from_bbox(bbox)
    snapshot = resolve_snapshot_date(event, cfg, snapshot_date)
    names = list(layers) if layers else list(cfg.layers)
    unknown = [n for n in names if n not in cfg.layers]
    if unknown:
        raise InvalidInputError(f"Unknown OSM layer(s) {unknown}; known: {sorted(cfg.layers)}")

    out_dir = Path(cfg.cache_root) / aoi.key()
    client = OhsomeClient(cfg, session, sleep)
    result = OsmSnapshot(aoi, event, snapshot, out_dir)
    emit = progress or (lambda msg, frac: None)
    stats_all: dict = {}

    for i, name in enumerate(names):
        flt = cfg.layers[name]
        emit(f"OSM {name}", i / len(names))
        t1 = time.time()
        path = layer_path(out_dir, name, snapshot)
        cached = read_layer(path, snapshot, flt, aoi)
        try:
            if cached is not None:                                       # cache hit
                feats, source = cached["features"], "cache"
            elif offline:
                raise OsmError("offline mode and no cached file for this AOI/date")
            else:
                feats, source = fetch_layer(client, aoi, flt, snapshot, cfg), "ohsome"
                write_layer(path, name, feats, aoi, flt, snapshot)
        except SnapshotDateError:
            raise                                                        # never "recover" from this
        except OsmError as exc:
            fb = _find_fallback(fallback_dir, name, snapshot, event)
            if fb is not None:
                path, content = fb
                feats, source = content["features"], "fallback"
                result.warnings.append(f"OSM layer '{name}' comes from the bundled sample "
                                       f"({path.name}, snapshot {content['osm_snapshot']}): "
                                       f"live fetch failed ({exc}).")
            elif name in cfg.required_layers:
                raise
            else:
                result.missing[name] = str(exc)
                log.warning("OSM layer %s skipped: %s", name, exc)
                continue
        stats = layer_stats(name, feats, aoi)
        stats_all[name] = stats
        result.layers[name] = LayerResult(name, path, stats, source)
        log.info("osm %-11s %6d features  %-8s %.1fs", name, stats["count"], source,
                 time.time() - t1)

    result.warnings += quality_warnings(stats_all, result.missing, cfg)
    assert_snapshot_before_event(result.snapshot_date, event)            # final guard
    result.requests_made, result.seconds = client.requests_made, round(time.time() - t0, 2)
    _atomic_write_json(out_dir / f"manifest_{snapshot.isoformat()}.json",
                       result.to_metadata(), indent=2)
    emit("OSM done", 1.0)
    return result


# ============================================================================
# 9. PACKAGE HAND-OFF (Part 3 folder)  +  PIPELINE ENTRY POINT
# ============================================================================

def _merge_unique(dst: list, items: Sequence[str]) -> list:
    return dst + [x for x in items if x not in dst]


def _merge_metadata(meta_path: Path, entry: dict, add_attribution: bool) -> None:
    """Add/replace the "osm" entry of Part 3's metadata.json (same job as Part 4's
    _merge_metadata for "optical"). Writes only if something changed."""
    if not meta_path.exists():
        log.warning("metadata.json not found; OSM status only in osm.json")
        return
    try:
        meta = json.loads(meta_path.read_text())
        new = dict(meta)
        new["osm"] = entry
        if add_attribution:
            new["attributions"] = _merge_unique(list(meta.get("attributions", [])), [OSM_ATTRIBUTION])
        new["warnings"] = _merge_unique(list(meta.get("warnings", [])), entry.get("warnings", []))
        if new != meta:
            _atomic_write_json(meta_path, new, indent=2)
    except (OSError, ValueError) as exc:
        log.warning("could not merge OSM into metadata.json: %s", exc)


def attach_to_package(snap: OsmSnapshot, package_dir, extra: Optional[dict] = None) -> Path:
    """Copy the layers into ``<package>/osm/<layer>.geojson`` (stable names for
    Teammate 3), write ``<package>/osm.json`` (like Part 4's optical.json) and merge
    the ``osm`` entry + attribution into metadata.json. Safe to call again.
    ``extra`` goes into osm.json only (params hash, config). Returns osm.json's path.

    If Part 3 rebuilds the package it rewrites metadata.json: call this again (or
    build_osm_layers, whose cache hit re-adds the entry)."""
    pkg = Path(package_dir)
    pkg.mkdir(parents=True, exist_ok=True)
    osm_dir = pkg / "osm"
    if snap.layers:
        osm_dir.mkdir(parents=True, exist_ok=True)
    for name, lr in snap.layers.items():
        dest = osm_dir / f"{name}.geojson"
        if lr.path.resolve() != dest.resolve():
            shutil.copyfile(lr.path, dest)
    entry = snap.to_metadata()
    entry["files"] = {n: str(osm_dir / f"{n}.geojson") for n in snap.layers}
    info = pkg / "osm.json"
    _atomic_write_json(info, {**entry, **(extra or {})}, indent=2)
    _merge_metadata(pkg / "metadata.json", entry, add_attribution=bool(snap.layers))
    return info


def _osm_hash(cfg: OsmConfig, aoi: AOI, snapshot: date, names: Sequence[str]) -> str:
    blob = json.dumps({"cfg": cfg.effective_params(), "aoi": aoi.as_tuple(),
                       "snapshot": snapshot.isoformat(), "layers": sorted(names)}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _load_cached_result(pkg: Path, ohash: str, cfg: OsmConfig, aoi: AOI,
                        event: date) -> Optional[OsmSnapshot]:
    """Reuse osm.json if settings match, files still exist, it was not an error and,
    unless it is a final "used", it is not older than ``cache_ttl_h`` (TRICK 12)."""
    try:
        d = json.loads((pkg / "osm.json").read_text())
        if d.get("params_hash") != ohash or d.get("reason_code") == "error":
            return None
        files = d.get("files", {})
        if not all(Path(f).exists() for f in files.values()):
            return None
        created = d["created_utc"]
        if d.get("status") != "used":
            age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(created)).total_seconds() / 3600
            if age_h > cfg.cache_ttl_h:
                return None
        snapshot = date.fromisoformat(d["snapshot_date"])
        assert_snapshot_before_event(snapshot, event)
        snap = OsmSnapshot(aoi, event, snapshot, Path(cfg.cache_root) / aoi.key(),
                           missing=dict(d.get("missing_layers", {})),
                           warnings=list(d.get("warnings", [])), from_cache=True, created_utc=created)
        for name, info in d.get("layers", {}).items():
            stats = {k: v for k, v in info.items() if k not in ("file", "source")}
            snap.layers[name] = LayerResult(name, Path(files[name]), stats, "cache")
        return snap
    except (OSError, ValueError, KeyError, TypeError):
        return None


def build_osm_layers(pair, out_dir, cfg: Optional[OsmConfig] = None, *, snapshot_date=None,
                     layers: Optional[Sequence[str]] = None,
                     session: Optional[requests.Session] = None,
                     progress: Optional[Callable[[str, float], None]] = None,
                     fallback_dir=None, offline: bool = False,
                     sleep: Callable[[float], None] = time.sleep,
                     raise_errors: bool = False) -> OsmSnapshot:
    """Part 5 PIPELINE entry point (same shape as Part 4's build_optical_layers).

    pair      Part 1's ScenePair (gives the AOI and the event date)
    out_dir   the Part 3 package folder (data/processed/<key>/)

    Does NOT raise on network/server problems unless ``raise_errors``: they become
    status "unavailable" with the reason, so the pipeline continues without OSM.
    ALWAYS raises SnapshotDateError / InvalidInputError (rule breach or bad input).
    """
    cfg = cfg or OsmConfig()
    pkg = Path(out_dir)
    pkg.mkdir(parents=True, exist_ok=True)
    aoi, event = pair.aoi, pair.event_date
    snapshot = resolve_snapshot_date(event, cfg, snapshot_date)       # may raise (by design)
    names = list(layers) if layers else list(cfg.layers)
    ohash = _osm_hash(cfg, aoi, snapshot, names)

    cached = _load_cached_result(pkg, ohash, cfg, aoi, event)
    if cached is not None:
        log.info("osm layers: cache hit (%s)", cached.status)
        _merge_metadata(pkg / "metadata.json", {**cached.to_metadata(),
                        "files": {n: str(p) for n, p in cached.files.items()}},
                        add_attribution=bool(cached.layers))          # Part 3 may have rebuilt it
        return cached

    try:
        snap = fetch_osm_snapshot(aoi.as_tuple(), event, snapshot_date=snapshot, layers=names,
                                  config=cfg, session=session, progress=progress,
                                  fallback_dir=fallback_dir, offline=offline, sleep=sleep)
    except (SnapshotDateError, InvalidInputError):
        raise
    except Exception as exc:                                           # noqa: BLE001 (deliberate)
        if raise_errors:
            raise
        log.exception("OSM layers failed; continuing without OSM")
        snap = OsmSnapshot(aoi, event, snapshot, Path(cfg.cache_root) / aoi.key(),
                           error=f"{type(exc).__name__}: {exc}")
        snap.warnings.append(f"OSM data unavailable ({exc}): road access, settlements and "
                             "hospitals cannot be assessed.")
    attach_to_package(snap, pkg, extra={"params_hash": ohash, "config": cfg.effective_params()})
    return snap


def fetch_and_attach(bbox: Sequence[float], event_date, *,
                     processed_root="data/processed", **kwargs) -> OsmSnapshot:
    """For callers without a ScenePair: fetch, then attach to
    ``<processed_root>/<aoi_key>_<event_date>`` if that package exists."""
    snap = fetch_osm_snapshot(bbox, event_date, **kwargs)
    pkg = Path(processed_root) / run_key(snap.aoi, snap.event_date)
    if pkg.is_dir():
        attach_to_package(snap, pkg)
    else:
        log.info("no analysis package at %s yet; run attach_to_package() after Part 3", pkg)
    return snap


# ============================================================================
# 10. CLI
# ============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fetch the pre-event OSM snapshot (ohsome) (Part 5).")
    p.add_argument("--scene-pair", help="scene_pair.json from Part 1 (AOI + event date + package)")
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    p.add_argument("--date", help="event date YYYY-MM-DD (with --bbox)")
    p.add_argument("--snapshot-date", default=None, help="override; must be before the event")
    p.add_argument("--layers", nargs="+", default=None, help=f"default: {' '.join(DEFAULT_LAYERS)}")
    p.add_argument("--cache-root", default="data/raw/osm")
    p.add_argument("--processed-root", default="data/processed")
    p.add_argument("--no-attach", action="store_true", help="with --bbox: only fetch raw files")
    p.add_argument("--fallback-dir", default=None, help="bundled sample layers (demo safety net)")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--selftest", action="store_true", help="run the offline unit tests and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    if a.selftest:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestOsm)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not a.scene_pair and not (a.bbox and a.date):
        p.error("give --scene-pair, or --bbox and --date (or use --selftest)")
    try:
        cfg = OsmConfig(cache_root=Path(a.cache_root))
        kw = dict(snapshot_date=a.snapshot_date, layers=a.layers,
                  fallback_dir=a.fallback_dir, offline=a.offline)
        if a.scene_pair:
            from acquisition.sentinel import ScenePair            # Part 1
            pair = ScenePair.load_json(a.scene_pair)
            out_dir = Path(a.processed_root) / run_key(pair.aoi, pair.event_date)
            snap = build_osm_layers(pair, out_dir, cfg, raise_errors=True, **kw)
        elif a.no_attach:
            snap = fetch_osm_snapshot(a.bbox, a.date, config=cfg, **kw)
        else:
            snap = fetch_and_attach(a.bbox, a.date, processed_root=a.processed_root, config=cfg, **kw)
    except (AcquisitionError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(snap.summary())
    return 0


# ============================================================================
# 11. SELF-TESTS (offline: fake HTTP session, temp folders)
#     Run:  python -m acquisition.ohsome --selftest
#     (Part 5 test list: snapshot date < event date, cache, attach, failures)
# ============================================================================

class _FakeResp:
    def __init__(self, code=200, payload=None, text=""):
        self.status_code, self._payload, self.text = code, payload, text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeSession:
    """Answers each POST from a handler(params) -> _FakeResp | exception."""

    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def post(self, url, data=None, timeout=None):
        self.calls.append(dict(data or {}, _url=url))
        out = self.handler(data)
        if isinstance(out, Exception):
            raise out
        return out


def _road(osm_id: str, x0: float, y0: float, x1: float, y1: float, edit="2026-06-01T00:00:00Z"):
    return {"type": "Feature", "id": osm_id,
            "properties": {"@osmId": osm_id, "highway": "track", "@lastEdit": edit},
            "geometry": {"type": "LineString", "coordinates": [[x0, y0], [x1, y1]]}}


def _ok(features):
    return _FakeResp(200, {"type": "FeatureCollection", "features": features})


_BBOX = (85.30, 28.00, 85.35, 28.05)      # ~5 km x 5.5 km, one cell
_EVENT = "2026-08-26"


class TestOsm(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.cfg = OsmConfig(cache_root=self.tmp / "osm", retry_base_delay_s=0.0)

    def tearDown(self):
        self._tmp.cleanup()

    def run_fetch(self, handler, **kw):
        s = _FakeSession(handler)
        snap = fetch_osm_snapshot(_BBOX, _EVENT, config=self.cfg, session=s,
                                  sleep=lambda _: None, **kw)
        return snap, s

    # ---- the rule that can disqualify the team
    def test_osm_snapshot_date(self):
        self.assertEqual(resolve_snapshot_date("2026-08-26", self.cfg), date(2026, 7, 27))
        self.assertEqual(resolve_snapshot_date("2026-07-01", self.cfg), date(2026, 6, 30))
        for bad in ("2026-08-26", "2026-09-01"):
            with self.assertRaises(SnapshotDateError):
                resolve_snapshot_date("2026-08-26", self.cfg, requested=bad)
        self.assertEqual(resolve_snapshot_date("2026-08-26", self.cfg, requested="2026-08-10"),
                         date(2026, 8, 10))
        with self.assertRaises(ValueError):                  # also an InvalidInputError
            assert_snapshot_before_event("2026-08-26", "2026-08-26")

    def test_request_asks_for_snapshot_not_now(self):
        snap, s = self.run_fetch(lambda p: _ok([_road("way/1", 85.31, 28.01, 85.34, 28.04)]))
        self.assertEqual(snap.snapshot_date, date(2026, 7, 27))
        self.assertLess(snap.snapshot_date, date.fromisoformat(_EVENT))
        first = s.calls[0]
        self.assertEqual(first["time"], "2026-07-27")
        self.assertEqual(first["clipGeometry"], "false")
        self.assertTrue(first["_url"].startswith("https://api.ohsome.org/v1"))
        self.assertEqual(first["bboxes"], "85.300000,28.000000,85.350000,28.050000")

    def test_late_edit_is_refused(self):
        late = [_road("way/1", 85.31, 28.01, 85.34, 28.04, edit="2026-09-02T10:00:00Z")]
        with self.assertRaises(SnapshotDateError):
            self.run_fetch(lambda p: _ok(late))

    # ---- geometry helpers
    def test_split_bbox_tiles_exactly(self):
        aoi = AOI.from_bbox((85.0, 27.0, 85.5, 27.3))
        cells = split_bbox(aoi, 0.2)
        self.assertEqual(len(cells), 3 * 2)
        self.assertTrue(all(c[2] - c[0] <= 0.2 + 1e-9 and c[3] - c[1] <= 0.2 + 1e-9 for c in cells))
        self.assertAlmostEqual(sum((c[2] - c[0]) * (c[3] - c[1]) for c in cells), 0.5 * 0.3)
        self.assertEqual(len(split_bbox(AOI.from_bbox(_BBOX), 0.2)), 1)

    def test_road_length(self):
        aoi = AOI.from_bbox((85.0, 28.0, 85.1, 28.1))
        f = _road("way/1", 85.0, 28.0, 85.0, 28.01)           # 0.01 deg of latitude
        self.assertAlmostEqual(road_length_km([f], aoi), 0.01 * KM_PER_DEG, places=3)
        out = _road("way/2", 86.0, 28.0, 86.0, 28.01)         # outside the AOI
        self.assertEqual(road_length_km([out], aoi), 0.0)

    # ---- cells, de-duplication, splitting
    def test_multi_cell_dedupe(self):
        self.cfg.split_deg = 0.03                              # 2 x 2 cells
        shared = _road("way/7", 85.31, 28.01, 85.34, 28.04)
        snap, s = self.run_fetch(lambda p: _ok([shared]), layers=["roads"])
        self.assertEqual(len(s.calls), 4)
        self.assertEqual(snap.layers["roads"].stats["count"], 1)

    def test_big_cell_is_split_on_timeout(self):
        def handler(p):
            x0, y0, x1, y1 = map(float, p["bboxes"].split(","))
            if x1 - x0 > 0.04:
                return requests.Timeout("slow")
            return _ok([_road(f"way/{x0:.3f}{y0:.3f}", x0 + 0.001, y0 + 0.001, x0 + 0.002, y0 + 0.002)])
        snap, s = self.run_fetch(handler, layers=["roads"])
        self.assertEqual(snap.layers["roads"].stats["count"], 4)   # 4 quarters
        self.assertEqual(len(s.calls), self.cfg.max_retries + 4)

    # ---- retries and failures
    def test_retry_then_ok(self):
        answers = iter([_FakeResp(429), _FakeResp(503), _ok([_road("way/1", 85.31, 28.01, 85.34, 28.04)])])
        snap, s = self.run_fetch(lambda p: next(answers), layers=["roads"])
        self.assertEqual(len(s.calls), 3)
        self.assertEqual(snap.layers["roads"].stats["count"], 1)

    def test_required_layer_failure_raises_with_server_text(self):
        with self.assertRaises(OsmError) as cm:
            self.run_fetch(lambda p: _FakeResp(400, None, "invalid filter"), layers=["roads"])
        self.assertIn("invalid filter", str(cm.exception))

    def test_optional_layer_failure_is_a_warning(self):
        def handler(p):
            if p["filter"].startswith("amenity"):
                return _FakeResp(400, None, "nope")
            return _ok([_road("way/1", 85.31, 28.01, 85.34, 28.04)])
        snap, _ = self.run_fetch(handler, layers=["roads", "hospitals"])
        self.assertIn("hospitals", snap.missing)
        self.assertTrue(any("hospitals" in w for w in snap.warnings))
        self.assertIn("roads", snap.files)

    def test_unknown_layer(self):
        with self.assertRaises(InvalidInputError):
            self.run_fetch(lambda p: _ok([]), layers=["roads", "volcanoes"])

    # ---- cache
    def test_cache_hit_and_key(self):
        h = lambda p: _ok([_road("way/1", 85.31, 28.01, 85.34, 28.04)])
        first, s1 = self.run_fetch(h, layers=["roads"])
        second, s2 = self.run_fetch(h, layers=["roads"])
        self.assertEqual(len(s1.calls), 1)
        self.assertEqual(len(s2.calls), 0)
        self.assertTrue(second.layers["roads"].from_cache)
        third, s3 = self.run_fetch(h, layers=["roads"], snapshot_date="2026-07-20")   # other key
        self.assertEqual(len(s3.calls), 1)
        path = first.layers["roads"].path                       # corrupt file => refetch
        path.write_text("{ not json")
        _, s4 = self.run_fetch(h, layers=["roads"])
        self.assertEqual(len(s4.calls), 1)

    def test_offline_mode(self):
        with self.assertRaises(OsmError):
            self.run_fetch(lambda p: self.fail("network used"), offline=True, layers=["roads"])
        h = lambda p: _ok([_road("way/1", 85.31, 28.01, 85.34, 28.04)])
        self.run_fetch(h, layers=["roads"])
        snap, s = self.run_fetch(lambda p: self.fail("network used"), offline=True, layers=["roads"])
        self.assertTrue(snap.layers["roads"].from_cache)

    # ---- demo-day fallback
    def test_fallback_sample(self):
        sample = self.tmp / "sample"
        write_layer(sample / "roads.geojson", "roads",
                    [_road("way/9", 85.31, 28.01, 85.34, 28.04)], AOI.from_bbox(_BBOX),
                    "x", date(2026, 7, 27))
        snap, _ = self.run_fetch(lambda p: requests.ConnectionError("down"), layers=["roads"],
                                 fallback_dir=sample)
        self.assertEqual(snap.layers["roads"].source, "fallback")
        self.assertTrue(any("bundled sample" in w for w in snap.warnings))

    def test_fallback_after_event_is_refused(self):
        sample = self.tmp / "sample"
        write_layer(sample / "roads.geojson", "roads", [_road("way/9", 85.31, 28.01, 85.34, 28.04)],
                    AOI.from_bbox(_BBOX), "x", date(2026, 9, 1))      # post-event!
        with self.assertRaises(OsmError):
            self.run_fetch(lambda p: requests.ConnectionError("down"), layers=["roads"],
                           fallback_dir=sample)

    # ---- quality warnings
    def test_sparse_and_empty_warnings(self):
        snap, _ = self.run_fetch(lambda p: _ok([]), layers=["roads", "settlements"])
        self.assertTrue(any("NO mapped roads" in w for w in snap.warnings))
        self.assertTrue(any("settlement" in w for w in snap.warnings))
        tiny = [_road("way/1", 85.31, 28.01, 85.311, 28.011)]
        snap2, _ = self.run_fetch(lambda p: _ok(tiny), layers=["roads"], snapshot_date="2026-07-10")
        self.assertTrue(any("sparse" in w for w in snap2.warnings))

    # ---- hand-off
    def test_attach_to_package(self):
        snap, _ = self.run_fetch(lambda p: _ok([_road("way/1", 85.31, 28.01, 85.34, 28.04)]),
                                 layers=["roads"])
        pkg = self.tmp / "pkg"
        pkg.mkdir()
        (pkg / "metadata.json").write_text(json.dumps(
            {"attributions": ["Contains modified Copernicus Sentinel data 2026."], "warnings": ["w0"]}))
        attach_to_package(snap, pkg)
        attach_to_package(snap, pkg)                            # idempotent
        meta = json.loads((pkg / "metadata.json").read_text())
        self.assertEqual(meta["osm"]["snapshot_date"], "2026-07-27")
        self.assertEqual(meta["attributions"].count(OSM_ATTRIBUTION), 1)
        self.assertIn("w0", meta["warnings"])
        self.assertTrue((pkg / "osm" / "roads.geojson").exists())
        self.assertTrue((pkg / "osm.json").exists())
        self.assertEqual(meta["osm"]["status"], "used")
        gj = json.loads((pkg / "osm" / "roads.geojson").read_text())
        self.assertEqual(gj["osm_snapshot"], "2026-07-27")
        self.assertEqual(gj["attribution"], OSM_ATTRIBUTION)

    def test_attach_without_metadata_does_not_crash(self):
        snap, _ = self.run_fetch(lambda p: _ok([_road("way/1", 85.31, 28.01, 85.34, 28.04)]),
                                 layers=["roads"])
        pkg = self.tmp / "empty_pkg"
        self.assertTrue(attach_to_package(snap, pkg).exists())

    def test_progress_callback(self):
        msgs = []
        self.run_fetch(lambda p: _ok([_road("way/1", 85.31, 28.01, 85.34, 28.04)]),
                       layers=["roads", "bridges"], progress=lambda m, f: msgs.append((m, f)))
        self.assertEqual(msgs[0][1], 0.0)
        self.assertEqual(msgs[-1], ("OSM done", 1.0))

    # ---- Part 4-style pipeline entry (ScenePair in, status out, never breaks the pipeline)
    def _pair(self, event=date(2026, 8, 26)):
        from datetime import datetime as dt
        from acquisition.sentinel import Acquisition, Scene, ScenePair
        ring = [[84.0, 27.0], [86.0, 27.0], [86.0, 29.0], [84.0, 29.0], [84.0, 27.0]]

        def sc(day):
            d = dt(2026, 8, day, 12, tzinfo=timezone.utc)
            n = f"S1A_IW_GRDH_1SDV_{d:%Y%m%dT%H%M%S}_{d:%Y%m%dT%H%M%S}_050000_06B000_ABCD.SAFE"
            return Scene(product_id=n, name=n, platform="S1A", start=d, end=d, rel_orbit=41,
                         direction="DESCENDING", rings=[ring], abs_orbit=50000 + day,
                         datatake_id="06B000", size_bytes=1000)
        return ScenePair(AOI.from_bbox(_BBOX), event, Acquisition([sc(14)]), Acquisition([sc(28)]),
                         1.0, "standard")

    def _pkg(self):
        pkg = self.tmp / "pkg"
        pkg.mkdir(exist_ok=True)
        (pkg / "metadata.json").write_text(json.dumps({"attributions": ["S1"], "warnings": []}))
        return pkg

    def _build(self, handler, pkg, **kw):
        s = _FakeSession(handler)
        res = build_osm_layers(self._pair(), pkg, self.cfg, session=s, sleep=lambda _: None, **kw)
        return res, s

    def test_build_used_then_cache_hit(self):
        pkg = self._pkg()
        h = lambda p: _ok([_road("way/1", 85.31, 28.01, 85.34, 28.04)])
        res, s1 = self._build(h, pkg)
        self.assertEqual((res.status, res.reason_code), ("used", "ok"))
        self.assertEqual(len(res.files), 5)
        meta = json.loads((pkg / "metadata.json").read_text())
        self.assertEqual(meta["osm"]["status"], "used")
        self.assertIn(OSM_ATTRIBUTION, meta["attributions"])
        self.assertIn("params_hash", json.loads((pkg / "osm.json").read_text()))
        # Part 3 rebuilt metadata.json: a cache hit must put the osm entry back
        (pkg / "metadata.json").write_text(json.dumps({"attributions": ["S1"], "warnings": []}))
        res2, s2 = self._build(lambda p: self.fail("network used"), pkg)
        self.assertTrue(res2.from_cache)
        self.assertEqual(len(s2.calls), 0)
        self.assertEqual(res2.status, "used")
        self.assertEqual(json.loads((pkg / "metadata.json").read_text())["osm"]["status"], "used")

    def test_build_never_breaks_pipeline(self):
        pkg = self._pkg()
        res, _ = self._build(lambda p: requests.ConnectionError("down"), pkg)
        self.assertEqual((res.status, res.reason_code), ("unavailable", "error"))
        self.assertIn("connection error", res.reason)
        meta = json.loads((pkg / "metadata.json").read_text())
        self.assertEqual(meta["osm"]["status"], "unavailable")
        self.assertNotIn(OSM_ATTRIBUTION, meta["attributions"])      # no OSM data was used
        self.assertTrue(any("unavailable" in w for w in meta["warnings"]))
        # errors are never cached: the next call tries again
        ok = lambda p: _ok([_road("way/1", 85.31, 28.01, 85.34, 28.04)])
        res2, s2 = self._build(ok, pkg)
        self.assertEqual(res2.status, "used")
        self.assertGreater(len(s2.calls), 0)

    def test_build_raise_errors_flag(self):
        with self.assertRaises(OsmError):
            self._build(lambda p: requests.ConnectionError("down"), self._pkg(), raise_errors=True)

    def test_build_always_raises_on_rule_breach(self):
        with self.assertRaises(SnapshotDateError):
            self._build(lambda p: _ok([]), self._pkg(), snapshot_date="2026-08-26")
        late = [_road("way/1", 85.31, 28.01, 85.34, 28.04, edit="2026-09-02T00:00:00Z")]
        with self.assertRaises(SnapshotDateError):
            self._build(lambda p: _ok(late), self._pkg())

    def test_build_partial_and_ttl(self):
        pkg = self._pkg()

        def h(p):
            if p["filter"].startswith("amenity"):
                return _FakeResp(400, None, "nope")
            return _ok([_road("way/1", 85.31, 28.01, 85.34, 28.04)])
        res, s1 = self._build(h, pkg)
        self.assertEqual((res.status, res.reason_code), ("partial", "missing_layers"))
        _, s2 = self._build(h, pkg)                                  # fresh partial: cached
        self.assertEqual(len(s2.calls), 0)
        info = json.loads((pkg / "osm.json").read_text())           # age it beyond the TTL
        info["created_utc"] = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat(timespec="seconds")
        (pkg / "osm.json").write_text(json.dumps(info))
        _, s3 = self._build(h, pkg)
        self.assertGreater(len(s3.calls), 0)                         # re-tried

    def test_hash_ignores_network_knobs_but_not_filters(self):
        aoi, snap = AOI.from_bbox(_BBOX), date(2026, 7, 27)
        base = _osm_hash(OsmConfig(), aoi, snap, ["roads"])
        self.assertEqual(base, _osm_hash(OsmConfig(timeout_s=5, max_retries=1), aoi, snap, ["roads"]))
        other = OsmConfig()
        other.layers["roads"] = "highway=primary"
        self.assertNotEqual(base, _osm_hash(other, aoi, snap, ["roads"]))
        self.assertNotEqual(base, _osm_hash(OsmConfig(), aoi, date(2026, 7, 20), ["roads"]))

    def test_cli_scene_pair(self):
        pair = self._pair()
        sp = pair.save_json(self.tmp / "scene_pair.json")
        proc = self.tmp / "processed"
        (proc / run_key(pair.aoi, pair.event_date)).mkdir(parents=True)
        rc = main(["--scene-pair", str(sp), "--processed-root", str(proc),
                   "--cache-root", str(self.tmp / "osm"), "--offline"])
        self.assertEqual(rc, 1)                                      # offline + empty cache => clear error


if __name__ == "__main__":
    raise SystemExit(main())
