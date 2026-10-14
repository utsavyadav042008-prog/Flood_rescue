"""
Offline tests for acquisition/sentinel.py  (no network, no CDSE account).

Run from the repo root:
    python -m unittest tests.test_acquisition -v
    (or:  pytest tests/test_acquisition.py)

Synthetic scenes use plain rectangles as footprints, so every expected
coverage number can be checked by hand.
"""
import hashlib
import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from acquisition.sentinel import (
    AOI, AcquisitionConfig, CDSEClient, CoverageGrid, DownloadError,
    InvalidInputError, NoValidPairError, Relaxation, Scene, build_odata_filter,
    group_acquisitions, parse_product, select_pair,
)

AOI_BOX = (85.1, 27.9, 85.5, 28.3)          # illustrative, 0.4 x 0.4 degrees
EVENT = date(2026, 8, 26)


def ring(minx, miny, maxx, maxy):
    return [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]


FULL = ring(84.0, 27.0, 86.0, 29.0)          # covers the whole AOI


def mk(day, track=41, direction="DESCENDING", platform="S1A", rings=None,
       abs_orbit=None, datatake="06B000", hour=12, pid=None):
    """Build a synthetic Scene. Different (platform, abs_orbit, datatake) = different pass."""
    d = datetime(2026, 8, 1, hour, 0, tzinfo=timezone.utc).replace(
        month=day[0], day=day[1])
    abs_orbit = abs_orbit if abs_orbit is not None else 50000 + day[0] * 100 + day[1]
    name = (f"{platform}_IW_GRDH_1SDV_{d:%Y%m%dT%H%M%S}_{d:%Y%m%dT%H%M%S}_"
            f"{abs_orbit:06d}_{datatake}_ABCD.SAFE")
    return Scene(product_id=pid or name, name=name, platform=platform, start=d, end=d,
                 rel_orbit=track, direction=direction, rings=rings or [FULL],
                 abs_orbit=abs_orbit, datatake_id=datatake, size_bytes=1000)


class TestAOI(unittest.TestCase):
    def test_valid_and_area(self):
        a = AOI.from_bbox(AOI_BOX)
        # 0.4 deg x 0.4 deg at ~28 N is roughly 44 km x 39 km
        self.assertTrue(1500 < a.area_km2() < 2000)

    def test_swapped_lat_lon_rejected(self):
        with self.assertRaises(InvalidInputError):
            # An Australian bbox with lat/lon swapped: "latitude" becomes 150 > 90
            AOI.from_bbox((-33.0, 150.0, -32.0, 151.0))

    def test_min_ge_max_rejected(self):
        with self.assertRaises(InvalidInputError):
            AOI.from_bbox((85.5, 27.9, 85.1, 28.3))

    def test_odata_filter_pieces(self):
        f = build_odata_filter(AOI.from_bbox(AOI_BOX), date(2026, 7, 1), date(2026, 9, 1),
                               AcquisitionConfig())
        self.assertIn("Collection/Name eq 'SENTINEL-1'", f)
        self.assertIn("SRID=4326;POLYGON((85.10000 27.90000", f)
        self.assertIn("IW_GRDH_1S", f)


class TestCoverageGrid(unittest.TestCase):
    def test_half_covered(self):
        aoi = AOI.from_bbox(AOI_BOX)
        grid = CoverageGrid(aoi, 200)
        half = mk((8, 14), rings=[ring(84, 27, 86, 28.1)])      # lat 27.9..28.1 of 27.9..28.3
        self.assertAlmostEqual(grid.fraction(grid.mask(half)), 0.5, delta=0.01)


class TestParse(unittest.TestCase):
    def item(self, name, pol="VV&VH"):
        return {
            "Id": "abc", "Name": name, "ContentLength": 12345, "Online": True,
            "ContentDate": {"Start": "2026-08-14T12:00:00.123Z", "End": "2026-08-14T12:00:25.000Z"},
            "GeoFootprint": {"type": "Polygon", "coordinates": [FULL]},
            "Checksum": [{"Algorithm": "MD5", "Value": "ABCDEF"}],
            "Attributes": [
                {"Name": "relativeOrbitNumber", "Value": "41"},
                {"Name": "orbitDirection", "Value": "descending"},
                {"Name": "polarisationChannels", "Value": pol}],
        }

    def test_dual_pol_parsed(self):
        s = parse_product(self.item("S1A_IW_GRDH_1SDV_20260814T120000_20260814T120025_055000_06B000_ABCD.SAFE"))
        self.assertEqual((s.platform, s.rel_orbit, s.direction), ("S1A", 41, "DESCENDING"))
        self.assertEqual(s.datatake_id, "06B000")
        self.assertEqual(s.md5, "abcdef")
        self.assertEqual(s.start.tzinfo, timezone.utc)

    def test_single_pol_skipped(self):
        self.assertIsNone(parse_product(self.item(
            "S1A_IW_GRDH_1SSV_20260814T120000_20260814T120025_055000_06B000_ABCD.SAFE")))

    def test_wkt_footprint_fallback(self):
        it = self.item("S1A_IW_GRDH_1SDV_20260814T120000_20260814T120025_055000_06B000_ABCD.SAFE")
        del it["GeoFootprint"]
        it["Footprint"] = "geography'SRID=4326;POLYGON ((84 27, 86 27, 86 29, 84 29, 84 27))'"
        s = parse_product(it)
        self.assertEqual(len(s.rings[0]), 5)

    def test_missing_track_skipped(self):
        it = self.item("S1A_IW_GRDH_1SDV_20260814T120000_20260814T120025_055000_06B000_ABCD.SAFE")
        it["Attributes"] = []
        self.assertIsNone(parse_product(it))


class TestSelection(unittest.TestCase):
    aoi = AOI.from_bbox(AOI_BOX)

    def test_basic_same_track_pair(self):
        sc = [mk((8, 2)), mk((8, 14)), mk((8, 28)), mk((9, 9))]
        p = select_pair(sc, self.aoi, EVENT)
        self.assertEqual(p.before.date, date(2026, 8, 14))     # latest before
        self.assertEqual(p.after.date, date(2026, 8, 28))      # earliest after
        self.assertEqual(p.gap_days, 14)
        self.assertGreater(p.coverage, 0.99)

    def test_never_mixes_tracks(self):
        # before only on track 41, after only on track 12 -> must FAIL
        sc = [mk((8, 14), track=41), mk((8, 28), track=12)]
        with self.assertRaises(NoValidPairError) as ctx:
            select_pair(sc, self.aoi, EVENT)
        text = str(ctx.exception)
        self.assertIn("track 41", text)
        self.assertIn("track 12", text)

    def test_never_mixes_directions(self):
        sc = [mk((8, 14), direction="ASCENDING"), mk((8, 28), direction="DESCENDING")]
        with self.assertRaises(NoValidPairError):
            select_pair(sc, self.aoi, EVENT)

    def test_prefers_same_platform_over_shorter_cross_platform_gap(self):
        sc = [mk((8, 14), platform="S1A"), mk((8, 20), platform="S1C"),
              mk((8, 28), platform="S1A")]
        p = select_pair(sc, self.aoi, EVENT)
        self.assertEqual(p.before.platform, p.after.platform)
        self.assertEqual(p.before.date, date(2026, 8, 14))

    def test_picks_other_track_when_nearest_covers_only_a_corner(self):
        corner = [ring(85.1, 27.9, 85.2, 28.0)]                # tiny corner of the AOI
        sc = [mk((8, 20), track=41, rings=corner), mk((8, 28), track=41, rings=corner),
              mk((8, 14), track=12), mk((8, 28), track=12)]
        p = select_pair(sc, self.aoi, EVENT)
        self.assertEqual(p.track[0], 12)

    def test_slices_of_one_pass_are_merged(self):
        south = ring(84, 27, 86, 28.12)
        north = ring(84, 28.08, 86, 29)                       # slices overlap a bit
        sc = []
        for day in [(8, 14), (8, 28)]:
            sc += [mk(day, rings=[south], datatake="AAAAAA", pid=f"s{day}"),
                   mk(day, rings=[north], datatake="AAAAAA", pid=f"n{day}")]
        # same datatake => same name?  mk() names are unique per call via pid only
        p = select_pair(sc, self.aoi, EVENT)
        self.assertEqual(len(p.before.slices), 2)
        self.assertEqual(len(p.after.slices), 2)
        self.assertGreater(p.coverage, 0.99)

    def test_useless_sliver_slice_is_not_downloaded(self):
        main = ring(84, 27, 86, 28.298)
        sliver = ring(84, 28.297, 86, 29)                     # adds ~0.5 % of AOI
        sc = []
        for day in [(8, 14), (8, 28)]:
            sc += [mk(day, rings=[main], datatake="AAAAAA", pid=f"m{day}"),
                   mk(day, rings=[sliver], datatake="AAAAAA", pid=f"x{day}")]
        p = select_pair(sc, self.aoi, EVENT)
        self.assertEqual(len(p.before.slices), 1)
        self.assertEqual(len(p.after.slices), 1)

    def test_event_day_scene_is_flagged(self):
        p = select_pair([mk((8, 14)), mk((8, 26))], self.aoi, EVENT)
        self.assertEqual(p.after.date, EVENT)
        self.assertTrue(any("event date itself" in w for w in p.warnings))

    def test_relaxation_ladder_partial_coverage(self):
        part = [ring(84, 27, 86, 28.18)]                      # ~70 % of the AOI (rung 3 needs 60 %)
        p = select_pair([mk((8, 14), rings=part), mk((8, 28), rings=part)],
                        self.aoi, EVENT)
        self.assertEqual(p.relaxation, "partial AOI coverage")
        self.assertTrue(any("relaxed" in w.lower() for w in p.warnings))

    def test_too_little_coverage_fails_with_reason(self):
        part = [ring(84, 27, 86, 27.95)]                      # ~12 %
        with self.assertRaises(NoValidPairError) as ctx:
            select_pair([mk((8, 14), rings=part), mk((8, 28), rings=part)], self.aoi, EVENT)
        self.assertIn("coverage", str(ctx.exception))

    def test_no_scenes_message(self):
        with self.assertRaises(NoValidPairError) as ctx:
            select_pair([], self.aoi, EVENT)
        self.assertIn("No Sentinel-1", str(ctx.exception))

    def test_no_after_scene_message(self):
        with self.assertRaises(NoValidPairError) as ctx:
            select_pair([mk((8, 2)), mk((8, 14))], self.aoi, EVENT)
        self.assertIn("no scene on/after", str(ctx.exception))

    def test_long_gap_warning(self):
        p = select_pair([mk((7, 20)), mk((8, 28))], self.aoi, EVENT)
        self.assertTrue(any("Long gap" in w for w in p.warnings))

    def test_before2_same_track(self):
        sc = [mk((8, 2)), mk((8, 14)), mk((8, 28)), mk((7, 22), track=12)]
        p = select_pair(sc, self.aoi, EVENT, want_before2=True)
        self.assertEqual(p.before2.date, date(2026, 8, 2))
        self.assertEqual(p.before2.track_key, p.before.track_key)

    def test_json_roundtrip(self):
        p = select_pair([mk((8, 14)), mk((8, 28))], self.aoi, EVENT)
        with tempfile.TemporaryDirectory() as td:
            path = p.save_json(Path(td) / "x" / "scene_pair.json")
            d = json.loads(path.read_text())
        self.assertEqual(d["gap_days"], 14)
        self.assertEqual(d["track"]["relative_orbit"], 41)

    def test_group_acquisitions_separates_passes_same_day(self):
        a = mk((8, 14), datatake="AAAAAA", pid="a", hour=5)
        b = mk((8, 14), datatake="BBBBBB", pid="b", hour=17, abs_orbit=99999)
        self.assertEqual(len(group_acquisitions([a, b])), 2)


# --------------------------------------------------------------------------
# Download tests with a fake HTTP session
# --------------------------------------------------------------------------
class FakeResp:
    def __init__(self, status, body=b"", headers=None, json_data=None):
        self.status_code, self.body = status, body
        self.headers, self._json = headers or {}, json_data
        self.text = ""

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]

    def json(self):
        return self._json

    def close(self):
        pass


class FakeSession:
    """Mimics: download host -> 302 -> file host (needs Authorization, supports Range)."""

    def __init__(self, payload, first_401=False, ignore_range=False):
        self.payload, self.first_401, self.ignore_range = payload, first_401, ignore_range
        self.logins, self.ranges, self.auth_seen = 0, [], []

    def post(self, url, data=None, timeout=None):
        self.logins += 1
        return FakeResp(200, json_data={"access_token": f"tok{self.logins}", "expires_in": 600})

    def get(self, url, headers=None, stream=False, allow_redirects=True, timeout=None, params=None):
        headers = headers or {}
        if "download.fake" in url:
            return FakeResp(302, headers={"Location": "https://files.fake/blob"})
        self.auth_seen.append(headers.get("Authorization"))
        if self.first_401:
            self.first_401 = False
            return FakeResp(401)
        rng = headers.get("Range")
        self.ranges.append(rng)
        if rng and not self.ignore_range:
            start = int(rng.split("=")[1].rstrip("-"))
            return FakeResp(206, self.payload[start:])
        return FakeResp(200, self.payload)


class TestDownload(unittest.TestCase):
    payload = bytes(range(256)) * 40            # 10 240 bytes

    def make_client(self, session):
        cfg = AcquisitionConfig(download_url="https://download.fake/odata/v1",
                                retry_base_delay_s=0.0, chunk_bytes=1000,
                                search_cache_dir=None)
        return CDSEClient(cfg, username="u", password="p", session=session)

    def scene(self, md5=None):
        s = mk((8, 14))
        s.size_bytes = len(self.payload)
        s.md5 = md5 or hashlib.md5(self.payload).hexdigest()
        return s

    def test_redirect_keeps_auth_header(self):
        sess = FakeSession(self.payload)
        with tempfile.TemporaryDirectory() as td:
            out = self.make_client(sess).download(self.scene(), Path(td))
            self.assertEqual(out.read_bytes(), self.payload)
            self.assertFalse(list(Path(td).glob("*.part")))
        self.assertEqual(sess.auth_seen, ["Bearer tok1"])

    def test_resume_uses_range(self):
        sess = FakeSession(self.payload)
        sc = self.scene()
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / f"{sc.stem}.zip.part").write_bytes(self.payload[:4000])
            out = self.make_client(sess).download(sc, Path(td))
            self.assertEqual(out.read_bytes(), self.payload)
        self.assertEqual(sess.ranges, ["bytes=4000-"])

    def test_server_ignoring_range_restarts_cleanly(self):
        sess = FakeSession(self.payload, ignore_range=True)
        sc = self.scene()
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / f"{sc.stem}.zip.part").write_bytes(self.payload[:4000])
            out = self.make_client(sess).download(sc, Path(td))
            self.assertEqual(out.read_bytes(), self.payload)

    def test_expired_token_refreshes_and_retries(self):
        sess = FakeSession(self.payload, first_401=True)
        with tempfile.TemporaryDirectory() as td:
            out = self.make_client(sess).download(self.scene(), Path(td))
            self.assertEqual(out.read_bytes(), self.payload)
        self.assertEqual(sess.logins, 2)                        # logged in again

    def test_cache_hit_skips_network(self):
        sess = FakeSession(self.payload)
        sc = self.scene()
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / f"{sc.stem}.zip").write_bytes(self.payload)
            self.make_client(sess).download(sc, Path(td))
        self.assertEqual(sess.logins, 0)

    def test_md5_mismatch_raises_and_deletes(self):
        sess = FakeSession(self.payload)
        sc = self.scene(md5="0" * 32)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(DownloadError):
                self.make_client(sess).download(sc, Path(td))
            self.assertEqual(list(Path(td).iterdir()), [])      # nothing left behind

    def test_missing_credentials(self):
        from acquisition.sentinel import AuthError
        c = CDSEClient(AcquisitionConfig(download_url="https://download.fake/odata/v1",
                                         retry_base_delay_s=0.0, search_cache_dir=None),
                       session=FakeSession(self.payload))
        c.username = c.password = None
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(AuthError):
                c.download(self.scene(), Path(td))


if __name__ == "__main__":
    unittest.main()
