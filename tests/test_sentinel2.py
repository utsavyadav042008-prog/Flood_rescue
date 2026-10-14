"""
Offline tests for Part 4:  acquisition/sentinel2.py  and  preprocessing/sentinel2.py
(no rasterio, no network, no CDSE account).

Run from the repo root:
    python -m unittest tests.test_sentinel2 -v

A tiny synthetic "world" stands in for Sentinel-2: tiles with known reflectances,
a flooded patch, a debris patch, a river, and clouds, all at known positions, so
every expected result can be checked by hand.
"""
import json
import struct
import tempfile
import unittest
import zipfile
import zlib
from datetime import date
from pathlib import Path

import numpy as np

from acquisition.sentinel import AOI, Acquisition, AcquisitionError, CoverageGrid, ScenePair
from acquisition.sentinel2 import (
    FallbackBandSource, NodeBandSource, OpticalConfig, S2FetchError, ZipBandSource, boa_offset,
    choose_pair, find_member, group_passes, parse_s2_product, pick_side, rank_candidates)
from preprocessing.alignment import build_grid
from preprocessing.sentinel1 import GeoRaster
from preprocessing.sentinel2 import (
    BANDS, TileInfo, build_optical_layers, change_evidence, clear_from_scl, clear_mask,
    combine_first, compute_indices, grid_bounds_lonlat, grid_from_metadata, grid_window_in_tile,
    make_quicklook, normalized_difference, read_band_on_grid, shared_stretch, to_reflectance,
    write_png_rgba, _downscale)
from preprocessing.utm import lonlat_to_utm, utm_to_lonlat
from tests.test_acquisition import EVENT, mk
from tests.test_alignment import AOI_S, CRS, MemIO

GRID = build_grid(AOI_S, CRS)
RING = [[84.0, 27.0], [86.0, 27.0], [86.0, 29.0], [84.0, 29.0], [84.0, 27.0]]
MARGIN = 50                                           # tile extends 50 px beyond the grid


def s2_item(day, platform="S2B", tile="45RUL", cloud=10.0, orbit=19, baseline="0511",
            ring=RING, hour="050701"):
    name = f"{platform}_MSIL2A_202608{day:02d}T{hour}_N{baseline}_R{orbit:03d}_T{tile}_202608{day:02d}T091245.SAFE"
    item = {"Id": name, "Name": name, "ContentLength": 1000, "Online": True,
            "ContentDate": {"Start": f"2026-08-{day:02d}T05:07:01.000Z", "End": f"2026-08-{day:02d}T05:07:01.000Z"},
            "Attributes": [{"Name": "cloudCover", "Value": cloud}]}
    if ring is not None:
        item["GeoFootprint"] = {"type": "Polygon", "coordinates": [ring]}
    return item


def stem_of(item):
    return item["Name"][:-5]


# --------------------------------------------------------------------------
# Synthetic world
# --------------------------------------------------------------------------
LAND = {"B02": 0.04, "B03": 0.08, "B04": 0.05, "B08": 0.30, "B11": 0.20}
WATER = {"B02": 0.05, "B03": 0.07, "B04": 0.05, "B08": 0.03, "B11": 0.01}
DEBRIS = {"B02": 0.10, "B03": 0.12, "B04": 0.15, "B08": 0.17, "B11": 0.30}
CLOUD = {"B02": 0.50, "B03": 0.50, "B04": 0.50, "B08": 0.50, "B11": 0.40}

FLOOD = (40, 100, 40, 100)          # rows r0:r1, cols c0:c1 in GRID coordinates (even numbers)
DEBRIS_R = (120, 160, 120, 170)
CLOUD_R = (20, 40, 150, 170)
RIVER = (100, 110, 0, GRID.width // 2 * 2)


class FakeReader:
    """Holds synthetic band files: key -> (TileInfo, array). Counts reads."""

    def __init__(self):
        self.files, self.reads = {}, 0

    def info(self, path):
        return self.files[path][0]

    def read(self, path, c0, r0, nc, nr):
        self.reads += 1
        return self.files[path][1][r0:r0 + nr, c0:c0 + nc].astype(np.float32).copy()


class FakeSource:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def path_for(self, product, band):
        self.calls.append((product.stem, band))
        if self.fail:
            raise RuntimeError("boom")
        return f"{product.stem}|{band}"


class FakeClient:
    def __init__(self, items):
        self.items, self.searches = items, 0

    def search(self, aoi, start, end):
        self.searches += 1
        return self.items


def paint(arr, region, value, margin=MARGIN):
    r0, r1, c0, c1 = region
    arr[margin + r0:margin + r1, margin + c0:margin + c1] = value


def add_scene(reader, item, water=(), debris=(), cloud=(), full_cloud=False):
    """Register band files for one product in the fake reader (baseline N0511: offset -1000)."""
    h, w = GRID.height + 2 * MARGIN, GRID.width + 2 * MARGIN
    x0, y0 = GRID.x0 - MARGIN * 10, GRID.y0 + MARGIN * 10
    refl = {b: np.full((h, w), v, np.float32) for b, v in LAND.items()}
    scl = np.full((h, w), 4, np.uint8)
    for reg in water:
        for b, v in WATER.items():
            paint(refl[b], reg, v)
        paint(scl, reg, 6)
    for reg in debris:
        for b, v in DEBRIS.items():
            paint(refl[b], reg, v)
        paint(scl, reg, 5)
    for reg in cloud:
        for b, v in CLOUD.items():
            paint(refl[b], reg, v)
        paint(scl, reg, 9)
    if full_cloud:
        for b, v in CLOUD.items():
            refl[b][:] = v
        scl[:] = 9
    stem = stem_of(item)
    for b in ("B02", "B03", "B04", "B08"):
        dn = np.round(refl[b] * 10000 + 1000).astype(np.float32)
        reader.files[f"{stem}|{b}"] = (TileInfo(CRS, 10.0, x0, y0, w, h), dn)
    dn11 = np.round(refl["B11"] * 10000 + 1000).astype(np.float32)[::2, ::2]
    reader.files[f"{stem}|B11"] = (TileInfo(CRS, 20.0, x0, y0, dn11.shape[1], dn11.shape[0]), dn11)
    scl20 = scl[::2, ::2].astype(np.float32)
    reader.files[f"{stem}|SCL"] = (TileInfo(CRS, 20.0, x0, y0, scl20.shape[1], scl20.shape[0]), scl20)


def make_pair():
    return ScenePair(AOI_S, EVENT, Acquisition([mk((8, 14), direction="DESCENDING")]),
                     Acquisition([mk((8, 28), direction="DESCENDING")]), 1.0, "standard")


# --------------------------------------------------------------------------
class TestParsing(unittest.TestCase):
    def test_parse_fields(self):
        p = parse_s2_product(s2_item(20, cloud=33.5))
        self.assertEqual((p.platform, p.tile, p.rel_orbit), ("S2B", "45RUL", 19))
        self.assertAlmostEqual(p.baseline, 5.11)
        self.assertEqual(p.date, date(2026, 8, 20))
        self.assertEqual(p.cloud_cover, 33.5)
        self.assertEqual(p.stem, p.name[:-5])

    def test_skips_other_levels_and_missing_footprint(self):
        it = s2_item(20)
        it["Name"] = it["Name"].replace("MSIL2A", "MSIL1C")
        self.assertIsNone(parse_s2_product(it))
        self.assertIsNone(parse_s2_product(s2_item(20, ring=None)))

    def test_missing_cloud_attribute(self):
        it = s2_item(20)
        it["Attributes"] = []
        self.assertIsNone(parse_s2_product(it).cloud_cover)

    def test_boa_offset_by_baseline(self):
        self.assertEqual(boa_offset(5.11), -1000.0)
        self.assertEqual(boa_offset(4.00), -1000.0)          # boundary: offset starts at 04.00
        self.assertEqual(boa_offset(3.01), 0.0)
        self.assertEqual(boa_offset(5.11, override=0.0), 0.0)


class TestFindMember(unittest.TestCase):
    names = ["X.SAFE/GRANULE/L2A_T45RUL_A1/IMG_DATA/R10m/T45RUL_20260820T050701_B02_10m.jp2",
             "X.SAFE/GRANULE/L2A_T45RUL_A1/IMG_DATA/R20m/T45RUL_20260820T050701_B02_20m.jp2",
             "X.SAFE/GRANULE/L2A_T45RUL_A1/IMG_DATA/R20m/T45RUL_20260820T050701_B11_20m.jp2",
             "X.SAFE/GRANULE/L2A_T45RUL_A1/IMG_DATA/R20m/T45RUL_20260820T050701_SCL_20m.jp2",
             "X.SAFE/GRANULE/L2A_T45RUL_A1/IMG_DATA/R60m/T45RUL_20260820T050701_B11_60m.jp2",
             "X.SAFE/MTD_MSIL2A.xml"]

    def test_prefers_native_resolution(self):
        self.assertTrue(find_member(self.names, "B02").endswith("R10m/T45RUL_20260820T050701_B02_10m.jp2"))
        self.assertIn("R20m", find_member(self.names, "B11"))
        self.assertIn("R20m", find_member(self.names, "SCL"))

    def test_falls_back_to_60m_and_none(self):
        self.assertIn("R60m", find_member([n for n in self.names if "R20m" not in n], "B11"))
        self.assertIsNone(find_member(self.names, "B08"))


class TestBandSources(unittest.TestCase):
    def test_zip_source_points_gdal_inside_the_zip(self):
        with tempfile.TemporaryDirectory() as td:
            prod = parse_s2_product(s2_item(20))
            z = Path(td) / f"{prod.stem}.zip"
            member = f"{prod.name}/GRANULE/L2A_X/IMG_DATA/R10m/T45RUL_20260820T050701_B02_10m.jp2"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr(member, b"x")

            class C:
                def download(self, product, dest):
                    return z
            src = ZipBandSource(C(), td)
            self.assertEqual(src.path_for(prod, "B02"), f"/vsizip/{z.as_posix()}/{member}")
            with self.assertRaises(S2FetchError):
                src.path_for(prod, "B08")

    def test_node_source_downloads_only_one_file(self):
        prod = parse_s2_product(s2_item(20))
        payload = b"jp2data"
        tree = {"GRANULE": [{"Name": "L2A_T45RUL_A1"}],
                "IMG_DATA": [{"Name": "R10m"}, {"Name": "R20m"}],
                "R10m": [{"Name": "T45RUL_20260820T050701_B02_10m.jp2", "ContentLength": len(payload)}]}

        class R:
            def __init__(self, status, body=None, raw=b""):
                self.status_code, self._b, self.raw = status, body, raw
            def json(self):
                return self._b
            def iter_content(self, n):
                yield self.raw
            def close(self):
                pass

        class C:
            class cfg:
                download_url = "https://dl.fake/odata/v1"
            urls = []
            def _token(self):
                return "tok"
            def _get_following_redirects(self, url, headers):
                self.urls.append(url)
                assert headers["Authorization"] == "Bearer tok"
                if url.endswith("/$value"):
                    return R(200, raw=payload)
                key = [k for k in ("R10m", "IMG_DATA", "GRANULE") if f"Nodes({k})/Nodes" in url][:1]
                return R(200, {"result": tree[key[0]]}) if key else R(404)

        with tempfile.TemporaryDirectory() as td:
            path = NodeBandSource(C(), td).path_for(prod, "B02")
            self.assertEqual(Path(path).read_bytes(), payload)
            self.assertFalse(list(Path(td).rglob("*.part")))
            n = len(C.urls)
            NodeBandSource(C(), td).path_for(prod, "B02")                       # cached
            self.assertEqual(len(C.urls), n)

    def test_node_source_failure_is_s2fetcherror(self):
        class C:
            class cfg:
                download_url = "x"
            def _token(self):
                return "t"
            def _get_following_redirects(self, url, headers):
                class R:
                    status_code = 404
                    def close(self):
                        pass
                return R()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(S2FetchError):
                NodeBandSource(C(), td).path_for(parse_s2_product(s2_item(20)), "B02")

    def test_fallback_chain(self):
        class Bad:
            def path_for(self, p, b):
                raise S2FetchError("nope")

        class Good:
            def path_for(self, p, b):
                return "ok"
        prod = parse_s2_product(s2_item(20))
        self.assertEqual(FallbackBandSource([Bad(), Good()]).path_for(prod, "B02"), "ok")
        with self.assertRaises(S2FetchError) as ctx:
            FallbackBandSource([Bad(), Bad()]).path_for(prod, "B02")
        self.assertIn("Bad: nope", str(ctx.exception))


class TestSelection(unittest.TestCase):
    cfg = OpticalConfig()

    def passes(self, *specs):
        items = [s2_item(d, cloud=c, platform=pl, tile=t) for d, c, pl, t in specs]
        return group_passes([parse_s2_product(i) for i in items])

    def test_tiles_of_one_pass_are_grouped(self):
        ps = self.passes((20, 10, "S2B", "45RUL"), (20, 20, "S2B", "45RVL"), (25, 5, "S2B", "45RUL"))
        self.assertEqual([len(p.tiles) for p in ps], [2, 1])
        self.assertAlmostEqual(ps[0].cloud_cover, 15.0)

    def test_two_satellites_same_day_are_two_passes(self):
        self.assertEqual(len(self.passes((20, 10, "S2A", "45RUL"), (20, 10, "S2B", "45RUL"))), 1 + 1)

    def test_pass_coverage_from_footprints(self):
        half = [[84.0, 27.0], [85.31, 27.0], [85.31, 29.0], [84.0, 29.0], [84.0, 27.0]]
        full = group_passes([parse_s2_product(s2_item(20))])[0]
        part = group_passes([parse_s2_product(s2_item(20, ring=half, tile="45RVL"))])[0]   # distinct id
        cov = CoverageGrid(AOI_S, 64)
        self.assertGreater(full.coverage(cov), 0.99)
        self.assertAlmostEqual(part.coverage(cov), 0.5, delta=0.05)

    def test_rank_sides_filter_and_order(self):
        ps = self.passes((15, 60, "S2B", "45RUL"), (20, 5, "S2A", "45RUL"), (24, 98, "S2B", "45RUL"),
                         (28, 10, "S2B", "45RUL"), (31, 10, "S2A", "45RUL"))
        before = rank_candidates(ps, EVENT, "before", self.cfg)
        after = rank_candidates(ps, EVENT, "after", self.cfg)
        self.assertEqual([p.date.day for p in before], [20, 15])      # 98 % tile cloud dropped; low cloud first
        self.assertEqual([p.date.day for p in after], [28, 31])
        self.assertTrue(all(p.date < EVENT for p in before) and all(p.date >= EVENT for p in after))

    def test_rank_limits_number_of_checks(self):
        ps = self.passes(*[(d, 10, "S2B", "45RUL") for d in range(1, 12)])
        self.assertEqual(len(rank_candidates(ps, EVENT, "before", OpticalConfig(max_scl_checks=3))), 3)

    def test_pick_side_early_exit(self):
        ps = self.passes((20, 5, "S2A", "45RUL"), (15, 10, "S2B", "45RUL"), (10, 20, "S2B", "45RUL"))
        calls = []

        def clear(p):
            calls.append(p.date.day)
            return 0.95
        best, ev = pick_side(rank_candidates(ps, EVENT, "before", self.cfg), clear, EVENT, self.cfg)
        self.assertEqual(calls, [20])                                  # stopped after the first good one
        self.assertEqual(best.date.day, 20)

    def test_pick_side_rejects_and_prefers_clearer(self):
        ps = self.passes((24, 0, "S2A", "45RUL"), (20, 5, "S2B", "45RUL"), (15, 10, "S2B", "45RUL"))
        clearness = {24: 0.10, 20: 0.70, 15: 0.85}
        best, ev = pick_side(rank_candidates(ps, EVENT, "before", self.cfg),
                             lambda p: clearness[p.date.day], EVENT, self.cfg)
        self.assertEqual(best.date.day, 15)                            # clearer beats nearer
        self.assertEqual(set(ev.values()), {0.1, 0.7, 0.85})

    def test_pick_side_none_when_all_too_cloudy(self):
        ps = self.passes((20, 5, "S2A", "45RUL"))
        best, ev = pick_side(ps, lambda p: 0.3, EVENT, self.cfg)
        self.assertIsNone(best)

    def test_choose_pair_statuses(self):
        ps = self.passes((20, 5, "S2A", "45RUL"), (28, 5, "S2B", "45RUL"))
        b, a = rank_candidates(ps, EVENT, "before", self.cfg), rank_candidates(ps, EVENT, "after", self.cfg)
        used = choose_pair(b, a, lambda p: 0.95, EVENT, self.cfg)
        self.assertEqual((used.status, used.reason_code), ("used", "ok"))
        part = choose_pair(b, a, lambda p: 0.95 if p.date >= EVENT else 0.1, EVENT, self.cfg)
        self.assertEqual((part.status, part.reason_code, part.before), ("partial", "no_clear_before", None))
        no_before = choose_pair([], a, lambda p: 0.95, EVENT, self.cfg)
        self.assertEqual((no_before.status, no_before.reason_code), ("partial", "no_candidates_before"))
        none = choose_pair(b, a, lambda p: 0.1, EVENT, self.cfg)
        self.assertEqual((none.status, none.reason_code), ("unavailable", "no_clear_after"))
        self.assertEqual(choose_pair(b, [], lambda p: 0.9, EVENT, self.cfg).reason_code, "no_candidates_after")

    def test_after_side_is_evaluated_first_and_before_is_skipped_on_failure(self):
        ps = self.passes((20, 5, "S2A", "45RUL"), (28, 5, "S2B", "45RUL"))
        b, a = rank_candidates(ps, EVENT, "before", self.cfg), rank_candidates(ps, EVENT, "after", self.cfg)
        seen = []
        choose_pair(b, a, lambda p: seen.append(p.date.day) or 0.1, EVENT, self.cfg)
        self.assertEqual(seen, [28])                                   # before (day 20) never tested


class TestReading(unittest.TestCase):
    def test_window_covers_grid_with_margin_and_clips(self):
        info = TileInfo(CRS, 10.0, GRID.x0 - 500, GRID.y0 + 500, GRID.width + 100, GRID.height + 100)
        c0, r0, nc, nr = grid_window_in_tile(GRID, info, margin_px=3)
        self.assertEqual((c0, r0), (50 - 3, 50 - 3))
        self.assertEqual((nc, nr), (GRID.width + 6, GRID.height + 6))
        small = TileInfo(CRS, 10.0, GRID.x0 + 100, GRID.y0 - 100, 50, 50)      # starts inside the grid
        win = grid_window_in_tile(GRID, small)
        self.assertEqual((win[0], win[1]), (0, 0))

    def test_window_none_when_tile_is_elsewhere(self):
        far = TileInfo(CRS, 10.0, GRID.x0 + 500000, GRID.y0, 100, 100)
        self.assertIsNone(grid_window_in_tile(GRID, far))

    def test_aligned_10m_band_is_an_exact_copy(self):
        rd = FakeReader()
        data = np.arange(150 * 300, dtype=np.float32).reshape(150, 300) + 1
        rd.files["a"] = (TileInfo(CRS, 10.0, GRID.x0 - 500, GRID.y0 + 500, 300, 150), data)
        # tile too small to cover the whole grid: part of the grid must be NaN
        out = read_band_on_grid(rd, "a", GRID, order=1, zero_is_nodata=True)
        self.assertEqual(out.shape, (GRID.height, GRID.width))
        self.assertTrue(np.array_equal(out[:100, 5:150], data[50:150, 55:200][:100, :145]))
        self.assertTrue(np.isnan(out[110:, :]).all())

    def test_zero_is_nodata_and_nan_does_not_leak(self):
        rd = FakeReader()
        h, w = GRID.height + 100, GRID.width + 100
        d = np.full((h, w), 100.0, np.float32)
        d[60:70, 60:70] = 0                                              # DN 0 = no data
        rd.files["a"] = (TileInfo(CRS, 10.0, GRID.x0 - 500, GRID.y0 + 500, w, h), d)
        out = read_band_on_grid(rd, "a", GRID, order=1, zero_is_nodata=True)
        self.assertTrue(np.isnan(out[10:20, 10:20]).all())
        self.assertTrue(np.isfinite(out[30:40, 30:40]).all() and np.allclose(out[30:40, 30:40], 100.0))

    def test_scl_nearest_never_invents_classes(self):
        rd = FakeReader()
        classes = np.random.default_rng(1).choice([4, 5, 6, 9], size=(130, 130)).astype(np.float32)
        rd.files["s"] = (TileInfo(CRS, 20.0, GRID.x0 - 500, GRID.y0 + 500, 130, 130), classes)
        near = read_band_on_grid(rd, "s", GRID, order=0, zero_is_nodata=False)
        self.assertTrue(set(np.unique(near[np.isfinite(near)])) <= {4.0, 5.0, 6.0, 9.0})
        lin = read_band_on_grid(rd, "s", GRID, order=1, zero_is_nodata=False)
        self.assertFalse(set(np.unique(lin[np.isfinite(lin)])) <= {4.0, 5.0, 6.0, 9.0})   # bilinear would break classes

    def test_tile_in_another_utm_zone(self):
        """Tile in zone 44, grid in zone 45; compare against a smooth lon field."""
        rd = FakeReader()
        n = 400
        x0, y0 = None, None
        e_c, n_c = lonlat_to_utm(AOI_S.min_lon, AOI_S.min_lat, 44)
        x0, y0 = float(e_c) - 2000, float(n_c) + 4000
        ee, nn = np.meshgrid(x0 + (np.arange(n) + .5) * 20, y0 - (np.arange(n) + .5) * 20)
        lon, _ = utm_to_lonlat(ee, nn, 44)
        rd.files["z"] = (TileInfo("EPSG:32644", 20.0, x0, y0, n, n), (lon * 100).astype(np.float32))
        out = read_band_on_grid(rd, "z", GRID, order=1, zero_is_nodata=False)
        ee2, nn2 = np.meshgrid(GRID.x0 + (np.arange(GRID.width) + .5) * 10, GRID.y0 - (np.arange(GRID.height) + .5) * 10)
        lon2, _ = utm_to_lonlat(ee2, nn2, 45)
        ok = np.isfinite(out)
        self.assertGreater(ok.mean(), 0.5)
        self.assertLess(np.abs(out[ok] - (lon2 * 100)[ok]).max(), 0.02)

    def test_no_overlap_returns_none(self):
        rd = FakeReader()
        rd.files["f"] = (TileInfo(CRS, 10.0, GRID.x0 + 400000, GRID.y0, 10, 10), np.ones((10, 10), np.float32))
        self.assertIsNone(read_band_on_grid(rd, "f", GRID, 1, True))

    def test_combine_first(self):
        a = np.array([1.0, np.nan, np.nan])
        b = np.array([9.0, 2.0, np.nan])
        np.testing.assert_array_equal(combine_first([a, None, b]), [1.0, 2.0, np.nan])
        self.assertIsNone(combine_first([None, None]))


class TestPixelMath(unittest.TestCase):
    cfg = OpticalConfig()

    def test_reflectance_offset_and_nodata(self):
        dn = np.array([1000.0, 2000.0, 11000.0, 500.0, np.nan], np.float32)
        r = to_reflectance(dn, -1000.0)
        np.testing.assert_allclose(r[:4], [0.0, 0.1, 1.0, 0.0])           # negative clipped to 0
        self.assertTrue(np.isnan(r[4]))
        np.testing.assert_allclose(to_reflectance(dn[:2], 0.0), [0.1, 0.2])   # old baseline

    def test_clear_classes(self):
        scl = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], np.uint8)
        clear = clear_from_scl(scl, OpticalConfig(cloud_buffer_px=0))
        self.assertEqual(clear[0].tolist(), [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0])

    def test_cloud_buffer_grows_clouds_but_not_snow(self):
        scl = np.full((21, 21), 4, np.uint8)
        scl[10, 10] = 9
        c = clear_from_scl(scl, OpticalConfig(cloud_buffer_px=3))
        self.assertFalse(c[7:14, 7:14].any())                  # 3 px square around the cloud is not clear
        self.assertTrue(c[0, 0] and c[10, 15])
        scl2 = np.full((21, 21), 4, np.uint8)
        scl2[10, 10] = 11                                       # snow: not clear itself, no buffer
        c2 = clear_from_scl(scl2, OpticalConfig(cloud_buffer_px=3))
        self.assertFalse(c2[10, 10])
        self.assertTrue(c2[10, 11])

    def refl(self, v=0.1, shape=(4, 4)):
        return {b: np.full(shape, v, np.float32) for b in BANDS}

    def test_bright_blue_and_nodata_are_not_clear(self):
        scl = np.full((4, 4), 4, np.uint8)
        r = self.refl()
        r["B02"][0, 0] = 0.45                                   # haze SCL missed
        r["B08"][1, 1] = np.nan
        c = clear_mask(scl, r, OpticalConfig(cloud_buffer_px=0))
        self.assertFalse(c[0, 0] or c[1, 1])
        self.assertTrue(c[2, 2])
        self.assertTrue(clear_mask(scl, r, OpticalConfig(cloud_buffer_px=0, bright_blue=None))[0, 0])

    def test_normalized_difference(self):
        a, b = np.array([0.3, 0.0, 0.5], np.float32), np.array([0.1, 0.0, 0.5], np.float32)
        out = normalized_difference(a, b)
        self.assertAlmostEqual(float(out[0]), 0.5, places=5)
        self.assertTrue(np.isnan(out[1]))                       # 0/0 guarded
        self.assertAlmostEqual(float(out[2]), 0.0)

    def test_indices_known_values(self):
        r = {b: np.full((2, 2), v, np.float32) for b, v in LAND.items()}
        idx = compute_indices(r)
        self.assertAlmostEqual(float(idx[0, 0, 0]), (0.08 - 0.30) / 0.38, places=5)   # NDWI
        self.assertAlmostEqual(float(idx[1, 0, 0]), (0.08 - 0.20) / 0.28, places=5)   # MNDWI
        self.assertAlmostEqual(float(idx[2, 0, 0]), (0.30 - 0.05) / 0.35, places=5)   # NDVI
        w = {b: np.full((2, 2), v, np.float32) for b, v in WATER.items()}
        self.assertGreater(float(compute_indices(w)[1, 0, 0]), 0.7)

    def test_change_evidence_scenarios(self):
        cfg = OpticalConfig(min_blob_px=1)
        h = w = 12
        z = np.zeros((h, w), bool)
        new_w, debris, cloud, river = z.copy(), z.copy(), z.copy(), z.copy()
        new_w[0:3, 0:3] = True
        debris[0:3, 4:7] = True
        cloud[0:3, 8:11] = True
        river[6:9, 0:3] = True

        def build(patches):
            r = {b: np.full((h, w), LAND[b], np.float32) for b in BANDS}
            for mask, spec in patches:
                for b in BANDS:
                    r[b][mask] = spec[b]
            return r
        before = build([(river, WATER)])
        after = build([(new_w, WATER), (debris, DEBRIS), (cloud, CLOUD), (river, WATER)])
        clear_b = np.ones((h, w), bool)
        clear_a = ~cloud
        ev, mode = change_evidence(compute_indices(before), compute_indices(after), clear_b, clear_a, None, cfg)
        self.assertEqual(mode, "change")
        self.assertTrue(ev[0][new_w].all() and not ev[0][debris].any())
        self.assertTrue(ev[1][debris].all() and not ev[1][new_w].any())
        self.assertFalse(ev[0][river].any())                         # water both times: not NEW water
        self.assertFalse(ev[0][cloud].any() or ev[1][cloud].any())   # unknown is never flagged
        self.assertEqual(int(ev.sum()), int(new_w.sum() + debris.sum()))

        # after-only mode: old river IS flagged unless permanent water says otherwise
        ev2, mode2 = change_evidence(None, compute_indices(after), None, clear_a, None, cfg)
        self.assertEqual(mode2, "after_only")
        self.assertTrue(ev2[0][river].all())
        self.assertFalse(ev2[1].any())                               # no vegetation evidence without a before image
        ev3, _ = change_evidence(None, compute_indices(after), None, clear_a, river, cfg)
        self.assertFalse(ev3[0][river].any())
        self.assertTrue(ev3[0][new_w].all())

    def test_small_specks_are_removed(self):
        h = w = 12
        r_b = {b: np.full((h, w), LAND[b], np.float32) for b in BANDS}
        r_a = {b: v.copy() for b, v in r_b.items()}
        for b in BANDS:
            r_a[b][5, 5] = WATER[b]                                   # a single water pixel
        ev, _ = change_evidence(compute_indices(r_b), compute_indices(r_a), np.ones((h, w), bool),
                                np.ones((h, w), bool), None, OpticalConfig(min_blob_px=15))
        self.assertEqual(int(ev.sum()), 0)


class TestPngAndQuicklook(unittest.TestCase):
    def test_png_roundtrip_own_decoder(self):
        rgba = np.random.default_rng(3).integers(0, 256, (7, 9, 4), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a" / "x.png"
            write_png_rgba(p, rgba)
            b = p.read_bytes()
        self.assertEqual(b[:8], b"\x89PNG\r\n\x1a\n")
        pos, idat, hdr = 8, b"", None
        while pos < len(b):
            n = struct.unpack(">I", b[pos:pos + 4])[0]
            tag, data = b[pos + 4:pos + 8], b[pos + 8:pos + 8 + n]
            crc = struct.unpack(">I", b[pos + 8 + n:pos + 12 + n])[0]
            self.assertEqual(crc, zlib.crc32(tag + data) & 0xFFFFFFFF)          # every chunk checksum valid
            if tag == b"IHDR":
                hdr = struct.unpack(">IIBBBBB", data)
            if tag == b"IDAT":
                idat += data
            pos += 12 + n
        self.assertEqual(hdr, (9, 7, 8, 6, 0, 0, 0))
        raw = zlib.decompress(idat)
        rows = [np.frombuffer(raw[i * (1 + 36) + 1:(i + 1) * (1 + 36)], np.uint8).reshape(9, 4) for i in range(7)]
        self.assertTrue(np.array_equal(np.stack(rows), rgba))

    @unittest.skipUnless(__import__("importlib").util.find_spec("PIL"), "Pillow not installed")
    def test_png_opens_in_pillow(self):
        from PIL import Image
        rgba = np.random.default_rng(4).integers(0, 256, (5, 6, 4), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.png"
            write_png_rgba(p, rgba)
            im = Image.open(p)
            self.assertEqual((im.size, im.mode), ((6, 5), "RGBA"))
            self.assertTrue(np.array_equal(np.array(im), rgba))

    def refl(self, v, shape=(10, 10)):
        return {b: np.full(shape, v, np.float32) for b in BANDS}

    def test_shared_stretch_keeps_pair_comparable(self):
        dark, bright = self.refl(0.05), self.refl(0.25)
        clear = np.ones((10, 10), bool)
        lo, hi = shared_stretch([dark, bright], [clear, clear])
        self.assertAlmostEqual(lo, 0.05, delta=0.01)
        self.assertAlmostEqual(hi, 0.25, delta=0.01)
        qd = make_quicklook(dark, (lo, hi), 100)
        qb = make_quicklook(bright, (lo, hi), 100)
        self.assertLess(int(qd[0, 0, 0]), 30)                       # dark image stays dark
        self.assertGreater(int(qb[0, 0, 0]), 225)

    def test_quicklook_alpha_and_downscale(self):
        r = self.refl(0.1, (40, 60))
        r["B04"][0:10, 0:10] = np.nan
        q = make_quicklook(r, (0.0, 0.2), max_px=30)
        self.assertEqual(q.shape, (20, 30, 4))                       # factor 2
        self.assertEqual(int(q[0, 0, 3]), 0)                         # no data: transparent
        self.assertEqual(int(q[15, 15, 3]), 255)
        self.assertEqual(_downscale(np.zeros((10, 10, 4)), 100).shape, (10, 10, 4))

    def test_stretch_without_clear_pixels_is_safe(self):
        lo, hi = shared_stretch([self.refl(0.1)], [np.zeros((10, 10), bool)])
        self.assertTrue(hi > lo)

    def test_bounds_lonlat_order_and_location(self):
        (s, w), (n, e) = grid_bounds_lonlat(GRID)
        self.assertTrue(s < n and w < e)
        self.assertAlmostEqual(w, AOI_S.min_lon, delta=0.002)
        self.assertAlmostEqual(n, AOI_S.max_lat, delta=0.002)


# --------------------------------------------------------------------------
class TestOrchestration(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.out = Path(self.td.name) / "pkg"
        self.out.mkdir()
        (self.out / "metadata.json").write_text(json.dumps({"grid": GRID.to_dict()}))
        self.io, self.reader, self.source = MemIO(), FakeReader(), FakeSource()
        self.pair = make_pair()

    def tearDown(self):
        self.td.cleanup()

    def run_build(self, items, **kw):
        self.client = FakeClient(items)
        return build_optical_layers(self.pair, GRID, self.out, kw.pop("cfg", None), self.client,
                                    kw.pop("source", self.source), self.reader, self.io, **kw)

    def scl_calls(self):
        return [c for c in self.source.calls if c[1] == "SCL"]

    def test_full_scenario_used(self):
        b, a = s2_item(15, cloud=5), s2_item(28, cloud=10)
        add_scene(self.reader, b, water=[RIVER])
        add_scene(self.reader, a, water=[FLOOD, RIVER], debris=[DEBRIS_R], cloud=[CLOUD_R])
        res = self.run_build([b, a])
        self.assertEqual((res.status, res.reason_code, res.evidence_mode), ("used", "ok", "change"))
        self.assertEqual(len(self.scl_calls()), 2)                       # early exit: one SCL test per side
        ev = self.io.store[res.files["s2_evidence"]].data
        valid = self.io.store[res.files["s2_valid_mask"]].data
        self.assertGreater(ev[0, 50:90, 50:90].mean(), 0.97)             # the flood
        self.assertGreater(ev[1, 130:150, 130:160].mean(), 0.97)         # the debris
        self.assertEqual(int(ev[0, 100:110, 10:150].sum()), 0)           # old river is not new water
        self.assertEqual(int(ev[:, 22:38, 152:168].sum()), 0)            # cloud: unknown, never flagged
        self.assertEqual(int(valid[1, 24:36, 154:166].sum()), 0)         # ...and marked not valid
        self.assertTrue(valid[1, 170:190, 20:60].all())
        for name in ("s2_before", "s2_after", "s2_indices_before", "s2_indices_after", "quicklook_before"):
            self.assertIn(name, res.files)
        self.assertEqual(self.io.writes["s2_evidence.tif"]["dtype"], "uint8")
        self.assertEqual(self.io.store[res.files["s2_after"]].bands, 5)
        # every optical file is on the Part 3 grid
        for k in ("s2_before", "s2_after", "s2_indices_after", "s2_valid_mask", "s2_evidence"):
            self.assertTrue(GRID.matches(self.io.store[res.files[k]]), k)
        self.assertEqual(res.selection["before"]["days_from_event"], -11)
        self.assertTrue(Path(res.files["quicklook_after"]).read_bytes().startswith(b"\x89PNG"))
        self.assertEqual(len(res.quicklook_bounds), 2)

    def test_reflectance_values_use_the_baseline_offset(self):
        a = s2_item(28, cloud=10)
        add_scene(self.reader, a)
        res = self.run_build([a])
        refl = self.io.store[res.files["s2_after"]].data
        np.testing.assert_allclose(refl[3, 80, 80], 0.30, atol=1e-3)     # B08 land reflectance recovered
        np.testing.assert_allclose(refl[0, 80, 80], 0.04, atol=1e-3)

    def test_tile_cloud_hint_does_not_decide(self):
        """Tile says 5 % cloud, but the AOI is fully cloudy: the SCL decides (TRICK 1)."""
        liar, good, a = s2_item(24, cloud=5), s2_item(20, cloud=40), s2_item(28, cloud=10)
        add_scene(self.reader, liar, full_cloud=True)
        add_scene(self.reader, good)
        add_scene(self.reader, a)
        res = self.run_build([liar, good, a])
        self.assertEqual(res.status, "used")
        self.assertEqual(res.selection["before"]["date_utc"], "2026-08-20")
        self.assertEqual(len([c for c in self.scl_calls() if "20260824" in c[0]]), 1)   # the liar WAS tested
        self.assertTrue(any("6 days away from the radar before" in w for w in res.warnings))

    def test_cloudy_after_means_unavailable_and_before_is_never_fetched(self):
        b, a1, a2 = s2_item(15), s2_item(28), s2_item(31)
        add_scene(self.reader, b)
        add_scene(self.reader, a1, full_cloud=True)
        add_scene(self.reader, a2, full_cloud=True)
        res = self.run_build([b, a1, a2])
        self.assertEqual((res.status, res.reason_code), ("unavailable", "no_clear_after"))
        self.assertEqual(res.files, {})
        self.assertFalse(any("20260815" in c[0] for c in self.source.calls))        # TRICK 12: no wasted downloads

    def test_no_candidates(self):
        res = self.run_build([])
        self.assertEqual((res.status, res.reason_code), ("unavailable", "no_candidates"))

    def test_insufficient_footprint_coverage(self):
        half = [[84.0, 27.0], [85.31, 27.0], [85.31, 29.0], [84.0, 29.0], [84.0, 27.0]]
        res = self.run_build([s2_item(28, ring=half)])
        self.assertEqual(res.reason_code, "no_coverage")

    def test_partial_after_only_uses_permanent_water(self):
        b, a = s2_item(15), s2_item(28)
        add_scene(self.reader, b, full_cloud=True)
        add_scene(self.reader, a, water=[FLOOD, RIVER])
        res = self.run_build([b, a])
        self.assertEqual((res.status, res.reason_code, res.evidence_mode), ("partial", "no_clear_before", "after_only"))
        self.assertNotIn("s2_before", res.files)
        self.assertTrue(any("permanent-water mask" in w for w in res.warnings))
        ev = self.io.store[res.files["s2_evidence"]].data
        self.assertGreater(ev[0, 100:110, 10:150].mean(), 0.9)           # river wrongly "new" without the mask
        # same again with Part 3's permanent-water mask supplied (fresh folder => no cache)
        self.out2 = Path(self.td.name) / "pkg2"
        pw = np.zeros((GRID.height, GRID.width), bool)
        pw[100:110, :] = True
        res2 = build_optical_layers(self.pair, GRID, self.out2, None, FakeClient([b, a]), FakeSource(),
                                    self.reader, self.io, permanent_water=pw)
        ev2 = self.io.store[res2.files["s2_evidence"]].data
        self.assertEqual(int(ev2[0, 100:110].sum()), 0)
        self.assertGreater(ev2[0, 50:90, 50:90].mean(), 0.97)
        self.assertFalse(any("permanent-water mask" in w for w in res2.warnings))

    def test_event_day_after_image_is_flagged(self):
        a = s2_item(26)
        add_scene(self.reader, a)
        res = self.run_build([a])
        self.assertTrue(any("event date itself" in w for w in res.warnings))

    def test_errors_never_escape_and_are_not_cached(self):
        a = s2_item(28)
        add_scene(self.reader, a)
        bad = FakeSource(fail=True)
        res = self.run_build([a], source=bad)
        self.assertEqual((res.status, res.reason_code), ("unavailable", "error"))
        self.assertIn("boom", res.reason)
        self.assertTrue((self.out / "optical.json").exists())
        n = len(bad.calls)
        self.run_build([a], source=bad)                                    # error result is NOT reused
        self.assertGreater(len(bad.calls), n)
        with self.assertRaises(RuntimeError):
            self.run_build([a], source=bad, raise_errors=True)

    def test_cache_for_used_and_metadata_merge(self):
        a = s2_item(28)
        add_scene(self.reader, a)
        r1 = self.run_build([a])
        reads, searches = self.reader.reads, self.client.searches
        r2 = self.run_build([a])
        self.assertTrue(r2.from_cache and not r1.from_cache)
        self.assertEqual(self.reader.reads, reads)                          # no band re-read
        self.assertEqual(self.client.searches, 0)                           # (new fake client: never searched)
        meta = json.loads((self.out / "metadata.json").read_text())
        self.assertEqual(meta["optical"]["status"], "partial")
        self.assertIn("grid", meta)                                        # Part 3's content preserved
        self.assertEqual(grid_from_metadata(meta), GRID)

    def test_changed_settings_invalidate_cache(self):
        a = s2_item(28)
        add_scene(self.reader, a)
        self.run_build([a])
        r = self.run_build([a], cfg=OpticalConfig(water_abs=0.2))
        self.assertFalse(r.from_cache)

    def test_unavailable_results_expire(self):
        res = self.run_build([])
        again = self.run_build([])
        self.assertTrue(again.from_cache)                                  # fresh unavailable: reused
        expired = self.run_build([], cfg=OpticalConfig(cache_ttl_h=0.0))
        self.assertFalse(expired.from_cache)                               # old enough: search again

    def test_missing_metadata_json_is_tolerated(self):
        (self.out / "metadata.json").unlink()
        res = self.run_build([])
        self.assertEqual(res.status, "unavailable")


if __name__ == "__main__":
    unittest.main()
