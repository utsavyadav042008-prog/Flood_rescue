"""
Offline tests for Part 3:  preprocessing/utm.py, acquisition/dem.py,
preprocessing/alignment.py   (no rasterio, no pyproj, no network).

Run from the repo root:
    python -m unittest tests.test_alignment -v

Every expected number is derived independently (an integral, a plane with a known
slope, a ridge with a known shadow width), not copied from the code's own output.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from acquisition.dem import (DemError, CopernicusDemDownloader, LatLonRaster, LocalDemSource,
                             mosaic_latlon, tile_name, tiles_needed, DEM_ATTRIBUTION)
from acquisition.sentinel import AOI, Acquisition, ScenePair
from preprocessing.alignment import (
    AlignmentConfig, AlignmentError, GridSpec, PreInputs, TERRAIN_BITS, TERRAIN_CAST_SHADOW,
    TERRAIN_FORESHORTEN, TERRAIN_LAYOVER, TERRAIN_SHADOW, TERRAIN_STEEP, _block_mean,
    assert_aligned, build_analysis_package, build_grid, build_terrain_mask, cast_shadow,
    choose_water_threshold, dem_to_grid, gradients, look_azimuth_deg, otsu_threshold,
    permanent_water_mask, range_slope_deg, remove_small_blobs, resample_to_grid,
    slope_degrees, valid_mask)
from preprocessing.sentinel1 import GeoRaster
from preprocessing.utm import (central_meridian, lonlat_to_utm, parse_utm_crs, utm_to_lonlat)
from tests.test_acquisition import EVENT, mk

CRS = "EPSG:32645"
AOI_S = AOI.from_bbox((85.30, 28.00, 85.32, 28.02))           # ~2 km x 2.2 km: fast tests
RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class MemIO:
    def __init__(self):
        self.store, self.writes = {}, {}

    def read(self, path):
        return self.store[str(path)]

    def write(self, path, r, descriptions=None, tags=None, dtype="float32", nodata=float("nan")):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"x")
        self.store[str(path)] = r
        self.writes[Path(path).name] = {"dtype": dtype, "desc": descriptions, "tags": tags}


def plane_dem(z_of_n, lon_rng=(85.20, 85.40), lat_rng=(27.90, 28.10), step=1 / 1200):
    """A lon/lat DEM whose height is a known function of UTM-45 NORTHING."""
    w = int(round((lon_rng[1] - lon_rng[0]) / step))
    h = int(round((lat_rng[1] - lat_rng[0]) / step))
    lons = lon_rng[0] + (np.arange(w) + 0.5) * step
    lats = lat_rng[1] - (np.arange(h) + 0.5) * step
    ll, la = np.meshgrid(lons, lats)
    _, n = lonlat_to_utm(ll, la, 45)
    return LatLonRaster(z_of_n(n).astype(np.float32), lon_rng[0], lat_rng[1], step, step)


N_REF = 3_098_000.0


# --------------------------------------------------------------------------
class TestUTM(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_utm_crs("EPSG:32645"), (45, True))
        self.assertEqual(parse_utm_crs("EPSG:32719"), (19, False))
        for bad in ("EPSG:4326", "EPSG:32600", "EPSG:32661", "utm45"):
            with self.assertRaises(ValueError):
                parse_utm_crs(bad)

    def test_central_meridian_zone45(self):
        self.assertEqual(central_meridian(45), 87.0)

    def test_on_central_meridian_northing_is_scaled_meridian_arc(self):
        """Independent check: integrate the WGS84 meridian arc numerically."""
        a, f = 6378137.0, 1 / 298.257223563
        e2 = f * (2 - f)
        phi = np.linspace(0, math.radians(28.0), 200001)
        y = a * (1 - e2) / (1 - e2 * np.sin(phi) ** 2) ** 1.5
        arc = float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(phi)))
        e, n = lonlat_to_utm(87.0, 28.0, 45)
        self.assertAlmostEqual(float(e), 500000.0, places=6)
        self.assertAlmostEqual(float(n), 0.9996 * arc, places=3)

    def test_east_west_scale(self):
        a, f = 6378137.0, 1 / 298.257223563
        e2 = f * (2 - f)
        lat = math.radians(28.0)
        radius = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)           # prime vertical radius
        d_lon = math.radians(0.01)
        e, _ = lonlat_to_utm(87.01, 28.0, 45)
        self.assertAlmostEqual(float(e) - 500000.0, 0.9996 * radius * math.cos(lat) * d_lon, delta=1e-2)

    def test_kathmandu_ballpark(self):
        e, n = lonlat_to_utm(85.3, 27.7, 45)
        self.assertTrue(330000 < float(e) < 336000, e)
        self.assertTrue(3_050_000 < float(n) < 3_080_000, n)

    def test_round_trip_arrays(self):
        lon = 87 + RNG.uniform(-3, 3, 500)
        lat = RNG.uniform(-60, 70, 500)
        for north in (True, False):
            e, n = lonlat_to_utm(lon, lat, 45, north)
            lo2, la2 = utm_to_lonlat(e, n, 45, north)
            self.assertLess(np.abs(lo2 - lon).max(), 1e-9)
            self.assertLess(np.abs(la2 - lat).max(), 1e-9)

    def test_southern_hemisphere_false_northing(self):
        _, n = lonlat_to_utm(-69.0, -33.0, 19, north=False)
        self.assertTrue(5_000_000 < float(n) < 10_000_000)

    @unittest.skipUnless(__import__("importlib").util.find_spec("pyproj"), "pyproj not installed")
    def test_matches_pyproj(self):                                     # runs on your machine
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
        e, n = lonlat_to_utm(85.31, 28.01, 45)
        pe, pn = t.transform(85.31, 28.01)
        self.assertAlmostEqual(float(e), pe, places=3)
        self.assertAlmostEqual(float(n), pn, places=3)


class TestDemTiles(unittest.TestCase):
    def test_tile_names(self):
        self.assertEqual(tile_name(27, 85), "Copernicus_DSM_COG_10_N27_00_E085_00_DEM")
        self.assertEqual(tile_name(-3, -70), "Copernicus_DSM_COG_10_S03_00_W070_00_DEM")

    def test_tiles_needed(self):
        self.assertEqual(tiles_needed((85.1, 27.1, 85.5, 27.9)), [(27, 85)])          # inside one tile
        self.assertEqual(tiles_needed((85.1, 27.9, 85.5, 28.3)), [(27, 85), (28, 85)])  # crosses lat 28
        # exact integer max edge must not request the next tile (TRICK 2)
        self.assertEqual(sorted(tiles_needed((84.5, 27.5, 86.0, 28.5))),
                         [(27, 84), (27, 85), (28, 84), (28, 85)])
        self.assertEqual(tiles_needed((85.0, 27.0, 85.0001, 27.0001)), [(27, 85)])

    def test_mosaic_integer_paste(self):
        a = LatLonRaster(np.full((4, 4), 1.0, np.float32), 85.0, 28.0, 0.25, 0.25)
        b = LatLonRaster(np.full((4, 4), 2.0, np.float32), 86.0, 28.0, 0.25, 0.25)
        m, worst = mosaic_latlon([a, b])
        self.assertEqual(m.data.shape, (4, 8))
        self.assertEqual(worst, 0.0)
        self.assertTrue((m.data[:, :4] == 1).all() and (m.data[:, 4:] == 2).all())
        self.assertEqual(m.bounds, (85.0, 27.0, 87.0, 28.0))

    def test_mosaic_different_pixel_size_rejected(self):
        a = LatLonRaster(np.zeros((2, 2), np.float32), 85, 28, 0.25, 0.25)
        b = LatLonRaster(np.zeros((2, 2), np.float32), 86, 28, 0.5, 0.25)
        with self.assertRaises(DemError):
            mosaic_latlon([a, b])

    def fake_reader(self, path):
        token = Path(path).name
        val = 10.0 if "N27_00_E085" in token else 20.0
        lon0 = 85.0 if "E085" in token else 86.0
        return LatLonRaster(np.full((4, 4), val, np.float32), lon0, 28.0, 0.25, 0.25)

    def test_local_folder_source(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "sub").mkdir()
            for name in ("Copernicus_DSM_COG_10_N27_00_E085_00_DEM.tif",
                         "Copernicus_DSM_COG_10_N27_00_E086_00_DEM.tif"):
                (Path(td) / "sub" / name).write_bytes(b"x")
            dem = LocalDemSource(td, reader=self.fake_reader).load((85.5, 27.5, 86.5, 27.9))
            self.assertEqual(dem.data.shape, (4, 8))

    def test_local_missing_tiles_raise(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(DemError):
                LocalDemSource(td, reader=self.fake_reader).load((85.5, 27.5, 85.6, 27.6))

    def test_local_single_file(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "dem.tif"
            f.write_bytes(b"x")
            dem = LocalDemSource(f, reader=self.fake_reader).load((0, 0, 1, 1))
            self.assertEqual(dem.data.shape, (4, 4))


class FakeResp:
    def __init__(self, status, body=b""):
        self.status_code, self.body = status, body

    def iter_content(self, n):
        yield self.body

    def close(self):
        pass


class FakeSession:
    def __init__(self, plan):
        self.plan, self.calls = plan, []

    def get(self, url, stream=False, timeout=None):
        self.calls.append(url)
        seq = self.plan[url.split("/")[-1].split(".")[0]]
        return seq.pop(0) if len(seq) > 1 else seq[0]


class TestDemDownloader(unittest.TestCase):
    def test_download_retry_404_and_cache(self):
        t1 = tile_name(27, 85)
        t2 = tile_name(27, 86)
        sess = FakeSession({t1: [FakeResp(503), FakeResp(200, b"tiledata")],
                            t2: [FakeResp(404)]})
        reader = lambda p: LatLonRaster(np.full((2, 2), 5.0, np.float32), 85.0, 28.0, 0.5, 0.5)
        with tempfile.TemporaryDirectory() as td:
            d = CopernicusDemDownloader(td, session=sess, reader=reader, retry_base_delay_s=0.0)
            dem = d.load((85.2, 27.2, 86.5, 27.8))                      # needs both tiles
            self.assertEqual(dem.data.shape, (2, 2))                    # only t1 existed
            self.assertTrue((Path(td) / f"{t1}.tif").exists())
            self.assertFalse(list(Path(td).glob("*.part")))
            n_calls = len(sess.calls)
            d.load((85.2, 27.2, 85.8, 27.8))                            # t1 cached: no network
            self.assertEqual(len(sess.calls), n_calls)

    def test_all_tiles_missing_raises(self):
        sess = FakeSession({tile_name(27, 85): [FakeResp(404)]})
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(DemError):
                CopernicusDemDownloader(td, session=sess, retry_base_delay_s=0.0).load((85.2, 27.2, 85.8, 27.8))


# --------------------------------------------------------------------------
class TestGridAndResample(unittest.TestCase):
    def test_grid_snapped_and_covers_aoi(self):
        g = build_grid(AOI_S, CRS, 10.0)
        self.assertEqual(g.x0 % 10, 0)
        self.assertEqual(g.y0 % 10, 0)
        zone, north = parse_utm_crs(CRS)
        corners = [(AOI_S.min_lon, AOI_S.min_lat), (AOI_S.max_lon, AOI_S.max_lat),
                   (AOI_S.min_lon, AOI_S.max_lat), (AOI_S.max_lon, AOI_S.min_lat)]
        for lo, la in corners:
            e, n = lonlat_to_utm(lo, la, zone, north)
            self.assertTrue(g.x0 <= e <= g.x0 + g.width * 10)
            self.assertTrue(g.y0 - g.height * 10 <= n <= g.y0)
        # ~0.02 deg lon * 98 km/deg ~ 1.97 km => ~197 px (+ at most a couple of px margin)
        self.assertTrue(195 <= g.width <= 202, g.width)
        self.assertTrue(218 <= g.height <= 226, g.height)

    def test_grid_deterministic(self):
        self.assertEqual(build_grid(AOI_S, CRS), build_grid(AOI_S, CRS))

    def test_fast_path_is_exact_copy(self):
        g = GridSpec(CRS, 10.0, 500000.0, 3100000.0, 6, 5)
        data = RNG.normal(size=(2, 12, 14)).astype(np.float32)
        src = GeoRaster(data, 499970.0, 3100020.0, 10.0, CRS)          # grid starts at src col 3, row 2
        out = resample_to_grid(src, g)
        self.assertTrue(np.array_equal(out.data, data[:, 2:7, 3:9]))   # bit-for-bit, no blur
        self.assertTrue(g.matches(out))

    def test_fast_path_pads_outside_with_nan(self):
        g = GridSpec(CRS, 10.0, 500000.0, 3100000.0, 6, 5)
        src = GeoRaster(np.ones((1, 3, 3), np.float32), 500020.0, 3099990.0, 10.0, CRS)
        out = resample_to_grid(src, g).data[0]
        self.assertTrue(np.isfinite(out[1:4, 2:5]).all() and np.isnan(out).sum() == 30 - 9)

    def test_general_path_bilinear_of_a_ramp_is_exact(self):
        cols = np.arange(40, dtype=np.float32)
        src = GeoRaster(np.tile(cols, (30, 1))[None].copy(), 0.0, 600.0, 20.0, CRS)    # 20 m, value = col
        g = GridSpec(CRS, 10.0, 0.0, 600.0, 80, 60)
        out = resample_to_grid(src, g, order=1).data[0]
        e = (np.arange(80) + 0.5) * 10.0
        expected = e / 20.0 - 0.5
        interior = slice(2, 76)
        np.testing.assert_allclose(out[30, interior], expected[interior], atol=1e-4)
        self.assertTrue(np.isnan(out[30, 0]))                           # beyond src edge: NaN

    def test_nan_is_not_smeared_into_numbers(self):
        d = np.ones((1, 10, 10), np.float32)
        d[0, 5, 5] = np.nan
        src = GeoRaster(d, 0.0, 200.0, 20.0, CRS)
        out = resample_to_grid(src, GridSpec(CRS, 10.0, 0.0, 200.0, 20, 20)).data[0]
        self.assertTrue(np.isnan(out).any())
        self.assertFalse(np.isinf(out).any())
        self.assertTrue(np.isclose(out[np.isfinite(out)], 1.0).all())    # finite values untouched

    def test_other_utm_zone_path(self):
        """Source in zone 44, grid in zone 45; compare to a smooth lon/lat field."""
        z44 = GridSpec("EPSG:32644", 20.0, 780000.0, 3100000.0, 60, 60)
        ee, nn = np.meshgrid(z44.x0 + (np.arange(60) + .5) * 20, z44.y0 - (np.arange(60) + .5) * 20)
        lo, la = utm_to_lonlat(ee, nn, 44)
        field = (lo * 10 + la).astype(np.float32)
        src = GeoRaster(field[None], z44.x0, z44.y0, 20.0, "EPSG:32644")
        lo_c, la_c = utm_to_lonlat(780000 + 600, 3100000 - 600, 44)
        e45, n45 = lonlat_to_utm(lo_c, la_c, 45)
        g = GridSpec(CRS, 10.0, math.floor(e45 / 10) * 10 - 200, math.ceil(n45 / 10) * 10 + 200, 40, 40)
        out = resample_to_grid(src, g, order=1).data[0]
        ee2, nn2 = np.meshgrid(g.x0 + (np.arange(40) + .5) * 10, g.y0 - (np.arange(40) + .5) * 10)
        lo2, la2 = utm_to_lonlat(ee2, nn2, 45)
        ok = np.isfinite(out)
        self.assertGreater(ok.sum(), 1000)
        self.assertLess(np.abs(out[ok] - (lo2 * 10 + la2)[ok]).max(), 5e-3)


class TestDemToGrid(unittest.TestCase):
    def test_plane_is_reproduced(self):
        dem = plane_dem(lambda n: 0.5 * (n - N_REF))
        g = build_grid(AOI_S, CRS)
        z = dem_to_grid(dem, g)
        n_t = g.y0 - (np.arange(g.height) + 0.5) * g.res
        expected = 0.5 * (n_t - N_REF)[:, None] * np.ones((1, g.width))
        self.assertTrue(np.isfinite(z).all())
        self.assertLess(np.abs(z - expected).max(), 0.3)

    def test_coarse_nodes_trick_matches_exact(self):
        dem = plane_dem(lambda n: 0.5 * (n - N_REF) + 3 * np.sin(n / 200.0))
        g = build_grid(AOI_S, CRS)
        fast = dem_to_grid(dem, g, node_step=32)
        exact = dem_to_grid(dem, g, node_step=1)
        self.assertLess(np.abs(fast - exact).max(), 0.02)             # centimetres at worst

    def test_outside_dem_is_nan(self):
        dem = plane_dem(lambda n: n * 0, lon_rng=(85.31, 85.40), lat_rng=(27.90, 28.10))
        z = dem_to_grid(dem, build_grid(AOI_S, CRS))
        self.assertTrue(np.isnan(z[:, :5]).all())                     # west of the DEM
        self.assertTrue(np.isfinite(z[:, -5:]).all())


class TestSlopeAndGeometry(unittest.TestCase):
    def ramp(self, p=0.0, q=0.0, shape=(30, 30), res=10.0):
        c, r = np.meshgrid(np.arange(shape[1]) * res, np.arange(shape[0]) * res)
        return (p * c - q * r).astype(np.float32)          # row down = south: z = p*E + q*N

    def test_gradients_and_slope(self):
        p, q = gradients(self.ramp(q=0.5), 10.0)
        self.assertAlmostEqual(float(q[10, 10]), 0.5, places=5)
        self.assertAlmostEqual(float(p[10, 10]), 0.0, places=5)
        self.assertAlmostEqual(float(slope_degrees(p, q)[10, 10]), math.degrees(math.atan(0.5)), places=3)
        p, q = gradients(self.ramp(p=0.3), 10.0)
        self.assertAlmostEqual(float(p[10, 10]), 0.3, places=5)

    def test_look_azimuth(self):
        asc, desc = look_azimuth_deg("ASCENDING", 28.0), look_azimuth_deg("DESCENDING", 28.0)
        self.assertAlmostEqual(asc, 80.7, delta=1.0)
        self.assertAlmostEqual(desc, 279.3, delta=1.0)
        self.assertEqual(look_azimuth_deg("A", 28.0), asc)

    def test_range_slope_sign(self):
        p, q = np.array(0.3), np.array(0.0)                 # ground rises to the east
        self.assertAlmostEqual(float(range_slope_deg(p, q, 270.0)), math.degrees(math.atan(0.3)), places=4)
        self.assertAlmostEqual(float(range_slope_deg(p, q, 90.0)), -math.degrees(math.atan(0.3)), places=4)

    def flags(self, dem, az=270.0, **cfg):
        cfg = AlignmentConfig(cast_shadow=False, **cfg)
        mask, slope, info = build_terrain_mask(dem, 10.0, az, cfg)
        return mask[5:-5, 5:-5]

    def test_flat_ground_unflagged(self):
        self.assertFalse(self.flags(self.ramp()).any())

    def test_layover_when_slope_faces_radar_steeper_than_beam(self):
        # sensor east (az 270); ground rising east at 45 deg => faces the sensor, > 39 deg beam
        m = self.flags(self.ramp(p=math.tan(math.radians(45))))
        self.assertTrue(((m & TERRAIN_LAYOVER) > 0).all() and ((m & TERRAIN_STEEP) > 0).all())

    def test_foreshortening(self):
        m = self.flags(self.ramp(p=math.tan(math.radians(35))))        # theta_loc = 4 deg < 10
        self.assertTrue(((m & TERRAIN_FORESHORTEN) > 0).all())
        self.assertFalse(((m & TERRAIN_LAYOVER) > 0).any())

    def test_local_shadow_when_facing_away(self):
        m = self.flags(self.ramp(p=-math.tan(math.radians(55))))       # theta_loc = 94 deg
        self.assertTrue(((m & TERRAIN_SHADOW) > 0).all())

    def test_gentle_slopes_unflagged_either_way(self):
        for p in (0.15, -0.15):
            self.assertFalse(self.flags(self.ramp(p=p)).any())

    def test_lia_overrides_dem_estimate(self):
        dem = self.ramp()
        for val, bit in ((85.0, TERRAIN_SHADOW), (5.0, TERRAIN_FORESHORTEN), (-1.0, TERRAIN_LAYOVER)):
            lia = np.full(dem.shape, val, np.float32)
            mask, _, info = build_terrain_mask(dem, 10.0, 270.0, AlignmentConfig(cast_shadow=False), lia)
            self.assertTrue(((mask & bit) > 0).all())
            self.assertEqual(info["local_geometry_source"], "snap_local_incidence_angle")

    def test_nan_dem_is_not_flagged(self):
        dem = self.ramp(p=0.9)
        dem[:, :10] = np.nan
        mask, _, _ = build_terrain_mask(dem, 10.0, 270.0, AlignmentConfig(cast_shadow=False))
        self.assertTrue((mask[10:20, :4] == 0).all())                  # unknown != bad


class TestCastShadow(unittest.TestCase):
    def plateau(self):
        dem = np.zeros((20, 200), np.float32)
        dem[:, 100:] = 1000.0
        return dem

    def test_shadow_behind_a_plateau_has_the_expected_width(self):
        # sensor EAST (look az 270), incidence 45 deg: a 1000 m wall shades 1000 m = 33 px of 30 m
        cs = cast_shadow(self.plateau(), 30.0, 270.0, 45.0, max_dist_m=3000, step_m=30, coarse=1)
        self.assertTrue(cs[:, 67:100].all())
        self.assertFalse(cs[:, :67].any())
        self.assertFalse(cs[:, 100:].any())                            # plateau top sees the radar

    def test_no_shadow_when_the_radar_is_on_the_low_side(self):
        cs = cast_shadow(self.plateau(), 30.0, 90.0, 45.0, max_dist_m=3000, step_m=30, coarse=1)
        self.assertFalse(cs.any())

    def test_steeper_incidence_means_shorter_shadow(self):
        a = cast_shadow(self.plateau(), 30.0, 270.0, 45.0, 3000, 30, 1).sum()
        b = cast_shadow(self.plateau(), 30.0, 270.0, 30.0, 3000, 30, 1).sum()   # tan(30) < tan(45)
        self.assertLess(b, a)

    def test_coarse_mode_keeps_shape_and_finds_shadow(self):
        cs = cast_shadow(self.plateau(), 30.0, 270.0, 45.0, 3000, 60, 3)
        self.assertEqual(cs.shape, (20, 200))
        self.assertGreater(cs[:, :100].sum(), 20 * 20)

    def test_nan_hole_does_not_disable_the_scan(self):
        dem = self.plateau()
        dem[5:8, 20:25] = np.nan
        cs = cast_shadow(dem, 30.0, 270.0, 45.0, 3000, 30, 1)
        self.assertTrue(cs[10:15, 67:100].all())

    def test_block_mean_handles_nan_and_remainder(self):
        a = np.arange(25, dtype=np.float32).reshape(5, 5)
        a[0, 0] = np.nan
        b = _block_mean(a, 2)
        self.assertEqual(b.shape, (3, 3))
        self.assertAlmostEqual(float(b[0, 0]), (1 + 5 + 6) / 3, places=5)
        self.assertAlmostEqual(float(b[2, 2]), 24.0, places=5)


class TestMasks(unittest.TestCase):
    cfg = AlignmentConfig()

    def test_otsu_bimodal(self):
        v = np.concatenate([RNG.normal(-8, 1.5, 7000), RNG.normal(-24, 1.5, 3000)])
        t, gap, dark = otsu_threshold(v)
        self.assertTrue(-18 < t < -13, t)
        self.assertGreater(gap, 12)
        self.assertAlmostEqual(dark, 0.3, delta=0.02)

    def test_otsu_clean_two_level_data_uses_middle_of_gap(self):
        v = np.concatenate([np.full(900, -8.0), np.full(100, -24.0)])
        t, _, _ = otsu_threshold(v)
        self.assertTrue(-20 < t < -12, t)                              # not stuck on the dark value

    def test_threshold_guard_falls_back_without_water(self):
        land = RNG.normal(-9, 1.2, 5000).astype(np.float32)
        t, how = choose_water_threshold(land, np.ones(land.shape, bool), self.cfg)
        self.assertEqual(t, self.cfg.water_default_db)
        self.assertTrue(how.startswith("default"))

    def test_threshold_accepts_real_bimodal(self):
        v = np.concatenate([RNG.normal(-8, 1.5, 4500), RNG.normal(-24, 1.5, 500)]).astype(np.float32)
        t, how = choose_water_threshold(v, np.ones(v.shape, bool), self.cfg)
        self.assertEqual(how, "otsu")

    def test_threshold_too_few_pixels(self):
        v = np.full(100, -9.0, np.float32)
        self.assertIn("too few", choose_water_threshold(v, np.ones(100, bool), self.cfg)[1])

    def test_remove_small_blobs(self):
        m = np.zeros((20, 20), bool)
        m[1:3, 1:3] = True                                             # 4 px
        m[8:14, 8:14] = True                                           # 36 px
        out = remove_small_blobs(m, 20)
        self.assertFalse(out[1:3, 1:3].any())
        self.assertTrue(out[8:14, 8:14].all())

    def scene(self):
        vv = RNG.normal(-8, 0.5, (60, 60)).astype(np.float32)
        pool = np.zeros((60, 60), bool)
        pool[10:30, 10:30] = True
        vv[pool] = RNG.normal(-24, 0.5, pool.sum())
        return vv, pool

    def test_permanent_water_finds_the_pool_only(self):
        vv, pool = self.scene()
        slope = np.zeros((60, 60), np.float32)
        terr = np.zeros((60, 60), np.uint8)
        # a dark but STEEP strip and a dark SHADOW strip must not become water
        vv[40:50, 5:30] = -24
        slope[40:50, 5:30] = 35
        vv[40:50, 35:55] = -24
        terr[40:50, 35:55] = TERRAIN_CAST_SHADOW
        w, info = permanent_water_mask([vv], slope, terr, np.ones((60, 60), bool), self.cfg)
        self.assertTrue(w[10:30, 10:30].mean() > 0.97)
        self.assertFalse(w[40:50].any())
        self.assertEqual(info["methods"], ["otsu"])

    def test_two_pre_event_images_are_anded(self):
        vv, pool = self.scene()
        vv2 = vv.copy()
        vv2[10:30, 20:30] = -8.0                                       # pool only half-dark the 2nd time
        ones = np.ones((60, 60), bool)
        w, _ = permanent_water_mask([vv, vv2], np.zeros((60, 60), np.float32),
                                    np.zeros((60, 60), np.uint8), ones, self.cfg)
        self.assertTrue(w[10:30, 10:20].mean() > 0.95)
        self.assertFalse(w[10:30, 21:30].any())

    def test_nan_slope_means_no_dem_and_water_is_still_found(self):
        vv, pool = self.scene()
        w, _ = permanent_water_mask([vv], np.full((60, 60), np.nan, np.float32),
                                    np.zeros((60, 60), np.uint8), np.ones((60, 60), bool), self.cfg)
        self.assertTrue(w[10:30, 10:30].mean() > 0.97)

    def test_valid_mask(self):
        a = np.ones((2, 4, 4), np.float32)
        b = np.ones((2, 4, 4), np.float32)
        b[1, 0, 0] = np.nan
        a[0, 3, 3] = np.inf
        v = valid_mask(a, b)
        self.assertFalse(v[0, 0] or v[3, 3])
        self.assertTrue(v[1, 1])

    def test_bit_flags_are_distinct_powers_of_two(self):
        vals = sorted(TERRAIN_BITS.values())
        self.assertEqual(vals, [1, 2, 4, 8, 16])


class TestAssertAligned(unittest.TestCase):
    def r(self, **kw):
        d = dict(data=np.zeros((1, 4, 5), np.float32), x0=0.0, y0=40.0, res=10.0, crs=CRS)
        d.update(kw)
        return GeoRaster(**d)

    def test_ok(self):
        assert_aligned({"a": self.r(), "b": self.r()})

    def test_reports_every_offender(self):
        with self.assertRaises(AlignmentError) as ctx:
            assert_aligned({"a": self.r(), "b": self.r(x0=5.0), "c": self.r(crs="EPSG:32644"),
                            "d": self.r(data=np.zeros((1, 3, 5), np.float32))})
        msg = str(ctx.exception)
        self.assertTrue(all(k in msg for k in ("b:", "c:", "d:")))
        self.assertIn("origin", msg)
        self.assertIn("crs", msg)
        self.assertIn("shape", msg)


# --------------------------------------------------------------------------
class FakeDem:
    def __init__(self, dem):
        self.dem, self.calls = dem, 0

    def describe(self):
        return "fake-dem"

    def load(self, bbox):
        self.calls += 1
        return self.dem


class BrokenDem:
    def describe(self):
        return "broken"

    def load(self, bbox):
        raise DemError("no tiles")


class TestPackage(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.tmp = Path(self.td.name)
        self.io = MemIO()
        self.grid = build_grid(AOI_S, CRS)
        self.pair = ScenePair(AOI_S, EVENT, Acquisition([mk((8, 14), direction="DESCENDING")]),
                              Acquisition([mk((8, 28), direction="DESCENDING")]), 1.0, "standard")

        g, m = self.grid, 30                                           # raster bigger than the grid (Part 2 margin)
        def img(vv_fn):
            h, w = g.height + 2 * m, g.width + 2 * m
            vv = np.full((h, w), -8.0, np.float32)
            vh = np.full((h, w), -15.0, np.float32)
            vv_fn(vv)
            return GeoRaster(np.stack([vv, vh]), g.x0 - m * 10, g.y0 + m * 10, 10.0, CRS)

        def pool(vv):
            vv[m + 20:m + 60, m + 20:m + 60] = -24.0                   # a lake before the event
        def flood(vv):
            pool(vv)
            vv[m + 80:m + 140, m + 80:m + 140] = -22.0                 # new flood after
        self.io.store[str(self.tmp / "pre" / "before.tif")] = img(pool)
        self.io.store[str(self.tmp / "pre" / "after.tif")] = img(flood)
        self.pre = PreInputs(self.tmp / "pre" / "before.tif", self.tmp / "pre" / "after.tif",
                             crs=CRS, params_hash="abc123", warnings=["from part 2"])
        self.dem = plane_dem(lambda n: 0.05 * (n - N_REF))             # gentle 2.9 degree plane

    def tearDown(self):
        self.td.cleanup()

    def build(self, dem=None, cfg=None):
        return build_analysis_package(self.pair, self.pre, cfg or AlignmentConfig(),
                                      dem or FakeDem(self.dem), self.io, self.tmp / "processed")

    def test_end_to_end_contract(self):
        pkg = self.build()
        g = self.grid
        for name in ("before", "after", "dem", "slope", "valid_mask", "terrain_mask", "permanent_water"):
            self.assertTrue(pkg.files[name].exists(), name)
            r = self.io.store[str(pkg.files[name])]
            self.assertTrue(g.matches(r), f"{name} is not on the grid")
        self.assertEqual(self.io.store[str(pkg.files["before"])].bands, 2)
        # masks are written as uint8
        for name in ("valid_mask", "terrain_mask", "permanent_water"):
            self.assertEqual(self.io.writes[f"{name}.tif"]["dtype"], "uint8")
        self.assertEqual(self.io.writes["before.tif"]["desc"], ["VV_dB", "VH_dB"])

    def test_permanent_water_is_the_pre_event_lake_not_the_new_flood(self):
        pkg = self.build()
        w = self.io.store[str(pkg.files["permanent_water"])].data[0].astype(bool)
        m = 0                                                           # grid-relative coordinates
        self.assertGreater(w[20:60, 20:60].mean(), 0.95)
        self.assertFalse(w[80:140, 80:140].any())                       # new flood is NOT permanent

    def test_slope_and_dem_values(self):
        pkg = self.build()
        slope = self.io.store[str(pkg.files["slope"])].data[0]
        self.assertAlmostEqual(float(np.nanmedian(slope)), math.degrees(math.atan(0.05)), delta=0.1)
        self.assertEqual(float((self.io.store[str(pkg.files["terrain_mask"])].data > 0).mean()), 0.0)

    def test_metadata_contract(self):
        pkg = self.build()
        meta = json.loads(pkg.metadata_path.read_text())
        self.assertEqual(meta["grid"]["crs"], CRS)
        self.assertEqual(meta["preprocess_params_hash"], "abc123")
        self.assertEqual(meta["terrain_bits"], TERRAIN_BITS)
        self.assertIn(DEM_ATTRIBUTION, meta["attributions"])
        self.assertIn("Contains modified Copernicus Sentinel data 2026.", meta["attributions"])
        self.assertIn("from part 2", meta["warnings"])                  # warnings are carried forward
        self.assertEqual(meta["track"]["direction"], "DESCENDING")
        self.assertAlmostEqual(meta["terrain"]["look_azimuth_deg"], 279.3, delta=1.5)

    def test_cache_hit_and_invalidation(self):
        dem = FakeDem(self.dem)
        p1 = self.build(dem)
        p2 = self.build(dem)
        self.assertFalse(p1.from_cache)
        self.assertTrue(p2.from_cache)
        self.assertEqual(dem.calls, 1)
        p3 = self.build(dem, AlignmentConfig(max_slope_deg=20.0))        # different settings
        self.assertFalse(p3.from_cache)

    def test_dem_failure_fails_soft_with_loud_warning(self):
        pkg = self.build(BrokenDem())
        self.assertTrue(any("DEM unavailable" in w for w in pkg.warnings))
        self.assertTrue(np.isnan(self.io.store[str(pkg.files["slope"])].data).all())
        self.assertEqual(int(self.io.store[str(pkg.files["terrain_mask"])].data.sum()), 0)
        w = self.io.store[str(pkg.files["permanent_water"])].data[0].astype(bool)
        self.assertGreater(w[20:60, 20:60].mean(), 0.95)                 # water still found without DEM

    def test_require_dem_raises(self):
        with self.assertRaises(DemError):
            self.build(BrokenDem(), AlignmentConfig(require_dem=True))

    def test_raster_outside_aoi_is_an_error(self):
        far = GeoRaster(np.zeros((2, 50, 50), np.float32), 100000.0, 5000000.0, 10.0, CRS)
        self.io.store[str(self.pre.before_tif)] = far
        with self.assertRaises(AlignmentError):
            self.build()

    def test_before2_is_processed_and_ands_water(self):
        b2 = self.io.store[str(self.pre.before_tif)]
        self.io.store[str(self.tmp / "pre" / "before2.tif")] = b2
        self.pre.before2_tif = self.tmp / "pre" / "before2.tif"
        pkg = self.build()
        self.assertIn("before2", pkg.files)
        self.assertTrue(self.grid.matches(self.io.store[str(pkg.files["before2"])]))

    def test_pre_inputs_from_summary_json(self):
        d = self.tmp / "s"
        d.mkdir()
        (d / "s1_before_h.tif").write_bytes(b"x")
        (d / "s1_before_lia_h.tif").write_bytes(b"x")
        (d / "preprocess_summary.json").write_text(json.dumps({
            "params_hash": "h", "crs": CRS, "before": str(d / "s1_before_h.tif"),
            "after": str(d / "s1_after_h.tif"), "before2": None, "warnings": ["w"]}))
        pre = PreInputs.from_summary_json(d / "preprocess_summary.json")
        self.assertEqual(pre.lia_tif, d / "s1_before_lia_h.tif")
        self.assertIsNone(pre.before2_tif)
        self.assertEqual(pre.warnings, ["w"])


if __name__ == "__main__":
    unittest.main()
