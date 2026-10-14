"""
Offline tests for preprocessing/sentinel1.py  (no SNAP, no rasterio, no network).

Run from the repo root:
    python -m unittest tests.test_preprocessing -v

How it works: a FakeRunner "runs SNAP" by reading the generated graph.xml and
writing synthetic linear VV/VH rasters to wherever the Write nodes point. A MemIO
stores rasters in a dict instead of GeoTIFF files. So the whole orchestration
(graph -> run -> mosaic -> dB -> QA -> cache -> metadata) is exercised for real.
One test also executes a real subprocess (a fake 'gpt' shell script).
"""
import json
import os
import stat
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from acquisition.sentinel import AOI, Acquisition, ScenePair
from preprocessing.sentinel1 import (
    GeoRaster, PreprocessConfig, SnapError, SnapNotFoundError, SnapRunner,
    NoUsableDataError, PreprocessingError, buffered_aoi, build_graph_xml,
    choose_subset_mode, clean_linear_to_db, diagnose_snap_log,
    distance_to_footprint_edge_km, erode_valid, find_gpt, grid_offset_px, mosaic,
    normalize_db, perimeter_points, preprocess_acquisition, preprocess_pair,
    qa_report, to_db, utm_epsg)
from tests.test_acquisition import AOI_BOX, EVENT, mk

AOI_ = AOI.from_bbox(AOI_BOX)
CRS = "EPSG:32645"


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class MemIO:
    """Raster 'files' kept in memory; also touches a real file so .exists() works."""

    def __init__(self):
        self.store = {}
        self.writes = []

    def read(self, path):
        return self.store[str(path)]

    def write(self, path, r, descriptions=None, tags=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"x")
        self.store[str(path)] = r
        self.writes.append((str(path), descriptions, tags))


class FakeRunner:
    """Pretends to be SNAP. ``geoms`` maps a zip file name to (x0, y0, width, height).

    VV is a bright land value with a dark 'river'; VH is ~7 dB below VV.
    Zero pixels (no-data) are put on the left 2 columns to test the edge handling.
    """

    def __init__(self, io, geoms, vv=0.10, vh=0.02, swap=False):
        self.io, self.geoms, self.vv, self.vh, self.swap = io, geoms, vv, vh, swap
        self.calls = 0

    def command(self, graph):
        return ["fake-gpt", str(graph)]

    def run(self, graph_path, log_path):
        self.calls += 1
        root = ET.parse(graph_path).getroot()
        nodes = {n.get("id"): n for n in root.findall("node")}
        zip_name = Path(nodes["Read"].find("parameters/file").text).name
        x0, y0, w, h = self.geoms[zip_name]
        vv = np.full((1, h, w), self.vv, np.float32)
        vh = np.full((1, h, w), self.vh, np.float32)
        vv[:, :, 10:12] = 0.003                       # dark river stripe
        vv[:, :, :2] = 0.0                            # no-data columns
        vh[:, :, :2] = 0.0
        if self.swap:
            vv, vh = vh, vv
        for key, arr in (("VV", vv), ("VH", vh)):
            out = Path(nodes[f"Write-{key}"].find("parameters/file").text)
            self.io.write(out, GeoRaster(arr.copy(), x0, y0, 10.0, CRS))
        if "Write-LIA" in nodes:
            out = Path(nodes["Write-LIA"].find("parameters/file").text)
            self.io.write(out, GeoRaster(np.full((1, h, w), 35.0, np.float32), x0, y0, 10.0, CRS))
        Path(log_path).write_text("ok")


def acq_with_zip(tmp, *names_and_days):
    """Acquisition whose slices have real (empty) zip files on disk."""
    scenes = []
    for i, (day, kw) in enumerate(names_and_days):
        s = mk(day, **kw)
        z = Path(tmp) / f"{s.stem}.zip"
        z.write_bytes(b"zip")
        s.local_path = str(z)
        scenes.append(s)
    return Acquisition(scenes)


# --------------------------------------------------------------------------
class TestHelpers(unittest.TestCase):
    def test_utm_epsg(self):
        self.assertEqual(utm_epsg(85.3, 27.7), 32645)       # Trishuli area
        self.assertEqual(utm_epsg(83.9, 28.0), 32644)       # western Nepal
        self.assertEqual(utm_epsg(-70.0, -33.0), 32719)     # southern hemisphere
        self.assertEqual(utm_epsg(179.99, 10), 32660)       # upper edge clamps to zone 60

    def test_to_db_nodata_is_nan_not_inf(self):
        out = to_db(np.array([0.1, 0.0, -1.0, np.nan, 1.0], np.float32))
        self.assertAlmostEqual(float(out[0]), -10.0, places=4)
        self.assertTrue(np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3]))
        self.assertAlmostEqual(float(out[4]), 0.0, places=5)
        self.assertFalse(np.isinf(out).any())

    def test_normalize_db(self):
        out = normalize_db(np.array([-40, -30, -15, 0, 5, np.nan], np.float32))
        np.testing.assert_allclose(out[:5], [0, 0, 0.5, 1, 1])
        self.assertTrue(np.isnan(out[5]))

    def test_erosion_keeps_array_border(self):
        valid = np.ones((20, 20), bool)
        valid[:, 10] = False                                 # an internal no-data column
        out = erode_valid(valid, 2)
        self.assertFalse(out[:, 8:13].any() and out[:, 8:13].all())
        self.assertTrue(out[5, 0] and out[5, 19])            # outer border NOT eaten
        self.assertFalse(out[5, 9] or out[5, 11])            # but the rim around the hole is

    def test_grid_offset(self):
        r = GeoRaster(np.zeros((1, 2, 2), np.float32), 500000.0, 3100000.0, 10.0, CRS)
        self.assertEqual(grid_offset_px(r), (0.0, 0.0))
        r2 = GeoRaster(np.zeros((1, 2, 2), np.float32), 500003.0, 3100000.0, 10.0, CRS)
        self.assertAlmostEqual(grid_offset_px(r2)[0], 0.3, places=6)


class TestFootprintMaths(unittest.TestCase):
    big = [[[84.0, 27.0], [86.0, 27.0], [86.0, 29.0], [84.0, 29.0], [84.0, 27.0]]]

    def test_inside_far_from_edge(self):
        d = distance_to_footprint_edge_km(perimeter_points(AOI_), self.big)
        # Nearest footprint edge is the EAST one: 86.0 - 85.5 = 0.5 deg * ~98 km ~ 49 km
        self.assertTrue(45 < d < 53, d)

    def test_near_edge_is_small(self):
        near = [[[85.09, 27.0], [86.0, 27.0], [86.0, 29.0], [85.09, 29.0], [85.09, 27.0]]]
        d = distance_to_footprint_edge_km(perimeter_points(AOI_), near)
        self.assertTrue(0 < d < 2)

    def test_outside_is_negative(self):
        part = [[[85.3, 27.0], [86.0, 27.0], [86.0, 29.0], [85.3, 29.0], [85.3, 27.0]]]
        self.assertLess(distance_to_footprint_edge_km(perimeter_points(AOI_), part), 0)

    def test_buffered_aoi_grows_about_right(self):
        b = buffered_aoi(AOI_, 3.0)
        self.assertAlmostEqual((AOI_.min_lat - b.min_lat) * 110.57, 3.0, places=2)

    def test_choose_subset_mode(self):
        s = mk((8, 14))                                       # FULL footprint: far from edge
        cfg = PreprocessConfig()
        self.assertTrue(choose_subset_mode(AOI_, s, cfg)[0])
        edge = mk((8, 14), rings=[[[85.09, 27.0], [86.0, 27.0], [86.0, 29.0],
                                   [85.09, 29.0], [85.09, 27.0]]])
        self.assertFalse(choose_subset_mode(AOI_, edge, cfg)[0])      # near edge: full scene
        cfg.subset_early = "always"
        self.assertTrue(choose_subset_mode(AOI_, edge, cfg)[0])
        cfg.subset_early = "never"
        self.assertFalse(choose_subset_mode(AOI_, s, cfg)[0])


class TestGraph(unittest.TestCase):
    def build(self, **over):
        cfg = PreprocessConfig(**over)
        outs = {"VV": Path("/o/vv.tif"), "VH": Path("/o/vh.tif")}
        if cfg.save_incidence_angle:
            outs["LIA"] = Path("/o/lia.tif")
        xml = build_graph_xml(Path("/z/a.zip"), outs, AOI_, CRS, cfg, crop_early=True)
        return ET.fromstring(xml)

    def test_wellformed_and_references_resolve(self):
        root = self.build()
        ids = {n.get("id") for n in root.findall("node")}
        for n in root.findall("node"):
            for src in n.find("sources"):
                self.assertIn(src.get("refid"), ids)

    def test_operator_chain_order(self):
        root = self.build()
        ops = {n.get("id"): n.find("operator").text for n in root.findall("node")}
        refs = {n.get("id"): [s.get("refid") for s in n.find("sources")] for n in root.findall("node")}
        chain, cur = [], "TC"
        while refs[cur]:
            cur = refs[cur][0]
            chain.append(ops[cur])
        self.assertEqual(chain[::-1], ["Read", "Subset", "Apply-Orbit-File", "Remove-GRD-Border-Noise",
                                       "ThermalNoiseRemoval", "Calibration", "Speckle-Filter"])

    def test_key_parameters(self):
        root = self.build()
        tc = {c.tag: c.text for c in root.find("node[@id='TC']/parameters")}
        self.assertEqual(tc["mapProjection"], CRS)
        self.assertEqual(tc["pixelSpacingInMeter"], "10.0")
        self.assertEqual(tc["nodataValueAtSea"], "false")
        self.assertEqual(tc["alignToStandardGrid"], "true")
        self.assertEqual(tc["demName"], "Copernicus 30m Global DEM")
        orb = {c.tag: c.text for c in root.find("node[@id='Orbit']/parameters")}
        self.assertEqual(orb["continueOnFail"], "true")
        cal = {c.tag: c.text for c in root.find("node[@id='Calib']/parameters")}
        self.assertEqual(cal["outputImageScaleInDb"], "false")               # stays linear

    def test_no_early_subset_when_not_safe(self):
        cfg = PreprocessConfig()
        xml = build_graph_xml(Path("/z/a.zip"), {"VV": Path("v"), "VH": Path("h")}, AOI_, CRS, cfg, False)
        self.assertNotIn("Subset-Early", xml)

    def test_external_dem(self):
        root = self.build(dem_path="/data/dem.tif")
        tc = {c.tag: c.text for c in root.find("node[@id='TC']/parameters")}
        self.assertEqual(tc["demName"], "External DEM")
        self.assertEqual(tc["externalDEMFile"], "/data/dem.tif")
        self.assertEqual(tc["externalDEMApplyEGM"], "true")

    def test_incidence_angle_branch_optional(self):
        self.assertIsNone(self.build().find("node[@id='Write-LIA']"))
        root = self.build(save_incidence_angle=True)
        self.assertIsNotNone(root.find("node[@id='Write-LIA']"))
        tc = {c.tag: c.text for c in root.find("node[@id='TC']/parameters")}
        self.assertEqual(tc["saveLocalIncidenceAngle"], "true")

    def test_paths_with_special_characters_are_escaped(self):
        cfg = PreprocessConfig()
        xml = build_graph_xml(Path("/z/a&b <1>.zip"), {"VV": Path("v"), "VH": Path("h")},
                              AOI_, CRS, cfg, True)
        ET.fromstring(xml)                                                   # must still parse
        self.assertIn("a&amp;b", xml)

    def test_hash_changes_with_settings_not_with_runtime(self):
        base = PreprocessConfig().params_hash(AOI_, CRS)
        self.assertEqual(base, PreprocessConfig(snap_mem_gb=16, timeout_s=5).params_hash(AOI_, CRS))
        self.assertNotEqual(base, PreprocessConfig(speckle_filter="Lee").params_hash(AOI_, CRS))
        other = AOI.from_bbox((85.0, 27.9, 85.5, 28.3))
        self.assertNotEqual(base, PreprocessConfig().params_hash(other, CRS))


class TestSnapRunner(unittest.TestCase):
    def fake_gpt(self, td, body):
        p = Path(td) / "gpt"
        p.write_text("#!/bin/sh\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        return str(p)

    def test_diagnose_log(self):
        self.assertIn("mem-gb", diagnose_snap_log("java.lang.OutOfMemoryError: Java heap space"))
        self.assertIn("intersect", diagnose_snap_log("Error: No intersection with product"))
        self.assertIn("corrupt", diagnose_snap_log("java.util.zip.ZipException: invalid"))
        self.assertEqual(diagnose_snap_log("all good"), "")

    def test_find_gpt_missing(self):
        old = os.environ.pop("SNAP_GPT", None)
        try:
            if Path("/opt/snap/bin/gpt").exists():
                self.skipTest("a real SNAP is installed here")
            with self.assertRaises(SnapNotFoundError):
                find_gpt("/definitely/not/here")
        finally:
            if old:
                os.environ["SNAP_GPT"] = old

    def test_command_has_memory_flags(self):
        with tempfile.TemporaryDirectory() as td:
            r = SnapRunner(PreprocessConfig(gpt_path=self.fake_gpt(td, "exit 0\n"),
                                            snap_mem_gb=10, snap_threads=4))
            cmd = r.command(Path("g.xml"))
            self.assertIn("-J-Xmx10G", cmd)
            self.assertIn("6G", cmd)
            self.assertEqual(cmd[-2:], ["-q", "4"])

    @unittest.skipIf(os.name == "nt", "the fake gpt is a POSIX shell script")
    def test_success_and_failure_with_hint(self):
        with tempfile.TemporaryDirectory() as td:
            ok = SnapRunner(PreprocessConfig(gpt_path=self.fake_gpt(td, "echo done\nexit 0\n")))
            ok.run(Path(td) / "g.xml", Path(td) / "ok.log")                     # no exception
            bad = SnapRunner(PreprocessConfig(gpt_path=self.fake_gpt(
                td, "echo 'java.lang.OutOfMemoryError: Java heap space'\nexit 1\n")))
            with self.assertRaises(SnapError) as ctx:
                bad.run(Path(td) / "g.xml", Path(td) / "bad.log")
            self.assertIn("memory", str(ctx.exception).lower())
            self.assertIn("bad.log", str(ctx.exception))

    @unittest.skipIf(os.name == "nt", "the fake gpt is a POSIX shell script")
    def test_error_text_with_exit_zero_still_fails(self):
        with tempfile.TemporaryDirectory() as td:
            r = SnapRunner(PreprocessConfig(gpt_path=self.fake_gpt(
                td, "echo 'Error: Unknown operator X'\nexit 0\n")))
            with self.assertRaises(SnapError):
                r.run(Path(td) / "g.xml", Path(td) / "l.log")


class TestMosaic(unittest.TestCase):
    def r(self, x0, y0, w, h, val):
        return GeoRaster(np.full((1, h, w), val, np.float32), x0, y0, 10.0, CRS)

    def test_integer_paste_and_first_wins(self):
        a = self.r(500000, 3100000, 10, 10, 1.0)               # covers rows 0..9
        b = self.r(500000, 3099950, 10, 10, 2.0)               # 5 rows lower: overlaps rows 5..9
        m, frac = mosaic([a, b])
        self.assertEqual(m.data.shape, (1, 15, 10))
        self.assertEqual(frac, 0.0)
        self.assertTrue((m.data[0, :10] == 1.0).all())          # earlier raster wins in overlap
        self.assertTrue((m.data[0, 10:] == 2.0).all())
        self.assertEqual((m.x0, m.y0), (500000, 3100000))

    def test_fills_nodata_from_second(self):
        a = self.r(500000, 3100000, 4, 4, 1.0)
        a.data[0, :, 2:] = np.nan
        b = self.r(500000, 3100000, 4, 4, 7.0)
        m, _ = mosaic([a, b])
        self.assertTrue((m.data[0, :, :2] == 1.0).all() and (m.data[0, :, 2:] == 7.0).all())

    def test_union_extends_west_and_north(self):
        a = self.r(500000, 3100000, 5, 5, 1.0)
        b = self.r(499980, 3100020, 5, 5, 2.0)                  # 2 px west, 2 px north
        m, _ = mosaic([a, b])
        self.assertEqual((m.x0, m.y0), (499980, 3100020))
        self.assertEqual(m.data.shape, (1, 7, 7))

    def test_fractional_offset_reported(self):
        a = self.r(500000, 3100000, 5, 5, 1.0)
        b = self.r(500013, 3100000, 5, 5, 2.0)                  # 1.3 px east
        _, frac = mosaic([a, b])
        self.assertAlmostEqual(frac, 0.3, places=5)

    def test_crs_mismatch_raises(self):
        a = self.r(0, 0, 3, 3, 1.0)
        b = GeoRaster(np.ones((1, 3, 3), np.float32), 0, 0, 10.0, "EPSG:32644")
        with self.assertRaises(PreprocessingError):
            mosaic([a, b])


class TestCleanAndQA(unittest.TestCase):
    cfg = PreprocessConfig()

    def lin(self, vv, vh, w=30, h=30):
        return (GeoRaster(np.full((1, h, w), vv, np.float32), 0, 0, 10.0, CRS),
                GeoRaster(np.full((1, h, w), vh, np.float32), 0, 0, 10.0, CRS))

    def test_db_values_and_bands(self):
        vv, vh = self.lin(0.1, 0.02)
        db = clean_linear_to_db(vv, vh, self.cfg)
        self.assertEqual(db.bands, 2)
        self.assertAlmostEqual(float(db.data[0, 15, 15]), -10.0, places=3)
        self.assertAlmostEqual(float(db.data[1, 15, 15]), 10 * np.log10(0.02), places=3)

    def test_zero_nodata_becomes_nan_and_rim_removed(self):
        vv, vh = self.lin(0.1, 0.02)
        vv.data[0, :, :5] = 0.0
        db = clean_linear_to_db(vv, vh, self.cfg)
        self.assertTrue(np.isnan(db.data[:, :, :5]).all())
        self.assertTrue(np.isnan(db.data[:, :, 5:8]).all())      # edge guard (3 px) beyond the hole
        self.assertTrue(np.isfinite(db.data[:, :, 9:]).all())
        self.assertFalse(np.isinf(db.data).any())

    def test_crop_border_not_eroded(self):
        vv, vh = self.lin(0.1, 0.02)
        db = clean_linear_to_db(vv, vh, self.cfg)
        self.assertTrue(np.isfinite(db.data).all())              # nothing lost at array edges

    def test_clip_range(self):
        vv, vh = self.lin(1e-9, 1e3)
        db = clean_linear_to_db(vv, vh, self.cfg)
        self.assertEqual(float(db.data[0, 15, 15]), -50.0)
        self.assertEqual(float(db.data[1, 15, 15]), 20.0)

    def test_qa_ok(self):
        vv, vh = self.lin(0.1, 0.02)
        stats, warns = qa_report(clean_linear_to_db(vv, vh, self.cfg), self.cfg)
        self.assertEqual(warns, [])
        self.assertAlmostEqual(stats["median_vv_minus_vh_db"], 7.0, places=1)

    def test_qa_detects_swapped_polarisations(self):
        vv, vh = self.lin(0.02, 0.1)                              # VV darker than VH: swapped
        _, warns = qa_report(clean_linear_to_db(vv, vh, self.cfg), self.cfg)
        self.assertTrue(any("swapped" in w for w in warns))

    def test_qa_fatal_when_empty(self):
        vv, vh = self.lin(0.0, 0.0)
        with self.assertRaises(NoUsableDataError):
            qa_report(clean_linear_to_db(vv, vh, self.cfg), self.cfg)

    def test_qa_warns_on_mostly_dark(self):
        vv, vh = self.lin(0.001, 0.0001)                          # -30 dB / -40 dB
        _, warns = qa_report(clean_linear_to_db(vv, vh, self.cfg), self.cfg)
        self.assertTrue(any("very dark" in w for w in warns))


class TestOrchestration(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.tmp = Path(self.td.name)
        self.cfg = PreprocessConfig()
        self.io = MemIO()

    def tearDown(self):
        self.td.cleanup()

    def test_single_slice_end_to_end(self):
        acq = acq_with_zip(self.tmp, ((8, 14), {}))
        name = Path(acq.slices[0].local_path).name
        runner = FakeRunner(self.io, {name: (500000.0, 3100000.0, 40, 30)})
        res = preprocess_acquisition(acq, AOI_, "before", self.tmp / "out", self.cfg,
                                     io=self.io, runner=runner)
        self.assertEqual(res.crs, "EPSG:32645")
        db = self.io.store[str(res.tif_path)]
        self.assertEqual(db.bands, 2)
        self.assertAlmostEqual(float(db.data[0, 15, 25]), -10.0, places=3)     # VV dB
        self.assertTrue(np.isnan(db.data[0, 15, 0]))                           # no-data columns
        meta = json.loads(res.meta_path.read_text())
        self.assertEqual(meta["bands"], ["VV", "VH"])
        self.assertEqual(meta["params_hash"], res.params_hash)
        desc = [w for w in self.io.writes if w[0] == str(res.tif_path)][0][1]
        self.assertEqual(desc, ["VV_dB", "VH_dB"])
        self.assertTrue((Path(res.slices[0]["graph"])).exists())               # receipt kept

    def test_cache_hit_skips_snap(self):
        acq = acq_with_zip(self.tmp, ((8, 14), {}))
        name = Path(acq.slices[0].local_path).name
        runner = FakeRunner(self.io, {name: (500000.0, 3100000.0, 40, 30)})
        out = self.tmp / "out"
        r1 = preprocess_acquisition(acq, AOI_, "before", out, self.cfg, self.io, runner)
        r2 = preprocess_acquisition(acq, AOI_, "before", out, self.cfg, self.io, runner)
        self.assertEqual(runner.calls, 1)
        self.assertTrue(r2.from_cache and not r1.from_cache)

    def test_changed_settings_reprocess(self):
        acq = acq_with_zip(self.tmp, ((8, 14), {}))
        name = Path(acq.slices[0].local_path).name
        runner = FakeRunner(self.io, {name: (500000.0, 3100000.0, 40, 30)})
        out = self.tmp / "out"
        preprocess_acquisition(acq, AOI_, "before", out, self.cfg, self.io, runner)
        preprocess_acquisition(acq, AOI_, "before", out, PreprocessConfig(speckle_filter="Lee"),
                               self.io, runner)
        self.assertEqual(runner.calls, 2)

    def test_two_slices_are_stitched(self):
        acq = acq_with_zip(self.tmp, ((8, 14), {"pid": "s1", "hour": 12}),
                           ((8, 14), {"pid": "s2", "hour": 13, "abs_orbit": 50814}))
        # make names unique & ordered: rebuild geoms by actual file names
        n1, n2 = (Path(s.local_path).name for s in acq.slices)
        runner = FakeRunner(self.io, {n1: (500000.0, 3100000.0, 40, 30),
                                      n2: (500000.0, 3099800.0, 40, 30)})       # 20 rows lower
        res = preprocess_acquisition(acq, AOI_, "before", self.tmp / "out", self.cfg,
                                     io=self.io, runner=runner)
        db = self.io.store[str(res.tif_path)]
        self.assertEqual(db.data.shape, (2, 50, 40))                            # 30 + 30 - 10 overlap
        self.assertEqual(runner.calls, 2)

    def test_pair_same_hash_and_summary_file(self):
        pair = ScenePair(AOI_, EVENT, acq_with_zip(self.tmp, ((8, 14), {})),
                         acq_with_zip(self.tmp, ((8, 28), {})), 1.0, "standard",
                         warnings=["from part 1"])
        geoms = {Path(s.local_path).name: (500000.0, 3100000.0, 40, 30) for s in pair.all_scenes()}
        runner = FakeRunner(self.io, geoms)
        res = preprocess_pair(pair, self.cfg, self.tmp / "interim", io=self.io, runner=runner)
        self.assertEqual(res.before.params_hash, res.after.params_hash)
        self.assertIn("from part 1", res.warnings)
        self.assertTrue((res.out_dir / "preprocess_summary.json").exists())
        self.assertIn("valid=", res.summary())

    def test_swapped_channels_warning_reaches_pair(self):
        pair = ScenePair(AOI_, EVENT, acq_with_zip(self.tmp, ((8, 14), {})),
                         acq_with_zip(self.tmp, ((8, 28), {})), 1.0, "standard")
        geoms = {Path(s.local_path).name: (500000.0, 3100000.0, 40, 30) for s in pair.all_scenes()}
        res = preprocess_pair(pair, self.cfg, self.tmp / "i", io=self.io,
                              runner=FakeRunner(self.io, geoms, swap=True))
        self.assertTrue(any("swapped" in w for w in res.warnings))

    def test_misaligned_grid_warns(self):
        acq = acq_with_zip(self.tmp, ((8, 14), {}))
        name = Path(acq.slices[0].local_path).name
        runner = FakeRunner(self.io, {name: (500003.0, 3100000.0, 40, 30)})     # 0.3 px off lattice
        res = preprocess_acquisition(acq, AOI_, "before", self.tmp / "o", self.cfg, self.io, runner)
        self.assertTrue(any("standard lattice" in w for w in res.warnings))

    def test_incidence_angle_output(self):
        cfg = PreprocessConfig(save_incidence_angle=True)
        acq = acq_with_zip(self.tmp, ((8, 14), {}))
        name = Path(acq.slices[0].local_path).name
        runner = FakeRunner(self.io, {name: (500000.0, 3100000.0, 40, 30)})
        res = preprocess_acquisition(acq, AOI_, "before", self.tmp / "o", cfg, self.io, runner)
        self.assertIsNotNone(res.lia_path)
        self.assertAlmostEqual(float(self.io.store[str(res.lia_path)].data[0, 5, 5]), 35.0)

    def test_dry_run_writes_graph_only(self):
        acq = acq_with_zip(self.tmp, ((8, 14), {}))
        runner = FakeRunner(self.io, {})
        res = preprocess_acquisition(acq, AOI_, "before", self.tmp / "o", self.cfg, self.io,
                                     runner, dry_run=True)
        self.assertEqual(runner.calls, 0)
        self.assertIsNone(res.tif_path)
        self.assertTrue(Path(res.slices[0]["graph"]).exists())
        self.assertIn("fake-gpt", res.slices[0]["command"])

    def test_missing_zip_is_a_clear_error(self):
        acq = acq_with_zip(self.tmp, ((8, 14), {}))
        Path(acq.slices[0].local_path).unlink()
        with self.assertRaises(PreprocessingError) as ctx:
            preprocess_acquisition(acq, AOI_, "before", self.tmp / "o", self.cfg, self.io,
                                   FakeRunner(self.io, {}))
        self.assertIn("download", str(ctx.exception))

    def test_snap_not_writing_output_is_detected(self):
        class Lazy(FakeRunner):
            def run(self, graph_path, log_path):
                Path(log_path).write_text("done")                # exits fine, writes nothing
        acq = acq_with_zip(self.tmp, ((8, 14), {}))
        with self.assertRaises(SnapError):
            preprocess_acquisition(acq, AOI_, "before", self.tmp / "o", self.cfg, self.io,
                                   Lazy(self.io, {}))


if __name__ == "__main__":
    unittest.main()
