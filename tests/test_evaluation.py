import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dashboard.raster_overlay import encode_png_rgba
from evaluation import case_study, metrics, oracle, reference
from pipeline.contracts import Artifact
from pipeline.errors import InputValidationError
from pipeline.fileio import read_json, write_geojson, feature
from pipeline.geo_math import lonlat_to_utm, utm_to_lonlat
from pipeline.rules import GUARDED_FOLDERS
from pipeline.run import RunPaths
from scripts.make_mock_outputs import build_mock_run

EPSG = 32645
REPO_ROOT = Path(__file__).resolve().parents[1]


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.ref = np.zeros((10, 10), bool)
        self.ref[2:6, 2:6] = True                    # 16 px
        self.ours = np.zeros((10, 10), bool)
        self.ours[3:7, 3:7] = True                   # 16 px, overlap 9

    def test_confusion_and_scores(self):
        c = metrics.confusion(self.ours, self.ref)
        self.assertEqual(c, {"tp": 9, "fp": 7, "fn": 7, "tn": 77})
        s = metrics.scores(c)
        self.assertAlmostEqual(s["iou"], 9 / 23)
        self.assertAlmostEqual(s["precision"], 9 / 16)
        self.assertAlmostEqual(s["recall"], 9 / 16)
        self.assertAlmostEqual(s["f1"], 18 / 32)

    def test_undefined_scores_are_none_not_zero(self):
        s = metrics.scores(metrics.confusion(np.zeros((4, 4), bool), np.zeros((4, 4), bool)))
        self.assertEqual(s, {"iou": None, "precision": None, "recall": None, "f1": None})

    def test_valid_region_excludes_pixels_the_reference_never_examined(self):
        valid = np.zeros((10, 10), bool)
        valid[:5, :] = True
        c = metrics.confusion(self.ours, self.ref, valid)
        self.assertEqual(c["fp"] + c["tp"] + c["fn"] + c["tn"], 50)
        self.assertLess(c["fp"], 7)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.confusion(np.zeros((3, 3)), np.zeros((4, 4)))

    def test_extent_metrics_areas(self):
        m = metrics.extent_metrics(self.ours, self.ref, pixel_km2=0.01)
        self.assertEqual(m["areas_km2"], {"ours": 0.16, "reference": 0.16, "overlap": 0.09})

    def test_agreement_codes_and_png(self):
        codes = metrics.agreement_codes(self.ours, self.ref)
        self.assertEqual(int((codes == 1).sum()), 9)
        self.assertEqual(int((codes == 2).sum()), 7)
        self.assertEqual(int((codes == 3).sum()), 7)
        png, bounds = metrics.agreement_png(codes, (50.0, 0.0, 320_000.0, 0.0, -50.0, 3_115_000.0), EPSG, 100)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(len(bounds), 4)

    def test_compare_ids(self):
        c = metrics.compare_ids(["a", "b", "c"], ["b", "c", "d", "e"])
        self.assertEqual(c, {"reference_flagged": 4, "also_flagged": 2, "extra_flagged": 1, "missed": 2,
                             "agreement": 0.5})
        self.assertIsNone(metrics.compare_ids(["a"], [])["agreement"])


class RasterizeTests(unittest.TestCase):
    def setUp(self):
        self.x0, self.y0 = 320_000.0, 3_115_000.0
        self.transform = (50.0, 0.0, self.x0, 0.0, -50.0, self.y0)
        self.shape = (200, 300)

    def square_lonlat(self, x1, y1, x2, y2):
        ring = [utm_to_lonlat(x, y, EPSG) for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1))]
        return [[list(p) for p in ring]]

    def test_square_is_burned_on_the_right_pixels(self):
        # UTM square 1000..3000 m east, 500..2500 m south of the top-left corner = pixels cols 20..60, rows 10..50
        coords = self.square_lonlat(self.x0 + 1000, self.y0 - 2500, self.x0 + 3000, self.y0 - 500)
        mask = reference.rasterize_polygons(reference.polygons_from_geojson(
            {"type": "Polygon", "coordinates": coords}), self.transform, self.shape, EPSG)
        rows, cols = np.nonzero(mask)
        self.assertEqual((cols.min(), cols.max() + 1, rows.min(), rows.max() + 1), (20, 60, 10, 50))
        self.assertEqual(int(mask.sum()), 40 * 40)

    def test_holes_are_respected(self):
        outer = self.square_lonlat(self.x0 + 1000, self.y0 - 2500, self.x0 + 3000, self.y0 - 500)[0]
        hole = self.square_lonlat(self.x0 + 1500, self.y0 - 2000, self.x0 + 2500, self.y0 - 1000)[0]
        mask = reference.rasterize_polygons([[[tuple(p) for p in outer], [tuple(p) for p in hole]]],
                                            self.transform, self.shape, EPSG)
        self.assertEqual(int(mask.sum()), 40 * 40 - 20 * 20)

    def test_polygon_outside_the_raster_is_ignored_and_partial_overlap_is_clipped(self):
        far = self.square_lonlat(self.x0 + 900_000, self.y0, self.x0 + 901_000, self.y0 - 1000)
        self.assertEqual(int(reference.rasterize_polygons([[[tuple(p) for p in far[0]]]],
                                                          self.transform, self.shape, EPSG).sum()), 0)
        edge = self.square_lonlat(self.x0 - 1000, self.y0 + 1000, self.x0 + 500, self.y0 - 500)
        mask = reference.rasterize_polygons([[[tuple(p) for p in edge[0]]]], self.transform, self.shape, EPSG)
        self.assertEqual(int(mask.sum()), 10 * 10)         # only the part inside the raster

    def test_geojson_shapes_are_all_read(self):
        poly = {"type": "Polygon", "coordinates": [[[85, 28], [85.1, 28], [85.1, 28.1], [85, 28]]]}
        self.assertEqual(len(reference.polygons_from_geojson(poly)), 1)
        self.assertEqual(len(reference.polygons_from_geojson({"type": "MultiPolygon",
                                                              "coordinates": [poly["coordinates"]] * 2})), 2)
        fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": poly}, {"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 2]}}]}
        self.assertEqual(len(reference.polygons_from_geojson(fc)), 1)

    def test_missing_file_and_unsupported_format(self):
        with self.assertRaises(InputValidationError):
            reference.load_reference_polygons("/does/not/exist.geojson")


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.rp = build_mock_run(Path(cls.tmp.name) / "r", with_rasters=False)
        rd = lambda a: read_json(cls.rp.path(a))
        cls.roads, cls.settle, cls.dest = rd(Artifact.OSM_ROADS), rd(Artifact.OSM_SETTLEMENTS), rd(Artifact.OSM_HOSPITALS)
        cls.aff = [f["properties"]["object_id"] for f in rd(Artifact.AFFECTED_ROADS)["features"]]
        cls.bridges = [f["properties"]["object_id"] for f in rd(Artifact.AFFECTED_BRIDGES)["features"]]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def analyse(self, removed):
        return oracle.analyse(self.roads, self.settle, self.dest, removed)

    def test_mock_scenario_matches_the_designed_story(self):
        res = self.analyse(oracle.expand_removed(self.aff, self.bridges, self.roads))
        status = {v["name"]: v["status"] for v in res.values()}
        self.assertEqual(status, {"Village A": "connected", "Village B": "cut_off", "Village C": "cut_off",
                                  "Village D": "cut_off"})

    def test_blocking_segments_name_the_real_cause(self):
        res = self.analyse(oracle.expand_removed(self.aff, self.bridges, self.roads))
        self.assertTrue(all(s.startswith("way/main#") for s in res["node/B"]["blocking_segment_ids"]))
        self.assertTrue(all(s.startswith("way/bridge-1#") for s in res["node/D"]["blocking_segment_ids"]))
        self.assertEqual(res["node/B"]["nearest_destination"], "Mock District Hospital")

    def test_nothing_removed_means_everything_connected(self):
        self.assertEqual({v["status"] for v in self.analyse([]).values()}, {"connected"})

    def test_a_flooded_bridge_removes_all_its_segments(self):
        removed = oracle.expand_removed([], ["way/bridge-1"], self.roads)
        self.assertTrue(all(r.startswith("way/bridge-1#") for r in removed) and len(removed) >= 5)
        status = {v["name"]: v["status"] for v in self.analyse(removed).values()}
        self.assertEqual(status["Village D"], "cut_off")
        self.assertEqual(status["Village B"], "connected")

    def test_already_unreachable_villages_are_not_blamed_on_the_flood(self):
        # a village far from every road: unconnected BEFORE the flood
        far = {"type": "FeatureCollection", "features": [feature(
            {"type": "Point", "coordinates": [85.9, 27.5]}, id="node/X", name="Remote Village")]}
        res = oracle.analyse(self.roads, far, self.dest, [])
        self.assertEqual(res["node/X"]["status"], "unconnected_before")

    def test_snap_distance_matters(self):
        res = oracle.analyse(self.roads, self.settle, self.dest, [], snap_m=5)
        self.assertIn("unconnected_before", {v["status"] for v in res.values()})


class CaseStudyTests(unittest.TestCase):
    """Orchestrator with injected loader / mask writer / damage method (no rasterio needed)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rp = build_mock_run(self.root / "run", with_rasters=False)
        self.transform = (50.0, 0.0, 320_000.0, 0.0, -50.0, 3_115_000.0)
        ours = np.zeros((200, 300), np.uint8)
        ours[10:50, 20:60] = 1                                    # 40 x 40 px
        self.ours = ours
        ring = [utm_to_lonlat(320_000 + 1000 + dx, 3_115_000 - 2500 + dy, EPSG)
                for dx, dy in ((0, 0), (2000, 0), (2000, 2000), (0, 2000), (0, 0))]   # shifted by 0 px: identical
        self.ref_file = self.root / "ref" / "extent.geojson"
        self.ref_file.parent.mkdir()
        self.ref_file.write_text(json.dumps({"type": "Polygon", "coordinates": [[list(p) for p in ring]]}))

    def tearDown(self):
        self.tmp.cleanup()

    def run_cs(self, **kw):
        written = {}
        def damage(scratch):
            # the "same method" applied to the reference mask: here a fake that flags a fixed subset
            self.assertTrue(scratch.path(Artifact.OSM_ROADS).exists())         # OSM layers were copied
            self.assertNotEqual(scratch.root, self.rp.root)                    # and it is a scratch folder
            write_geojson(scratch.path(Artifact.AFFECTED_ROADS), [
                feature({"type": "Point", "coordinates": [85, 28]}, object_id="way/main#101"),
                feature({"type": "Point", "coordinates": [85, 28]}, object_id="way/main#999")])
            write_geojson(scratch.path(Artifact.AFFECTED_BRIDGES), [])
            write_geojson(scratch.path(Artifact.AFFECTED_BUILDINGS), [])
        return case_study.run_case_study(
            self.rp, self.ref_file, out_json=self.root / "results.json", out_png=self.root / "agree.png",
            loader=lambda run, art: (self.ours, self.transform, EPSG),
            write_mask=lambda path, arr, tf, epsg: written.update(mask=arr.copy()) or path.write_bytes(b"x"),
            damage_fn=damage, **kw), written

    def test_identical_extent_gives_perfect_scores_and_the_results_file(self):
        results, written = self.run_cs(notes=("Human note.",))
        fe = results["flood_extent"]
        self.assertEqual((fe["iou"], fe["precision"], fe["recall"], fe["f1"]), (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(results["areas_km2"]["ours"], 4.0)             # 1600 px * 0.0025 km2
        self.assertTrue(results["check_only"])
        self.assertIn("Human note.", results["notes"])
        self.assertTrue(any("different pixel" in n or "reference area of interest" in n for n in results["notes"]))
        self.assertEqual(json.loads((self.root / "results.json").read_text())["reference"], "EMSR927")
        self.assertEqual((self.root / "agree.png").read_bytes()[:4], b"\x89PNG")
        self.assertEqual(int(written["mask"].sum()), 1600)              # the reference mask went to the scratch run

    def test_infrastructure_comparison_uses_the_same_method_on_the_reference_mask(self):
        results, _ = self.run_cs()
        roads = results["infrastructure"]["roads"]
        ours = len(read_json(self.rp.path(Artifact.AFFECTED_ROADS))["features"])
        self.assertEqual(roads["reference_flagged"], 2)
        self.assertEqual(roads["also_flagged"], 1)                      # only way/main#101 is in both
        self.assertEqual(roads["extra_flagged"], ours - 1)
        self.assertIn("same method", results["method"])

    def test_scratch_folder_is_removed_and_the_run_folder_untouched(self):
        before = sorted(p.name for p in self.rp.root.rglob("*"))
        self.run_cs()
        self.assertEqual(sorted(p.name for p in self.rp.root.rglob("*")), before)

    def test_partial_overlap_scores(self):
        self.ours[:] = 0
        self.ours[30:70, 40:80] = 1                                     # shifted by 20 px each way: overlap 20x20
        results, _ = self.run_cs(infrastructure=False)
        self.assertAlmostEqual(results["flood_extent"]["iou"], 400 / 2800, places=3)
        self.assertEqual(results["infrastructure"], {})

    def test_reference_inside_the_run_folder_is_refused(self):
        inside = self.rp.root / "leak.geojson"
        inside.write_text(self.ref_file.read_text())
        with self.assertRaises(InputValidationError):
            case_study.assert_check_only(inside, self.rp)

    def test_results_match_what_the_dashboard_tab_expects(self):
        from dashboard.case_study import extent_rows, infrastructure_rows, load_case_study
        self.run_cs()
        cs = load_case_study(self.root / "results.json")
        rows = {r["Metric"]: r["Value"] for r in extent_rows(cs)}
        self.assertEqual(rows["IoU (overlap of flood extent)"], "1.00")
        self.assertEqual(infrastructure_rows(cs)[0]["Flagged in reference"], "2")


class DataRuleStructureTests(unittest.TestCase):
    def test_no_pipeline_code_imports_evaluation(self):
        offenders = []
        for folder in GUARDED_FOLDERS:
            for py in (REPO_ROOT / folder).rglob("*.py"):
                if re.search(r"^\s*(from|import)\s+evaluation\b", py.read_text(encoding="utf-8"), re.MULTILINE):
                    offenders.append(py.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(offenders, [])

    def test_reference_folder_is_git_ignored(self):
        self.assertIn("data/reference/", (REPO_ROOT / ".gitignore").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
