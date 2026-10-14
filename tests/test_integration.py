"""Tests for the integration glue (stages that connect the four teammates' code).

Everything here is offline.  Tests that need rasterio skip without it; the
vector/graph tests run with geopandas + networkx only.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.config import load_config, load_dotenv
from pipeline.contracts import CONTRACTS, Artifact, check_stage
from pipeline.fileio import rasterio_available, read_json


def _have(*modules) -> bool:
    try:
        for m in modules:
            __import__(m)
        return True
    except ImportError:
        return False


GEO = _have("geopandas", "shapely", "networkx", "scipy")


class EntrypointTests(unittest.TestCase):
    def test_every_stage_entrypoint_module_exists(self):
        import importlib.util
        for name, c in CONTRACTS.items():
            if c.entrypoint:
                module = c.entrypoint.split(":")[0]
                self.assertIsNotNone(importlib.util.find_spec(module), f"{name}: {module} missing")


@unittest.skipUnless(GEO, "geopandas/shapely/networkx/scipy not installed")
class OsmLayerTests(unittest.TestCase):
    @staticmethod
    def raw():
        from acquisition.osm import _raw_from_features

        def line(i, coords, **tags):
            return {"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"@osmId": f"way/{i}", **tags}}

        def pt(i, c, **tags):
            return {"type": "Feature", "geometry": {"type": "Point", "coordinates": c},
                    "properties": {"@osmId": f"node/{i}", **tags}}
        d = 0.0001
        house = {"type": "Feature", "properties": {"@osmId": "way/10", "building": "house"},
                 "geometry": {"type": "Polygon", "coordinates": [[[85.305, 28.101], [85.305 + d, 28.101],
                                                                   [85.305 + d, 28.101 + d], [85.305, 28.101 + d],
                                                                   [85.305, 28.101]]]}}
        roads = [line(1, [[85.30, 28.10], [85.31, 28.10], [85.32, 28.10]], highway="primary", name="Main"),
                 line(2, [[85.31, 28.10], [85.31, 28.11]], highway="track"),
                 line(3, [[85.315, 28.095], [85.315, 28.105]], highway="secondary", bridge="yes")]
        return {"roads": _raw_from_features(roads, "line"),
                "buildings": _raw_from_features([house], "polygon"),
                "settlements": _raw_from_features([pt(20, [85.311, 28.111], place="village", name="Gaun",
                                                      **{"name:ne": "गाउँ"})], "point"),
                "hospitals": _raw_from_features([pt(30, [85.301, 28.101], amenity="hospital", name="H")], "point"),
                "towns": _raw_from_features([], "point")}

    def test_roads_split_only_at_shared_nodes_with_stable_ids(self):
        from acquisition.osm import build_layers
        from pipeline.schemas import AOI
        layers = build_layers(self.raw(), AOI(85.29, 28.09, 85.33, 28.12, "t"))
        ids = sorted(layers["roads"]["id"])
        self.assertEqual(ids, ["way/1#0", "way/1#1", "way/2#0", "way/3#0"])   # bridge crosses, not split
        self.assertEqual(list(layers["bridges"]["id"]), ["way/3"])
        self.assertEqual(list(layers["hospitals"]["destination_type"]), ["hospital"])
        self.assertEqual(list(layers["settlements"]["name_ne"]), ["गाउँ"])

    def test_snapshot_is_always_before_the_event(self):
        from datetime import date
        from acquisition.osm import snapshot_for
        cfg = load_config()
        self.assertEqual(snapshot_for(date(2026, 8, 26), cfg).isoformat(), "2026-07-27")
        self.assertEqual(snapshot_for(date(2025, 3, 1), cfg).isoformat(), "2025-02-28")


@unittest.skipUnless(GEO, "geopandas/shapely/networkx/scipy not installed")
class ConnectivityStageTests(unittest.TestCase):
    def test_mock_scenario_matches_the_independent_oracle(self):
        from evaluation import oracle
        from gis.connectivity import run_connectivity
        from scripts.make_mock_outputs import build_mock_run
        with tempfile.TemporaryDirectory() as tmp:
            run = build_mock_run(Path(tmp) / "mock", with_rasters=False)
            for a in CONTRACTS["connectivity"].produces:
                run.path(a).unlink()
            run_connectivity(run, run.aoi, run.event_date, load_config())
            self.assertEqual([i for i in check_stage(run, "connectivity") if i.severity == "error"], [])
            ours = {f["properties"]["name"] for f in read_json(run.path(Artifact.CUTOFF_SETTLEMENTS))["features"]}
            roads = read_json(run.path(Artifact.OSM_ROADS))
            removed = oracle.expand_removed(
                [f["properties"]["object_id"] for f in read_json(run.path(Artifact.AFFECTED_ROADS))["features"]],
                [f["properties"]["object_id"] for f in read_json(run.path(Artifact.AFFECTED_BRIDGES))["features"]],
                roads)
            ref = oracle.analyse(roads, read_json(run.path(Artifact.OSM_SETTLEMENTS)),
                                 read_json(run.path(Artifact.OSM_HOSPITALS)), removed)
            self.assertEqual(ours, {v["name"] for v in ref.values() if v["status"] == oracle.CUT_OFF})
            summary = read_json(run.path(Artifact.CONNECTIVITY_SUMMARY))
            self.assertEqual(summary["cutoff_settlements"], 3)
            self.assertEqual(summary["destinations_used"], ["Mock District Hospital"])

    def test_bridge_blocking_can_be_switched_off(self):
        from gis.connectivity import run_connectivity
        from scripts.make_mock_outputs import build_mock_run
        from pipeline.fileio import write_geojson
        with tempfile.TemporaryDirectory() as tmp:
            run = build_mock_run(Path(tmp) / "mock", with_rasters=False)
            # keep only the main-road segments as affected roads (drop the bridge ones)
            fc = read_json(run.path(Artifact.AFFECTED_ROADS))
            write_geojson(run.path(Artifact.AFFECTED_ROADS),
                          [f for f in fc["features"] if f["properties"]["object_id"].startswith("way/main#")])
            cfg = load_config(overrides={"connectivity": {"block_flooded_bridges": False}})
            run_connectivity(run, run.aoi, run.event_date, cfg)
            names = set(read_json(run.path(Artifact.CONNECTIVITY_SUMMARY))["cutoff_names"])
            self.assertEqual(names, {"Village B", "Village C"})      # D still reachable over the bridge


class ChangeDetectionTests(unittest.TestCase):
    def test_new_water_and_debris_are_found_and_unchanged_ground_is_not(self):
        from models.change_detection import detect
        rng = np.random.default_rng(0)
        before = rng.normal(-10, 0.5, (2, 80, 80)).astype("float32")
        before[:, :, 38:42] = -22.0                                      # river (permanent water)
        after = before + rng.normal(0, 0.3, before.shape).astype("float32")
        after[0, 10:30, 20:36] = -21.0                                   # new ponded water
        after[0, 50:70, 42:60] = -2.0                                    # debris next to the river
        res = detect(before, after, permanent_water=(before[0] < -20), slope=np.zeros((80, 80)))
        self.assertGreater(res["water"][10:30, 20:36].mean(), 0.95)
        self.assertGreater(res["debris"][50:70, 42:60].mean(), 0.95)
        self.assertEqual(res["mask"][:5, :].sum(), 0)
        self.assertTrue(((res["confidence"] >= 0) & (res["confidence"] <= 1)).all())

    def test_brightening_far_from_any_water_is_not_debris(self):
        from models.change_detection import detect
        before = np.full((2, 60, 60), -10.0, np.float32)
        after = before.copy()
        after[0, 5:20, 5:20] = -2.0
        self.assertEqual(detect(before, after)["debris"].sum(), 0)


class FloodPathTests(unittest.TestCase):
    def test_path_follows_the_valley_and_breaches_a_pit(self):
        from gis.flood_path import trace
        rr, cc = np.mgrid[0:200, 0:120]
        dem = (2000 - rr * 5.0 + np.abs(cc - 60) * 20).astype("float32")
        dem[100, 55:66] += 40                                           # dam across the valley
        path, why = trace(dem, (5, 30), 30.0)
        self.assertEqual(path[-1], (199, 60))
        self.assertIn("edge", why)


class DotenvTests(unittest.TestCase):
    def test_copernicus_aliases_fill_cdse_names(self):
        keys = ("CDSE_USERNAME", "CDSE_PASSWORD", "COPERNICUS_USERNAME", "COPERNICUS_PASSWORD")
        saved = {k: os.environ.pop(k, None) for k in keys}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                env = Path(tmp) / ".env"
                env.write_text("# comment\nCOPERNICUS_USERNAME=alice\nCOPERNICUS_PASSWORD='s3cret'\n", encoding="utf-8")
                load_dotenv(env)
                self.assertEqual(os.environ["CDSE_USERNAME"], "alice")
                self.assertEqual(os.environ["CDSE_PASSWORD"], "s3cret")
        finally:
            for k in keys:
                os.environ.pop(k, None)
                if saved[k] is not None:
                    os.environ[k] = saved[k]


@unittest.skipUnless(rasterio_available() and GEO, "rasterio not installed")
class RasterStageTests(unittest.TestCase):
    """infer + damage + figures on the mock scenario (needs rasterio)."""

    def test_infer_damage_figures_on_the_mock_run(self):
        from gis.damage import run_damage
        from models.stage import run_infer
        from pipeline.figures import run_figures
        from scripts.make_mock_outputs import build_mock_run
        with tempfile.TemporaryDirectory() as tmp:
            run = build_mock_run(Path(tmp) / "mock", with_rasters=True)
            cfg = load_config(overrides={"model": {"use_unet": False}})
            for stage, fn in (("infer", run_infer), ("damage", run_damage), ("figures", run_figures)):
                for a in CONTRACTS[stage].produces:
                    run.path(a).unlink(missing_ok=True)
                fn(run, run.aoi, run.event_date, cfg)
                errors = [str(i) for i in check_stage(run, stage) if i.severity == "error"]
                self.assertEqual(errors, [], stage)
            meta = read_json(run.root / "extras" / "inference_meta.json")
            self.assertGreater(meta["flood_pixels"], 0)
            self.assertTrue(any("AI model not used" in w for w in meta["warnings"]))


if __name__ == "__main__":
    unittest.main()
