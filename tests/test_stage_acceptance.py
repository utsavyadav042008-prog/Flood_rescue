"""Acceptance tests for the stages T1-T3 will write.

They run the REAL stage entry points on the mock scenario and check the results
against the independent oracle (``evaluation/oracle.py``).  While a stage is
still a stub they SKIP (they never fail on missing work), so they are green
today and become meaningful as soon as a teammate delivers.  They need rasterio
because the stages read GeoTIFFs.
"""

import os
import tempfile
import unittest
from pathlib import Path

from evaluation import oracle
from pipeline.config import load_config
from pipeline.contracts import CONTRACTS, Artifact, check_stage
from pipeline.fileio import rasterio_available, read_json
from pipeline.stages import LazyEntrypoint
from scripts.make_mock_outputs import build_mock_run


@unittest.skipUnless(rasterio_available(), "rasterio not installed")
class StageAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.template = build_mock_run(Path(cls.tmp.name) / "template", with_rasters=True)
        cls.cfg = load_config()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fresh_run(self, remove: tuple = ()):
        """A copy of the mock run in which the outputs of the stage under test are deleted."""
        import shutil
        dest = Path(self.tmp.name) / f"run_{len(list(Path(self.tmp.name).iterdir()))}"
        shutil.copytree(self.template.root, dest)
        from pipeline.run import RunPaths
        run = RunPaths(dest)
        for artifact in remove:
            run.path(artifact).unlink(missing_ok=True)
        return run

    def call(self, stage, run):
        c = CONTRACTS[stage]
        try:
            LazyEntrypoint(stage, c.entrypoint, c.owner)(run, run.aoi, run.event_date, self.cfg)
        except NotImplementedError as exc:
            self.skipTest(str(exc))

    def test_damage_stage_flags_the_flooded_bridge_and_road_break(self):
        run = self.fresh_run(CONTRACTS["damage"].produces)
        self.call("damage", run)
        self.assertEqual([i for i in check_stage(run, "damage") if i.severity == "error"], [])
        bridges = {f["properties"]["object_id"] for f in read_json(run.path(Artifact.AFFECTED_BRIDGES))["features"]}
        roads = {f["properties"]["object_id"] for f in read_json(run.path(Artifact.AFFECTED_ROADS))["features"]}
        self.assertIn("way/bridge-1", bridges)
        self.assertTrue(any(r.startswith("way/main#") for r in roads), "the flooded stretch of the main road")
        self.assertFalse(any(r.startswith("way/branch-") for r in roads), "approach roads are outside the flood")

    def test_connectivity_stage_agrees_with_the_oracle(self):
        run = self.fresh_run(CONTRACTS["connectivity"].produces)
        self.call("connectivity", run)
        self.assertEqual([i for i in check_stage(run, "connectivity") if i.severity == "error"], [])
        ours = {f["properties"]["name"]: f["properties"] for f in
                read_json(run.path(Artifact.CUTOFF_SETTLEMENTS))["features"]}
        self.assertEqual(set(ours), {"Village B", "Village C", "Village D"})
        for props in ours.values():
            self.assertTrue(props["blocking_segment_ids"], "every cut-off settlement names its blocking segments")
            self.assertTrue(props["reason"].strip())
        summary = read_json(run.path(Artifact.CONNECTIVITY_SUMMARY))
        self.assertEqual(summary["cutoff_settlements"], 3)
        self.assertEqual(sorted(summary["cutoff_names"]), ["Village B", "Village C", "Village D"])

    def test_infer_stage_output_is_contract_valid_and_overlaps_the_corridor(self):
        run = self.fresh_run(CONTRACTS["infer"].produces)
        self.call("infer", run)
        self.assertEqual([i for i in check_stage(run, "infer") if i.severity == "error"], [])

    @unittest.skipUnless(os.environ.get("FLOODRESQ_NETWORK_TESTS") == "1",
                         "downloads live OSM data; set FLOODRESQ_NETWORK_TESTS=1 to run")
    def test_osm_stage_output_is_contract_valid(self):
        run = self.fresh_run(CONTRACTS["osm"].produces)
        self.call("osm", run)
        self.assertEqual([i for i in check_stage(run, "osm") if i.severity == "error"], [])


if __name__ == "__main__":
    unittest.main()
