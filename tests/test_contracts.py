import json
import os
import tempfile
import unittest
from pathlib import Path

from pipeline import fileio
from pipeline.contracts import (ALLOWED_IMPACT_STATUS, CONTRACTS, ERROR, STAGE_ORDER, WARNING, Artifact,
                                check_artifact, check_run, check_stage, contracts_markdown,
                                raise_for_issues, stage_is_fresh)
from pipeline.errors import ContractError
from pipeline.run import RunPaths
from scripts.make_mock_outputs import build_mock_run

VECTOR_STAGES = ("init", "osm", "damage", "connectivity", "summarize")


def errors(issues):
    return [i for i in issues if i.severity == ERROR]


class ContractTableTests(unittest.TestCase):
    def test_every_artifact_has_a_spec_and_every_stage_output_is_known(self):
        from pipeline.contracts import SPECS
        self.assertEqual(set(SPECS), set(Artifact))
        for c in CONTRACTS.values():
            for a in c.needs + c.produces + c.optional:
                self.assertIn(a, SPECS)

    def test_stage_needs_are_produced_by_earlier_stages(self):
        produced = set()
        for name in STAGE_ORDER:
            c = CONTRACTS[name]
            if name != "osm":     # osm only needs run.json, order irrelevant, still fine
                pass
            for need in c.needs:
                self.assertTrue(need in produced or need is Artifact.RUN_MANIFEST,
                                f"{name} needs {need.value} that no earlier stage produces")
            produced |= set(c.produces)

    def test_entrypoints_use_module_colon_function(self):
        for c in CONTRACTS.values():
            if c.entrypoint:
                module, func = c.entrypoint.split(":")
                self.assertTrue(module and func.startswith("run_"))

    def test_markdown_mentions_every_file(self):
        md = contracts_markdown()
        for a in Artifact:
            self.assertIn(a.value, md)


# NOTE: never name a TestCase attribute "run": unittest already has a run() method.
class MockRunValidationTests(unittest.TestCase):
    """The mock run must satisfy its own contract (vector/JSON part: no rasterio needed)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.rp = build_mock_run(Path(cls.tmp.name) / "run", with_rasters=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_vector_stages_are_clean(self):
        for stage in VECTOR_STAGES:
            self.assertEqual(check_stage(self.rp, stage), [], stage)

    def test_missing_rasters_are_reported_not_crashed(self):
        problems = check_stage(self.rp, "acquire")
        self.assertTrue(any("before.tif" in i.artifact and "missing" in i.message for i in problems))

    def test_check_run_skips_stages_without_output(self):
        result = check_run(self.rp)
        self.assertNotIn("figures", result)
        self.assertNotIn("report", result)

    def test_forbid_mock(self):
        issues = check_stage(self.rp, "damage", forbid_mock=True)
        self.assertTrue(errors(issues))
        self.assertTrue(all("MOCK" in i.message for i in errors(issues)))

    def test_mock_numbers_agree_between_files(self):
        dmg = fileio.read_json(self.rp.path(Artifact.DAMAGE_SUMMARY))
        roads = fileio.read_json(self.rp.path(Artifact.AFFECTED_ROADS))
        summary = fileio.read_json(self.rp.path(Artifact.SUMMARY))
        self.assertEqual(dmg["affected_roads"], len(roads["features"]))
        self.assertEqual(summary["affected_roads"], dmg["affected_roads"])
        self.assertEqual(summary["cutoff_names"], ["Village B", "Village C", "Village D"])

    def test_nepali_names_survive_a_round_trip(self):
        data = fileio.read_json(self.rp.path(Artifact.OSM_SETTLEMENTS))
        self.assertIn("गाउँ A", [f["properties"]["name_ne"] for f in data["features"]])


class BrokenOutputTests(unittest.TestCase):
    """Each test copies the mock and breaks exactly one thing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rp = build_mock_run(Path(self.tmp.name) / "run", with_rasters=False)

    def tearDown(self):
        self.tmp.cleanup()

    def edit(self, artifact, fn):
        path = self.rp.path(artifact)
        data = fileio.read_json(path)
        fn(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_destroyed_is_not_allowed(self):
        def f(d): d["features"][0]["properties"]["impact_status"] = "destroyed"
        self.edit(Artifact.AFFECTED_ROADS, f)
        msgs = [i.message for i in errors(check_artifact(self.rp, Artifact.AFFECTED_ROADS))]
        self.assertTrue(any("destroyed" in m or "not allowed" in m for m in msgs))

    def test_projected_coordinates_in_geojson_are_caught(self):
        def f(d): d["features"][0]["geometry"]["coordinates"] = [[323166.0, 3109588.0], [323266.0, 3109588.0]]
        self.edit(Artifact.AFFECTED_ROADS, f)
        msgs = [i.message for i in errors(check_artifact(self.rp, Artifact.AFFECTED_ROADS))]
        self.assertTrue(any("lon/lat" in m for m in msgs))

    def test_missing_property_and_out_of_range_value(self):
        def f(d):
            del d["features"][0]["properties"]["confidence"]
            d["features"][1]["properties"]["impact_fraction"] = 1.7
        self.edit(Artifact.AFFECTED_ROADS, f)
        text = " | ".join(i.message for i in errors(check_artifact(self.rp, Artifact.AFFECTED_ROADS)))
        self.assertIn("missing property 'confidence'", text)
        self.assertIn("impact_fraction must be a number in [0, 1]", text)

    def test_all_problems_are_reported_at_once_and_grouped(self):
        def f(d):
            for feat in d["features"]:
                feat["properties"]["impact_status"] = "destroyed"
        self.edit(Artifact.AFFECTED_ROADS, f)
        issues = errors(check_artifact(self.rp, Artifact.AFFECTED_ROADS))
        self.assertEqual(len(issues), 1)                        # one grouped line, not N
        self.assertRegex(issues[0].message, r"\(\d+ features, e\.g\. ")

    def test_wrong_geometry_type(self):
        def f(d): d["features"][0]["geometry"] = {"type": "Point", "coordinates": [85.2, 28.1]}
        self.edit(Artifact.AFFECTED_ROADS, f)
        self.assertTrue(any("geometry type" in i.message
                            for i in errors(check_artifact(self.rp, Artifact.AFFECTED_ROADS))))

    def test_osm_snapshot_after_event_is_a_rule_violation(self):
        def f(d): d["snapshot_date"] = "2026-09-01"
        self.edit(Artifact.OSM_SNAPSHOT, f)
        issues = errors(check_artifact(self.rp, Artifact.OSM_SNAPSHOT))
        self.assertTrue(issues and "not before event" in issues[0].message)

    def test_metadata_orbit_mismatch_and_bracketing(self):
        def f(d):
            d["relative_orbit_after"] = 12
            d["after_date"] = "2026-08-24"          # before the 26 Aug event
        self.edit(Artifact.METADATA, f)
        text = " | ".join(i.message for i in errors(check_artifact(self.rp, Artifact.METADATA)))
        self.assertIn("relative orbit differs", text)
        self.assertIn("bracket the event", text)

    def test_odd_gap_is_only_a_warning(self):
        def f(d): d["after_date"] = "2026-09-03"     # 14 days, not a multiple of 6
        self.edit(Artifact.METADATA, f)
        issues = check_artifact(self.rp, Artifact.METADATA)
        self.assertFalse(errors(issues))
        self.assertTrue(any(i.severity == WARNING for i in issues))

    def test_missing_provenance_is_a_warning_and_strict_promotes_it(self):
        path = self.rp.path(Artifact.DAMAGE_SUMMARY)
        data = fileio.read_json(path)
        del data["provenance"]
        path.write_text(json.dumps(data), encoding="utf-8")
        issues = check_stage(self.rp, "damage")
        self.assertTrue(any(i.severity == WARNING and "provenance" in i.message for i in issues))
        raise_for_issues(issues, "damage")                              # not blocking
        with self.assertRaises(ContractError):
            raise_for_issues(issues, "damage", strict=True)             # blocking in strict mode

    def test_invalid_json_and_empty_file(self):
        self.rp.path(Artifact.DAMAGE_SUMMARY).write_text("{not json", encoding="utf-8")
        self.assertTrue(errors(check_artifact(self.rp, Artifact.DAMAGE_SUMMARY)))
        self.rp.path(Artifact.DAMAGE_SUMMARY).write_text("", encoding="utf-8")
        self.assertIn("empty", check_artifact(self.rp, Artifact.DAMAGE_SUMMARY)[0].message)

    def test_raise_for_issues_carries_friendly_and_detailed_text(self):
        self.rp.path(Artifact.DAMAGE_SUMMARY).unlink()
        issues = check_stage(self.rp, "damage")
        with self.assertRaises(ContractError) as ctx:
            raise_for_issues(issues, "damage")
        self.assertIn("damage", ctx.exception.user_message)
        self.assertIn("damage_summary.json", ctx.exception.detail)


class FreshnessTests(unittest.TestCase):
    def test_make_style_staleness(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = build_mock_run(Path(tmp) / "run", with_rasters=False)
            # damage needs rasters (flood_mask); fake the missing inputs by touching files.
            for art in (Artifact.FLOOD_MASK,):
                run.path(art).write_bytes(b"x")
            old = 1_700_000_000
            for art in CONTRACTS["damage"].produces:
                os.utime(run.path(art), (old, old))
            for art in CONTRACTS["damage"].needs:
                os.utime(run.path(art), (old - 100, old - 100))
            self.assertTrue(stage_is_fresh(run, "damage"))
            os.utime(run.path(Artifact.FLOOD_MASK), (old + 50, old + 50))   # input re-generated
            self.assertFalse(stage_is_fresh(run, "damage"))
            run.path(Artifact.DAMAGE_SUMMARY).unlink()
            self.assertFalse(stage_is_fresh(run, "damage"))


@unittest.skipUnless(fileio.rasterio_available(), "rasterio not installed")
class RasterContractTests(unittest.TestCase):
    """Runs on machines with rasterio (the team's laptops, CI)."""

    def setUp(self):
        import numpy as np
        self.np = np
        self.tmp = tempfile.TemporaryDirectory()
        self.rp = build_mock_run(Path(self.tmp.name) / "run", with_rasters=True)

    def tearDown(self):
        self.tmp.cleanup()

    def rewrite(self, artifact, array, *, epsg=None, shift=0.0):
        from rasterio.transform import from_origin
        import rasterio
        with rasterio.open(self.rp.path(artifact)) as src:
            t, e = src.transform, src.crs.to_epsg()
        tf = from_origin(t.c + shift, t.f, t.a, -t.e)
        fileio.write_geotiff(self.rp.path(artifact), array, transform=tf, epsg=epsg or e)

    def test_full_mock_passes_all_stages(self):
        for stage in ("acquire", "infer", *VECTOR_STAGES):
            self.assertEqual(errors(check_stage(self.rp, stage)), [], stage)

    def test_mask_must_be_binary(self):
        import rasterio
        with rasterio.open(self.rp.path(Artifact.FLOOD_MASK)) as src:
            arr = src.read(1)
        arr[0, 0] = 7
        self.rewrite(Artifact.FLOOD_MASK, arr)
        self.assertTrue(any("0 or 1" in i.message for i in check_artifact(self.rp, Artifact.FLOOD_MASK)))

    def test_misaligned_mask_is_caught(self):
        import rasterio
        with rasterio.open(self.rp.path(Artifact.FLOOD_MASK)) as src:
            arr = src.read(1)
        self.rewrite(Artifact.FLOOD_MASK, arr, shift=25.0)       # half a pixel
        self.assertTrue(any("same grid" in i.message for i in check_artifact(self.rp, Artifact.FLOOD_MASK)))

    def test_geographic_crs_is_rejected(self):
        import rasterio
        with rasterio.open(self.rp.path(Artifact.FLOOD_MASK)) as src:
            arr = src.read(1)
        self.rewrite(Artifact.FLOOD_MASK, arr, epsg=4326)
        self.assertTrue(any("geographic" in i.message for i in check_artifact(self.rp, Artifact.FLOOD_MASK)))

    def test_mask_area_matches_summary(self):
        area = fileio.mask_area_km2(self.rp.path(Artifact.FLOOD_MASK))
        summary = fileio.read_json(self.rp.path(Artifact.SUMMARY))
        self.assertAlmostEqual(area, summary["flood_area_km2"], places=2)


if __name__ == "__main__":
    unittest.main()
