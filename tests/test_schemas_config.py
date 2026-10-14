import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from pipeline.config import (DEFAULT_CONFIG, DEFAULT_YAML, config_fingerprint, deep_merge, load_config)
from pipeline.errors import InputValidationError
from pipeline.run import RunPaths
from pipeline.schemas import AOI, Summary, parse_date, parse_event_date, run_id

TRISHULI = AOI(85.10, 27.85, 85.45, 28.30, "Trishuli")


class DateTests(unittest.TestCase):
    def test_datetime_is_converted_to_date(self):
        # datetime is a subclass of date: the order of isinstance checks matters.
        self.assertEqual(parse_date(datetime(2026, 8, 26, 5, 30)), date(2026, 8, 26))
        self.assertNotIsInstance(parse_date(datetime(2026, 8, 26, 5, 30)), datetime)

    def test_timestamp_string_is_accepted(self):
        self.assertEqual(parse_date("2026-08-26T05:30:00Z"), date(2026, 8, 26))

    def test_garbage_is_rejected(self):
        for bad in ("26/08/2026", "", None, 20260826.5):
            with self.assertRaises(InputValidationError):
                parse_date(bad)

    def test_event_date_range(self):
        today = date(2026, 10, 4)
        self.assertEqual(parse_event_date("2026-08-26", today=today), date(2026, 8, 26))
        with self.assertRaises(InputValidationError):
            parse_event_date("2026-10-05", today=today)       # future
        with self.assertRaises(InputValidationError):
            parse_event_date("2010-01-01", today=today)       # before Sentinel-1


class AOITests(unittest.TestCase):
    def test_trishuli_utm_zone(self):
        self.assertEqual(TRISHULI.utm_epsg(), 32645)

    def test_utm_edge_cases(self):
        self.assertEqual(AOI(179.0, 10.0, 180.0, 11.0).utm_epsg(), 32660)   # not 32661
        self.assertEqual(AOI(10.0, -20.0, 11.0, -19.0).utm_epsg(), 32732)   # southern hemisphere

    def test_area_is_plausible(self):
        self.assertTrue(1600 < TRISHULI.area_km2() < 1800)

    def test_validation(self):
        TRISHULI.validate()
        bad = [AOI(85.5, 27.0, 85.1, 28.0), AOI(85, 28, 86, 27), AOI(85, 27, 86, 95),
               AOI(float("nan"), 27, 86, 28)]
        for aoi in bad:
            with self.assertRaises(InputValidationError):
                aoi.validate()
        with self.assertRaises(InputValidationError):
            AOI(80, 20, 90, 30).validate(max_km2=2500)

    def test_numbers_are_coerced_but_text_is_rejected(self):
        self.assertEqual(AOI("85.1", 27, 86, 28).west, 85.1)
        with self.assertRaises(InputValidationError):
            AOI("abc", 27, 86, 28)

    def test_from_dict_accepts_bbox_form(self):
        self.assertEqual(AOI.from_dict({"bbox": [1, 2, 3, 4], "name": "x"}).north, 4.0)
        with self.assertRaises(InputValidationError):
            AOI.from_dict({"west": 1})

    def test_slug(self):
        self.assertEqual(AOI(0, 0, 1, 1, "Trishuli corridor (placeholder)").slug(),
                         "trishuli-corridor-placeholder")
        self.assertEqual(AOI(0, 0, 1, 1, "***").slug(), "aoi")


class SummaryTests(unittest.TestCase):
    def test_none_is_not_zero(self):
        s = Summary(affected_roads=0)
        self.assertIn("flood_area_km2", s.unavailable())
        self.assertNotIn("affected_roads", s.unavailable())

    def test_unknown_keys_are_ignored(self):
        s = Summary.from_dict({"affected_roads": 3, "field_from_the_future": 1})
        self.assertEqual(s.affected_roads, 3)

    def test_round_trip(self):
        s = Summary(flood_area_km2=1.5, cutoff_names=["A"], is_mock=True)
        self.assertEqual(Summary.from_dict(s.to_dict()), s)


class RunIdAndConfigTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.fp = config_fingerprint(self.cfg)

    def test_float_noise_does_not_change_id(self):
        a = run_id(TRISHULI, date(2026, 8, 26), self.fp, "m")
        b = run_id(AOI(85.10000000001, 27.85, 85.45, 28.30, "other name"), date(2026, 8, 26), self.fp, "m")
        self.assertEqual(a, b)

    def test_inputs_change_id(self):
        base = run_id(TRISHULI, date(2026, 8, 26), self.fp, "m")
        self.assertNotEqual(base, run_id(TRISHULI, date(2026, 8, 27), self.fp, "m"))
        self.assertNotEqual(base, run_id(TRISHULI, date(2026, 8, 26), self.fp, "m2"))

    def test_fingerprint_ignores_output_section_only(self):
        changed_output = deep_merge(self.cfg, {"output": {"root": "elsewhere"}})
        changed_damage = deep_merge(self.cfg, {"damage": {"min_impact_fraction": 0.5}})
        self.assertEqual(self.fp, config_fingerprint(changed_output))
        self.assertNotEqual(self.fp, config_fingerprint(changed_damage))

    def test_default_yaml_is_the_default_config(self):
        import yaml
        self.assertEqual(yaml.safe_load(DEFAULT_YAML), DEFAULT_CONFIG)

    def test_partial_yaml_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.yaml"
            p.write_text("limits:\n  max_aoi_km2: 10\n", encoding="utf-8")
            cfg = load_config(p)
        self.assertEqual(cfg["limits"]["max_aoi_km2"], 10)
        self.assertEqual(cfg["limits"]["min_aoi_km2"], DEFAULT_CONFIG["limits"]["min_aoi_km2"])


class RunPathsTests(unittest.TestCase):
    def test_create_is_idempotent_and_ignores_renames(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = RunPaths.create(TRISHULI, "2026-08-26", out_root=tmp)
            created = a.manifest()["created_utc"]
            b = RunPaths.create(AOI(85.10000000001, 27.85, 85.45, 28.30, "renamed"), "2026-08-26", out_root=tmp)
            self.assertEqual(a.root, b.root)
            self.assertEqual(b.manifest()["created_utc"], created)
            self.assertEqual(b.event_date, date(2026, 8, 26))
            self.assertEqual(b.aoi.utm_epsg(), 32645)

    def test_oversized_aoi_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(InputValidationError):
                RunPaths.create(AOI(80, 20, 90, 30), "2026-08-26", out_root=tmp)

    def test_open_requires_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(InputValidationError):
                RunPaths.open(tmp)

    def test_unknown_artifact_name_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = RunPaths(Path(tmp))
            with self.assertRaises(ValueError):
                run.path("flod_mask.tif")        # typo


if __name__ == "__main__":
    unittest.main()
