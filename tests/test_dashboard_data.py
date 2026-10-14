import json
import tempfile
import unittest
from pathlib import Path

from dashboard import data
from dashboard.data import LAYER_BY_KEY, LAYERS
from pipeline.contracts import Artifact
from pipeline.fileio import read_json, write_json
from pipeline.schemas import AOI, Summary
from scripts.make_mock_outputs import build_mock_run


class FormatTests(unittest.TestCase):
    def test_format_prop(self):
        self.assertEqual(data.format_prop("impact_fraction", 0.4166), "42%")
        self.assertEqual(data.format_prop("confidence", 0.8123), "0.81")
        self.assertEqual(data.format_prop("bridge", True), "yes")
        self.assertEqual(data.format_prop("name", None), "-")
        self.assertEqual(data.format_prop("name", ""), "-")
        self.assertEqual(data.format_prop("blocking_segment_ids", list("abcdefg")), "a, b, c, d, e (+2 more)")

    def test_html_is_escaped_so_untrusted_names_cannot_inject_markup(self):
        out = data.format_prop("name", '<img src=x onerror="alert(1)">')
        self.assertNotIn("<", out)
        self.assertIn("&lt;img", out)

    def test_nepali_text_is_preserved(self):
        self.assertEqual(data.format_prop("name_ne", "गाउँ A"), "गाउँ A")


class GeometryTests(unittest.TestCase):
    def test_line_is_thinned_but_keeps_ends(self):
        line = {"type": "LineString", "coordinates": [[85.0 + i * 1e-6, 28.0] for i in range(100)] + [[85.01, 28.0]]}
        out = data.thin_geometry(line)
        self.assertLess(len(out["coordinates"]), 10)
        self.assertEqual(out["coordinates"][0], [85.0, 28.0])
        self.assertEqual(out["coordinates"][-1], [85.01, 28.0])

    def test_polygon_stays_closed_and_valid(self):
        ring = [[85.0, 28.0], [85.0000001, 28.0], [85.00001, 28.00001], [85.0, 28.0000001], [85.0, 28.0]]
        out = data.thin_geometry({"type": "Polygon", "coordinates": [ring]})
        r = out["coordinates"][0]
        self.assertGreaterEqual(len(r), 4)
        self.assertEqual(r[0], r[-1])

    def test_point_and_multi_geometries_and_unknown(self):
        self.assertEqual(data.thin_geometry({"type": "Point", "coordinates": [85.123456789, 28.1]})["coordinates"],
                         [85.12346, 28.1])
        multi = {"type": "MultiLineString", "coordinates": [[[85, 28], [85.1, 28.1]], [[86, 28], [86.1, 28.1]]]}
        self.assertEqual(len(data.thin_geometry(multi)["coordinates"]), 2)
        self.assertEqual(data.thin_geometry({"type": "Weird"}), {"type": "Weird"})


class PrepareFeatureTests(unittest.TestCase):
    def feats(self, n):
        return [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [85 + i * 1e-4, 28]},
                 "properties": {"id": f"n{i}", "name": f"N{i}", "extra": "dropped"}} for i in range(n)]

    def test_properties_are_trimmed_to_popup_fields_and_complete(self):
        spec = LAYER_BY_KEY["settlements"]
        out, total, shown = data.prepare_features(spec, self.feats(3), 5000)
        self.assertEqual((total, shown), (3, 3))
        for f in out:
            self.assertEqual(set(f["properties"]), {"name", "name_ne", "id"})   # same keys on every feature
            self.assertNotIn("extra", f["properties"])
        self.assertEqual(out[0]["properties"]["name_ne"], "-")                    # missing -> placeholder

    def test_context_layers_are_capped_with_an_even_stride(self):
        spec = LAYER_BY_KEY["osm_buildings"]
        out, total, shown = data.prepare_features(spec, self.feats(10_000), 1000)
        self.assertEqual(total, 10_000)
        self.assertLessEqual(shown, 1000)
        ids = [int(f["properties"]["id"][1:]) for f in out]
        self.assertGreater(max(ids), 9000)               # covers the whole range, not just the first rows

    def test_important_layers_are_never_capped(self):
        spec = LAYER_BY_KEY["affected_roads"]
        self.assertFalse(spec.capped)
        _, total, shown = data.prepare_features(spec, self.feats(7000), 1000)
        self.assertEqual(shown, 7000)

    def test_bounds(self):
        feats = [{"geometry": {"type": "Point", "coordinates": [85.1, 28.0]}},
                 {"geometry": {"type": "LineString", "coordinates": [[85.3, 28.2], [85.2, 28.1]]}}]
        self.assertEqual(data.features_bounds(feats), (85.1, 28.0, 85.3, 28.2))
        self.assertIsNone(data.features_bounds([]))


class LayerRegistryTests(unittest.TestCase):
    def test_every_layer_has_unique_key_and_a_contract_artifact(self):
        keys = [s.key for s in LAYERS]
        self.assertEqual(len(keys), len(set(keys)))
        for s in LAYERS:
            self.assertIsInstance(s.artifact, Artifact)

    def test_cutoff_is_drawn_last_so_it_is_on_top(self):
        self.assertEqual(LAYERS[-1].key, "cutoff")

    def test_popup_fields_exist_in_the_contract_properties(self):
        from pipeline.contracts import SPECS
        for s in LAYERS:
            if s.kind.startswith("raster") or not s.popup:
                continue
            allowed = set(SPECS[s.artifact].required_props) | {"name_ne"}   # name_ne is optional extra
            for prop, _ in s.popup:
                self.assertIn(prop, allowed, f"{s.key}: popup field {prop!r} is not in the contract")

    def test_colours_are_distinct_for_the_main_overlays(self):
        main = [LAYER_BY_KEY[k].color for k in ("affected_roads", "affected_bridges", "affected_buildings", "cutoff")]
        self.assertEqual(len(main), len(set(main)))

    def test_legend_lists_enabled_layers_top_first(self):
        items = data.legend_items(["flood_mask", "cutoff"])
        self.assertEqual(items[0][1], LAYER_BY_KEY["cutoff"].legend)
        self.assertIn("Legend", data.legend_html(items))
        self.assertIn("&lt;", data.legend_html([("#fff", "<b>")]))


class MockRunBasedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.rp = build_mock_run(Path(cls.tmp.name) / "run", with_rasters=False)
        cls.cfg = {"dashboard": {"max_buildings_drawn": 5000}}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_prepare_layers_loads_vectors_and_notes_missing_rasters(self):
        keys = [s.key for s in LAYERS]
        prepared = data.prepare_layers(self.rp, self.cfg, keys)
        self.assertEqual(len(prepared["cutoff"].features), 3)
        self.assertEqual(len(prepared["settlements"].features), 4)
        self.assertEqual(len(prepared["affected_roads"].features), 12)
        self.assertIn("not available", prepared["flood_mask"].note)     # no raster in this run
        self.assertEqual(prepared["flood_mask"].features, [])

    def test_disabled_layers_are_not_loaded(self):
        prepared = data.prepare_layers(self.rp, self.cfg, ["cutoff"])
        self.assertEqual(list(prepared), ["cutoff"])

    def test_cutoff_popup_shows_reason_and_segment_summary(self):
        cut = data.prepare_layers(self.rp, self.cfg, ["cutoff"])["cutoff"]
        village_b = next(f for f in cut.features if f["properties"]["name"] == "Village B")
        self.assertIn("Main road is flooded", village_b["properties"]["reason"])
        self.assertIn("way/main#", village_b["properties"]["blocking_segment_ids"])

    def test_map_bounds_frame_the_data(self):
        prepared = data.prepare_layers(self.rp, self.cfg, list(data.FRAMING_LAYERS))
        (south, west), (north, east) = data.map_bounds(prepared, None)
        self.assertTrue(85.15 < west < east < 85.45 and 28.0 < south < north < 28.25)

    def test_bounds_fall_back_to_the_aoi_then_to_a_default(self):
        aoi = AOI(85.0, 27.0, 85.5, 27.5)
        (south, west), (north, east) = data.map_bounds({}, aoi)
        self.assertLess(south, 27.0)
        self.assertGreater(east, 85.5)
        self.assertEqual(len(data.map_bounds({}, None)), 2)

    def test_tables_and_csv_with_bom(self):
        tables = data.build_tables(self.rp)
        self.assertEqual(set(tables), {"roads", "bridges", "buildings", "cutoff"})
        self.assertEqual(tables["cutoff"].total, 3)
        overlaps = [r["Overlap"] for r in tables["roads"].rows]
        self.assertEqual(overlaps, sorted(overlaps, reverse=True))
        raw = data.table_to_csv(tables["cutoff"])
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))                 # BOM for Excel
        self.assertIn("Settlement,Why cut off", raw.decode("utf-8-sig"))

    def test_table_cap(self):
        tables = data.build_tables(self.rp, max_rows=2)
        self.assertEqual(len(tables["roads"].rows), 2)
        self.assertGreater(tables["roads"].total, 2)
        self.assertIn("first 2", tables["roads"].note)

    def test_missing_files_give_no_table_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = build_mock_run(Path(tmp) / "r", with_rasters=False)
            run.path(Artifact.AFFECTED_BRIDGES).unlink()
            self.assertNotIn("bridges", data.build_tables(run))

    def test_run_version_changes_when_summary_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = build_mock_run(Path(tmp) / "r", with_rasters=False)
            v1 = data.run_version(run)
            import os, time
            future = time.time() + 100
            os.utime(run.path(Artifact.SUMMARY), (future, future))
            self.assertGreater(data.run_version(run), v1)


class CardAndBannerTests(unittest.TestCase):
    def test_none_is_not_available_and_zero_is_zero(self):
        cards = {c.key: c for c in data.kpi_cards(Summary(affected_roads=0, cutoff_settlements=None))}
        self.assertEqual(cards["roads"].value, "0")
        self.assertTrue(cards["roads"].available)
        self.assertEqual(cards["cutoff"].value, "not available")
        self.assertFalse(cards["cutoff"].available)
        self.assertEqual(len(cards), 6)

    def test_cards_use_the_shared_formatter(self):
        from reports.format import fmt
        s = Summary(flood_area_km2=1234.567, affected_roads=12, affected_road_length_km=3.456,
                    cutoff_settlements=5, cutoff_names=["A", "B", "C", "D", "E"], mean_confidence=0.912)
        cards = {c.key: c for c in data.kpi_cards(s)}
        f = fmt(s)
        self.assertEqual(cards["flood"].value, f["flood_area_km2"])
        self.assertEqual(cards["flood"].value, "1,234.6")
        self.assertEqual(cards["roads"].sub, "3.5 km")
        self.assertEqual(cards["cutoff"].sub, "A, B, C (+2)")
        self.assertEqual(cards["confidence"].value, "0.91")

    def test_banners_mock_and_failed_stages(self):
        class O:  # minimal outcome
            def __init__(self, name, status, message=""): self.name, self.status, self.message = name, status, message
        class R: outcomes = [O("osm", "failed", "ohsome timeout"), O("report", "skipped"), O("damage", "skipped")]
        s = Summary(is_mock=True, warnings=["MOCK DATA: x", "Stage 'osm' failed: ohsome timeout"])
        texts = [b.text for b in data.banners(s, R())]
        self.assertTrue(texts[0].startswith("MOCK DATA"))
        self.assertEqual(sum("MOCK DATA" in t for t in texts), 1)                 # not repeated
        self.assertTrue(any("ohsome timeout" in t for t in texts))
        self.assertFalse(any("report" in t for t in texts))                       # unbuilt report is not nagged about

    def test_imagery_caption(self):
        s = Summary(before_date="2026-08-20", after_date="2026-09-01", sentinel1_orbit="99 ASCENDING",
                    osm_snapshot_date="2026-07-27", model_version="m1")
        cap = data.imagery_caption(s)
        for piece in ("2026-08-20", "2026-09-01", "99 ASCENDING", "2026-07-27", "m1"):
            self.assertIn(piece, cap)
        self.assertEqual(data.imagery_caption(Summary()), "")


class InputAndRunListTests(unittest.TestCase):
    def test_aoi_inputs_validate_live(self):
        aoi, msg = data.aoi_from_inputs(85.1, 27.85, 85.45, 28.3, "T")
        self.assertIsNotNone(aoi)
        self.assertIn("km", msg)
        for bad in ((85.5, 27, 85.1, 28), (80, 20, 90, 30), ("x", 1, 2, 3)):
            aoi, msg = data.aoi_from_inputs(*bad)
            self.assertIsNone(aoi)
            self.assertTrue(msg)

    def test_presets_contain_the_trishuli_placeholder_and_a_mock_valley(self):
        presets = data.load_presets()
        self.assertTrue(any("Trishuli" in k for k in presets))
        self.assertIn("Mock valley (demo)", presets)

    def test_list_runs_ignores_template_and_sorts_newest_first(self):
        import os, time
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = build_mock_run(root / "_mock" / "a_2026-08-26_aaaa1111", with_rasters=False)
            b = build_mock_run(root / "b_2026-08-26_bbbb2222", with_rasters=False)
            build_mock_run(root / "_mock" / "_mock_template", with_rasters=False)
            os.utime(a.path(Artifact.RUN_MANIFEST), (time.time() + 60,) * 2)
            runs = data.list_runs(root)
            self.assertEqual([r.path.name for r in runs], ["a_2026-08-26_aaaa1111", "b_2026-08-26_bbbb2222"])
            self.assertTrue(all(r.is_mock for r in runs))        # both manifests came from the mock builder
            self.assertEqual(data.list_runs(root / "missing"), [])

    def test_find_demo_run_prefers_trishuli_then_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(data.find_demo_run(root))
            build_mock_run(root / "data" / "sample" / "mock_run", with_rasters=False)
            self.assertEqual(data.find_demo_run(root).name, "mock_run")
            build_mock_run(root / "data" / "sample" / "trishuli_run", with_rasters=False)
            self.assertEqual(data.find_demo_run(root).name, "trishuli_run")

    def test_progress_markdown_and_limitations(self):
        text = data.stage_progress_markdown({"acquire": "running", "osm": "ok"})
        self.assertIn("Satellite images", text)
        self.assertIn("pending", text)                          # stages without a state yet
        self.assertGreaterEqual(len(data.LIMITATIONS), 6)


if __name__ == "__main__":
    unittest.main()
