"""Run the real dashboard code against fake Streamlit/Folium (see tests/fakes.py)."""

import importlib
import tempfile
import unittest
from datetime import date
from pathlib import Path

from tests import fakes

from pipeline.analyze import PipelineResult, StageOutcome, load_result
from pipeline.contracts import Artifact
from pipeline.schemas import AOI
from reports.format import ATTRIBUTION_LINES
from scripts.make_mock_outputs import build_mock_run


def kpi_values(st) -> dict:
    """{label: value} parsed from the KPI cards HTML (dashboard/style.kpi_html)."""
    import html as _html
    import re
    out = {}
    for text in st.texts("markdown"):
        for label, value in re.findall(r'class="fr-kpi-icon">[^<]*</span>([^<]*)</div><div class="fr-kpi-value">([^<]*)',
                                       text):
            out[_html.unescape(label)] = _html.unescape(value)
    return out


class UiTestCase(unittest.TestCase):
    def setUp(self):
        fakes.uninstall()
        self.st, self.folium, self.sf = fakes.install()
        self.tmp = tempfile.TemporaryDirectory()
        self.run = build_mock_run(Path(self.tmp.name) / "mock_run", with_rasters=False)
        # placeholder rasters: present on disk (so the run counts as complete) but unreadable
        # without rasterio, which exercises the "could not be drawn" paths
        for artifact in (Artifact.BEFORE, Artifact.AFTER, Artifact.FLOOD_MASK, Artifact.FLOOD_CONFIDENCE):
            self.run.path(artifact).write_bytes(b"not a real tiff")
        self.app = importlib.import_module("dashboard.app")
        self.components = importlib.import_module("dashboard.components")
        # point the demo/open-run discovery at our temp mock run and keep tests off the repo's outputs
        # (restored in tearDown, otherwise later tests in the same process see the fakes)
        self._saved = (self.app.data.find_demo_run, self.app.data.list_runs)
        self.app.data.find_demo_run = lambda *a, **k: self.run.root
        self.app.data.list_runs = lambda *a, **k: []

    def tearDown(self):
        self.app.data.find_demo_run, self.app.data.list_runs = self._saved
        fakes.uninstall()
        self.tmp.cleanup()

    def result(self):
        return load_result(self.run.root)


class EmptyStateTests(UiTestCase):
    def test_first_screen(self):
        self.app.main()
        self.assertEqual(self.st.of("set_page_config")[0][2]["layout"], "wide")
        self.assertTrue(any("Educational prototype" in t for t in self.st.texts("markdown")))
        self.assertTrue(any("Choose an area" in t for t in self.st.texts("markdown")))
        self.assertTrue(any("Try a real flood event" in t for t in self.st.texts("markdown")))
        self.assertEqual(self.st.of("tabs"), [])

    def test_footer_has_the_three_required_attributions_word_for_word(self):
        self.app.main()
        captions = self.st.texts("caption")
        for line in ATTRIBUTION_LINES:
            self.assertIn(line, captions)
        self.assertEqual(len(ATTRIBUTION_LINES), 3)
        self.assertIn("\u00a9 OpenStreetMap contributors.", captions)


class ResultViewTests(UiTestCase):
    def open_demo(self):
        self.st.buttons["Load cached demo"] = True
        self.app.main()

    def test_demo_renders_kpis_tabs_map_and_tables(self):
        self.open_demo()
        values = kpi_values(self.st)
        self.assertEqual(len(values), 6)
        self.assertEqual(values["Settlements cut off"], "3")
        self.assertEqual(values["Roads affected"], "12")
        self.assertEqual(values["Flooded / debris area"], "not available")   # no rasters in this run: not a fake 0
        self.assertEqual(self.st.of("tabs")[0][1][0], ["Map", "Before / After", "Tables", "Report", "Flood path (bonus)", "Case study", "About"])
        # the map: 3 cut-off + 4 settlements as circles, 1 hospital icon, fit to the data
        self.assertEqual(self.folium.count("CircleMarker"), 7)
        self.assertEqual(self.folium.count("Marker"), 1)
        self.assertGreaterEqual(self.folium.count("GeoJson"), 3)       # roads, bridges, buildings
        self.assertEqual(self.folium.count("fit_bounds"), 1)
        self.assertEqual(self.folium.count("ImageOverlay"), 0)     # rasters unreadable here
        # tables with CSV downloads
        labels = [c[1][0] for c in self.st.of("download_button")]
        self.assertEqual(len([l for l in labels if "(CSV)" in l]), 4)

    def test_mock_banner_and_missing_raster_notes_are_visible(self):
        self.open_demo()
        self.assertTrue(any("MOCK DATA" in t for t in self.st.texts("markdown")))
        self.assertTrue(any("could not be drawn" in t for t in self.st.texts("caption")))   # raster layer note

    def test_missing_report_and_images_show_friendly_messages(self):
        self.open_demo()
        info = self.st.texts("info")
        self.assertTrue(any("report has not been generated" in t for t in info))
        self.assertTrue(any("Preview unavailable" in t for t in info))

    def test_unchecked_layer_is_not_drawn(self):
        self.st.values["Cut-off settlements"] = False
        self.open_demo()
        self.assertEqual(self.folium.count("CircleMarker"), 4)

    def test_popups_contain_escaped_text_and_the_reason(self):
        self.open_demo()
        popups = [c[1][0] for c in self.folium.calls if c[0] == "Popup"]
        self.assertTrue(any("Main road is flooded" in p for p in popups))
        self.assertTrue(any("Village D" in p for p in popups))

    def test_legend_is_added_and_lists_cutoff(self):
        self.open_demo()
        elements = [c[1][0] for c in self.folium.calls if c[0] == "Element"]
        self.assertTrue(any("No road path to any destination" in e for e in elements))

    def test_map_call_survives_a_streamlit_folium_without_stretch_support(self):
        self.sf.reject_stretch = True
        self.open_demo()
        self.assertEqual(len(self.sf.calls), 1)
        self.assertEqual(self.sf.calls[0]["returned_objects"], [])       # never rerun on pan/zoom
        self.assertTrue(self.sf.calls[0].get("use_container_width"))

    def test_tables_survive_old_dataframe_signature(self):
        self.st.reject_stretch = True
        self.open_demo()
        self.assertGreaterEqual(len(self.st.of("dataframe")), 4)

    def test_missing_map_libraries_give_a_message_not_a_crash(self):
        import sys
        self.open_demo()                       # populate session state first
        self.st.calls.clear()
        sys.modules["streamlit_folium"] = None       # makes `import streamlit_folium` raise ImportError
        self.app.main()
        self.assertTrue(any("pip install folium streamlit-folium" in t for t in self.st.texts("error")))


class RunActionTests(UiTestCase):
    def test_pipeline_failure_shows_friendly_error_and_steps(self):
        def fake_analyze(aoi, event, **kw):
            kw["on_progress"]("acquire", "running")
            kw["on_progress"]("acquire", "failed")
            return PipelineResult(None, None, [StageOutcome("acquire", "failed", 1.2, "No same-orbit pair found.")],
                                  False, "No same-orbit pair found.", "acquire")
        self.app.analyze_flood = fake_analyze
        self.st.buttons["Run analysis"] = True
        self.app.main()
        self.assertIn("No same-orbit pair found.", self.st.texts("error"))
        self.assertTrue(any("Satellite images" in t for t in self.st.texts("markdown")))
        self.assertEqual(self.st.of("tabs"), [])
        self.assertEqual(self.st.of("status.update")[0][2]["state"], "error")

    def test_successful_run_passes_the_choices_through(self):
        seen = {}
        def fake_analyze(aoi, event, **kw):
            seen.update(aoi=aoi, event=event, mock=kw["mock"], cache=kw["use_cache"])
            return load_result(self.run.root)
        self.app.analyze_flood = fake_analyze
        self.st.buttons["Run analysis"] = True
        self.st.values["Use mock data for these steps"] = ["osm", "damage"]
        self.st.values["Reuse earlier results (cache)"] = False
        self.app.main()
        self.assertEqual(seen["mock"], {"osm", "damage"})
        self.assertFalse(seen["cache"])
        self.assertEqual(seen["event"], date(2026, 8, 26))
        self.assertIsInstance(seen["aoi"], AOI)
        self.assertEqual(len(kpi_values(self.st)), 6)

    def test_no_mock_stages_means_a_real_run(self):
        seen = {}
        def fake_analyze(aoi, event, **kw):
            seen["mock"] = kw["mock"]
            return load_result(self.run.root)
        self.app.analyze_flood = fake_analyze
        self.st.buttons["Run analysis"] = True
        self.app.main()
        self.assertIs(seen["mock"], False)

    def test_invalid_custom_area_disables_run_and_explains(self):
        self.st.values["Area"] = "Custom"
        self.st.values["West (lon)"] = 86.0          # west > east
        self.st.values["East (lon)"] = 85.0
        self.app.main()
        self.assertTrue(self.st.texts("error"))
        run_button = [c for c in self.st.of("button") if c[1][0] == "Run analysis"][0]
        self.assertTrue(run_button[2]["disabled"])

    def test_partial_result_degrades_visibly(self):
        result = self.result()
        result.outcomes = [StageOutcome("osm", "failed", 2.0, "ohsome timeout"),
                           StageOutcome("damage", "skipped", 0.0, "Needs output from 'osm'")]
        result.summary.warnings = ["Stage 'osm' failed: ohsome timeout"]
        self.st.session_state["result"] = result
        self.app.main()
        errors = [t for t in self.st.texts("markdown") if "fr-callout error" in t]
        self.assertTrue(any("ohsome timeout" in t for t in errors))
        self.assertTrue(any("ohsome timeout" in t for t in self.st.texts("markdown")))


class ComponentTests(UiTestCase):
    def test_call_compat_tries_new_old_then_plain(self):
        calls = []
        def only_old(*a, **kw):
            calls.append(kw)
            if "width" in kw:
                raise TypeError("no width")
        self.components.call_compat(only_old, 1)
        self.assertEqual(calls, [{"width": "stretch"}, {"use_container_width": True}])

        def plain_only(*a, **kw):
            if kw:
                raise TypeError("no kwargs")
            return "ok"
        self.assertEqual(self.components.call_compat(plain_only), "ok")

        def broken(*a, **kw):
            raise ValueError("real failure")
        with self.assertRaises(ValueError):
            self.components.call_compat(broken)

    def test_kpi_unavailable_is_labelled(self):
        from pipeline.schemas import Summary
        self.components.render_kpis(self.app.data.kpi_cards(Summary()))
        self.assertEqual(set(kpi_values(self.st).values()), {"not available"})

    def test_report_tab_shows_text_and_downloads(self):
        self.run.path(Artifact.REPORT_TXT).write_text("SITUATION REPORT\nflood: 4.2 km2\n", encoding="utf-8")
        self.components.render_report_tab(self.run)
        self.assertTrue(any("SITUATION REPORT" in t for t in self.st.texts("code")))
        self.assertEqual(len(self.st.of("download_button")), 1)          # no PDF yet
        self.run.path(Artifact.REPORT_PDF).write_bytes(b"%PDF-1.4 fake")
        self.components.render_report_tab(self.run)
        self.assertEqual(len(self.st.of("download_button")), 3)

    def test_report_tab_offers_nepali_and_html_when_they_exist(self):
        from reports.generator import write_reports
        from pipeline.fileio import read_json
        from pipeline.schemas import Summary
        summary = Summary.from_dict(read_json(self.run.path(Artifact.SUMMARY)))
        write_reports(self.run, summary, {"report": {"languages": ["en", "ne"]}})
        self.components.render_report_tab(self.run)
        labels = [c[1][0] for c in self.st.of("download_button")]
        self.assertIn("Download report (text)", labels)
        self.assertIn("Download report (PDF)", labels)
        self.assertIn("Download printable version (HTML)", labels)
        # switch to Nepali
        self.st.calls.clear()
        self.st.values["Language"] = "\u0928\u0947\u092a\u093e\u0932\u0940"
        self.components.render_report_tab(self.run)
        labels = [c[1][0] for c in self.st.of("download_button")]
        self.assertIn("Download report, Nepali (text)", labels)
        self.assertNotIn("Download report (PDF)", labels)            # no Nepali PDF: the font cannot shape it
        self.assertTrue(any("Print > Save as PDF" in t for t in self.st.texts("caption")))

    def test_about_lists_limitations_and_responsible_use(self):
        self.components.render_about(self.result())
        text = " ".join(self.st.texts("markdown"))
        self.assertIn("cannot do", text)
        self.assertIn("Sentinel satellites revisit", text)
        self.assertIn("no images of people", text)


class CaseStudyTests(unittest.TestCase):
    def test_parser_is_tolerant_and_derives_agreement(self):
        from dashboard.case_study import parse_case_study, extent_rows, infrastructure_rows
        cs = parse_case_study({"flood_extent": {"iou": 0.6, "f1": "bad"},
                               "infrastructure": {"roads": {"reference_flagged": 40, "also_flagged": 30,
                                                            "extra_flagged": 5}, "bridges": "oops"},
                               "notes": ["radar shadow", 3]})
        rows = {r["Metric"]: r["Value"] for r in extent_rows(cs)}
        self.assertEqual(rows["IoU (overlap of flood extent)"], "0.60")
        self.assertEqual(rows["F1 score"], "not available")
        infra = {r["Object"]: r for r in infrastructure_rows(cs)}
        self.assertEqual(infra["Roads"]["Reference objects found"], "75%")
        self.assertEqual(infra["Bridges"]["Flagged in reference"], "not available")
        self.assertEqual(cs.notes, ["radar shadow", "3"])

    def test_garbage_input_gives_an_empty_case_study(self):
        from dashboard.case_study import parse_case_study, load_case_study
        self.assertEqual(parse_case_study("nope").extent, {})
        self.assertIsNone(load_case_study(Path("/does/not/exist.json")))

    def test_render_without_results_file(self):
        fakes.uninstall()
        st, _, _ = fakes.install()
        try:
            cs = importlib.import_module("dashboard.case_study")
            cs.RESULTS_PATH = Path("/does/not/exist.json")
            cs.load_case_study.__defaults__ = (cs.RESULTS_PATH,)
            cs.render()
            self.assertTrue(any("not available yet" in t for t in st.texts("info")))
            self.assertTrue(any("Check-only" in t for t in st.texts("caption")))
            self.assertTrue(any("Copernicus Emergency Management Service" in t for t in st.texts("caption")))
        finally:
            fakes.uninstall()


def tearDownModule():
    fakes.uninstall()


if __name__ == "__main__":
    unittest.main()
