import io
import re
import tempfile
import unittest
from pathlib import Path

from pipeline.contracts import Artifact, check_stage
from pipeline.fileio import write_json
from pipeline.run import RunPaths
from pipeline.schemas import AOI, Provenance, Summary
from reports import generator as gen
from reports.format import (ATTRIBUTION_LINES, NA, count, cutoff_list, fmt, num, to_ascii_digits,
                            to_devanagari_digits)
from reports.guard import allowed_numbers, extract_numbers, static_numbers, unexpected_numbers
from reports.strings import STRINGS
from scripts.make_mock_outputs import build_mock_run


def full_summary(**over) -> Summary:
    base = dict(flood_area_km2=1234.567, affected_roads=12, affected_road_length_km=3.456, affected_bridges=1,
                affected_buildings=17, cutoff_settlements=3, cutoff_names=["Village B", "Village C", "Village D"],
                mean_confidence=0.912, aoi_name="mock-valley", event_date="2026-08-26",
                before_date="2026-08-20", after_date="2026-09-01", sentinel1_orbit="99 ASCENDING",
                model_version="unet-baseline-0", osm_snapshot_date="2026-07-27",
                processing_date="2026-10-07T10:00:00+00:00")
    base.update(over)
    return Summary(**base)


def worst_case() -> Summary:
    names = [f"Settlement with a rather long name number {i} " + "x" * 80 for i in range(40)]
    return full_summary(cutoff_settlements=40, cutoff_names=names, is_mock=True,
                        warnings=["MOCK DATA: x"] + [f"Note {i}: " + "word " * 80 for i in range(12)])


class FormatTests(unittest.TestCase):
    def test_none_is_not_available_and_zero_is_zero(self):
        self.assertEqual(num(None), NA)
        self.assertEqual(count(None), NA)
        self.assertEqual(count(0), "0")
        self.assertEqual(num(0.0), "0.0")

    def test_rounding_and_thousands(self):
        self.assertEqual(num(1234.567), "1,234.6")
        self.assertEqual(num(0.912, 2), "0.91")
        self.assertEqual(count(1234567), "1,234,567")

    def test_cutoff_list_variants(self):
        self.assertEqual(cutoff_list([], 0), "None found")
        self.assertEqual(cutoff_list([], None), NA)
        text = cutoff_list([f"n{i}" for i in range(13)], 13)
        self.assertEqual(text.count("\n- "), 10)          # 10 items, the 11th line is the "more" line
        self.assertTrue(text.endswith("... and 3 more"))

    def test_digit_conversion_round_trip(self):
        self.assertEqual(to_devanagari_digits("2026-08-26"), "\u0968\u0966\u0968\u096c-\u0966\u096e-\u0968\u096c")
        self.assertEqual(to_ascii_digits(to_devanagari_digits("12,345.6 km")), "12,345.6 km")

    def test_attribution_is_word_for_word(self):
        self.assertEqual(ATTRIBUTION_LINES[0], "Contains modified Copernicus Sentinel data 2026.")
        self.assertTrue(ATTRIBUTION_LINES[1].startswith("Produced using Copernicus WorldDEM-30 \u00a9 DLR e.V. "
                                                         "2010\u20132014 and \u00a9 Airbus Defence and Space GmbH "
                                                         "2014\u20132018 provided under COPERNICUS by the European "
                                                         "Union and ESA; all rights reserved."))
        self.assertEqual(ATTRIBUTION_LINES[2], "\u00a9 OpenStreetMap contributors.")


class StringsTests(unittest.TestCase):
    def test_every_language_defines_the_same_keys(self):
        self.assertEqual(set(STRINGS["en"]), set(STRINGS["ne"]))

    def test_placeholders_match_between_languages(self):
        for key in STRINGS["en"]:
            en = set(re.findall(r"\{(\w+)\}", STRINGS["en"][key]))
            ne = set(re.findall(r"\{(\w+)\}", STRINGS["ne"][key]))
            self.assertEqual(en, ne, key)

    def test_wording_never_claims_what_we_cannot_know(self):
        banned = ("destroyed", "confirmed", "casualt", "victim", "dead", "died", "killed", "fatalit",
                  "people affected", "population")
        for lang in ("en",):
            blob = " ".join(STRINGS[lang].values()).lower()
            for word in banned:
                self.assertNotIn(word, blob, word)


class TextReportTests(unittest.TestCase):
    def test_contains_every_displayed_number(self):
        s = full_summary()
        text = gen.generate_report(s)
        f = fmt(s)
        for key in ("flood_area_km2", "affected_roads", "affected_road_length_km", "affected_bridges",
                    "affected_buildings", "cutoff_settlements", "mean_confidence", "event_date",
                    "before_date", "after_date", "osm_snapshot_date", "model_version"):
            self.assertIn(f[key], text, key)
        for name in s.cutoff_names:
            self.assertIn(name, text)
        self.assertNotIn("{", text)                      # no unfilled placeholder
        self.assertNotIn("}", text)

    def test_none_prints_not_available_never_zero_and_units_are_not_attached(self):
        s = full_summary(flood_area_km2=None, affected_roads=None, affected_bridges=None,
                         cutoff_settlements=None, cutoff_names=[], mean_confidence=None)
        text = gen.generate_report(s)
        self.assertIn("Flooded / debris-covered area: not available\n", text)
        self.assertIn("Road segments affected: not available\n", text)
        self.assertIn("Not available: the cut-off analysis could not be completed.", text)
        self.assertIn("Model confidence: not available.", text)
        self.assertNotIn("not available km", text)

    def test_zero_cutoff_is_a_positive_statement_not_missing(self):
        text = gen.generate_report(full_summary(cutoff_settlements=0, cutoff_names=[]))
        self.assertIn("No settlement without a road connection was found", text)

    def test_roads_without_length_have_no_empty_brackets(self):
        text = gen.generate_report(full_summary(affected_road_length_km=None))
        self.assertIn("Road segments affected: 12\n", text)

    def test_cutoff_list_is_capped_and_counts_the_rest(self):
        names = [f"Hamlet {i}" for i in range(25)]
        text = gen.generate_report(full_summary(cutoff_settlements=25, cutoff_names=names))
        self.assertIn("- Hamlet 9", text)
        self.assertNotIn("- Hamlet 10", text)
        self.assertIn("... and 15 more", text)

    def test_names_longer_than_the_limit_are_shortened(self):
        text = gen.generate_report(full_summary(cutoff_names=["N" * 200], cutoff_settlements=1))
        self.assertNotIn("N" * 100, text)

    def test_mock_banner_only_for_mock_data_and_not_duplicated(self):
        mock = gen.generate_report(full_summary(is_mock=True, warnings=["MOCK DATA: synthetic", "Real note"]))
        real = gen.generate_report(full_summary())
        self.assertEqual(mock.count("MOCK DATA"), 1)
        self.assertIn("Real note", mock)
        self.assertNotIn("MOCK DATA", real)

    def test_warnings_are_listed_and_capped(self):
        warnings = [f"w{i}" for i in range(10)]
        text = gen.generate_report(full_summary(warnings=warnings))
        self.assertIn("- w5", text)
        self.assertNotIn("- w6", text)
        self.assertIn("... and 4 more notes", text)

    def test_stays_roughly_one_page(self):
        self.assertLessEqual(len(gen.generate_report(worst_case()).splitlines()), 100)
        self.assertLessEqual(len(gen.generate_report(full_summary()).splitlines()), 56)

    def test_wording_has_no_unsupported_claims(self):
        text = gen.generate_report(worst_case()).lower()
        for word in ("destroyed", "confirmed", "casualt", "victim", "people affected"):
            self.assertNotIn(word, text)

    def test_includes_the_three_attribution_lines_and_limitations(self):
        text = gen.generate_report(full_summary())
        for line in ATTRIBUTION_LINES:
            self.assertIn(line.split(" all rights")[0][:60], text)
        self.assertIn("LIMITATIONS", text)

    def test_unknown_language_is_rejected(self):
        with self.assertRaises(ValueError):
            gen.build_model(full_summary(), "fr")


class NumberGuardTests(unittest.TestCase):
    SUMMARIES = [full_summary(), full_summary(cutoff_settlements=0, cutoff_names=[]),
                 full_summary(flood_area_km2=None, affected_roads=None), worst_case(), Summary()]

    def test_no_number_in_any_report_is_unexplained(self):
        for s in self.SUMMARIES:
            for lang in ("en", "ne"):
                text = gen.generate_report(s, lang)
                self.assertEqual(unexpected_numbers(text, s), [], f"{lang}: {text[:80]!r}")

    def test_guard_catches_an_invented_number(self):
        s = full_summary()
        text = gen.generate_report(s) + "\nAn estimated 8,400 people are affected.\n"
        self.assertEqual(unexpected_numbers(text, s), ["8400"])

    def test_guard_understands_both_digit_systems_and_thousands(self):
        self.assertEqual(extract_numbers("1,234.6 km and \u0967\u0968 villages"), {"1234.6", "12"})

    def test_hidden_count_is_derived_not_invented(self):
        s = full_summary(cutoff_settlements=25, cutoff_names=[f"H{i}" for i in range(25)])
        self.assertIn("15", allowed_numbers(s))
        self.assertEqual(unexpected_numbers(gen.generate_report(s), s), [])

    def test_static_numbers_include_the_pixel_size_and_attribution_years(self):
        self.assertTrue({"10", "2026", "2010", "2014", "2018"} <= static_numbers())


class NepaliTests(unittest.TestCase):
    def test_digits_are_devanagari_except_the_required_english_attribution(self):
        text = gen.generate_report(full_summary(), "ne")
        body = text.split(STRINGS["ne"]["attribution_title"])[0]
        self.assertEqual(re.findall(r"[0-9]", body.replace("unet-baseline-0", "").replace("99", "")), [])
        self.assertIn("\u0967\u0968", body)                 # the 12 road segments
        for line in ATTRIBUTION_LINES:                       # legal text stays exactly as required
            self.assertIn(line.split(" all rights")[0][:50], text)

    def test_numbers_match_the_english_report(self):
        s = full_summary()
        ne_numbers = extract_numbers(gen.generate_report(s, "ne"))
        for key in ("flood_area_km2", "affected_roads", "affected_bridges", "affected_buildings",
                    "cutoff_settlements"):
            self.assertTrue(extract_numbers(fmt(s)[key]) <= ne_numbers, key)

    def test_unreviewed_notice_depends_on_the_config_flag(self):
        s = full_summary()
        self.assertIn("Translation not yet reviewed", gen.generate_report(s, "ne"))
        self.assertNotIn("Translation not yet reviewed", gen.generate_report(s, "ne", nepali_reviewed=True))
        self.assertNotIn("Translation not yet reviewed", gen.generate_report(s, "en"))

    def test_missing_values_use_the_nepali_phrase(self):
        text = gen.generate_report(full_summary(flood_area_km2=None), "ne")
        self.assertIn(STRINGS["ne"]["na"], text)
        self.assertNotIn("not available", text.split(STRINGS["ne"]["attribution_title"])[0])


class HtmlTests(unittest.TestCase):
    def test_untrusted_text_is_escaped(self):
        s = full_summary(cutoff_names=['<script>alert(1)</script>'], cutoff_settlements=1,
                         aoi_name='A & B <i>', warnings=['<img src=x onerror=1>'])
        page = gen.render_html(gen.build_model(s))
        self.assertNotIn("<script>", page)
        self.assertNotIn("<img src=x", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("A &amp; B &lt;i&gt;", page)

    def test_language_attribute_and_charset(self):
        self.assertIn('lang="ne"', gen.render_html(gen.build_model(full_summary(), "ne")))
        self.assertIn('<meta charset="utf-8">', gen.render_html(gen.build_model(full_summary())))

    def test_embedded_image(self):
        from dashboard.raster_overlay import encode_png_rgba
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "m.png"
            png.write_bytes(encode_png_rgba(np.full((8, 12, 4), 200, dtype=np.uint8)))
            page = gen.render_html(gen.build_model(full_summary(), image=png))
        self.assertIn("data:image/png;base64,", page)


class PdfTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "r.pdf"

    def tearDown(self):
        self.tmp.cleanup()

    def pdf_text(self):
        from pypdf import PdfReader
        reader = PdfReader(str(self.path))
        return len(reader.pages), "\n".join(p.extract_text() for p in reader.pages)

    def test_normal_report_is_one_page_with_the_numbers(self):
        gen.render_pdf(gen.build_model(full_summary()), self.path)
        pages, text = self.pdf_text()
        self.assertEqual(pages, 1)
        for piece in ("1,234.6", "12 (3.5 km)", "Village B", "Contains modified Copernicus Sentinel data 2026."):
            self.assertIn(piece, text)
        self.assertIn("\u00a9 OpenStreetMap contributors.", text)

    def test_worst_case_still_fits_one_page(self):
        gen.render_pdf(gen.build_model(worst_case()), self.path)
        pages, text = self.pdf_text()
        self.assertEqual(pages, 1)
        self.assertIn("MOCK DATA", text)

    def test_non_latin_names_do_not_crash_and_are_flagged(self):
        s = full_summary(cutoff_names=["\u0917\u093e\u0909\u0901 A", "Village B"], cutoff_settlements=2)
        gen.render_pdf(gen.build_model(s), self.path)
        pages, text = self.pdf_text()
        self.assertEqual(pages, 1)
        self.assertIn("cannot print", text)
        self.assertIn("Village B", text)

    def test_pdf_safe(self):
        self.assertEqual(gen.pdf_safe("km\u00b2 \u00a9 \u2013 \u2022"), "km\u00b2 \u00a9 \u2013 \u2022")
        self.assertEqual(gen.pdf_safe("\u0917"), "?")

    def test_xml_special_characters_in_names_do_not_break_the_pdf(self):
        s = full_summary(cutoff_names=["A & B <tag> ]]>"], cutoff_settlements=1, aoi_name="x<y")
        gen.render_pdf(gen.build_model(s), self.path)
        self.assertEqual(self.pdf_text()[0], 1)
        self.assertIn("A & B <tag>", self.pdf_text()[1])

    def test_embedded_map_image_keeps_one_page(self):
        from dashboard.raster_overlay import encode_png_rgba
        import numpy as np
        png = Path(self.tmp.name) / "m.png"
        rng = np.random.default_rng(0)
        png.write_bytes(encode_png_rgba(rng.integers(0, 255, (300, 500, 4), dtype=np.uint8)))
        gen.render_pdf(gen.build_model(worst_case(), image=png), self.path)
        self.assertEqual(self.pdf_text()[0], 1)

    def test_nothing_is_left_behind_in_the_folder(self):
        gen.render_pdf(gen.build_model(full_summary()), self.path)
        self.assertEqual([p.name for p in Path(self.tmp.name).iterdir()], ["r.pdf"])     # no .tmp files


class RunReportTests(unittest.TestCase):
    def make_run(self, tmp, **cfg_over):
        run = build_mock_run(Path(tmp) / "run", with_rasters=False)
        cfg = {"report": {"languages": ["en", "ne"], "nepali_reviewed": False}}
        cfg["report"].update(cfg_over)
        return run, cfg

    def test_stage_writes_files_that_pass_the_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = self.make_run(tmp)
            gen.run_report(run, AOI(85.2, 28.05, 85.4, 28.2), None, cfg)
            for artifact in (Artifact.REPORT_TXT, Artifact.REPORT_PDF, Artifact.REPORT_HTML,
                             Artifact.REPORT_NE_TXT, Artifact.REPORT_NE_HTML):
                self.assertTrue(run.path(artifact).exists(), artifact.value)
            self.assertEqual(check_stage(run, "report"), [])

    def test_english_only_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = self.make_run(tmp, languages=["en"])
            gen.run_report(run, None, None, cfg)
            self.assertFalse(run.path(Artifact.REPORT_NE_TXT).exists())

    def test_english_is_always_written_even_if_not_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = self.make_run(tmp, languages=["ne"])
            gen.run_report(run, None, None, cfg)
            self.assertTrue(run.path(Artifact.REPORT_TXT).exists())

    def test_report_numbers_equal_summary_json(self):
        from pipeline.fileio import read_json
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = self.make_run(tmp)
            gen.run_report(run, None, None, cfg)
            summary = read_json(run.path(Artifact.SUMMARY))
            text = run.path(Artifact.REPORT_TXT).read_text(encoding="utf-8")
            self.assertIn(f"Road segments affected: {summary['affected_roads']}", text)
            self.assertEqual(unexpected_numbers(text, summary), [])
            self.assertIn("MOCK DATA", text)              # the mock run is labelled in the report too

    def test_flood_map_figure_is_embedded_when_it_exists(self):
        from dashboard.raster_overlay import encode_png_rgba
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = self.make_run(tmp)
            run.ensure_parent(Artifact.FIG_FLOOD).write_bytes(
                encode_png_rgba(np.full((20, 30, 4), 128, dtype=np.uint8)))
            gen.run_report(run, None, None, cfg)
            self.assertIn("data:image/png;base64,", run.path(Artifact.REPORT_HTML).read_text(encoding="utf-8"))

    def test_make_report_script(self):
        from scripts import make_report
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = self.make_run(tmp)
            self.assertEqual(make_report.main([str(run.root), "--lang", "en"]), 0)
            self.assertTrue(run.path(Artifact.REPORT_TXT).exists())
            self.assertFalse(run.path(Artifact.REPORT_NE_TXT).exists())
            self.assertEqual(make_report.main([str(Path(tmp) / "nope")]), 1)
            run.path(Artifact.SUMMARY).unlink()
            self.assertEqual(make_report.main([str(run.root)]), 1)


if __name__ == "__main__":
    unittest.main()
