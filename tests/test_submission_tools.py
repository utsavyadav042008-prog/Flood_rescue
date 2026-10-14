import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pipeline.contracts import Artifact
from pipeline.errors import InputValidationError
from pipeline.fileio import rasterio_available, read_json, write_json
from pipeline.run import RunPaths
from reports.format import ATTRIBUTION_LINES
from scripts import audit_submission as audit
from scripts import export_demo_run, report_facts
from scripts.make_mock_outputs import build_mock_run

REPO_ROOT = Path(__file__).resolve().parents[1]
ATTR = "\n\n".join(ATTRIBUTION_LINES)


def status(checks, name):
    return next(c for c in checks if c.name.startswith(name))


class AuditTests(unittest.TestCase):
    def make_tree(self, root: Path):
        """A minimal passing submission tree (built from the real repo's docs)."""
        for rel in ("README.md", "docs/attribution.md", "docs/limitations.md", "docs/data_sources.md",
                    "requirements.txt", ".gitignore"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO_ROOT / rel, root / rel)
        (root / "dashboard").mkdir()
        shutil.copy(REPO_ROOT / "dashboard" / "components.py", root / "dashboard" / "components.py")
        # real mock rasters when rasterio exists (the contract opens them); placeholders otherwise.
        # Either way the files are present, so the MOCK flag is what fails.
        demo = build_mock_run(root / "data" / "sample" / "trishuli_run", with_rasters=rasterio_available())
        if not rasterio_available():
            for artifact in (Artifact.BEFORE, Artifact.AFTER, Artifact.FLOOD_MASK, Artifact.FLOOD_CONFIDENCE):
                demo.path(artifact).write_bytes(b"placeholder")
        (root / "evaluation").mkdir()
        (root / "evaluation" / "case_study_results.json").write_text(json.dumps(
            {"flood_extent": {"iou": 0.61}, "notes": ["radar shadow on steep slopes"]}))
        (root / "submission").mkdir()
        (root / "submission" / "demo.mp4").write_bytes(b"\0" * 1000)

    def test_the_real_repo_docs_pass_their_own_checks(self):
        for fn in (audit.check_readme, audit.check_attribution, audit.check_citations, audit.check_limitations,
                   audit.check_todos, audit.check_data_rules):
            c = fn(REPO_ROOT)
            self.assertEqual(c.status, audit.PASS, f"{c.name}: {c.detail}")

    def test_passing_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_tree(root)
            checks = audit.run_audit(root, skip_tests=True, report=root / "nope.pdf")
            self.assertEqual(status(checks, "README").status, audit.PASS)
            self.assertEqual(status(checks, "Case-study").status, audit.PASS)
            self.assertEqual(status(checks, "Demo video").status, audit.MANUAL)
            # the demo run is a MOCK run: the audit must refuse it
            demo = status(checks, "Cached demo")
            self.assertEqual(demo.status, audit.FAIL)
            self.assertIn("MOCK", demo.detail)
            self.assertEqual(status(checks, "Written report").status, audit.FAIL)      # no pdf in the tree

    def test_empty_tree_fails_every_content_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            checks = audit.run_audit(Path(tmp), skip_tests=True)
            for name in ("README", "Required attribution", "Dataset citations", "Limitations document",
                         "Cached demo", "Case-study", "Written report", "Demo video"):
                self.assertEqual(status(checks, name).status, audit.FAIL, name)

    def test_attribution_check_catches_a_missing_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_tree(root)
            text = (root / "README.md").read_text(encoding="utf-8").replace(ATTRIBUTION_LINES[2], "")
            (root / "README.md").write_text(text, encoding="utf-8")
            c = audit.check_attribution(root)
            self.assertEqual(c.status, audit.FAIL)
            self.assertIn("README.md", c.detail)

    def test_citation_check_requires_bonafilia_when_sen1floods11_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_tree(root)
            path = root / "docs" / "attribution.md"
            path.write_text(path.read_text(encoding="utf-8").replace("Bonafilia", "Someone"), encoding="utf-8")
            c = audit.check_citations(root)
            self.assertEqual(c.status, audit.FAIL)
            self.assertIn("Sen1Floods11", c.detail)

    def test_data_rule_check_catches_reference_use_and_evaluation_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gis").mkdir()
            (root / "gis" / "bad.py").write_text('p = "data/reference/x.geojson"\n')
            (root / "models").mkdir()
            (root / "models" / "worse.py").write_text("from evaluation import metrics\n")
            c = audit.check_data_rules(root)
            self.assertEqual(c.status, audit.FAIL)
            self.assertIn("gis/bad.py", c.detail)
            self.assertIn("models/worse.py imports evaluation", c.detail)

    def test_todo_and_stub_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "README.md").write_text("fine\nTODO: add metrics\n")
            (root / "docs" / "limitations.md").write_text("# Limitations\n\n(T4 fills this in)\n")
            self.assertEqual(audit.check_todos(root).status, audit.WARN)
            self.assertEqual(audit.check_limitations(root).status, audit.FAIL)

    def test_case_study_without_notes_is_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evaluation").mkdir()
            (root / "evaluation" / "case_study_results.json").write_text(json.dumps({"flood_extent": {"iou": 0.5}, "notes": []}))
            self.assertEqual(audit.check_case_study(root).status, audit.WARN)

    def test_report_page_limit(self):
        try:
            from reportlab.pdfgen import canvas
            import pypdf  # noqa: F401
        except ImportError:
            self.skipTest("reportlab/pypdf not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "r.pdf"
            for pages, expected in ((3, audit.PASS), (7, audit.FAIL)):
                c = canvas.Canvas(str(pdf))
                for _ in range(pages):
                    c.drawString(72, 700, "Limitations of this system")
                    c.showPage()
                c.save()
                self.assertEqual(audit.check_report(root, pdf).status, expected)
            c = canvas.Canvas(str(pdf)); c.drawString(72, 700, "no such section"); c.showPage(); c.save()
            self.assertEqual(audit.check_report(root, pdf).status, audit.FAIL)

    def test_requirements_pinning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "requirements.txt").write_text("numpy\nrasterio==1.3.9\n")
            self.assertEqual(audit.check_requirements(root).status, audit.WARN)
            (root / "requirements.txt").write_text("numpy==2.1.0\nrasterio>=1.3\n")
            self.assertEqual(audit.check_requirements(root).status, audit.PASS)

    def test_main_exit_code_is_1_when_something_fails(self):
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = audit.main(["--skip-tests"])
        self.assertEqual(code, 1)                                # the repo is not submission-ready yet
        self.assertIn("FAIL", out.getvalue())


class ExportDemoTests(unittest.TestCase):
    def test_mock_runs_are_refused_unless_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = build_mock_run(Path(tmp) / "run", with_rasters=False)
            with self.assertRaises(InputValidationError):
                export_demo_run.export_demo_run(run.root, Path(tmp) / "demo")
            copied = export_demo_run.export_demo_run(run.root, Path(tmp) / "demo", allow_mock=True)
            self.assertIn("summary.json", copied)

    def test_cache_files_are_skipped_and_big_files_stop_the_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = build_mock_run(Path(tmp) / "run", with_rasters=False)
            (run.root / "figures" / "_overlay_flood_mask.png").write_bytes(b"x")
            copied = export_demo_run.export_demo_run(run.root, Path(tmp) / "demo", allow_mock=True)
            self.assertFalse(any("_overlay_" in c for c in copied))
            (run.root / "before.tif").write_bytes(b"\0" * (6 * 1024 * 1024))
            with self.assertRaises(InputValidationError) as ctx:
                export_demo_run.export_demo_run(run.root, Path(tmp) / "demo2", allow_mock=True)
            self.assertIn("before.tif", ctx.exception.user_message)

    def test_a_real_run_is_exported_and_the_old_demo_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = build_mock_run(Path(tmp) / "run", with_rasters=False)
            data = read_json(run.path(Artifact.SUMMARY))
            data["is_mock"] = False
            data.pop("provenance", None)
            write_json(run.path(Artifact.SUMMARY), data)
            manifest = read_json(run.path(Artifact.RUN_MANIFEST))
            manifest["is_mock"] = False
            write_json(run.path(Artifact.RUN_MANIFEST), manifest)
            dest = Path(tmp) / "demo"
            dest.mkdir()
            (dest / "stale.txt").write_text("old")
            export_demo_run.export_demo_run(run.root, dest)
            self.assertFalse((dest / "stale.txt").exists())
            self.assertTrue((dest / "run.json").exists())


class ReportFactsTests(unittest.TestCase):
    def test_facts_come_from_the_summary_and_the_case_study_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = build_mock_run(Path(tmp) / "run", with_rasters=False)
            results = Path(tmp) / "cs.json"
            results.write_text(json.dumps({
                "reference": "EMSR927", "credit": "European Union, Copernicus Emergency Management Service data",
                "flood_extent": {"iou": 0.612, "precision": 0.7, "recall": 0.8, "f1": 0.75},
                "infrastructure": {"roads": {"reference_flagged": 40, "also_flagged": 31, "extra_flagged": 9}},
                "notes": ["radar shadow"]}))
            md = report_facts.facts_markdown(run, results)
            self.assertIn("Settlements with no road connection | 3 |", md)
            self.assertIn("| 0.61 | 0.70 | 0.80 | 0.75 |", md)
            self.assertIn("| roads | 40 | 31 | 9 |", md)
            self.assertIn("- radar shadow", md)
            for line in ATTRIBUTION_LINES:
                self.assertIn(line, md)
            self.assertIn("not available", md)                       # flood area: no rasters in this run

    def test_missing_case_study_is_explained_not_faked(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = build_mock_run(Path(tmp) / "run", with_rasters=False)
            self.assertIn("No case-study results yet", report_facts.facts_markdown(run, Path(tmp) / "none.json"))


if __name__ == "__main__":
    unittest.main()
