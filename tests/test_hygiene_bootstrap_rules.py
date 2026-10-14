import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

from pipeline.errors import DataRuleViolation
from pipeline.rules import assert_osm_snapshot_pre_event, scan_for_forbidden_inputs
from scripts import bootstrap_repo
from scripts.check_repo_hygiene import classify_path, find_secrets

REPO_ROOT = Path(__file__).resolve().parents[1]
MB = 1024 * 1024


class ClassifyPathTests(unittest.TestCase):
    def test_normal_files_pass(self):
        self.assertEqual(classify_path("gis/damage.py", 4000), [])
        self.assertEqual(classify_path("data/sample/mock_run/before.tif", 1 * MB), [])

    def test_big_file_blocked(self):
        self.assertTrue(classify_path("docs/figure.png", 6 * MB))

    def test_checkpoints_and_archives_blocked_outside_release(self):
        self.assertTrue(classify_path("models/unet_epoch3.pt", 1000))
        self.assertTrue(classify_path("data/sample/scene.zip", 1000))
        self.assertEqual(classify_path("models/release/unet_best.pt", 20 * MB), [])
        self.assertTrue(classify_path("models/release/unet_best.pt", 60 * MB))

    def test_raw_data_folders_blocked(self):
        for p in ("data/raw/s1.tif", "outputs/run/x.json", "data/reference/ref.geojson"):
            self.assertTrue(classify_path(p, 10), p)

    def test_tif_only_in_sample_or_docs(self):
        self.assertTrue(classify_path("preprocessing/tmp.tif", 100))

    def test_secret_files_and_windows_separators(self):
        self.assertTrue(classify_path(".env", 10))
        self.assertEqual(classify_path(".env.example", 10), [])
        self.assertTrue(classify_path("data\\raw\\x.json", 10))      # backslashes are normalised


class SecretScanTests(unittest.TestCase):
    # Fake secrets are assembled from pieces so this file does not trip the hook itself.
    def test_detects_hardcoded_credential(self):
        line = "pass" + "word = '" + "hunter2hunter2" + "'"
        self.assertEqual(find_secrets(line)[0][1], "hard-coded credential")

    def test_detects_aws_key_and_private_key(self):
        self.assertTrue(find_secrets("key = AK" + "IA" + "ABCDEFGHIJKLMNOP"))
        self.assertTrue(find_secrets("-----BEGIN " + "RSA PRIVATE KEY-----"))

    def test_allow_marker_and_clean_text(self):
        line = "pass" + "word = '" + "hunter2hunter2" + "'  # hygiene: allow"
        self.assertEqual(find_secrets(line), [])
        self.assertEqual(find_secrets("os.environ['COPERNICUS_PASSWORD']"), [])


@unittest.skipUnless(shutil.which("git"), "git not installed")
class GitIntegrationTests(unittest.TestCase):
    """Real git repo: the hook logic must block a big staged file but allow deleting it."""

    def run_git(self, *args, cwd):
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)

    def hygiene(self, cwd):
        return subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "check_repo_hygiene.py"), "--staged"],
                              cwd=cwd, capture_output=True, text=True)

    def test_blocks_big_file_but_not_its_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_git("init", "-q", cwd=tmp)
            self.run_git("config", "user.email", "t@example.com", cwd=tmp)
            self.run_git("config", "user.name", "t", cwd=tmp)
            (Path(tmp) / "ok.py").write_text("print(1)\n")
            self.run_git("add", "ok.py", cwd=tmp)
            self.assertEqual(self.hygiene(tmp).returncode, 0)

            big = Path(tmp) / "big.bin"
            big.write_bytes(b"\0" * (6 * MB))
            self.run_git("add", "big.bin", cwd=tmp)
            res = self.hygiene(tmp)
            self.assertEqual(res.returncode, 1)
            self.assertIn("big.bin", res.stdout)

            # Commit it with the hook bypassed, then stage its DELETION: must be allowed.
            self.run_git("commit", "-q", "--no-verify", "-m", "x", cwd=tmp)
            self.run_git("rm", "-q", "big.bin", cwd=tmp)
            self.assertEqual(self.hygiene(tmp).returncode, 0)

    def test_checks_staged_content_not_working_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_git("init", "-q", cwd=tmp)
            f = Path(tmp) / "cfg.py"
            f.write_text("pass" + "word = '" + "hunter2hunter2" + "'\n")
            self.run_git("add", "cfg.py", cwd=tmp)
            f.write_text("x = 1\n")               # working tree is clean now, index is not
            self.assertEqual(self.hygiene(tmp).returncode, 1)


class RulesTests(unittest.TestCase):
    def test_osm_snapshot_must_be_strictly_before_event(self):
        ev = date(2026, 8, 26)
        self.assertEqual(assert_osm_snapshot_pre_event("2026-07-27", ev), date(2026, 7, 27))
        for bad in ("2026-08-26", "2026-09-01"):
            with self.assertRaises(DataRuleViolation):
                assert_osm_snapshot_pre_event(bad, ev)

    def test_scan_finds_forbidden_reference_in_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gis").mkdir()
            (root / "gis" / "bad.py").write_text('x = "data/reference/ref.geojson"\n')
            (root / "gis" / "good.py").write_text("x = 1\n")
            hits = scan_for_forbidden_inputs(root)
            self.assertEqual([h[0] for h in hits], ["gis/bad.py"])
            self.assertEqual(scan_for_forbidden_inputs(root, allow=["gis/bad.py"]), [])

    def test_this_repo_is_clean(self):
        """The real guard from Part 5, already active: pipeline code never names check-only data."""
        self.assertEqual(scan_for_forbidden_inputs(REPO_ROOT, allow=["dashboard/case_study.py"]), [])


class BootstrapTests(unittest.TestCase):
    def test_creates_then_skips_then_respects_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = dict(bootstrap_repo.bootstrap(root))
            self.assertEqual(first["config/default.yaml"], "created")
            self.assertTrue((root / "gis" / "damage.py").exists())
            self.assertTrue((root / "data" / "raw").is_dir())
            self.assertFalse((root / "data" / "raw" / ".gitkeep").exists())

            (root / "gis" / "damage.py").write_text("# teammate wrote real code\n")
            second = dict(bootstrap_repo.bootstrap(root))
            self.assertEqual(second["gis/damage.py"], "skipped")           # never clobbered
            self.assertEqual(second["docs/contracts.md"], "updated")       # generated: refreshed
            self.assertIn("real code", (root / "gis" / "damage.py").read_text())

            forced = dict(bootstrap_repo.bootstrap(root, force=True))
            self.assertEqual(forced["gis/damage.py"], "updated")

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            bootstrap_repo.bootstrap(Path(tmp), dry_run=True)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_generated_stubs_compile_and_raise_not_implemented(self):
        for stage in ("acquire", "osm", "infer", "damage", "connectivity"):
            rel, text = bootstrap_repo.stub_source(stage)
            compile(text, rel, "exec")
            self.assertIn("NotImplementedError", text)

    def test_generated_default_yaml_equals_code_defaults(self):
        import yaml
        from pipeline.config import DEFAULT_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            bootstrap_repo.bootstrap(Path(tmp))
            self.assertEqual(yaml.safe_load((Path(tmp) / "config" / "default.yaml").read_text()), DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
