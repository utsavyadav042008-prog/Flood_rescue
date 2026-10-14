"""Pipeline tests.  Stage functions are replaced by small fakes, so no satellite
data, network or rasterio is needed.  Raster files are placeholder bytes
(the validator only warns "rasterio not installed" for them)."""

import json
import os
import tempfile
import threading
import time
import unittest
from datetime import date, timedelta
from pathlib import Path

from pipeline import analyze, mocks, stages
from pipeline.analyze import CACHED, FAILED, OK, SKIPPED, analyze_flood, load_result, stage_graph
from pipeline.config import load_config
from pipeline.contracts import CONTRACTS, Artifact
from pipeline.errors import AcquisitionError, GISError
from pipeline.fileio import rasterio_available, read_json, write_json
from pipeline.run import RunPaths
from pipeline.schemas import AOI, Provenance

AOI_T = AOI(85.10, 27.85, 85.45, 28.30, "Trishuli")
EVENT = "2026-08-26"


def fake_raster(run, artifact):
    """A contract-valid 4x4 GeoTIFF when rasterio is installed, else placeholder bytes
    (the raster checks are skipped without rasterio, so the bytes are never opened)."""
    if not rasterio_available():
        run.ensure_parent(artifact).write_bytes(b"II*\x00 placeholder raster")
        return
    import numpy as np
    from pipeline.fileio import north_up_transform, write_geotiff
    if artifact is Artifact.FLOOD_MASK:
        arr = np.zeros((4, 4), "uint8")
    elif artifact is Artifact.FLOOD_CONFIDENCE:
        arr = np.zeros((4, 4), "float32")
    else:
        arr = np.full((4, 4), -10.0, "float32")
    write_geotiff(run.ensure_parent(artifact), arr, transform=north_up_transform(320000.0, 3100000.0, 10.0),
                  epsg=AOI_T.utm_epsg())


def fake_acquire(run, aoi, event, cfg):
    fake_raster(run, Artifact.BEFORE)
    fake_raster(run, Artifact.AFTER)
    write_json(run.path(Artifact.METADATA), {
        "source": "fake", "before_date": (event - timedelta(days=6)).isoformat(),
        "after_date": (event + timedelta(days=6)).isoformat(), "relative_orbit_before": 5,
        "relative_orbit_after": 5, "orbit_direction": "ASCENDING", "crs": "EPSG:32645"},
        Provenance("fake.acquire", "fake"))


def fake_infer(run, aoi, event, cfg):
    fake_raster(run, Artifact.FLOOD_MASK)
    fake_raster(run, Artifact.FLOOD_CONFIDENCE)


class PipelineTestCase(unittest.TestCase):
    """Base class: temp output folder, mock vector stages, fake raster stages."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "out"
        # Redirect the mock template into the temp dir so tests never touch the repo.
        self._old_template = mocks.TEMPLATE_DIR
        mocks.TEMPLATE_DIR = Path(self.tmp.name) / "template"
        self.overrides = {"acquire": fake_acquire, "infer": fake_infer}
        # Stub the raster statistics (needs rasterio) with fixed numbers.
        import pipeline.summary as summary_mod
        self._old_stats = summary_mod._raster_stats
        summary_mod._raster_stats = lambda run: (4.25, 0.9)

    def tearDown(self):
        import pipeline.summary as summary_mod
        summary_mod._raster_stats = self._old_stats
        mocks.TEMPLATE_DIR = self._old_template
        self.tmp.cleanup()

    def go(self, **kw):
        kw.setdefault("mock", {"osm", "damage", "connectivity"})
        kw.setdefault("stage_overrides", self.overrides)
        kw.setdefault("out_root", self.out)
        return analyze_flood(AOI_T, EVENT, **kw)

    def statuses(self, result):
        return {o.name: o.status for o in result.outcomes}


class HappyPathTests(PipelineTestCase):
    def test_full_run_and_numbers(self):
        r = self.go()
        self.assertTrue(r.ok, r.error)
        st = self.statuses(r)
        for name in ("acquire", "osm", "infer", "damage", "connectivity", "summarize"):
            self.assertEqual(st[name], OK, name)
        # report and figures are real code (figures are tolerant: they draw "not available" panels)
        self.assertEqual(st["report"], OK)
        self.assertEqual(st["figures"], OK)
        run = RunPaths(r.run_dir)
        for artifact in (Artifact.REPORT_TXT, Artifact.REPORT_HTML, Artifact.REPORT_NE_TXT, Artifact.REPORT_NE_HTML):
            self.assertTrue(run.path(artifact).exists(), artifact.value)
        s = r.summary
        self.assertEqual((s.flood_area_km2, s.mean_confidence), (4.25, 0.9))
        self.assertEqual(s.cutoff_names, ["Village B", "Village C", "Village D"])
        self.assertEqual(s.cutoff_settlements, 3)
        self.assertTrue(s.is_mock)
        self.assertEqual(s.event_date, EVENT)
        self.assertIn("MOCK DATA", s.warnings[0])

    def test_mock_runs_live_in_a_separate_folder(self):
        r = self.go()
        self.assertEqual(r.run_dir.parent.name, "_mock")
        r2 = self.go(mock=False, stage_overrides={**self.overrides, "osm": lambda *a: (_ for _ in ()).throw(GISError("x"))})
        self.assertNotEqual(r.run_dir, r2.run_dir)

    def test_run_log_is_written(self):
        r = self.go()
        log = read_json(RunPaths(r.run_dir).path(Artifact.RUN_LOG))
        self.assertTrue(log["ok"])
        self.assertEqual({s["name"] for s in log["stages"]}, set(analyze.STAGE_ORDER))
        self.assertEqual(log["mock_stages"], ["connectivity", "damage", "osm"])

    def test_progress_callback_runs_on_the_calling_thread(self):
        main, seen, events = threading.get_ident(), set(), []
        def cb(stage, state):
            seen.add(threading.get_ident())
            events.append((stage, state))
        self.go(on_progress=cb)
        self.assertEqual(seen, {main})
        self.assertIn(("acquire", "running"), events)
        self.assertIn(("acquire", "ok"), events)

    def test_load_result_roundtrip(self):
        r = self.go()
        loaded = load_result(r.run_dir)
        self.assertTrue(loaded.ok)
        self.assertEqual(loaded.summary, r.summary)
        self.assertEqual({o.name: o.status for o in loaded.outcomes}, self.statuses(r))


class CacheTests(PipelineTestCase):
    def test_second_run_is_fully_cached(self):
        self.go()
        r2 = self.go()
        st = self.statuses(r2)
        for name in ("acquire", "osm", "infer", "damage", "connectivity", "summarize"):
            self.assertEqual(st[name], CACHED, name)

    def test_touching_an_input_reruns_only_downstream(self):
        r = self.go()
        run = RunPaths(r.run_dir)
        future = time.time() + 100
        os.utime(run.path(Artifact.BEFORE), (future, future))      # "T1 regenerated before.tif"
        st = self.statuses(self.go())
        self.assertEqual(st["osm"], CACHED)           # independent of the satellite data
        self.assertEqual(st["acquire"], CACHED)       # its own outputs are not older than its input (run.json)
        self.assertEqual(st["infer"], OK)             # needs before.tif -> stale -> re-run
        self.assertEqual(st["damage"], OK)            # needs flood_mask, now newer
        self.assertEqual(st["connectivity"], OK)
        self.assertEqual(st["summarize"], OK)

    def test_report_is_never_served_from_the_cache(self):
        self.go()
        st = self.statuses(self.go())
        self.assertEqual(st["summarize"], CACHED)
        self.assertEqual(st["report"], OK)            # cheap and setting-dependent: always regenerated

    def test_no_cache_flag_reruns_everything(self):
        self.go()
        st = self.statuses(self.go(use_cache=False))
        self.assertEqual({st[n] for n in ("acquire", "osm", "infer", "damage")}, {OK})

    def test_corrupted_cache_heals(self):
        r = self.go()
        run = RunPaths(r.run_dir)
        run.path(Artifact.DAMAGE_SUMMARY).write_text("{broken", encoding="utf-8")
        future = time.time() + 50
        os.utime(run.path(Artifact.DAMAGE_SUMMARY), (future, future))   # looks "fresh" by mtime
        st = self.statuses(self.go())
        self.assertEqual(st["damage"], OK)
        self.assertIsInstance(read_json(run.path(Artifact.DAMAGE_SUMMARY)), dict)

    def test_mock_output_is_never_reused_by_a_real_stage(self):
        r = self.go()                                              # damage etc. are mock
        real_calls = []
        def real_damage(run, aoi, event, cfg):
            real_calls.append(1)
            raise GISError("real damage failed")
        # same folder trick: reuse the mock folder by mocking only osm/connectivity now
        r2 = self.go(mock={"osm", "connectivity"}, stage_overrides={**self.overrides, "damage": real_damage})
        self.assertEqual(real_calls, [1])           # mock cache ignored; the real stage ran
        self.assertEqual(r2.outcome("damage").status, FAILED)


class FailureTests(PipelineTestCase):
    def test_required_stage_failure_stops_the_run(self):
        def bad_acquire(run, aoi, event, cfg):
            raise AcquisitionError("No same-orbit pair found.", "searched 14 scenes")
        r = self.go(stage_overrides={**self.overrides, "acquire": bad_acquire})
        self.assertFalse(r.ok)
        self.assertEqual((r.error, r.failed_stage), ("No same-orbit pair found.", "acquire"))
        st = self.statuses(r)
        self.assertEqual(st["acquire"], FAILED)
        for name in ("infer", "damage", "connectivity", "summarize"):
            self.assertEqual(st[name], SKIPPED, name)
        log = read_json(RunPaths(r.run_dir).path(Artifact.RUN_LOG))
        self.assertFalse(log["ok"])
        self.assertEqual(r.outcome("acquire").detail, "searched 14 scenes")

    def test_stale_summary_from_an_earlier_run_is_not_returned_after_a_failure(self):
        first = self.go()
        self.assertIsNotNone(first.summary)
        def bad_acquire(run, aoi, event, cfg):
            raise AcquisitionError("No same-orbit pair found.")
        second = self.go(use_cache=False, stage_overrides={**self.overrides, "acquire": bad_acquire})
        self.assertFalse(second.ok)
        self.assertIsNone(second.summary)          # summary.json still exists on disk, but is stale

    def test_load_result_gives_a_readable_error_for_a_run_missing_a_required_stage(self):
        r = self.go()
        RunPaths(r.run_dir).path(Artifact.RUN_LOG).unlink()
        RunPaths(r.run_dir).path(Artifact.FLOOD_MASK).unlink()
        loaded = load_result(r.run_dir)
        self.assertFalse(loaded.ok)
        self.assertIn("infer", loaded.error)

    def test_degradable_failure_keeps_going_and_shows_none_not_zero(self):
        def bad_damage(run, aoi, event, cfg):
            raise GISError("Road data unavailable.")
        r = self.go(stage_overrides={**self.overrides, "damage": bad_damage})
        self.assertTrue(r.ok)
        st = self.statuses(r)
        self.assertEqual(st["damage"], FAILED)
        self.assertEqual(st["connectivity"], SKIPPED)         # needs affected_roads
        self.assertEqual(st["summarize"], OK)                  # tolerant: still runs
        s = r.summary
        self.assertIsNone(s.affected_roads)
        self.assertIsNone(s.cutoff_settlements)
        self.assertEqual(s.flood_area_km2, 4.25)               # the part that worked survives
        self.assertTrue(any("damage" in w and "Road data unavailable" in w for w in s.warnings))
        self.assertEqual(sorted(r.degraded), ["connectivity", "damage"])     # figures still run (tolerant)

    def test_osm_failure_skips_both_gis_stages(self):
        def bad_osm(run, aoi, event, cfg):
            raise GISError("ohsome timeout")
        r = self.go(mock={"damage", "connectivity"}, stage_overrides={**self.overrides, "osm": bad_osm})
        st = self.statuses(r)
        self.assertEqual((st["osm"], st["damage"], st["connectivity"]), (FAILED, SKIPPED, SKIPPED))
        self.assertTrue(r.ok)

    def test_stub_raising_not_implemented_is_skipped_with_owner(self):
        def stub(run, aoi, event, cfg):
            raise NotImplementedError("stage 'osm' is not implemented yet (owner: T3)")
        r = self.go(mock={"damage", "connectivity"}, stage_overrides={**self.overrides, "osm": stub})
        self.assertEqual(r.outcome("osm").status, SKIPPED)
        self.assertIn("owner: T3", r.outcome("osm").message)

    def test_unexpected_exception_is_contained(self):
        def buggy(run, aoi, event, cfg):
            raise RuntimeError("boom internal detail")
        r = self.go(stage_overrides={**self.overrides, "osm": buggy}, mock={"damage", "connectivity"})
        o = r.outcome("osm")
        self.assertEqual(o.status, FAILED)
        self.assertNotIn("boom", o.message)                    # friendly text only
        self.assertIn("boom internal detail", o.detail)        # traceback kept for the log
        self.assertTrue(r.ok)                                  # osm is degradable

    def test_missing_dependency_is_reported_as_load_failure(self):
        def needs_torch(run, aoi, event, cfg):
            import definitely_not_installed_pkg  # noqa: F401
        r = self.go(stage_overrides={**self.overrides, "infer": needs_torch})
        self.assertFalse(r.ok)
        self.assertIn("could not be loaded", r.error)

    def test_contract_violation_fails_the_stage(self):
        def bad_infer(run, aoi, event, cfg):
            fake_raster(run, Artifact.FLOOD_MASK)              # confidence file missing
        r = self.go(stage_overrides={**self.overrides, "infer": bad_infer})
        self.assertFalse(r.ok)
        self.assertIn("data contract", r.error)
        self.assertIn("flood_confidence.tif", r.outcome("infer").detail)

    def test_stale_outputs_are_removed_before_a_stage_reruns(self):
        r = self.go()
        run = RunPaths(r.run_dir)
        self.assertTrue(run.path(Artifact.DAMAGE_SUMMARY).exists())
        def bad_damage(run_, aoi, event, cfg):
            raise GISError("fails now")
        self.go(use_cache=False, stage_overrides={**self.overrides, "damage": bad_damage})
        self.assertFalse(run.path(Artifact.DAMAGE_SUMMARY).exists())   # no stale file left behind

    def test_strict_mode_turns_warnings_into_failures(self):
        r_normal = self.go()
        self.assertTrue(r_normal.ok)
        if not rasterio_available():       # placeholder rasters only warn -> strict must fail
            r = self.go(strict=True, use_cache=False)
            self.assertFalse(r.ok)

    def test_forbid_mock(self):
        r = self.go(forbid_mock=True, use_cache=False)
        self.assertFalse(r.ok)


class InputRuleTests(PipelineTestCase):
    def test_bad_inputs_return_a_friendly_failure_and_create_nothing(self):
        for aoi, date_ in ((AOI(80, 20, 90, 30), EVENT), (AOI_T, "26/08/2026"), (AOI_T, "2999-01-01")):
            r = analyze_flood(aoi, date_, out_root=self.out, mock=True)
            self.assertFalse(r.ok)
            self.assertIsNone(r.run_dir)
            self.assertTrue(r.error)
        self.assertFalse(self.out.exists())

    def test_osm_snapshot_is_clamped_to_the_day_before_the_event(self):
        # a configured snapshot AFTER the event must never be used: the run uses the day before
        cfg = load_config(overrides={"osm": {"snapshot_date": "2026-09-30"}})
        r = self.go(cfg=cfg)
        self.assertTrue(r.ok, r.error)
        snap = read_json(RunPaths(r.run_dir).path(Artifact.OSM_SNAPSHOT))["snapshot_date"]
        self.assertEqual(snap, "2026-08-25")

    def test_unknown_mock_stage_is_rejected(self):
        r = analyze_flood(AOI_T, EVENT, out_root=self.out, mock={"summarize"})
        self.assertFalse(r.ok)


class ParallelismAndGraphTests(PipelineTestCase):
    def test_graph_is_derived_from_the_contract(self):
        g = stage_graph()
        self.assertEqual(g["osm"], set())                       # runs next to acquire
        self.assertEqual(g["acquire"], set())
        self.assertEqual(g["infer"], {"acquire"})
        self.assertEqual(g["damage"], {"infer", "osm"})
        self.assertEqual(g["connectivity"], {"osm", "damage"})
        self.assertEqual(g["summarize"], {"acquire", "osm", "infer", "damage", "connectivity"})
        self.assertEqual(g["report"], {"summarize"})

    def test_osm_and_acquire_really_run_at_the_same_time(self):
        barrier = threading.Barrier(2, timeout=5)      # both must be inside their stage together
        def acquire(run, aoi, event, cfg):
            barrier.wait()
            fake_acquire(run, aoi, event, cfg)
        mock_osm = mocks.mock_stage("osm")
        def osm(run, aoi, event, cfg):
            barrier.wait()
            mock_osm(run, aoi, event, cfg)
        r = self.go(mock={"damage", "connectivity"},
                    stage_overrides={**self.overrides, "acquire": acquire, "osm": osm})
        self.assertTrue(r.ok, r.error)                 # a sequential scheduler -> BrokenBarrierError

    def test_max_workers_one_serialises_stages(self):
        cfg = load_config(overrides={"pipeline": {"max_workers": 1}})
        r = self.go(cfg=cfg)
        self.assertTrue(r.ok, r.error)


class SummaryTests(PipelineTestCase):
    def test_none_stays_none_for_bad_values(self):
        from pipeline.summary import build_summary
        r = self.go()
        run = RunPaths(r.run_dir)
        data = read_json(run.path(Artifact.DAMAGE_SUMMARY))
        data["affected_roads"] = "many"
        write_json(run.path(Artifact.DAMAGE_SUMMARY), data)
        s = build_summary(run)
        self.assertIsNone(s.affected_roads)
        self.assertTrue(any("not a number" in w for w in s.warnings))

    def test_suspiciously_large_flood_is_flagged(self):
        import pipeline.summary as summary_mod
        summary_mod._raster_stats = lambda run: (900.0, 0.8)    # > 30% of ~1700 km2
        r = self.go(use_cache=False)
        self.assertTrue(any("unusually large" in w for w in r.summary.warnings))

    def test_zero_flood_is_reported_as_zero_with_a_note(self):
        import pipeline.summary as summary_mod
        summary_mod._raster_stats = lambda run: (0.0, None)
        r = self.go(use_cache=False)
        self.assertEqual(r.summary.flood_area_km2, 0.0)
        self.assertTrue(any("No flooded pixels" in w for w in r.summary.warnings))


class RealRegistryTests(PipelineTestCase):
    """No overrides: the real, integrated stages (they download data, so opt-in)."""

    @unittest.skipUnless(os.environ.get("FLOODRESQ_NETWORK_TESTS") == "1",
                         "runs the real stages (downloads data); set FLOODRESQ_NETWORK_TESTS=1")
    def test_real_pipeline_finishes_or_fails_with_a_friendly_message(self):
        small = AOI(85.30, 28.10, 85.34, 28.13, "network-test")
        r = analyze_flood(small, EVENT, out_root=self.out)
        self.assertEqual({o.name for o in r.outcomes}, set(analyze.STAGE_ORDER))
        if not r.ok:
            self.assertTrue(r.error)
            self.assertNotIn("Traceback", r.error)

    def test_lazy_entrypoint_distinguishes_missing_module_from_missing_dependency(self):
        lazy = stages.LazyEntrypoint("x", "no_such_pkg_xyz.mod:run_x", "T9")
        with self.assertRaises(NotImplementedError):
            lazy(None, None, None, None)


class CliTests(PipelineTestCase):
    def test_cli_exit_codes(self):
        aoi_file = Path(self.tmp.name) / "aoi.json"
        aoi_file.write_text(json.dumps(AOI_T.to_dict()), encoding="utf-8")
        code = analyze.main(["--aoi", str(aoi_file), "--date", EVENT, "--out", str(self.out),
                             "--mock-stages", "osm", "damage", "connectivity"],
                            stage_overrides=self.overrides)
        self.assertEqual(code, 0)
        self.assertEqual(analyze.main(["--aoi", str(aoi_file), "--date", "2999-01-01",
                                       "--out", str(self.out)]), 1)
        self.assertEqual(analyze.main(["--aoi", str(Path(self.tmp.name) / "nope.json"),
                                       "--date", EVENT]), 1)


@unittest.skipUnless(rasterio_available(), "rasterio not installed")
class FullMockWithRastersTests(unittest.TestCase):
    """`--mock` end to end with real GeoTIFFs and the real raster statistics."""

    def test_full_mock_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = mocks.TEMPLATE_DIR
            mocks.TEMPLATE_DIR = Path(tmp) / "template"
            try:
                r = analyze_flood(AOI_T, EVENT, out_root=Path(tmp) / "out", mock=True)
            finally:
                mocks.TEMPLATE_DIR = old
            self.assertTrue(r.ok, r.error)
            self.assertGreater(r.summary.flood_area_km2, 0)
            self.assertTrue(0 < r.summary.mean_confidence <= 1)


if __name__ == "__main__":
    unittest.main()
