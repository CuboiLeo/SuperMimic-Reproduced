"""CPU-only tests for strict suite aggregation and GPU admission checks."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("run_evaluation_suite", SCRIPTS / "run_evaluation_suite.py")
suite_tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite_tools)


class SuiteTests(unittest.TestCase):
    def test_cleanup_continues_when_first_process_group_disappears(self):
        first = {"process": mock.Mock(pid=101), "stream": mock.Mock()}
        second = {"process": mock.Mock(pid=102), "stream": mock.Mock()}
        first["process"].poll.return_value = None
        second["process"].poll.return_value = None
        with mock.patch.object(suite_tools.os, "killpg", side_effect=[ProcessLookupError(), None]) as kill:
            suite_tools.terminate_processes({0: first, 4: second})
        self.assertEqual(kill.call_count, 2)
        for item in (first, second):
            item["process"].wait.assert_called_once_with(timeout=15)
            item["stream"].close.assert_called_once()

    def test_aggregation_rejects_mixed_checkpoints(self):
        plan = {"protocol": "training corpus", "baseline_commit": "abc", "subset": {"clips": 2},
                "jobs": [{"job_id": name, "kind": "full", "seed": 0, "iteration": 320000}
                         for name in ("first", "second")]}
        completed = {name: {"checkpoint_sha256": name} for name in ("first", "second")}
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "mixed checkpoints"):
            suite_tools.publish_report(Path(temporary), plan, completed, "running")

    def test_fingerprint_rejects_runtime_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "python"
            executable.write_bytes(b"python executable")
            plan = {"python": str(executable), "baseline_source": str(root), "baseline_commit": "abc"}
            with mock.patch.object(suite_tools.subprocess, "check_output", side_effect=["abc\n", "", '{"version":1}']):
                original, _ = suite_tools.freeze_evaluator(root, plan, root)
            self.assertEqual(len(original), 64)
            with (
                mock.patch.object(suite_tools.subprocess, "check_output", side_effect=["abc\n", "", '{"version":2}']),
                self.assertRaisesRegex(ValueError, "Evaluator/runtime changed"),
            ):
                suite_tools.freeze_evaluator(root, plan, root)

    def test_gpu_admission_refuses_an_existing_process(self):
        with (
            mock.patch.object(suite_tools.subprocess, "check_output", side_effect=["0, GPU-a\n", "GPU-a, 123\n"]),
            self.assertRaisesRegex(ValueError, "already has a compute process"),
        ):
            suite_tools.available_gpus([0])

    def test_gpu_admission_allows_idle_requested_devices(self):
        with mock.patch.object(suite_tools.subprocess, "check_output", side_effect=["0, GPU-a\n1, GPU-b\n", "GPU-b, 123\n"]):
            self.assertEqual(suite_tools.available_gpus([0]), {0: "GPU-a", 1: "GPU-b"})

    def test_gpu_admission_checks_missing_device(self):
        with (
            mock.patch.object(suite_tools.subprocess, "check_output", return_value="0, GPU-a\n"),
            self.assertRaisesRegex(ValueError, "unavailable"),
        ):
            suite_tools.available_gpus([4])

    def test_atomic_json_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            suite_tools.write_json(path, {"status": "running"})
            suite_tools.write_json(path, {"status": "completed"})
            self.assertEqual(json.loads(path.read_text()), {"status": "completed"})
            self.assertFalse(path.with_name("status.json.tmp").exists())

    def test_incomplete_group_does_not_become_a_curve_point(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = {"protocol": "training corpus", "baseline_commit": "abc", "subset": {"clips": 2},
                    "jobs": [{"job_id": "part1", "kind": "curve", "seed": 0, "iteration": 40000},
                             {"job_id": "part2", "kind": "curve", "seed": 0, "iteration": 40000}]}
            report = suite_tools.publish_report(root, plan, {"part1": {}}, "running")
            self.assertEqual(report["curve"], [])
            self.assertEqual(report["completed_jobs"], 1)
            self.assertEqual(report["status"], "running")

    def test_changed_job_manifest_is_rejected_before_output_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "job.json"
            path.write_text("{}")
            job = {"path": str(path), "sha256": "incorrect", "job_id": "bad"}
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                suite_tools.validate_output(job, {})


if __name__ == "__main__":
    unittest.main()
