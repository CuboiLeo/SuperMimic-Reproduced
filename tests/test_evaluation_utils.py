"""CPU-only strict evaluation tests; run with unittest, without site packages."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import make_dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from scripts import eval_manifest as worker
from scripts import evaluation_utils as utils


def clip(name: str = "walk", frames: int = 10) -> dict:
    return {"name": name, "frames": frames, "sha256": "a" * 64, "bytes": 10}


def row(name: str = "walk", frames: int = 10, steps: int = 12) -> dict:
    return {
        "name": name,
        "frames": frames,
        "steps": steps,
        "norm_return": 0.75,
        "failed": False,
        "fail_step": -1,
        "terms": {"tracking": 0.02, "penalty": -0.01},
    }


def job() -> dict:
    return {
        "schema_version": 1,
        "job_id": "seed7-iteration4000-shard0",
        "kind": "full",
        "seed": 7,
        "checkpoint_iteration": 4000,
        "checkpoint": "checkpoint.pt",
        "checkpoint_sha256": "b" * 64,
        "npz_dir": "npz",
        "num_envs": 32,
        "output": "result.json",
        "clips": [clip()],
    }


def checkpoint() -> dict:
    return {
        "iteration": 4000,
        "config": {
            "seed": 7,
            "env": {"future_frames": 10, "anchor_obs": False, "compact_library": True},
            "policy": {"history": 10, "goal_dim": 680},
        },
    }


class ResultTests(unittest.TestCase):
    def test_exact_order_and_optional_aggregate_order(self):
        expected = [clip("walk"), clip("run")]
        rows = [row("walk"), row("run")]
        self.assertIs(utils.validate_results(rows, expected), rows)
        with self.assertRaisesRegex(ValueError, "order"):
            utils.validate_results(rows[::-1], expected)
        self.assertEqual(
            utils.validate_results(rows[::-1], expected, require_order=False), rows[::-1]
        )
        self.assertEqual(utils.aggregate_results(rows[::-1], expected)["clips"], rows)

    def test_steps_can_exceed_original_frames(self):
        for steps in (1, 9, 10, 11, 12):
            with self.subTest(steps=steps):
                utils.validate_results([row(steps=steps)], [clip()])

    def test_invalid_clip_names(self):
        invalid = ("", ".", "..", "../walk", "/walk", "walk/run", "walk\\run", "C:walk",
                   "walk.npz", "walk.NPZ", "walk\x00", "walk\n", " walk", "walk ", "walk.")
        for name in invalid:
            with self.subTest(name=name), self.assertRaises(ValueError):
                utils.validate_results([row(name)], [clip(name)])
        for name in ("walk1_subject1", "dance-02", "subject.walk", "clip with spaces"):
            self.assertEqual(utils.validate_clip_name(name), name)

    def test_coverage_duplicate_missing_extra_and_empty(self):
        cases = [
            ([row(), row()], [clip()]),
            ([row()], [clip(), clip()]),
            ([row()], [clip(), clip("run")]),
            ([row(), row("run")], [clip()]),
            ([], [clip()]),
            ([row()], []),
        ]
        for rows, expected in cases:
            with self.subTest(rows=rows, expected=expected), self.assertRaises(ValueError):
                utils.validate_results(rows, expected, require_order=False)

    def test_missing_and_invalid_result_fields(self):
        for field in row():
            bad = row()
            del bad[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                utils.validate_results([bad], [clip()])
        invalid = {
            "frames": (0, -1, True, 10.0, 11, None),
            "steps": (0, -1, True, 2.0, None),
            "failed": (0, 1, "false", None),
            "fail_step": (0, 12, True, -1.0, None),
            "norm_return": (math.nan, math.inf, -math.inf, None, True, "0.5", 10**400),
            "terms": (None, {}, [], {"tracking": None}, {"tracking": math.inf}),
        }
        for field, values in invalid.items():
            for value in values:
                bad = row()
                bad[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    utils.validate_results([bad], [clip()])

    def test_failure_step_consistency(self):
        failed = dict(row(steps=3), failed=True, fail_step=3)
        utils.validate_results([failed], [clip()])
        for step in (-1, 0, 2, 4):
            with self.subTest(step=step), self.assertRaises(ValueError):
                utils.validate_results([dict(failed, fail_step=step)], [clip()])

    def test_term_sets_and_values_are_strict(self):
        mismatched = (
            {"tracking": 0.1},
            {"other": 0.1},
            {"tracking": 0.1, "penalty": 0, "extra": 1},
        )
        for terms in mismatched:
            with self.subTest(terms=terms), self.assertRaises(ValueError):
                utils.validate_results(
                    [row(), dict(row("run"), terms=terms)], [clip(), clip("run")]
                )
        for value in (math.nan, math.inf, -math.inf, True, None, "0.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                utils.summarize_results([dict(row(), terms={"tracking": value})])

    def test_unequal_shards_are_aggregated_from_rows(self):
        short = dict(row("short", frames=2, steps=2), norm_return=0.0,
                     terms={"tracking": 0.0}, failed=True, fail_step=2)
        long = dict(row("long", frames=8, steps=8), norm_return=1.0, terms={"tracking": 0.02})
        middle = dict(row("middle", frames=2, steps=2), norm_return=0.5, terms={"tracking": 0.01})
        shards = [
            {"summary": {"mean_norm_return": -999}, "clips": [short]},
            {"summary": {"mean_norm_return": 999}, "clips": [long, middle]},
        ]
        expected = [clip("short", 2), clip("middle", 2), clip("long", 8)]
        rows = [result for shard in shards for result in shard["clips"]]
        summary = utils.aggregate_results(rows, expected)["summary"]
        self.assertEqual(summary["clips"], 3)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["completed"], 2)
        self.assertAlmostEqual(summary["completion_rate"], 2 / 3)
        self.assertEqual(summary["total_frames"], 12)
        self.assertEqual(summary["total_steps"], 12)
        self.assertAlmostEqual(summary["mean_norm_return"], 0.5)
        self.assertAlmostEqual(summary["step_weighted_norm_return"], 0.75)
        self.assertAlmostEqual(summary["terms"]["tracking"], 0.015)
        self.assertAlmostEqual(summary["mean_terms"]["tracking"], 0.01)

    def test_summary_does_not_accept_invalid_rows(self):
        for rows in ([], [row(), row()], [dict(row(), norm_return=math.nan)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                utils.summarize_results(rows)


class CheckpointTests(unittest.TestCase):
    def test_canonical_checkpoint_and_optional_historical_metadata(self):
        options = utils.validate_checkpoint(checkpoint(), job())
        self.assertEqual(options, {"future_frames": 10, "anchor_obs": False})
        old = checkpoint()
        del old["iteration"]
        del old["config"]["seed"]
        self.assertEqual(utils.validate_checkpoint(old, job()), options)

    def test_seed_iteration_and_history_mismatches(self):
        paths = (("iteration",), ("config", "seed"), ("config", "policy", "history"),
                 ("config", "env", "history_length"), ("config", "policy", "goal_dim"))
        for path in paths:
            bad = checkpoint()
            target = bad
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = 2
            with self.subTest(path=path), self.assertRaises(ValueError):
                utils.validate_checkpoint(bad, job())

    def test_future_frames_and_anchor_configuration(self):
        custom = checkpoint()
        custom["config"]["env"].update(future_frames=5, anchor_obs=True)
        custom["config"]["policy"]["goal_dim"] = 349
        self.assertEqual(utils.validate_checkpoint(custom, job()),
                         {"future_frames": 5, "anchor_obs": True})
        custom["config"]["env"]["anchor_obs"] = 1
        with self.assertRaises(ValueError):
            utils.validate_checkpoint(custom, job())


class FileFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.job = job()
        (self.root / "npz").mkdir()
        self.npz = self.root / "npz" / "walk.npz"
        self.npz.write_bytes(b"immutable NPZ fixture")
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"immutable checkpoint fixture")
        self.job["checkpoint_sha256"] = utils.sha256_file(self.checkpoint)
        self.job["clips"][0].update(
            sha256=utils.sha256_file(self.npz), bytes=self.npz.stat().st_size
        )
        self.job_path = self.root / "job.json"
        self.job_path.write_text(json.dumps(self.job), encoding="utf-8")
        self.output = self.root / "result.json"


class FileTests(FileFixture):
    def test_load_job_digest_and_relative_paths(self):
        loaded, digest = utils.load_job(self.job_path)
        self.assertEqual(digest, hashlib.sha256(self.job_path.read_bytes()).hexdigest())
        self.assertEqual(loaded["checkpoint"], str(self.checkpoint))
        self.assertEqual(utils.verify_inputs(loaded), [str(self.npz)])

    def test_job_schema_and_missing_fields(self):
        for field in set(self.job) - {"kind"}:
            bad = copy.deepcopy(self.job)
            del bad[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                utils.validate_job(bad)
        invalid = {"schema_version": [2, True], "seed": [-1, True], "num_envs": [0, True],
                   "checkpoint_iteration": [None, 0.5], "checkpoint_sha256": ["abc", "g" * 64],
                   "clips": [[], [clip(), clip()]], "kind": [None, ""]}
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    utils.validate_job(dict(self.job, **{field: value}))

    def test_duplicate_keys_and_nonstandard_json(self):
        texts = (
            '{"schema_version":1,"schema_version":1}', '{"value":NaN}', '{"value":Infinity}'
        )
        for text in texts:
            self.job_path.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(ValueError):
                utils.load_job(self.job_path)

    def test_changed_size_and_same_size_digest(self):
        loaded, _ = utils.load_job(self.job_path)
        original = self.npz.read_bytes()
        for data in (original + b"x", b"x" * len(original)):
            self.npz.write_bytes(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                utils.verify_inputs(loaded)

    def test_missing_directory_and_symlink_inputs(self):
        loaded, _ = utils.load_job(self.job_path)
        self.npz.unlink()
        with self.assertRaises(FileNotFoundError):
            utils.verify_inputs(loaded)
        self.npz.mkdir()
        with self.assertRaises((ValueError, OSError)):
            utils.verify_inputs(loaded)
        self.npz.rmdir()
        self.npz.symlink_to(self.checkpoint)
        with self.assertRaises((ValueError, OSError)):
            utils.verify_inputs(loaded)

    def test_checkpoint_hash_mismatch(self):
        loaded, _ = utils.load_job(self.job_path)
        self.checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            utils.verify_inputs(loaded)

    def test_atomic_publication_and_existing_output(self):
        payload = {"clips": [row()], "summary": utils.summarize_results([row()])}
        utils.atomic_write_json(self.output, payload)
        original = self.output.read_bytes()
        self.assertEqual(json.loads(original), payload)
        with self.assertRaises(FileExistsError):
            utils.atomic_write_json(self.output, {"different": True})
        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_atomic_failure_and_publication_race(self):
        original_link = os.link

        def concurrent_link(source, destination):
            Path(destination).write_bytes(b"other worker")
            original_link(source, destination)

        with patch.object(utils.os, "link", side_effect=concurrent_link), self.assertRaises(FileExistsError):
            utils.atomic_write_json(self.output, {"ours": True})
        self.assertEqual(self.output.read_bytes(), b"other worker")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])
        self.output.unlink()
        with patch.object(utils.os, "link", side_effect=OSError("publication failure")), self.assertRaises(OSError):
            utils.atomic_write_json(self.output, {"ours": True})
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_nonfinite_json_and_dangling_output_symlink(self):
        with self.assertRaises(ValueError):
            utils.atomic_write_json(self.output, {"bad": math.nan})
        self.assertFalse(self.output.exists())
        self.output.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            utils.atomic_write_json(self.output, {"valid": True})


class WorkerTests(FileFixture):
    def runtime(self):
        saved = checkpoint()
        fields = [(key, object) for key in row()]
        result_type = make_dataclass("Result", fields)
        cuda = SimpleNamespace(is_available=Mock(return_value=True),
                               device_count=Mock(return_value=2), set_device=Mock())
        fake_torch = ModuleType("torch")
        fake_torch.cuda = cuda
        fake_torch.load = Mock(return_value=saved)
        fake_envs = ModuleType("supermimic.envs")
        fake_envs.SuperMimicEnvCfgArgs = lambda **kwargs: SimpleNamespace(history_length=10)
        fake_benchmark = ModuleType("supermimic.eval.benchmark")
        fake_benchmark.load_policy_from_checkpoint = Mock(
            return_value=(object(), saved["config"])
        )
        fake_benchmark.run_benchmark = Mock(return_value=[result_type(**row())])
        modules = {
            "torch": fake_torch,
            "supermimic.envs": fake_envs,
            "supermimic.eval.benchmark": fake_benchmark,
        }
        self.enterContext(patch.dict(sys.modules, modules))
        return fake_torch, fake_benchmark, result_type

    def test_worker_uses_checkpoint_runtime_config_with_paths_and_tuples(self):
        fake_torch, benchmark, _ = self.runtime()
        config = fake_torch.load.return_value["config"]
        config["log_root"] = Path("logs")
        config["policy"]["encoder_hidden"] = (2048, 1024, 512, 256)
        benchmark.load_policy_from_checkpoint.return_value = (object(), copy.deepcopy(config))
        stale_config = copy.deepcopy(config)
        stale_config["env"]["compact_library"] = False
        (self.root / "config.json").write_text(
            json.dumps(stale_config, default=str), encoding="utf-8"
        )
        payload = worker.run_job(self.job_path, "cuda:0")
        self.assertEqual(payload["summary"]["clips"], 1)
        self.assertTrue(config["env"]["compact_library"])
        self.assertNotIn("compact_library", benchmark.run_benchmark.call_args.kwargs)

    def test_worker_passes_options_and_publishes_complete_schema(self):
        fake_torch, benchmark, _ = self.runtime()
        payload = worker.run_job(self.job_path, "cuda:1")
        self.assertEqual(json.loads(self.output.read_bytes()), payload)
        self.assertEqual(
            payload["input_manifest_sha256"], utils.sha256_file(self.job_path)
        )
        self.assertEqual(payload["protocol"], utils.PROTOCOL)
        self.assertEqual(payload["kind"], "full")
        metadata = ("schema_version", "job_id", "seed", "checkpoint_iteration", "checkpoint_sha256")
        for key in metadata:
            self.assertEqual(payload[key], self.job[key])
        self.assertLessEqual(payload["started_at"], payload["finished_at"])
        self.assertGreaterEqual(payload["elapsed_seconds"], 0)
        fake_torch.load.assert_called_once_with(
            str(self.checkpoint), map_location="cpu", weights_only=False
        )
        fake_torch.cuda.set_device.assert_called_once_with("cuda:1")
        self.assertEqual(benchmark.run_benchmark.call_args.args[1], [str(self.npz)])
        self.assertEqual(
            benchmark.run_benchmark.call_args.kwargs,
            {"device": "cuda:1", "num_envs": 32, "future_frames": 10, "anchor_obs": False},
        )

    def test_worker_rejects_invalid_results_without_publishing(self):
        _, benchmark, result_type = self.runtime()
        for bad in (dict(row(), frames=11), dict(row(), name="other"), dict(row(), terms=None)):
            benchmark.run_benchmark.return_value = [result_type(**bad)]
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                worker.run_job(self.job_path, "cuda:0")
            self.assertFalse(self.output.exists())

    def test_worker_rechecks_inputs_after_evaluation(self):
        _, benchmark, _ = self.runtime()
        original_results = benchmark.run_benchmark.return_value

        def mutate(*args, **kwargs):
            self.npz.write_bytes(b"changed during evaluation")
            return original_results

        benchmark.run_benchmark.side_effect = mutate
        with self.assertRaises(ValueError):
            worker.run_job(self.job_path, "cuda:0")
        self.assertFalse(self.output.exists())

    def test_worker_rechecks_job_manifest_after_evaluation(self):
        _, benchmark, _ = self.runtime()
        original_results = benchmark.run_benchmark.return_value

        def mutate(*args, **kwargs):
            self.job_path.write_text(json.dumps(self.job) + "\n", encoding="utf-8")
            return original_results

        benchmark.run_benchmark.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            worker.run_job(self.job_path, "cuda:0")
        self.assertFalse(self.output.exists())

    def test_worker_rejects_checkpoint_metadata_before_evaluation(self):
        fake_torch, benchmark, _ = self.runtime()
        fake_torch.load.return_value["iteration"] = 3999
        with self.assertRaisesRegex(ValueError, "iteration"):
            worker.run_job(self.job_path, "cuda:0")
        benchmark.load_policy_from_checkpoint.assert_not_called()
        benchmark.run_benchmark.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_worker_rejects_reordered_evaluator_results(self):
        _, benchmark, result_type = self.runtime()
        second = self.root / "npz" / "aaa.npz"
        second.write_bytes(b"second NPZ fixture")
        self.job["clips"].append(
            dict(clip("aaa"), sha256=utils.sha256_file(second), bytes=second.stat().st_size)
        )
        self.job_path.write_text(json.dumps(self.job), encoding="utf-8")
        benchmark.run_benchmark.return_value = [
            result_type(**row("aaa")), result_type(**row("walk"))
        ]
        with self.assertRaisesRegex(ValueError, "order"):
            worker.run_job(self.job_path, "cuda:0")
        self.assertEqual(benchmark.run_benchmark.call_args.args[1], [str(self.npz), str(second)])
        self.assertFalse(self.output.exists())

    def test_worker_rejects_existing_output_before_runtime_loading(self):
        self.output.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            worker.run_job(self.job_path, "cuda:0")
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_worker_requires_explicit_available_cuda_device(self):
        for device in ("cpu", "cuda", "mps", "cuda:-1", "cuda:0,1"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                worker.run_job(self.job_path, device)
        self.runtime()
        with self.assertRaisesRegex(ValueError, "unavailable"):
            worker.run_job(self.job_path, "cuda:2")
        self.assertFalse(self.output.exists())


class DependencyTests(unittest.TestCase):
    def test_imports_and_cli_help_without_site_packages(self):
        root = Path(__file__).resolve().parents[1]
        code = ("import sys; from scripts import evaluation_utils, eval_manifest; "
                "assert not any(name in sys.modules for name in ('torch', 'mjlab', 'numpy'))")
        subprocess.run(
            [sys.executable, "-S", "-c", code], cwd=root, check=True, capture_output=True
        )
        completed = subprocess.run([sys.executable, "-S", "scripts/eval_manifest.py", "--help"],
                                   cwd=root, check=True, capture_output=True, text=True)
        self.assertIn("--job", completed.stdout)
        self.assertIn("--device", completed.stdout)


if __name__ == "__main__":
    unittest.main()
