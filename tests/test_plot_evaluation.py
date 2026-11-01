"""Standard-library policy-report tests; rendering is optional when matplotlib is installed."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_evaluation.py"
SPEC = importlib.util.spec_from_file_location("plot_evaluation", SCRIPT)
plot_evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_evaluation)


def measurement(seed=0, iteration=40000, clips=2000, failed=500, mean=0.65):
    return {
        "seed": seed,
        "iteration": iteration,
        "checkpoint_sha256": hashlib.sha256(f"{seed}:{iteration}".encode()).hexdigest(),
        "summary": {
            "clips": clips,
            "failed": failed,
            "completion_rate": 1 - failed / clips,
            "mean_norm_return": mean,
            "step_weighted_norm_return": mean + 0.02,
        },
    }


def report_fixture():
    return {
        "schema_version": 1,
        "updated_at": "2026-10-31T20:15:00Z",
        "status": "running",
        "protocol": plot_evaluation.PROTOCOL,
        "baseline_commit": "1234567",
        "subset": {"clips": 2000, "sha256": "a" * 64},
        "curve": [],
        "full": [],
    }


class ValidationTests(unittest.TestCase):
    def test_all_report_states_keep_completed_measurements(self):
        for status in ("running", "completed", "failed"):
            with self.subTest(status=status):
                report = report_fixture()
                report["status"] = status
                report["curve"] = [measurement()]
                self.assertIsNone(plot_evaluation.validate_report(report))
                self.assertEqual(plot_evaluation.prepare_series(report)[0]["iterations"], [40000])

    def test_invalid_top_level_fields(self):
        invalid = {
            "schema_version": (None, True, 1.0, 2),
            "updated_at": (None, "yesterday", "2026-10-31", "2026-10-31T20:15:00",
                           "2026-11-01T04:15:00+08:00"),
            "status": (None, "pending", [], {}),
            "protocol": (None, "held-out evaluation"),
            "baseline_commit": (None, 123, " "),
            "subset": (None, [], {}),
            "curve": (None, {}, "pending"),
            "full": (None, {}, "pending"),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    report = report_fixture()
                    report[field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        plot_evaluation.validate_report(report)

    def test_missing_required_fields_and_nonobject_reports(self):
        for field in report_fixture():
            with self.subTest(field=field):
                report = report_fixture()
                del report[field]
                with self.assertRaises(ValueError):
                    plot_evaluation.validate_report(report)
        for value in (None, [], 1, "report"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "object"):
                plot_evaluation.validate_report(value)

    def test_subset_identity_and_count(self):
        for field, value in (("clips", 1999), ("clips", True), ("sha256", "z" * 64),
                             ("sha256", "abc"), ("sha256", None)):
            with self.subTest(field=field, value=value):
                report = report_fixture()
                report["subset"][field] = value
                with self.assertRaisesRegex(ValueError, f"subset.{field}"):
                    plot_evaluation.validate_report(report)

    def test_invalid_entry_fields(self):
        invalid = {
            "seed": (-1, 2, True, 0.0, None),
            "iteration": (39999, 320001, True, 40000.0, None),
            "checkpoint_sha256": (None, "", "z" * 64),
            "summary": (None, [], {}),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    report = report_fixture()
                    report["curve"] = [measurement()]
                    report["curve"][0][field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        plot_evaluation.validate_report(report)
        for collection in ("curve", "full"):
            report = report_fixture()
            report[collection] = [None]
            with self.subTest(collection=collection), self.assertRaises(ValueError):
                plot_evaluation.validate_report(report)

    def test_partial_and_nonfinite_summaries_are_rejected(self):
        invalid = {
            "clips": (1999, 2001, 129491, True, 2000.0, None),
            "failed": (-1, 2001, True, 1.5, None),
            "completion_rate": (-0.1, 1.1, 0.25, True, float("nan"), float("inf"), None),
            "mean_norm_return": (True, float("nan"), float("inf"), float("-inf"), None, "0.5"),
            "step_weighted_norm_return": (False, float("nan"), float("inf"), None),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    report = report_fixture()
                    report["curve"] = [measurement()]
                    report["curve"][0]["summary"][field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        plot_evaluation.validate_report(report)

    def test_every_summary_metric_is_required(self):
        for field in measurement()["summary"]:
            with self.subTest(field=field):
                report = report_fixture()
                entry = measurement()
                del entry["summary"][field]
                report["curve"] = [entry]
                with self.assertRaisesRegex(ValueError, field):
                    plot_evaluation.validate_report(report)

    def test_duplicates_rejected_in_each_collection(self):
        for collection, clips in (("curve", 2000), ("full", 129491)):
            report = report_fixture()
            entry = measurement(iteration=320000, clips=clips)
            report[collection] = [entry, copy.deepcopy(entry)]
            with self.subTest(collection=collection), self.assertRaisesRegex(ValueError, "duplicates"):
                plot_evaluation.validate_report(report)

    def test_full_requires_final_iteration_and_whole_corpus(self):
        for iteration, clips in ((280000, 129491), (320000, 2000), (320000, 129490)):
            report = report_fixture()
            report["full"] = [measurement(iteration=iteration, clips=clips)]
            with self.subTest(iteration=iteration, clips=clips), self.assertRaises(ValueError):
                plot_evaluation.validate_report(report)

    def test_return_is_not_a_probability_and_completion_can_be_zero(self):
        report = report_fixture()
        report["updated_at"] = "2026-10-31T20:15:00+00:00"
        report["curve"] = [measurement(failed=2000, mean=-0.2),
                           measurement(seed=1, failed=0, mean=1.2)]
        series = plot_evaluation.prepare_series(report)
        self.assertEqual(series[0]["completion_rate_percent"], [0.0])
        self.assertEqual(series[1]["completion_rate_percent"], [100.0])
        self.assertEqual(series[0]["mean_norm_return"], [-0.2])
        self.assertEqual(series[1]["mean_norm_return"], [1.2])


class SeriesTests(unittest.TestCase):
    def test_sorted_aligned_measured_points_only_without_mutation(self):
        report = report_fixture()
        report["curve"] = [measurement(1, 120000, mean=0.8), measurement(0, 320000, mean=0.9),
                           measurement(0, 40000, mean=0.3)]
        report["full"] = [measurement(0, 320000, clips=129491, mean=0.77)]
        before = copy.deepcopy(report)
        series = plot_evaluation.prepare_series(report)
        self.assertEqual(report, before)
        self.assertEqual([entry["seed"] for entry in series], [0, 1])
        self.assertEqual(series[0]["iterations"], [40000, 320000])
        self.assertEqual(series[1]["iterations"], [120000])
        self.assertEqual(series[0]["mean_norm_return"], [0.3, 0.9])
        self.assertEqual(series[0]["completion_rate_percent"], [75.0, 75.0])
        self.assertEqual(series[0]["clips"], [2000, 2000])
        self.assertEqual(series[0]["failed"], [500, 500])
        self.assertEqual(series[0]["checkpoint_sha256"],
                         [report["curve"][2]["checkpoint_sha256"],
                          report["curve"][1]["checkpoint_sha256"]])
        self.assertEqual(series[0]["step_weighted_norm_return"], [0.32, 0.92])

    def test_no_unmeasured_seed_or_full_point_on_curve(self):
        report = report_fixture()
        report["full"] = [measurement(seed=0, iteration=320000, clips=129491)]
        self.assertEqual(plot_evaluation.prepare_series(report), [])
        report["curve"] = [measurement(seed=1)]
        series = plot_evaluation.prepare_series(report)
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0]["seed"], 1)

    def test_empty_report_has_no_series(self):
        self.assertEqual(plot_evaluation.prepare_series(report_fixture()), [])


class ArtifactTests(unittest.TestCase):
    def test_summary_provenance_exact_source_bytes_and_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = report_fixture()
            report["curve"] = [measurement()]
            report["full"] = [measurement(1, 320000, clips=129491)]
            source = (json.dumps(report, indent=4) + "\n\n").encode()
            report_path = root / "evaluation_report.json"
            report_path.write_bytes(source)
            output = root / "plots"
            with mock.patch.object(plot_evaluation, "render_plots", return_value=[]) as renderer:
                paths = plot_evaluation.generate_artifacts(report_path, output)
            self.assertEqual(paths, [output.resolve() / plot_evaluation.SUMMARY_NAME])
            summary = json.loads(paths[0].read_text())
            self.assertEqual(summary["report_source_sha256"], hashlib.sha256(source).hexdigest())
            self.assertEqual(summary["report_source"], str(report_path.resolve()))
            self.assertEqual(
                summary["report_updated_at_asia_shanghai"], "2026-11-01T04:15:00+08:00"
            )
            self.assertEqual(summary["series"], plot_evaluation.prepare_series(report))
            self.assertEqual(summary["full"], report["full"])
            self.assertEqual(summary["subset"], report["subset"])
            self.assertEqual(summary["baseline_commit"], report["baseline_commit"])
            self.assertEqual(summary["protocol"], plot_evaluation.PROTOCOL)
            self.assertEqual(summary, renderer.call_args.args[0])
            self.assertEqual(report_path.read_bytes(), source)

    def test_invalid_report_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "report.json"
            output = root / "plots"
            for content in ("{", '{"schema_version": 2}', "null"):
                source.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    plot_evaluation.generate_artifacts(source, output)
                self.assertFalse(output.exists())

    def test_output_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / plot_evaluation.SUMMARY_NAME
            contents = json.dumps(report_fixture())
            source.write_text(contents)
            with self.assertRaisesRegex(ValueError, "generated artifact path"):
                plot_evaluation.generate_artifacts(source, root)
            self.assertEqual(source.read_text(), contents)

    def test_validation_and_preparation_import_no_plotting_dependencies(self):
        program = (
            "import runpy, sys; "
            f"namespace = runpy.run_path({str(SCRIPT)!r}); "
            f"namespace['prepare_series']({report_fixture()!r}); "
            "assert 'matplotlib' not in sys.modules; assert 'numpy' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-S", "-c", program], capture_output=True, text=True, check=False
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional matplotlib is not installed")
class RenderingTests(unittest.TestCase):
    def test_plotted_coordinates_are_measured_markers_not_interpolated_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.dict(os.environ, {"MPLCONFIGDIR": str(root / "matplotlib")}):
                import matplotlib

                matplotlib.use("Agg", force=True)
                from matplotlib.figure import Figure

                captured = []

                def capture_figure(figure, destination, **kwargs):
                    captured.append([
                        [(list(line.get_xdata()), list(line.get_ydata()), line.get_linestyle(),
                          line.get_color(), line.get_marker()) for line in axis.lines]
                        for axis in figure.axes
                    ])

                report = report_fixture()
                report["curve"] = [measurement(0, 40000, mean=0.2),
                                   measurement(0, 320000, mean=0.9),
                                   measurement(1, 120000, mean=0.5)]
                source = root / "report.json"
                source.write_text(json.dumps(report))
                with mock.patch.object(Figure, "savefig", capture_figure):
                    plot_evaluation.generate_artifacts(source, root / "plots")
                self.assertEqual(len(captured), 2)
                self.assertEqual(captured[0], captured[1])
                returns, completions = captured[0]
                self.assertEqual(returns[0][:2], ([40000, 320000], [0.2, 0.9]))
                self.assertEqual(returns[1][:2], ([120000], [0.5]))
                self.assertEqual(completions[0][:2], ([40000, 320000], [75.0, 75.0]))
                self.assertNotEqual(returns[0][3], returns[1][3])
                self.assertNotEqual(returns[0][4], returns[1][4])
                for plotted in returns + completions:
                    self.assertEqual(plotted[2], "None")
                captured.clear()
                source.write_text(json.dumps(report_fixture()))
                with mock.patch.object(Figure, "savefig", capture_figure):
                    plot_evaluation.generate_artifacts(source, root / "plots")
                self.assertEqual(captured, [[[], []], [[], []]])

    def test_cli_renders_measured_empty_and_full_only_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "plots"
            output.mkdir()
            legacy = [output / f"evaluation_curves.{suffix}" for suffix in ("png", "svg", "json")]
            for destination in legacy:
                destination.write_bytes(b"untouched online diagnostic")
            environment = dict(os.environ, MPLCONFIGDIR=str(root / "matplotlib"),
                               PYTHONDONTWRITEBYTECODE="1")
            report_path = root / "evaluation_report.json"
            for scenario in ("measured", "empty", "full-only"):
                with self.subTest(scenario=scenario):
                    report = report_fixture()
                    if scenario == "measured":
                        report["curve"] = [measurement(0, 40000), measurement(0, 320000),
                                           measurement(1, 80000, mean=0.7)]
                    if scenario != "empty":
                        report["full"] = [measurement(0, 320000, clips=129491),
                                          measurement(1, 320000, clips=129491, mean=0.7)]
                    report_path.write_text(json.dumps(report))
                    result = subprocess.run(
                        [sys.executable, "-B", str(SCRIPT), "--report", str(report_path),
                         "--output-dir", str(output)],
                        env=environment, capture_output=True, text=True, check=False, timeout=90,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    png = output / "policy_evaluation_curves.png"
                    self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                    self.assertGreater(png.stat().st_size, 10000)
                    svg = (output / "policy_evaluation_curves.svg").read_text()
                    for label in ("NOT held-out generalization", "NOT success", "Asia/Shanghai",
                                  "no smoothing or interpolation", "2,000 clips"):
                        self.assertIn(label, svg)
                    if scenario != "measured":
                        self.assertIn("No completed fixed-subset evaluations yet", svg)
                        self.assertNotIn("measured)", svg)
                    else:
                        self.assertNotIn("No completed fixed-subset evaluations yet", svg)
                        self.assertIn("Seed 0 (2 measured)", svg)
                        self.assertIn("Seed 1 (1 measured)", svg)
                    self.assertEqual((output / plot_evaluation.FULL_PLOT_NAME).exists(),
                                     scenario != "empty")
                    for destination in legacy:
                        self.assertEqual(destination.read_bytes(), b"untouched online diagnostic")
                    summary = json.loads((output / plot_evaluation.SUMMARY_NAME).read_text())
                    self.assertEqual(summary["series"], plot_evaluation.prepare_series(report))


if __name__ == "__main__":
    unittest.main()
