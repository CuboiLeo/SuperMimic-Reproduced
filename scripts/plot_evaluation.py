"""Plot measured policy evaluations, independently of online training diagnostics.

Usage: python scripts/plot_evaluation.py --report evaluation_report.json --output-dir plots
Only complete, fixed-subset measurements belong in the curve. Full-corpus results
are preserved in the summary and, when present, shown in a separate comparison.
Validation and series preparation require only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PROTOCOL = "Deterministic training-corpus policy evaluation; NOT held-out generalization"
SUBSET_CLIPS = 2000
FULL_CLIPS = 129491
FIRST_ITERATION = 40000
FINAL_ITERATION = 320000
SEED_COLORS = {0: "#0072B2", 1: "#D55E00"}
SEED_MARKERS = {0: "o", 1: "s"}
SUMMARY_NAME = "policy_evaluation_summary.json"
FULL_PLOT_NAME = "policy_evaluation_full_final.png"


def _integer(value: object, field: str, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")


def _finite_number(value: object, field: str) -> None:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{field} must be a finite number")


def _sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256")


def _utc_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("updated_at must be an ISO timestamp in UTC")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("updated_at must be an ISO timestamp in UTC") from error
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError("updated_at must be an ISO timestamp in UTC")
    return timestamp


def validate_report(report: object) -> None:
    """Reject malformed, partial, duplicate, or incompatible measurements.

    A whole-subset entry includes every clip, including failed clips. Report-level
    running/failed status does not invalidate measurements already completed.
    Normalized return is finite but is not treated as a bounded success rate.
    """
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    _integer(report.get("schema_version"), "schema_version", 1, 1)
    _utc_timestamp(report.get("updated_at"))
    if report.get("status") not in ("running", "completed", "failed"):
        raise ValueError("status must be running, completed, or failed")
    if report.get("protocol") != PROTOCOL:
        raise ValueError(f"protocol must be {PROTOCOL!r}")
    if not isinstance(report.get("baseline_commit"), str) or not report["baseline_commit"].strip():
        raise ValueError("baseline_commit must be a nonempty string")
    subset = report.get("subset")
    if not isinstance(subset, dict):
        raise ValueError("subset must be an object")
    _integer(subset.get("clips"), "subset.clips", SUBSET_CLIPS, SUBSET_CLIPS)
    _sha256(subset.get("sha256"), "subset.sha256")

    for collection, expected_clips in (("curve", SUBSET_CLIPS), ("full", FULL_CLIPS)):
        entries = report.get(collection)
        if not isinstance(entries, list):
            raise ValueError(f"{collection} must be an array")
        seen = set()
        for index, entry in enumerate(entries):
            field = f"{collection}[{index}]"
            if not isinstance(entry, dict):
                raise ValueError(f"{field} must be an object")
            _integer(entry.get("seed"), f"{field}.seed", 0, 1)
            first = FIRST_ITERATION if collection == "curve" else FINAL_ITERATION
            _integer(entry.get("iteration"), f"{field}.iteration", first, FINAL_ITERATION)
            key = (entry["seed"], entry["iteration"])
            if key in seen:
                raise ValueError(f"{field} duplicates seed/iteration {key}")
            seen.add(key)
            _sha256(entry.get("checkpoint_sha256"), f"{field}.checkpoint_sha256")
            summary = entry.get("summary")
            if not isinstance(summary, dict):
                raise ValueError(f"{field}.summary must be a completed summary object")
            _integer(summary.get("clips"), f"{field}.summary.clips", expected_clips, expected_clips)
            _integer(summary.get("failed"), f"{field}.summary.failed", 0, expected_clips)
            for metric in ("completion_rate", "mean_norm_return", "step_weighted_norm_return"):
                _finite_number(summary.get(metric), f"{field}.summary.{metric}")
            completion = summary["completion_rate"]
            if not 0 <= completion <= 1:
                raise ValueError(f"{field}.summary.completion_rate must be in [0, 1]")
            expected_completion = 1 - summary["failed"] / expected_clips
            if not math.isclose(completion, expected_completion, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"{field}.summary.completion_rate disagrees with failed/clips")


def prepare_series(report: dict) -> list[dict]:
    """Return sorted, aligned arrays of actual subset measurements, without filling gaps."""
    validate_report(report)
    series = []
    for seed in sorted({entry["seed"] for entry in report["curve"]}):
        entries = sorted(
            (entry for entry in report["curve"] if entry["seed"] == seed),
            key=lambda entry: entry["iteration"],
        )
        measured = {
            "seed": seed,
            "iterations": [entry["iteration"] for entry in entries],
            "checkpoint_sha256": [entry["checkpoint_sha256"] for entry in entries],
        }
        for metric in (
            "clips", "failed", "completion_rate", "mean_norm_return", "step_weighted_norm_return"
        ):
            measured[metric] = [entry["summary"][metric] for entry in entries]
        measured["completion_rate_percent"] = [100 * rate for rate in measured["completion_rate"]]
        series.append(measured)
    return series


def _decorate_figure(figure, summary: dict, title: str) -> None:
    timestamp = datetime.fromisoformat(summary["report_updated_at_asia_shanghai"])
    figure.suptitle(f"{title}\n{PROTOCOL}", fontsize=13, fontweight="bold", y=0.98)
    figure.text(
        0.5, 0.855,
        f"Report updated: {timestamp:%Y-%m-%d %H:%M:%S} Asia/Shanghai (UTC+08:00)"
        f"  |  Status: {summary['status']}",
        ha="center", fontsize=10,
    )
    figure.text(
        0.5, 0.025,
        "Markers only; no smoothing or interpolation. Missing checkpoints have no values.\n"
        "Normalized return is NOT success; clip completion is reported separately.",
        ha="center", fontsize=10,
    )
    figure.subplots_adjust(left=0.08, right=0.975, bottom=0.2, top=0.76, wspace=0.28)


def _style_axes(axes) -> None:
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)


def render_plots(summary: dict, output_dir: Path) -> list[Path]:
    """Render subset PNG/SVG and an optional, separate full-corpus comparison."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    outputs = []
    with plt.rc_context({"font.size": 11, "axes.labelsize": 11, "svg.fonttype": "none"}):
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.8))
        try:
            _decorate_figure(figure, summary, "Policy evaluation | fixed subset: 2,000 clips")
            _style_axes(axes)
            for measured in summary["series"]:
                seed = measured["seed"]
                for axis, metric in zip(axes, ("mean_norm_return", "completion_rate_percent")):
                    axis.plot(
                        measured["iterations"], measured[metric], linestyle="none",
                        marker=SEED_MARKERS[seed], color=SEED_COLORS[seed], markersize=7,
                        markeredgecolor="white", markeredgewidth=0.7,
                        label=f"Seed {seed} ({len(measured['iterations'])} measured)",
                    )
            for axis in axes:
                axis.set_xlim(FIRST_ITERATION - 20000, FINAL_ITERATION + 20000)
                ticks = list(range(FIRST_ITERATION, FINAL_ITERATION + 1, 40000))
                axis.set_xticks(ticks, [f"{iteration // 1000}k" for iteration in ticks])
                axis.set_xlabel("Training iteration")
                if summary["series"]:
                    axis.legend(loc="best", frameon=True, fontsize=9)
                else:
                    axis.text(
                        0.5, 0.5, "No completed fixed-subset evaluations yet\n"
                        "No checkpoint values are available",
                        transform=axis.transAxes, ha="center", va="center", fontsize=11,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 8},
                    )
            axes[0].set_title("Fixed-subset mean normalized return")
            axes[0].set_ylabel("Mean normalized return (NOT success)")
            if not summary["series"]:
                axes[0].set_yticks([])
            axes[1].set_title("Fixed-subset clip completion")
            axes[1].set_ylabel("Clip completion rate (%)")
            axes[1].set_ylim(-2, 102)
            for suffix in ("png", "svg"):
                destination = output_dir / f"policy_evaluation_curves.{suffix}"
                figure.savefig(destination, dpi=200, facecolor="white")
                outputs.append(destination)
        finally:
            plt.close(figure)

        full_destination = output_dir / FULL_PLOT_NAME
        if summary["full"]:
            figure, axes = plt.subplots(1, 2, figsize=(13, 5.8))
            try:
                _decorate_figure(
                    figure, summary, "Final policy evaluation | full training corpus: 129,491 clips"
                )
                _style_axes(axes)
                labels = []
                for position, entry in enumerate(summary["full"]):
                    seed = entry["seed"]
                    result = entry["summary"]
                    labels.append(f"Seed {seed}\n{entry['iteration']:,} iterations")
                    values = (result["mean_norm_return"], 100 * result["completion_rate"])
                    for axis, value in zip(axes, values):
                        axis.plot(
                            position, value, linestyle="none", marker=SEED_MARKERS[seed],
                            color=SEED_COLORS[seed], markersize=9,
                        )
                        label = f"{value:.4f}" if axis is axes[0] else f"{value:.2f}%"
                        axis.annotate(
                            label, (position, value), xytext=(0, 12), textcoords="offset points",
                            ha="center", fontsize=10,
                        )
                        axis.text(
                            position, 0.04,
                            f"n={result['clips']:,}\nfailed={result['failed']:,}",
                            transform=axis.get_xaxis_transform(), ha="center", fontsize=9,
                        )
                for axis in axes:
                    axis.set_xticks(range(len(labels)), labels)
                    axis.set_xlim(-0.6, len(labels) - 0.4)
                    axis.margins(y=0.35)
                axes[0].set_title("Full-corpus mean normalized return")
                axes[0].set_ylabel("Mean normalized return (NOT success)")
                axes[1].set_title("Full-corpus clip completion")
                axes[1].set_ylabel("Clip completion rate (%)")
                axes[1].set_ylim(-20, 115)
                axes[1].set_yticks(range(0, 101, 20))
                figure.savefig(full_destination, dpi=200, facecolor="white")
                outputs.append(full_destination)
            finally:
                plt.close(figure)
        else:
            full_destination.unlink(missing_ok=True)
    return outputs


def generate_artifacts(report_path: Path, output_dir: Path) -> list[Path]:
    """Read one report snapshot and record its exact byte SHA alongside plotted arrays."""
    report_path = Path(report_path).resolve()
    output_dir = Path(output_dir).resolve()
    source = report_path.read_bytes()
    report = json.loads(source)
    series = prepare_series(report)
    local_updated_at = _utc_timestamp(report["updated_at"]).astimezone(ZoneInfo("Asia/Shanghai"))
    summary = {
        "schema_version": 1,
        "report_source": str(report_path),
        "report_source_sha256": hashlib.sha256(source).hexdigest(),
        "report_updated_at": report["updated_at"],
        "report_updated_at_asia_shanghai": local_updated_at.isoformat(),
        "status": report["status"],
        "protocol": report["protocol"],
        "baseline_commit": report["baseline_commit"],
        "subset": report["subset"],
        "series": series,
        "full": sorted(report["full"], key=lambda entry: (entry["seed"], entry["iteration"])),
        "plot_semantics": "Measured checkpoints only; markers only; no smoothing or interpolation.",
        "return_semantics": "Normalized return is NOT success.",
    }
    owned_names = {SUMMARY_NAME, FULL_PLOT_NAME, "policy_evaluation_curves.png"}
    owned_names.add("policy_evaluation_curves.svg")
    if any(report_path == (output_dir / name).resolve() for name in owned_names):
        raise ValueError("report must not use a generated artifact path")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = render_plots(summary, output_dir)
    summary_path = output_dir / SUMMARY_NAME
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return [*outputs, summary_path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="Policy evaluation report JSON")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Destination for policy plots"
    )
    arguments = parser.parse_args(argv)
    try:
        outputs = generate_artifacts(arguments.report, arguments.output_dir)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    for destination in outputs:
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
