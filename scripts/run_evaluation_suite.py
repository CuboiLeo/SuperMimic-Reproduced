"""Run a frozen manifest suite on independent GPUs with a mandatory sanity gate."""

from __future__ import annotations

import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from evaluation_utils import summarize_results, validate_results


def timestamp():
    return datetime.datetime.now(datetime.UTC).isoformat()


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def group_key(job):
    return job["kind"], job["seed"], job["iteration"]


def validate_output(job, expected, fingerprint_sha256=None):
    require(digest(job["path"]) == job["sha256"], f"Job manifest changed: {job['job_id']}")
    result = json.loads(Path(job["output"]).read_text())
    require(result["job_id"] == job["job_id"] and result["seed"] == job["seed"] and
            result["checkpoint_iteration"] == job["iteration"], "Output job identity mismatch")
    require(result["checkpoint_sha256"] == expected["checkpoint_sha256"] and
            result["input_manifest_sha256"] == job["sha256"], "Output provenance mismatch")
    if fingerprint_sha256 is not None:
        require(result.get("evaluation_fingerprint_sha256") == fingerprint_sha256,
                "Output evaluator/runtime fingerprint mismatch")
    validate_results(result["clips"], expected["clips"])
    require(result["summary"] == summarize_results(result["clips"]), "Output summary mismatch")
    return result


def available_gpus(indices):
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    mapping = {int(line.split(",")[0]): line.split(",")[1].strip()
               for line in inventory.strip().splitlines()}
    require(all(index in mapping for index in indices), "Requested GPU is unavailable")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    occupied = {line.split(",")[0].strip() for line in apps.strip().splitlines() if line.strip()}
    require(not occupied.intersection(mapping[index] for index in indices),
            "A requested GPU already has a compute process; refusing to interfere")
    return mapping


def publish_report(suite, plan, completed, state):
    grouped = collections.defaultdict(list)
    for job in plan["jobs"]:
        grouped[group_key(job)].append(job)
    report = {"schema_version": 1, "updated_at": timestamp(), "status": state,
              "protocol": plan["protocol"], "baseline_commit": plan["baseline_commit"],
              "subset": plan["subset"], "curve": [], "full": [], "sanity": [],
              "completed_jobs": len(completed), "total_jobs": len(plan["jobs"])}
    for (kind, seed, iteration), jobs in sorted(grouped.items()):
        if not all(job["job_id"] in completed for job in jobs):
            continue
        hashes = {completed[job["job_id"]]["checkpoint_sha256"] for job in jobs}
        require(len(hashes) == 1, "Cannot aggregate mixed checkpoints")
        rows = [row for job in jobs for row in completed[job["job_id"]]["clips"]]
        expected = json.loads((suite / "inputs" / f"{'subset' if kind == 'curve' else kind}.json").read_text())
        rows.sort(key=lambda row: row["name"])
        expected.sort(key=lambda row: row["name"])
        validate_results(rows, expected)
        summary = summarize_results(rows)
        result = {"seed": seed, "iteration": iteration,
                  "checkpoint_sha256": completed[jobs[0]["job_id"]]["checkpoint_sha256"],
                  "summary": summary}
        report[kind].append(result)
        aggregate = suite / "aggregates" / f"{kind}-seed{seed}-it{iteration:07d}.json"
        if not aggregate.exists():
            write_json(aggregate, {**result, "protocol": plan["protocol"], "clips": rows,
                                   "shards": [{"job_id": job["job_id"], "sha256": digest(job["output"])}
                                              for job in jobs]})
    write_json(suite / "evaluation_report.json", report)
    return report


def freeze_evaluator(suite, plan, tools):
    source = plan["baseline_source"]
    commit = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    require(commit == plan["baseline_commit"], "Baseline source commit changed")
    require(not subprocess.check_output(["git", "-C", source, "status", "--porcelain"], text=True),
            "Baseline source has modifications")
    runtime = json.loads(subprocess.check_output([
        plan["python"], "-I", "-B", "-c",
        ("import importlib.metadata,json,sys; print(json.dumps({'python':sys.version,"
         "'packages':sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())}))"),
    ], text=True))
    fingerprint = {"schema_version": 1, "baseline_commit": commit, "runtime": runtime,
                   "python_sha256": digest(plan["python"]),
                   "tools": {path.name: digest(path) for path in sorted(tools.glob("*.py"))}}
    path = suite / "evaluation-fingerprint.json"
    if path.exists():
        require(json.loads(path.read_text()) == fingerprint,
                "Evaluator/runtime changed; use a separate evaluation suite")
    else:
        write_json(path, fingerprint)
        path.chmod(0o440)
    return digest(path), fingerprint


def terminate_processes(active):
    for item in active.values():
        if item["process"].poll() is None:
            try:
                os.killpg(item["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for item in active.values():
        try:
            item["process"].wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(item["process"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            item["process"].wait(timeout=15)
        item["stream"].close()


def execute(suite, sanity_only=False):
    suite = suite.resolve()
    lock = (suite / ".controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((suite / "plan.json").read_text())
    require(plan["schema_version"] == 1, "Unsupported plan")
    jobs = plan["jobs"]
    require(len({job["job_id"] for job in jobs}) == len(jobs), "Duplicate job IDs")
    require({job["kind"] for job in jobs} <= {"sanity", "curve", "full"}, "Unknown evaluation kind")
    expected = {}
    for job in jobs:
        require(digest(job["path"]) == job["sha256"], "Frozen job hash mismatch")
        expected[job["job_id"]] = json.loads(Path(job["path"]).read_text())
    tools = Path(__file__).resolve().parent
    fingerprint_sha256, fingerprint = freeze_evaluator(suite, plan, tools)
    grouped_hashes = collections.defaultdict(set)
    for job in jobs:
        grouped_hashes[group_key(job)].add(expected[job["job_id"]]["checkpoint_sha256"])
    require(all(len(hashes) == 1 for hashes in grouped_hashes.values()), "Plan mixes checkpoint hashes")
    completed = {job["job_id"]: validate_output(job, expected[job["job_id"]], fingerprint_sha256)
                 for job in jobs if Path(job["output"]).exists()}
    indices = plan["gpu_indices"]
    mapping = available_gpus(indices)
    active = {}
    failures = []
    state = {"status": "running", "started_at": timestamp(), "controller_pid": os.getpid(),
             "plan_sha256": digest(suite / "plan.json"), "gpu_uuids": mapping,
             "evaluation_fingerprint_sha256": fingerprint_sha256,
             "tools_sha256": fingerprint["tools"]}
    state["completed_jobs"] = len(completed)
    write_json(suite / "status.json", state)
    publish_report(suite, plan, completed, "running")

    def interrupted(signum, frame):
        raise InterruptedError(f"Controller interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for stage in ("sanity", "production"):
            pending = [job for job in jobs if job["job_id"] not in completed and
                       ((job["kind"] == "sanity") == (stage == "sanity"))]
            pending.sort(key=lambda job: (job["kind"] != "curve", -job["iteration"], -job["frames"]))
            state["stage"] = stage
            while pending or active:
                changed = False
                for gpu, item in list(active.items()):
                    process = item["process"]
                    require(time.monotonic() - item["started"] < plan["job_timeout_seconds"],
                            f"Job timeout: {item['job']['job_id']}")
                    code = process.poll()
                    if code is None:
                        continue
                    item["stream"].close()
                    job = item["job"]
                    del active[gpu]
                    require(code == 0, f"Worker failed ({code}): {job['job_id']}; inspect its log")
                    completed[job["job_id"]] = validate_output(job, expected[job["job_id"]], fingerprint_sha256)
                    print(f"Completed {job['job_id']} ({len(completed)}/{len(jobs)})", flush=True)
                    changed = True
                for gpu in indices:
                    if gpu in active:
                        continue
                    candidates = [job for job in pending if job["seed"] == (0 if gpu < 4 else 1)]
                    if not candidates:
                        continue
                    job = candidates[0]
                    pending.remove(job)
                    environment = os.environ.copy()
                    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "PYTHONPATH", "VIRTUAL_ENV"):
                        environment.pop(name, None)
                    environment.update(CUDA_VISIBLE_DEVICES=mapping[gpu], PYTHONUNBUFFERED="1",
                                       PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                    environment["SM_EVALUATION_FINGERPRINT"] = fingerprint_sha256
                    environment["XDG_CACHE_HOME"] = str(suite / "cache" / f"gpu-{gpu}")
                    stream = (suite / "logs" / f"{job['job_id']}.log").open("a")
                    command = [plan["python"], "-B", str(tools / "eval_manifest.py"),
                               "--job", job["path"], "--device", "cuda:0"]
                    process = subprocess.Popen(command, cwd=plan["baseline_source"], env=environment,
                                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    active[gpu] = {"process": process, "job": job, "stream": stream,
                                   "started": time.monotonic()}
                    print(f"Started {job['job_id']} on GPU {gpu}, pid {process.pid}", flush=True)
                state.update(updated_at=timestamp(), completed_jobs=len(completed), total_jobs=len(jobs),
                             active=[{"gpu": gpu, "pid": item["process"].pid, "job_id": item["job"]["job_id"]}
                                     for gpu, item in active.items()])
                write_json(suite / "status.json", state)
                if changed:
                    publish_report(suite, plan, completed, "running")
                if pending or active:
                    time.sleep(3)
            report = publish_report(suite, plan, completed, "running")
            if stage == "sanity":
                require(len(report["sanity"]) == 2 and all(entry["summary"]["completion_rate"] > 0
                                                          for entry in report["sanity"]),
                        "Sanity gate failed: every clip terminated for a seed")
                state["sanity_passed_at"] = timestamp()
                if sanity_only:
                    break
        state.update(status="sanity_passed" if sanity_only else "completed", finished_at=timestamp(), active=[])
        report = publish_report(suite, plan, completed, state["status"])
        if not sanity_only:
            require(len(report["full"]) == 2 and len(report["curve"]) == 16, "Incomplete final report")
        write_json(suite / "status.json", state)
        plotter = tools / "plot_evaluation.py"
        if plotter.exists() and not sanity_only:
            subprocess.run([plan["python"], "-B", str(plotter), "--report", str(suite / "evaluation_report.json"),
                            "--output-dir", str(suite / "plots")], check=True)
    except BaseException as error:
        terminate_processes(active)
        failures.append(str(error))
        state.update(status="failed", finished_at=timestamp(), errors=failures, active=[])
        write_json(suite / "status.json", state)
        publish_report(suite, plan, completed, "failed")
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    execute(args.suite, args.sanity_only)


if __name__ == "__main__":
    main()
