"""Evaluate one immutable job with the unchanged canonical benchmark.

Usage: python scripts/eval_manifest.py --job /path/job.json --device cuda:0
Requires the installed SuperMimic GPU runtime only when executing a job.
An existing output, even malformed or incomplete, is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

if __package__:
    from .evaluation_utils import (
        PROTOCOL,
        SCHEMA_VERSION,
        atomic_write_json,
        load_job,
        sha256_file,
        summarize_results,
        validate_checkpoint,
        validate_results,
        verify_inputs,
    )
else:
    from evaluation_utils import (
        PROTOCOL,
        SCHEMA_VERSION,
        atomic_write_json,
        load_job,
        sha256_file,
        summarize_results,
        validate_checkpoint,
        validate_results,
        verify_inputs,
    )


def run_job(job_path: str | Path, device: str) -> dict:
    """Verify inputs, evaluate on the controller-selected GPU, and publish once."""
    if re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device) is None:
        raise ValueError("device must be an explicit CUDA device, such as cuda:0")
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    job, manifest_sha256 = load_job(job_path)
    output = Path(job["output"])
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    files = verify_inputs(job)

    import torch

    if not torch.cuda.is_available() or int(device.split(":")[1]) >= torch.cuda.device_count():
        raise ValueError(f"controller-selected device is unavailable: {device}")

    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark

    defaults = SuperMimicEnvCfgArgs(npz_files=files)
    checkpoint = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
    options = validate_checkpoint(checkpoint, job, history_length=defaults.history_length)
    expected_config = checkpoint["config"]
    del checkpoint
    torch.cuda.set_device(device)
    policy, config = load_policy_from_checkpoint(job["checkpoint"], device)
    if config != expected_config:
        raise ValueError("checkpoint configuration changed during loading")
    validate_checkpoint({"config": config}, job, history_length=defaults.history_length)
    results = run_benchmark(policy, files, device=device, num_envs=job["num_envs"], **options)
    rows = validate_results([asdict(result) for result in results], job["clips"])
    summary = summarize_results(rows)
    verify_inputs(job)
    if sha256_file(job_path) != manifest_sha256:
        raise ValueError("job manifest changed during evaluation")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job["job_id"],
        "seed": job["seed"],
        "checkpoint_iteration": job["checkpoint_iteration"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "protocol": PROTOCOL,
        "summary": summary,
        "clips": rows,
        "input_manifest_sha256": manifest_sha256,
        "evaluation_fingerprint_sha256": os.environ.get("SM_EVALUATION_FINGERPRINT"),
    }
    if "kind" in job:
        payload["kind"] = job["kind"]
    atomic_write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="immutable version-1 job JSON")
    parser.add_argument("--device", required=True, help="controller-selected GPU, e.g. cuda:0")
    args = parser.parse_args()
    try:
        payload = run_job(args.job, args.device)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"evaluation failed: {error}\n")
    print(json.dumps(payload["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
