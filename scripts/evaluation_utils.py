"""Strict, dependency-free validation and aggregation for evaluation workers.

Relative job paths are relative to the job JSON's directory. Clip names are
single NPZ stems, not paths. ``terms`` in a summary are step-weighted means;
``mean_terms`` are equally weighted per-clip means. Completion rates are in [0, 1].
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
PROTOCOL = "deterministic training-corpus evaluation; not held out"


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty string without NUL characters")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _finite(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def validate_clip_name(name: object) -> str:
    """Reject traversal, separators, ambiguous basenames, and filename extensions."""
    name = _text(name, "clip name")
    if (
        name in {".", ".."}
        or name != name.strip()
        or any(character in name for character in '/\\:<>"|?*')
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
        or name.endswith(".")
        or name.lower().endswith(".npz")
    ):
        raise ValueError(f"unsafe NPZ stem: {name!r}")
    return name


def _expected_clips(clips: object, *, fingerprints: bool = False) -> list[dict]:
    if not isinstance(clips, list) or not clips:
        raise ValueError("clips must be a nonempty list")
    names = set()
    for clip in clips:
        _mapping(clip, "clip")
        name = validate_clip_name(clip.get("name"))
        if name in names:
            raise ValueError(f"duplicate expected clip: {name}")
        names.add(name)
        _integer(clip.get("frames"), f"{name}.frames", 1)
        if fingerprints:
            _digest(clip.get("sha256"), f"{name}.sha256")
            _integer(clip.get("bytes"), f"{name}.bytes", 1)
    return clips


def validate_job(job: object) -> dict:
    """Validate version-1 jobs, allowing controller metadata such as ``kind``."""
    job = _mapping(job, "job")
    if _integer(job.get("schema_version"), "schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported job schema_version")
    _text(job.get("job_id"), "job_id")
    _integer(job.get("seed"), "seed")
    _integer(job.get("checkpoint_iteration"), "checkpoint_iteration")
    _integer(job.get("num_envs"), "num_envs", 1)
    for field in ("checkpoint", "npz_dir", "output"):
        _text(job.get(field), field)
    _digest(job.get("checkpoint_sha256"), "checkpoint_sha256")
    _expected_clips(job.get("clips"), fingerprints=True)
    if "kind" in job:
        _text(job["kind"], "kind")
    return job


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def load_job(path: str | Path) -> tuple[dict, str]:
    """Return a validated job and the SHA-256 of its exact original JSON bytes."""
    path = Path(path).absolute()
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"job must be a regular file, not a symlink: {path}")
    raw = path.read_bytes()
    job = validate_job(
        json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    )
    for field in ("checkpoint", "npz_dir", "output"):
        job[field] = str(path.parent / job[field])
    return job, hashlib.sha256(raw).hexdigest()


def _fingerprint_file(path: str | Path) -> tuple[str, int]:
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(source.fileno())
    current = path.lstat()
    signatures = {
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        for info in (before, after, current)
    }
    if not stat.S_ISREG(current.st_mode) or len(signatures) != 1 or size != before.st_size:
        raise ValueError(f"file changed while hashing: {path}")
    return digest.hexdigest(), size


def sha256_file(path: str | Path) -> str:
    """Hash a stable regular file, refusing symlinks and special files."""
    return _fingerprint_file(path)[0]


def verify_file(path: str | Path, sha256: str, size: int | None = None) -> None:
    """Compare a regular file against immutable expected SHA-256 and byte count."""
    _digest(sha256, "expected sha256")
    if size is not None:
        _integer(size, "expected bytes", 1)
    actual_sha256, actual_size = _fingerprint_file(path)
    if size is not None and actual_size != size:
        raise ValueError(f"byte count mismatch for {path}: expected {size}, got {actual_size}")
    if actual_sha256 != sha256:
        raise ValueError(f"SHA-256 mismatch for {path}")


def verify_inputs(job: dict) -> list[str]:
    """Verify checkpoint and NPZ fingerprints, returning NPZ paths in job order."""
    validate_job(job)
    npz_dir = Path(job["npz_dir"])
    if not npz_dir.is_dir():
        raise ValueError(f"npz_dir is not a directory: {npz_dir}")
    verify_file(job["checkpoint"], job["checkpoint_sha256"])
    paths = []
    for clip in job["clips"]:
        path = npz_dir / f"{clip['name']}.npz"
        verify_file(path, clip["sha256"], clip["bytes"])
        paths.append(str(path))
    return paths


def validate_checkpoint(
    checkpoint: dict, job: dict, *, history_length: int = 10
) -> dict:
    """Check Trainer.save metadata and return supported run_benchmark options.

    The canonical evaluator does not expose history_length. Reject incompatible
    histories rather than changing the evaluator or silently evaluating a
    different observation interface. Missing historical seed/iteration metadata
    is permitted; metadata that is present must match the immutable job.
    """
    _mapping(checkpoint, "checkpoint")
    if "iteration" in checkpoint:
        iteration = _integer(checkpoint["iteration"], "checkpoint.iteration")
        if iteration != job["checkpoint_iteration"]:
            raise ValueError("checkpoint iteration does not match job")
    config = _mapping(checkpoint.get("config"), "checkpoint.config")
    if "seed" in config:
        seed = _integer(config["seed"], "checkpoint.config.seed")
        if seed != job["seed"]:
            raise ValueError("checkpoint seed does not match job")
    env = _mapping(config.get("env"), "checkpoint.config.env")
    policy = _mapping(config.get("policy"), "checkpoint.config.policy")
    future_frames = _integer(env.get("future_frames"), "env.future_frames", 1)
    anchor_obs = env.get("anchor_obs", False)
    if type(anchor_obs) is not bool:
        raise ValueError("env.anchor_obs must be boolean")
    _integer(history_length, "supported history_length", 1)
    policy_history = _integer(policy.get("history", 10), "policy.history", 1)
    env_history = _integer(env.get("history_length", history_length), "env.history_length", 1)
    if policy_history != history_length or env_history != history_length:
        raise ValueError(f"canonical evaluator requires history_length={history_length}")
    goal_dim = _integer(policy.get("goal_dim", 680), "policy.goal_dim", 1)
    if goal_dim != future_frames * 68 + (9 if anchor_obs else 0):
        raise ValueError("policy.goal_dim does not match env.future_frames/anchor_obs")
    return {"future_frames": future_frames, "anchor_obs": anchor_obs}


def _validate_rows(results: object) -> list[dict]:
    if not isinstance(results, list) or not results:
        raise ValueError("results must be a nonempty list")
    names = set()
    term_names = None
    for row in results:
        _mapping(row, "result")
        name = validate_clip_name(row.get("name"))
        if name in names:
            raise ValueError(f"duplicate result clip: {name}")
        names.add(name)
        _integer(row.get("frames"), f"{name}.frames", 1)
        steps = _integer(row.get("steps"), f"{name}.steps", 1)
        _finite(row.get("norm_return"), f"{name}.norm_return")
        failed = row.get("failed")
        if type(failed) is not bool:
            raise ValueError(f"{name}.failed must be boolean")
        fail_step = _integer(row.get("fail_step"), f"{name}.fail_step", -1)
        if fail_step != (steps if failed else -1):
            raise ValueError(f"{name}.fail_step is inconsistent with failed/steps")
        terms = _mapping(row.get("terms"), f"{name}.terms")
        if not terms:
            raise ValueError(f"{name}.terms must not be empty")
        for term, value in terms.items():
            _text(term, "term name")
            _finite(value, f"{name}.terms.{term}")
        if term_names is None:
            term_names = set(terms)
        elif set(terms) != term_names:
            raise ValueError(f"{name}.terms do not match the other clips")
    return results


def validate_results(
    results: list[dict], expected_clips: list[dict], *, require_order: bool = True
) -> list[dict]:
    """Require exact unique coverage and original frames; optionally enforce order.

    Expected clips need at least name and frames. Steps must be positive, but
    are not capped by frames: command advancement and timeout observation can
    yield one or two extra evaluator steps. No rows are sorted or repaired.
    """
    expected_clips = _expected_clips(expected_clips)
    results = _validate_rows(results)
    expected = {clip["name"]: clip["frames"] for clip in expected_clips}
    actual = {row["name"] for row in results}
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ValueError(f"result coverage mismatch: missing={missing}, extra={extra}")
    for row in results:
        if row["frames"] != expected[row["name"]]:
            raise ValueError(f"original frame count mismatch for {row['name']}")
    if require_order and [row["name"] for row in results] != list(expected):
        raise ValueError("result order does not exactly match input clip order")
    return results


def summarize_results(results: list[dict]) -> dict:
    """Summarize raw clip rows, never shard averages; coverage is checked separately."""
    results = _validate_rows(results)
    count = len(results)
    failed = sum(row["failed"] for row in results)
    total_steps = sum(row["steps"] for row in results)

    def mean(values: list[float], weights: list[int]) -> float:
        total = sum(weights)
        try:
            value = math.fsum(value * (weight / total) for value, weight in zip(values, weights))
        except OverflowError as error:
            raise ValueError("nonfinite aggregate") from error
        return _finite(value, "aggregate")

    weights = [row["steps"] for row in results]
    uniform = [1] * count
    returns = [float(row["norm_return"]) for row in results]
    terms = results[0]["terms"]
    return {
        "clips": count,
        "failed": failed,
        "completed": count - failed,
        "completion_rate": (count - failed) / count,
        "total_frames": sum(row["frames"] for row in results),
        "total_steps": total_steps,
        "mean_norm_return": mean(returns, uniform),
        "step_weighted_norm_return": mean(returns, weights),
        "terms": {
            term: mean([float(row["terms"][term]) for row in results], weights)
            for term in terms
        },
        "mean_terms": {
            term: mean([float(row["terms"][term]) for row in results], uniform)
            for term in terms
        },
    }


def aggregate_results(results: list[dict], expected_clips: list[dict]) -> dict:
    """Validate concatenated shard rows and return canonical-order rows and summary."""
    validate_results(results, expected_clips, require_order=False)
    by_name = {row["name"]: row for row in results}
    ordered = [by_name[clip["name"]] for clip in expected_clips]
    return {"summary": summarize_results(ordered), "clips": ordered}


def atomic_write_json(path: str | Path, payload: dict) -> None:
    """Publish complete JSON via a same-directory hard link, never replacing a file.

    Unlike os.replace, link creation fails if another worker wins publication.
    Readers see either no output or the complete, flushed document.
    """
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
