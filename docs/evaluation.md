# Frozen, sharded policy evaluation

The training replay diagnostics are not policy evaluation. These tools run the
existing deterministic evaluator on explicit motion manifests without changing
training, rewards, termination rules, or model parameters.

## Protocol

- Start each clip at its first frame, without observation/action/reset noise or
  domain randomization, using the settings in `run_benchmark`.
- Freeze checkpoint hashes and clip names, frame counts, byte sizes, and SHA256
  hashes before running. Every worker verifies those inputs.
- Report normalized return together with clip completion under the evaluator's
  termination rules. Normalized return is not success rate; an early-failing
  clip can still have high mean reward over its observed steps.
- A subset drawn from the training corpus is an in-distribution evaluation,
  **not a held-out generalization test**.
- Use the same fixed subset at every selected checkpoint. Do not join results
  from different subsets into one learning curve or select checkpoints on the
  full final benchmark.

## Worker

`scripts/eval_manifest.py --job JOB.json --device cuda:0` evaluates one shard.
The JSON job identifies the checkpoint, expected hashes, NPZ directory, vector
environment count, output path, and complete ordered clip metadata. Results are
validated and atomically published only after the whole shard finishes.

The simulation uses the installed `supermimic` source. Keep that environment on
the frozen baseline commit while running evaluation utilities from this branch.
This separation allows evaluation tooling changes without modifying the policy
or simulator being compared.

## Suite

`scripts/run_evaluation_suite.py --suite DIRECTORY` runs an already frozen suite.
Its `plan.json` references immutable jobs and groups them by seed, checkpoint,
and purpose (`sanity`, `curve`, or `full`). The current suite scheduler targets
two training seeds: GPUs 0–3 evaluate seed 0; GPUs 4–7 evaluate seed 1.

The sanity jobs must finish first, with finite, complete outputs and at least
one completed clip for each seed. `--sanity-only` stops at this gate. Production
runs selected checkpoint/subset jobs and then the final full-corpus shards.
Each GPU executes one independent process, not distributed data parallel.

The stock evaluator builds an expanded reference library. Shard by clip count
and total frames: reducing `num_envs` alone does not bound library memory.
Avoid overlapping device allocations; the controller refuses occupied GPUs.
It only terminates its own child process groups on failure or interruption.
An immutable evaluator fingerprint binds the tools, baseline Git commit, Python
binary, and installed package versions. Resuming with a different fingerprint
is rejected, and each worker output records the fingerprint that launched it.

Each job has a log and a validated output. The controller writes `status.json`
and `evaluation_report.json` atomically. A checkpoint appears in the report only
after every expected shard is present. Aggregates are recomputed from per-clip
records rather than averaging shard averages. A restarted controller validates
and reuses complete outputs; invalid outputs require investigation, not silent
replacement. Inspect worker logs before retrying a failed run.

## Curves

`scripts/plot_evaluation.py --report evaluation_report.json --output-dir plots`
produces PNG/SVG plots from completed fixed-subset results. Full-corpus final
results are displayed separately. Missing checkpoints are not filled with
training-return or world-model metrics.

Keep checkpoints, datasets, environments, and bulk run artifacts outside Git.
Commit reusable utilities, tests, and protocol documentation. Preserve source,
dependency, input, and checkpoint fingerprints alongside each experiment.
