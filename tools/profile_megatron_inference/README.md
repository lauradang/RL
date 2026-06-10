# Megatron Inference Profiling

This toolkit profiles Megatron policy inference inside normal NeMo-RL Slurm
recipe launches. It is intentionally launch-level because Nsight worker runtime
configuration must be present before Ray actors import their worker classes, and
Slurm log sync must be configured before `ray.sub` starts Ray.

## Quick Start

```bash
PROFILE=megatron-logprobs \
PROFILE_RANGE=4:6 \
CONTAINER=/path/to/container.sqsh \
ACCOUNT=<slurm-account> \
PARTITION=<slurm-partition> \
tools/launch tests/test_suites/llm/performance/grpo-llama3.1-8b-instruct-2n4g-async-1off.sh
```

Use `DRYRUN=2` to create a code snapshot and inspect `continue.sh` without
submitting:

```bash
PROFILE=megatron-logprobs PROFILE_RANGE=4:6 DRYRUN=2 \
CONTAINER=/path/to/container.sqsh ACCOUNT=<account> PARTITION=<partition> \
tools/launch tests/test_suites/llm/performance/grpo-llama3.1-8b-instruct-2n4g-async-1off.sh
```

## Profile Presets

- `PROFILE=none`: normal launch behavior.
- `PROFILE=megatron-logprobs`: profile Megatron policy workers. Use this for
  async GRPO logprob inference.
- `PROFILE=megatron-all`: same worker selection as `megatron-logprobs`; the
  report keeps all parsed Megatron NVTX ranges.
- `PROFILE=e2e-async`: profile Megatron policy workers and vLLM generation
  workers to inspect the end-to-end async loop.

`PROFILE_RANGE` uses the same `start:stop` convention as `NRL_NSYS_PROFILE_STEP_RANGE`.
The default is `4:6`, which profiles steps 4 and 5.

## Outputs

Artifacts are written to the recipe's normal result directory, for example:

```text
tests/test_suites/llm/performance/grpo-llama3.1-8b-instruct-2n4g-async-1off/
  metrics.json
  profile_metadata.json
  profile_summary.json
  profile_summary.csv
  report.html
```

Nsight reports are discovered from the synced Ray log tree:

```text
$SLURM_SUBMIT_DIR/$SLURM_JOB_ID-logs/ray/**/nsight/*.nsys-rep
```

If `nsys` is available where collection runs, `nvtxsum`, `gpukernsum`, and
`cudaapisum` CSV outputs are saved under `nsys_stats/` and summarized in
`report.html`. If `nsys` is not available, the report still includes timing
metrics and `.nsys-rep` inventory.

## Useful Environment Variables

- `PROFILE_RANGE=4:6`: profile window.
- `PROFILE_POST_RUN_LOG_SYNC_SLEEP=90`: seconds to wait after training exits so
  Ray log sync can copy profile files from `/tmp/ray`.
- `RAY_LOG_SYNC_FREQUENCY=30`: sync cadence. `tools/launch` sets this when
  profiling unless it is already set.
