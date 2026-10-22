# Managed Dynamo generation on Slurm

NeMo RL can launch and own Dynamo's control plane, frontend, and a fixed vLLM
worker fleet inside a Slurm-backed Ray allocation. This mode supports direct
GRPO and NeMo-Gym rollouts, NCCL and `nccl_reshard` weight refits, cache invalidation, and
`generation_metrics/*` telemetry sent to enabled loggers such as W&B.
See the [Dynamo integration design](../design-docs/dynamo-integration.md) for
the service ownership, startup, and weight-refit architecture.

This integration is managed and vLLM-only. It does not connect to an external
Dynamo deployment and does not support Kubernetes, DGD, SGLang, TensorRT-LLM,
FP8 KV caches, ModelOpt quantization, or model-parallel engine groups that span
nodes. FP8 and MXFP8 weights (`vllm_cfg.precision: fp8`) and speculative
decoding are supported; see [FP8 generation](#fp8-generation) and
[Speculative decoding](#speculative-decoding).

## Build the image

The normal image is unchanged unless `BUILD_DYNAMO` is set:

```bash
docker buildx build \
  --build-context nemo-rl=. \
  --build-arg BUILD_DYNAMO=1 \
  --target release \
  --file docker/Dockerfile \
  --tag registry.example.com/nemo-rl:dynamo \
  .
```

`BUILD_DYNAMO=1` prebuilds the `DynamoVllmWorker` actor venv from the root
project's `dynamo` extra, and installs etcd v3.5.21 and NATS Server v2.11.6 into
`/usr/local/bin` (`docker/install_dynamo_services.sh`). The `dynamo` extra is
the `vllm` extra plus the base `ai-dynamo==1.5.1` package, so the Dynamo
workers run the same vLLM (0.29.0), NCCL and `nemo_rl` code as NeMo RL's own
vLLM workers. That is what lets each engine load NeMo RL's vLLM worker
extension for refit (see [Weight refit](#weight-refit)). Only the base
`ai-dynamo` package is installed: its `[vllm]` extra pins vLLM 0.28.0.

From a source checkout, the venv is built on first use under
`venvs/nemo_rl.models.generation.dynamo.dynamo_worker.DynamoVllmWorker`, like
every other actor venv. etcd and nats-server are looked up in
`NEMO_RL_DYNAMO_BIN_DIR` when it is set, otherwise on `PATH`;
`NEMO_RL_DYNAMO_BIN_DIR=$PWD/bin bash docker/install_dynamo_services.sh`
installs them.

A Dynamo upgrade must update and reverify these coupled locations:

- root `pyproject.toml` and `uv.lock`: the `ai-dynamo` and
  `ai-dynamo-runtime` pins in the `dynamo` extra, which must otherwise stay
  identical to the `vllm` extra (`tests/unit/test_dependency_pins.py`)
- `tests/functional/grpo_dynamo*.sh`: the asserted Dynamo and vLLM versions
- `nemo_rl/models/generation/dynamo/refit.py`: the `/engine/*` RL routes it
  posts to, which must still forward any `engine_rpc` to vLLM `collective_rpc`
  with the request body as keyword arguments
  (`test_every_engine_rpc_the_driver_sends_binds_to_the_extension`)
- `nemo_rl/models/generation/dynamo/token_wrapper.py`: the real Dynamo response
  keys `nvext.engine_data.{prompt_token_ids,completion_token_ids,completion_logprobs}`;
  reverify them against real Dynamo output because unit tests validate only the
  expected local response shape
- `nemo_rl/models/generation/dynamo/managed_runtime.py`: the managed
  `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2=1` setting. Dynamo 1.3.0's legacy tool
  jail removes `nvext.engine_data`; remove this setting only after an upgraded
  Dynamo preserves the token metadata for `tool_choice=auto`
- this guide and the Dynamo design document: the stated versions

## Configure Dynamo

Start with [`examples/configs/grpo_math_1B_dynamo.yaml`](../../examples/configs/grpo_math_1B_dynamo.yaml).
The important boundary is:

```yaml
policy:
  generation:
    backend: dynamo
    dynamo_cfg:
      engine: vllm
      frontend_args:
        router_mode: kv
    vllm_cfg:
      tensor_parallel_size: 1
      pipeline_parallel_size: 1
      expert_parallel_size: 1
    colocated:
      enabled: false
      resources:
        gpus_per_node: 1
        num_nodes: 1
```

NeMo RL derives each engine's world size from TP times PP. EP must be one or
equal to TP. Parser settings belong under `dynamo_cfg.worker_args`; inherited
vLLM HTTP-parser settings are rejected with the corresponding Dynamo field.
Service ports and the namespace are runtime-owned rather than public config.

`vllm_cfg` settings are handled in eight explicit classes:

| Class | Behavior | Examples |
| --- | --- | --- |
| Translated | Forwarded to `dynamo.vllm` | TP, PP, EP, dtype, model length |
| Refit | Read by NeMo RL's worker extension inside each engine | `refit_with_reload_api` |
| Moved | Startup error naming the Dynamo replacement | tool and reasoning parsers, HTTP serving chat kwargs |
| FP8 | Read by the vLLM backend's FP8 builder when `precision: fp8`; an error otherwise | `is_mx`, mixed BF16/FP8 layers, DeepGEMM, pow2 scales |
| Engine patch | The vLLM backend's source patch, enabled in each engine's environment | `fp32_lm_head` (Nemotron-H fp32 LM head) |
| Unsupported | Warning when active | tokenizer skipping, context-length token capping |
| Managed runtime | Consumed or enforced by NeMo RL rather than forwarded | HTTP-wrapper enablement, metrics sampling, processed rollout logprobs |
| Inapplicable | Ignored because the managed path owns that behavior | async mode, progress bars, NeMo RL HTTP/ZMQ refit ports |

The shared GRPO base config also supplies `mcore_generation_config` and
`refit_cfg`. Dynamo accepts these inherited sections but does not use them;
worker arguments come from `vllm_cfg`.

Set `policy.generation.refit_transport` to `null` for the packed NCCL broadcast
(`CollectiveWeightSynchronizer`), or to `nccl_reshard` to send each Megatron
training shard straight into the Dynamo engines' shards
([NCCL reshard refit](../design-docs/nccl-reshard-refit.md)). `nccl_reshard`
has the same requirements as with the vLLM backend: a Megatron policy, train
ETP 1, generation PP 1, and EP equal to 1 or TP. Other transports are rejected.

Enable and filter worker telemetry with both managed configuration sections:

```yaml
policy:
  generation:
    vllm_cfg:
      enable_vllm_metrics_logger: true
      vllm_metrics_logger_interval: 1.0
    dynamo_cfg:
      metrics_include_prefixes: null  # null selects the curated defaults
      metrics_exclude_prefixes: null  # null excludes python_ and process_
```

## FP8 generation

`vllm_cfg.precision: fp8` (optionally with `is_mx`, mixed BF16 layers, DeepGEMM
or pow2 scales) quantizes generation weights exactly as the vLLM backend does.
The Dynamo worker actor runs the vLLM backend's FP8 builder and passes the
result to `dynamo.vllm` as `--quantization fp8` and `--hf-overrides`; NeMo RL's
worker extension applies the FP8 refit patches inside each engine before it
loads the model. Every refit then sends BF16 weights that the engine quantizes
on receive. As with the vLLM backend, importance-sampling correction is
required. FP8 KV caches stay unsupported, because their scales are refit only
with synchronous rollouts.

## Speculative decoding

Set `policy.generation.vllm_kwargs.speculative_config` exactly as for the vLLM
backend; it is passed to `dynamo.vllm` as `--speculative-config`, and
`num_speculative_tokens: 0` turns it off. Drafter weights follow the vLLM
backend's rules, because the same worker extension receives every refit:

- A co-trained MTP head is fed by every refit.
- Otherwise an MTP drafter is read from the checkpoint once at startup
  (`load_mtp_weights_from_disk`), so dummy-loaded engines still draft correctly.
- Other methods (for example Eagle3) without a refit-fed draft force
  `load_format: auto`.

Request budgets stop `num_speculative_tokens + 1` tokens short of
`max_model_len`, so the drafter never reads past the context.

## Prefill/decode disaggregation

Set the shared `policy.generation.disaggregation` block, the same one the
TRT-LLM backend reads, to split the fleet into prefill and decode engines:

```yaml
policy:
  generation:
    backend: dynamo
    disaggregation:
      enabled: true
      num_prefill_engines: 1  # per replica
      num_decode_engines: 1   # per replica, independent of prefill
```

A replica is `num_prefill_engines + num_decode_engines` engines of `TP x PP`
GPUs each, and the inference GPUs must hold a whole number of replicas. Every
engine uses the same `vllm_cfg` and `vllm_kwargs`; per-role settings are not
supported yet. The managed frontend discovers both roles and routes each
request through a prefill engine and then a decode engine across the whole
fleet, so replicas only fix the ratio and keep each group's engines together.
`num_frontend_workers` must stay 1 and `frontend_tokenize` false, and
speculative decoding is rejected.

NeMo RL manages each engine's `--disaggregation-mode`, its NIXL
`--kv-transfer-config`, and the NIXL side-channel host and port. With
`router_mode: kv`, prefill engines also publish KV events; only the prefill hop
is KV-routed, because decode engines receive their KV from prefill. Engines on
one node exchange KV over CUDA; across nodes NIXL needs a working UCX/RDMA
path, which you can tune through `vllm_cfg.env_vars` (`UCX_TLS`,
`UCX_NET_DEVICES`).

Refit gives every engine of both roles the full weights. vLLM cannot drop KV
that a decode engine is still receiving, so with
`async_rl.recompute_kv_cache_after_weight_updates` (the default) the pause before
each refit drains instead of freezing. It pauses prefill engines first, so the
decode engines are still running when they pull the KV for requests that
prefill already finished. Decode engines are paused once they have started
every handed-off request, and resume runs in reverse order.
`dynamo_cfg.control_timeout_s` bounds that drain, so it must be at least
`dynamo_cfg.request_timeout_s`. With the flag false the fleet freezes in place
instead, prefill first, and frozen requests keep the KV computed on the old
weights. Run the three-GPU
smoke (one training GPU, one prefill engine, one decode engine) with
`examples/configs/grpo_math_1B_dynamo_disagg.yaml`.

## Weight refit

Every managed engine loads NeMo RL's vLLM worker extension
(`--worker-extension-cls`), the same receiver the vLLM backend uses. The driver
calls it through Dynamo's per-worker `/engine/*` RL routes, which forward an
`engine_rpc` name to vLLM `collective_rpc`. HTTP carries only control calls;
the weights move over NCCL, using NeMo RL's own communicator protocol and
packing on both sides. Each refit has a deadline: GRPO uses
`dynamo_cfg.control_timeout_s`, and the SingleController uses
`async_rl.generation_fleet_health.refit_timeout_s` (300 s by default; `null`
disables it). When it passes, the extension and the trainers each abort their
own communicators and raise instead of hanging.

The fixed port layout is:

- `1313-1399`: driver-local etcd and NATS control plane
- `3000-3999`: frontend and token-wrapper HTTP endpoints
- `4000-4099`: node-local `DYN_SYSTEM_PORT`
- `4100 + slot`: node-local NIXL side channel of each prefill or decode engine
- `4200 + slot`: node-local KV-event stream of each prefill engine
- `7000 + slot * 100`: node-local vLLM rendezvous ports

## Run the two-GPU smoke

Convert the image to the format required by the Slurm site, then submit from
the repository root:

```bash
export CONTAINER=/shared/images/nemo-rl-dynamo.sqsh
export MOUNTS="$PWD:$PWD"
export GPUS_PER_NODE=2
export BASE_LOG_DIR="$PWD/results/dynamo-smoke/logs"
printf -v COMMAND '%q ' \
  /opt/nemo_rl_venv/bin/python -u "$PWD/examples/run_grpo.py" \
  --config "$PWD/examples/configs/grpo_math_1B_dynamo.yaml"
export COMMAND

sbatch \
  --nodes=1 \
  --gres=gpu:2 \
  --exclusive \
  --account=<account> \
  --partition=<partition> \
  ray.sub
```

The recipe assigns one GPU to training and one to a TP1 Dynamo worker. Its two
steps exercise generation, refit, post-refit cache invalidation, telemetry,
and cleanup. For a matched control, run the same seed/model/batch settings with
the standard non-colocated vLLM backend and compare post-refit output validity.

## Run under the SingleController

The async SingleController trainer can also use managed Dynamo. It follows the
same data path as the native vLLM backend: the SingleController sends each turn
to the Dynamo frontend, builds the rollout record from the response, and commits
it to the TransferQueue data plane, from which the policy workers read their
shards. No Dynamo-side component writes to TransferQueue.

Start from
[`examples/configs/grpo_math_1B_dynamo_single_controller.yaml`](../../examples/configs/grpo_math_1B_dynamo_single_controller.yaml)
and launch it with `examples/run_grpo_single_controller.py`. Weight updates pause
the engines in place, as on the vLLM backend: the SingleController stops new
dispatches, freezes every engine's requests, runs the NCCL refit, and resumes
them on the new weights. With `async_rl.recompute_kv_cache_after_weight_updates`
(the default) the pause also clears each engine's KV and prefix cache, so frozen
requests recompute their KV on the new weights; set it to false to keep both.

Current limits on this path:

- Native rollouts only. NeMo-Gym under the SingleController supports only
  `backend: vllm` and `backend: megatron`.
- `async_rl.generation_fleet_health.enabled` must stay false, because the
  Dynamo frontend owns request routing. Its `refit_timeout_s` is honored: the
  worker extension in each engine enforces it.
- `vllm_cfg.enable_vllm_metrics_logger` requires
  `rollout_checkpointing.telemetry_interval_s`: the rollout telemetry pump drains
  the Dynamo metrics sampler and reports each engine's generated tokens, running
  and waiting requests, and KV-cache usage.

Setup rejects any of these before allocating clusters.

## Run Megatron prefill/decode engines under the SingleController

The SingleController can also put NeMo RL's own Megatron inference engines
behind the Dynamo frontend, split into a prefill pool and a decode pool. Start
from
[`examples/configs/grpo_math_1B_dynamo_pd_megatron_single_controller.yaml`](../../examples/configs/grpo_math_1B_dynamo_pd_megatron_single_controller.yaml)
and launch it with `examples/run_grpo_single_controller.py`. The config
boundary is:

```yaml
async_rl:
  recompute_kv_cache_after_weight_updates: false   # the engines keep their caches
policy:
  megatron_cfg:
    enabled: true            # the engines receive logical weights from a Megatron trainer
  generation:
    backend: dynamo
    refit_transport: null    # collective refit only
    dynamo_cfg:
      engine: megatron
      megatron_root: null    # Megatron-LM tree the adapter imports; null = the Bridge submodule's
    disaggregation:
      enabled: true
      num_prefill_engines: 1
      num_decode_engines: 1
    mcore_generation_config:
      enable_prefix_caching: true
      kv_cache_management_mode: persist
      expose_http_server: false
    vllm_cfg: null
    vllm_kwargs: {}
```

What runs:

- **Engines.** `disaggregation` tiles the inference GPUs into replicas of
  `num_prefill_engines` prefill engines followed by `num_decode_engines` decode
  engines, exactly as for the vLLM fleet. Each engine slot becomes one dedicated
  Megatron inference policy (its own worker group, data parallel one), the same
  object `backend: megatron` builds when `colocated.enabled` is false. Every
  engine has the layout of `policy.megatron_cfg` with
  `mcore_generation_config` applied, plus its own entry in
  `dynamo_cfg.prefill_engines` / `decode_engines` (below); each engine's width
  (TP x PP) must fit on one node.
- **Adapter sidecars.** Next to each engine, a CPU Ray actor runs Megatron-LM's
  Dynamo adapter (`megatron.inference.integrations.dynamo`) from the Dynamo
  virtual environment in external launch mode. It registers the engine with the
  frontend, forwards requests to the engine's coordinator, and carries the KV
  handoff between prefill and decode over NIXL. It never owns GPUs or weights.
- **Frontend, etcd, NATS.** Driver-owned, as for the vLLM fleet. Prefill
  engines register as the `prefill` component and decode engines as `backend`;
  the frontend's prefill router splits each request between them.

Every engine may use its own layout. `dynamo_cfg.prefill_engines` and
`dynamo_cfg.decode_engines` list one entry per prefill and decode engine of a
replica, in launch order; each entry gives that engine's
`tensor_model_parallel_size` and `pipeline_model_parallel_size` (both
required; everything else comes from the shared layout). A non-empty list must have exactly `num_prefill_engines` (`num_decode_engines`)
entries, and every replica repeats the same layouts; an empty list (the
default) runs every engine of that role at the shared layout. Megatron reshards
the KV cache during each handoff, so any prefill engine may hand off to any
decode engine. For example, one prefill engine at TP 4 next to a decode engine
at TP 2 and one at TP 1 x PP 2 is an 8-GPU replica:

```yaml
policy:
  generation:
    dynamo_cfg:
      engine: megatron
      prefill_engines:
        - tensor_model_parallel_size: 4
          pipeline_model_parallel_size: 1
      decode_engines:
        - tensor_model_parallel_size: 2
          pipeline_model_parallel_size: 1
        - tensor_model_parallel_size: 1
          pipeline_model_parallel_size: 2
    disaggregation:
      enabled: true
      num_prefill_engines: 1
      num_decode_engines: 2
    mcore_generation_config:
      tensor_model_parallel_size: 2
```

With `T` trainer ranks, the refit collective places the trainer ranks first,
then the engines in launch order, each starting where the previous one ended:
the prefill engine at `T..T+3`, the decode engines at `T+4..T+5` and
`T+6..T+7`. Per-engine layouts are implemented for `engine: megatron` only; the
managed vLLM fleet runs every worker at the `vllm_cfg` layout and rejects a
non-empty list.

Refit and startup differ from the vLLM fleet:

- The engines are built at the first weight update (they start without
  weights), so the sidecars start at setup but the frontend becomes ready only
  after that first refit. `dynamo_cfg.startup_timeout_s` must cover trainer
  construction plus the first refit; the first resume after a refit waits for
  the frontend to list every engine and the model before rollouts start. That
  first resume also builds each engine and captures its CUDA graphs, so it is
  bounded by at least `startup_timeout_s` instead of the controller's resume
  timeout.
- A request spans two engines, and a prefill engine must not be suspended while
  a decode engine may still be pulling one of its KV blocks (a "handoff pin").
  So the pause stops the prefill engines from scheduling first, keeps decode
  running until every pin on prefill has cleared, then suspends prefill, then
  decode. The SingleController bounds that wait with its pause timeout:
  `async_rl.generation_fleet_health.refit_timeout_s` plus a 60-second grace, or
  360 seconds when `refit_timeout_s` is null. (`dynamo_cfg.control_timeout_s`
  applies only when a caller passes no timeout.) If the
  pins do not clear in time the prefill engines are unpaused again and the
  refit fails rather than leaving the fleet half stopped. Resume brings decode
  back first, then prefill, because decode is the reader of prefill's KV. The
  controller's refit sequence is unchanged.
- The collective is one NCCL group over the trainers plus all engines. Engine
  `i` sits at rank prefix `T` plus the widths of the engines before it, where
  `T` is the trainer world size (`T + i * w` when every engine has width `w`,
  the same placement a vLLM engine would have). The engines receive logical
  (unfused) weights, as the Megatron backend does.
- Shutdown stops decode engines first, confirms their termination to every
  prefill coordinator so pinned blocks are released, then stops prefill.

Limits:

- Prefill/decode only: aggregated Megatron engines behind Dynamo are rejected
  (use `backend: megatron` for that).
- Native rollouts only; NeMo-Gym under the SingleController does not support
  `backend: dynamo`.
- Each engine fits on one node; `refit_transport` must be null;
  `recompute_kv_cache_after_weight_updates` must be false (the engines keep
  their caches, mode `persist`); setup rejects the last two otherwise.
- Data-plane storage units cannot be placed on the `inference` cluster
  (`storage_unit_placement: all` or `[inference]`): the engines reserve those
  GPUs through their own clusters. Place them on `train`.
- Of `dynamo_cfg.worker_args`, the Megatron adapter reads only
  `endpoint_types` and `extra_cli_args`; setting a tool-call or reasoning
  parser, a custom chat template, or `enable_structural_tag` is rejected.
- The engine workers need the `nixl` package in the Megatron environment for
  the KV handoff, and the adapter sidecar needs a Megatron-LM tree with the
  Dynamo integration (`megatron/inference/integrations/dynamo`).

The functional test is
`tests/functional/grpo_dynamo_pd_megatron_single_controller.sh` (three GPUs:
trainer, prefill, decode).

## Run SWE1 with W&B

The three-node nightly recipe targets
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. It uses two 8-GPU training
nodes and one 8-GPU inference node. The inference node runs two TP4/EP4 Dynamo
engines. Download the standard SWE1 split under
`${HF_HOME}/superv3_data/swe1`, then run the registered test-suite driver:

```bash
HF_HOME=/shared/huggingface \
WANDB_API_KEY=<key> \
bash tests/test_suites/llm/grpo-nanov3-30ba3b-3n8g-megatron-dynamo-swe1.sh
```

A successful acceptance run completes four training steps, produces valid
generations after refit, and records worker timelines under
`generation_metrics/*` in TensorBoard and W&B.

## Operational notes

- The driver owns all services. Do not start a separate etcd, NATS, frontend,
  or worker fleet for this mode.
- Startup validates fixed worker membership; a dead or replaced worker fails
  refit instead of serving mixed model versions.
- Shutdown is idempotent and terminates whole subprocess groups, including
  partial-startup failures.
- Fault tolerance and a multi-controller architecture remain follow-up work.
