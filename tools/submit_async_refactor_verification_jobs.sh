#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/..")
cd "$PROJECT_ROOT"

ACCOUNT=${ACCOUNT:-nemotron_rl_systems}
PARTITION=${PARTITION:-batch_long}
NUM_NODES=${NUM_NODES:-24}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TIME_LIMIT=${TIME_LIMIT:-24:00:00}
WANDB_PROJECT=${WANDB_PROJECT:-nano-v3-megatron-inference}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/lustre/fsw/portfolios/nemotron/users/laurad}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
DRYRUN=${DRYRUN:-0}
NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true}
NRL_IGNORE_VERSION_MISMATCH=${NRL_IGNORE_VERSION_MISMATCH:-1}

if [[ "$DRYRUN" != "1" ]]; then
    : "${CONTAINER:?Set CONTAINER to the image used by ray.sub.}"
    : "${MOUNTS:?Set MOUNTS to the mount list required by ray.sub.}"
fi

if [[ "$DRYRUN" != "1" ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "[ERROR] sbatch is not available in PATH. Run this from a Slurm submit host." >&2
    exit 1
fi

COMMON_ARGS=(
    uv run python examples/nemo_gym/run_grpo_nemo_gym.py
    --config examples/nemo_gym/grpo_nanov3_vllm.yaml
    grpo.async_grpo.max_trajectory_age_steps=6
    +grpo.async_grpo.in_flight_weight_updates=true
    grpo.num_prompts_per_step=128
    grpo.num_generations_per_prompt=16
    grpo.val_period=256
    policy.train_global_batch_size=128
    policy.generation.colocated.resources.num_nodes=8
    cluster.num_nodes=24
    cluster.gpus_per_node=4
    policy.generation.colocated.resources.gpus_per_node=4
    data.train.data_path=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/tene/nemo-rl-workspace/data/train-split.jsonl
    data.validation.data_path=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/tene/nemo-rl-workspace/data/val-split.jsonl
    checkpointing.enabled=true
    checkpointing.save_period=256
    +env.nemo_gym.skip_venv_if_present=true
    logger.wandb.project="$WANDB_PROJECT"
)

submit_job() {
    local lag_mode=$1
    local run_name="nemoRL_async_unified_cls_refactor_${lag_mode}_age6_seqpack_mfix_val256_ckpt256_p128_g16_${TS}"
    local checkpoint_dir="${CHECKPOINT_ROOT}/${run_name}"

    local cmd=(
        "${COMMON_ARGS[@]}"
        "+grpo.async_grpo.lag_mode=${lag_mode}"
        "checkpointing.checkpoint_dir=${checkpoint_dir}"
        "logger.wandb.name=${run_name}"
    )

    local command
    printf -v command "%q " "${cmd[@]}"

    local launch_command
    printf -v launch_command \
        "cd /opt/nemo-rl && export PYTHONUNBUFFERED=1 && export NRL_FORCE_REBUILD_VENVS=%q && export NRL_IGNORE_VERSION_MISMATCH=%q && export PYTHONPATH=/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:\\\${PYTHONPATH:-} && %s" \
        "$NRL_FORCE_REBUILD_VENVS" \
        "$NRL_IGNORE_VERSION_MISMATCH" \
        "$command"

    echo "[INFO] Prepared ${lag_mode} verification job: ${run_name}"
    echo "[INFO] Checkpoint dir: ${checkpoint_dir}"
    echo "[INFO] Command: ${launch_command}"

    if [[ "$DRYRUN" == "1" ]]; then
        return
    fi

    COMMAND="$launch_command" \
    CONTAINER="$CONTAINER" \
    GPUS_PER_NODE="$GPUS_PER_NODE" \
    MOUNTS="$MOUNTS" \
    sbatch \
        --nodes="$NUM_NODES" \
        --account="$ACCOUNT" \
        --job-name="${ACCOUNT}:${run_name}" \
        --partition="$PARTITION" \
        --time="$TIME_LIMIT" \
        --gres="gpu:${GPUS_PER_NODE}" \
        ray.sub
}

submit_job forced
submit_job unforced
