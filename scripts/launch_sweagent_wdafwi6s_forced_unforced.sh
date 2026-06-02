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

BASE_DIR="${BASE_DIR:-/scratch/fsw/portfolios/nemotron/projects/nemotron_rl_systems/users/laurad/nemo/worktrees/RL-nemo-gym-validation-limiter}"
cd "${BASE_DIR}"

export CONTAINER="${CONTAINER:-/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/tene/nemo_rl_0521.sqsh}"
export MOUNTS="${MOUNTS:-/lustre:/lustre,/scratch:/scratch,/home/laurad:/home/laurad,/scratch/fsw/portfolios/nemotron/users/laurad/swegym-sifs/apptainer-conda-home:/home/conda,${BASE_DIR}:/opt/nemo-rl}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-64}"

NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-24}"
SLURM_PARTITION="${SLURM_PARTITION:-batch_long}"
SUBMIT_ACCOUNT="${SUBMIT_ACCOUNT:-nemotron_rl_systems}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LAG_MODES="${LAG_MODES:-unforced forced}"
CONFIG_PATH="${CONFIG_PATH:-examples/nemo_gym/grpo_nanov3_sweagent_wdafwi6s.yaml}"
EXCLUDE_NODES="${EXCLUDE_NODES:-nvl72d068-T01,nvl72d003-T07,nvl72d078-T[01-08],nvl72d091-T[01-17]}"

METADATA_FILE="${BASE_DIR}/sweagent_wdafwi6s_${RUN_TAG}.jobs.env"
LATEST_METADATA_FILE="${BASE_DIR}/sweagent_wdafwi6s.latest.jobs.env"

unset WANDB_RUN_ID WANDB_RESUME
: > "${METADATA_FILE}"

for lag_mode in ${LAG_MODES}; do
  case "${lag_mode}" in
    forced|unforced) ;;
    *)
      echo "Unsupported lag mode: ${lag_mode}" >&2
      exit 1
      ;;
  esac

  JOB_NAME="nemoRL_wdafwi6s_sweagent_${lag_mode}_age6_prompts128_gens16_5arm64_${RUN_TAG}"
  CHECKPOINT_DIR="/lustre/fsw/portfolios/nemotron/users/laurad/${JOB_NAME}"
  label="sweagent_${lag_mode}"

  export COMMAND="cd /opt/nemo-rl && JOB_NAME=${JOB_NAME} LAG_MODE=${lag_mode} CHECKPOINT_DIR=${CHECKPOINT_DIR} CONFIG_PATH=${CONFIG_PATH} bash scripts/run_sweagent_wdafwi6s_system_vllm.sh"

  echo "Submitting ${JOB_NAME}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'COMMAND=%s\n' "${COMMAND}"
    printf 'label=%q job_id=%q job_name=%q checkpoint_dir=%q lag_mode=%q\n' \
      "${label}" "DRY_RUN" "${JOB_NAME}" "${CHECKPOINT_DIR}" "${lag_mode}" | tee -a "${METADATA_FILE}"
    continue
  fi

  sbatch_args=(
    --nodes="${NUM_ACTOR_NODES}"
    --account="${SUBMIT_ACCOUNT}"
    --job-name="${SUBMIT_ACCOUNT}:${JOB_NAME}"
    --partition="${SLURM_PARTITION}"
    --time="${TIME_LIMIT}"
    --gres="gpu:${GPUS_PER_NODE}"
  )
  if [[ -n "${EXCLUDE_NODES}" ]]; then
    sbatch_args+=(--exclude="${EXCLUDE_NODES}")
  fi

  output="$(sbatch "${sbatch_args[@]}" ray.sub)"
  echo "${output}"
  job_id="$(awk '/Submitted batch job/ {print $NF}' <<< "${output}")"
  if [[ -z "${job_id}" ]]; then
    echo "Could not parse job id from sbatch output: ${output}" >&2
    exit 1
  fi
  printf 'label=%q job_id=%q job_name=%q checkpoint_dir=%q lag_mode=%q\n' \
    "${label}" "${job_id}" "${JOB_NAME}" "${CHECKPOINT_DIR}" "${lag_mode}" | tee -a "${METADATA_FILE}"
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  cp "${METADATA_FILE}" "${LATEST_METADATA_FILE}"
fi

echo "metadata=${METADATA_FILE}"
