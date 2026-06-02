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

: "${JOB_NAME:?JOB_NAME must be set}"
: "${LAG_MODE:?LAG_MODE must be set to forced or unforced}"

case "${LAG_MODE}" in
  forced|unforced) ;;
  *)
    echo "Unsupported LAG_MODE=${LAG_MODE}; expected forced or unforced" >&2
    exit 1
    ;;
esac

cd /opt/nemo-rl

CONFIG_PATH="${CONFIG_PATH:-examples/nemo_gym/grpo_nanov3_sweagent_wdafwi6s.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/lustre/fsw/portfolios/nemotron/users/laurad/${JOB_NAME}}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
# shellcheck disable=SC2206
EXTRA_OVERRIDES_ARGS=(${EXTRA_OVERRIDES})

export PYTHONUNBUFFERED=1
export NRL_IGNORE_VERSION_MISMATCH=1
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export NEMO_RL_DISTRIBUTED_TIMEOUT_SECONDS="${NEMO_RL_DISTRIBUTED_TIMEOUT_SECONDS:-3600}"
export WANDB_ENTITY=adlr
export WANDB_PROJECT=nano-v3-megatron-inference
export PATH=/scratch/fsw/portfolios/nemotron/users/laurad/swegym-sifs/bootstrap-apptainer/apptainer-conda-linux-aarch64/bin:${PATH}
export NEMO_GYM_APPTAINER_BIN=/scratch/fsw/portfolios/nemotron/users/laurad/swegym-sifs/bootstrap-apptainer/apptainer-conda-linux-aarch64/bin/apptainer
export NEMO_GYM_APPTAINER_SESSIONDIR=/home/conda/feedstock_root/build_artifacts/apptainer_1764716053350/_h_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_p/var/apptainer/mnt/session
export APPTAINER_TMPDIR=/tmp/laurad/apptainer-${SLURM_JOB_ID:-manual}
export APPTAINER_CACHEDIR=/tmp/laurad/apptainer-cache-${SLURM_JOB_ID:-manual}
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

"${NEMO_GYM_APPTAINER_BIN}" --version

SIF_DIR=/scratch/fsw/portfolios/nemotron/users/laurad/swegym-arm64-sifs
test -s "${SIF_DIR}/sweb.eval.arm64.facebookresearch_s_hydra-1616.sif"
test -s "${SIF_DIR}/sweb.eval.arm64.getmoto_s_moto-7365.sif"
test -s "${SIF_DIR}/sweb.eval.arm64.iterative_s_dvc-6893.sif"
test -s "${SIF_DIR}/sweb.eval.arm64.pandas-dev_s_pandas-50714.sif"
test -s "${SIF_DIR}/sweb.eval.arm64.python_s_mypy-15045.sif"

TRAIN_DATA=/scratch/fsw/portfolios/nemotron/users/laurad/nemo-gym-swe-agent-data/swe_agents/swegym_for_sweagent_and_openhands.nrl.arm64-5sif-train.jsonl
VAL_DATA=/scratch/fsw/portfolios/nemotron/users/laurad/nemo-gym-swe-agent-data/swe_agents/swegym_for_sweagent_and_openhands.nrl.arm64-5sif-val.jsonl
test "$(wc -l < "${TRAIN_DATA}")" -eq 5
test "$(wc -l < "${VAL_DATA}")" -eq 5

SWE_AGENT_EXTRA_DEPS=/scratch/fsw/portfolios/nemotron/users/laurad/nemo-gym-swe-agent-data/sweagent-python-deps/py313-aarch64
export PYTHONPATH=${SWE_AGENT_EXTRA_DEPS}:/scratch/fsw/portfolios/nemotron/projects/nemotron_rl_systems/users/laurad/nemo/RL/3rdparty/vllm:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}

SWE_AGENT_DIR=/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents
(
  cd "${SWE_AGENT_DIR}"
  (
    flock 9
    if [[ ! -f "${SWE_AGENT_EXTRA_DEPS}/.ready-v5" ]]; then
      rm -rf "${SWE_AGENT_EXTRA_DEPS}"
      mkdir -p "${SWE_AGENT_EXTRA_DEPS}"
      uv pip install --target "${SWE_AGENT_EXTRA_DEPS}" devtools rich omegaconf hydra-core orjson yappi tomlkit gprof2dot pydot
      touch "${SWE_AGENT_EXTRA_DEPS}/.ready-v5"
    fi

    rm -rf .venv
    mkdir -p .venv/bin
    ln -s /opt/nemo_rl_venv/bin/python .venv/bin/python
    {
      echo "export VIRTUAL_ENV=${SWE_AGENT_DIR}/.venv"
      echo "export PATH=/opt/nemo_rl_venv/bin:\${PATH}"
      echo "export PYTHONPATH=${SWE_AGENT_EXTRA_DEPS}:\${PYTHONPATH:-}"
    } > .venv/bin/activate
  ) 9>.venv.lock
  source .venv/bin/activate
  python - <<'PY'
import devtools
import fastapi
import gprof2dot
import hydra
import omegaconf
import orjson
import pydot
import ray
import rich
import nemo_gym
import tomlkit
import uvicorn
import yappi

if ray.__version__ != "2.54.0":
    raise SystemExit(f"Expected ray 2.54.0 in SWE-agent venv, got {ray.__version__}")
print(f"swe_agents venv ready with ray {ray.__version__}")
PY
)

python - <<'PY'
import sys
import vllm
from nemo_gym.cli import RunHelper
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env

print(f"driver_python={sys.executable}")
print(f"vllm={getattr(vllm, '__version__', 'unknown')} from {vllm.__file__}")
print(f"nemo_gym RunHelper={RunHelper}")
expected_system = {
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
    "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
    "nemo_rl.algorithms.async_utils.ReplayBuffer",
    "nemo_rl.environments.nemo_gym.NemoGym",
}
for actor in [
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
    "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
    "nemo_rl.algorithms.async_utils.ReplayBuffer",
    "nemo_rl.environments.nemo_gym.NemoGym",
]:
    py_exec = get_actor_python_env(actor)
    print(f"{actor} py_executable={py_exec}")
    if actor in expected_system and py_exec != sys.executable:
        raise SystemExit(f"Expected system executable for {actor}, got: {py_exec}")
    if actor.endswith("MegatronPolicyWorker") and "--extra mcore" not in py_exec:
        raise SystemExit(f"Expected uv mcore executable for {actor}, got: {py_exec}")
PY

python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${CONFIG_PATH}" \
  grpo.async_grpo.lag_mode="${LAG_MODE}" \
  checkpointing.checkpoint_dir="${CHECKPOINT_DIR}" \
  logger.wandb.name="${JOB_NAME}" \
  logger.log_dir="results/${JOB_NAME}" \
  "${EXTRA_OVERRIDES_ARGS[@]}"
