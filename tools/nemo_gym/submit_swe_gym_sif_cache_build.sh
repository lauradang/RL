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

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}"
: "${SLURM_PARTITION:?Set SLURM_PARTITION}"

DATA_PATH="${DATA_PATH:-/scratch/fsw/portfolios/nemotron/users/laurad/nemo-gym-swe-agent-data/swe_agents/swegym_for_sweagent_and_openhands.nrl.available-sifs.jsonl}"
SIF_DIR="${SIF_DIR:-/scratch/fsw/portfolios/nemotron/users/laurad/swegym-sifs}"
SCRIPT_PATH="${SCRIPT_PATH:-/scratch/fsw/portfolios/nemotron/users/laurad/nemo-gym-swe-agent-tools/build_swe_gym_sif_cache.py}"
NUM_SHARDS="${NUM_SHARDS:-256}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"
MAX_WORKERS="${MAX_WORKERS:-1}"
TIME_LIMIT="${TIME_LIMIT:-8:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
JOB_NAME="${JOB_NAME:-swegym-sif-cache}"
QOS_ARGS=()
if [[ -n "${SLURM_QOS:-}" ]]; then
    QOS_ARGS=(--qos="${SLURM_QOS}")
fi

mkdir -p "${SIF_DIR}/logs" "${SIF_DIR}/.apptainer-cache" "${SIF_DIR}/.apptainer-tmp"

TASK_SCRIPT="${SIF_DIR}/build_swe_gym_sif_cache_task.sh"
cat > "${TASK_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p '${SIF_DIR}/.apptainer-tmp/'"\${SLURM_ARRAY_TASK_ID}"
export APPTAINER_CACHEDIR='${SIF_DIR}/.apptainer-cache'
export APPTAINER_TMPDIR='${SIF_DIR}/.apptainer-tmp/'"\${SLURM_ARRAY_TASK_ID}"
python '${SCRIPT_PATH}' \
  --input '${DATA_PATH}' \
  --output-dir '${SIF_DIR}' \
  --build \
  --num-shards '${NUM_SHARDS}' \
  --shard-index "\${SLURM_ARRAY_TASK_ID}" \
  --max-workers '${MAX_WORKERS}'
EOF
chmod +x "${TASK_SCRIPT}"

sbatch \
    --job-name="${JOB_NAME}" \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION}" \
    "${QOS_ARGS[@]}" \
    --time="${TIME_LIMIT}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --array="0-$((NUM_SHARDS - 1))%${MAX_CONCURRENT}" \
    --output="${SIF_DIR}/logs/%x-%A_%a.out" \
    --error="${SIF_DIR}/logs/%x-%A_%a.err" \
    "${TASK_SCRIPT}"
