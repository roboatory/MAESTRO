#!/bin/bash
#SBATCH --job-name=maestro-all-samples
#SBATCH --account=arjunraj-rohitproject
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=28
#SBATCH --mem=896G
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00
#SBATCH --output=experiments/%x_%j.log
#SBATCH --error=experiments/%x_%j.log

set -euo pipefail

PROJECT_ROOT="/vast/projects/arjunraj/rohitproject/maestro"

RUN_PREFIX="${MAESTRO_RUN_PREFIX:-maestro-allof-prepro}"
if [[ ! "${RUN_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid MAESTRO_RUN_PREFIX: ${RUN_PREFIX}" >&2
    exit 1
fi

PROJECT_NAME="${RUN_PREFIX}-${SLURM_JOB_ID}"
HYDRA_OVERRIDES=("run.name=${PROJECT_NAME}")
if [[ -n "${MAESTRO_NUMBER_EPOCHS:-}" ]]; then
    if [[ ! "${MAESTRO_NUMBER_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "MAESTRO_NUMBER_EPOCHS must be a positive integer" >&2
        exit 1
    fi
    HYDRA_OVERRIDES+=("training.number_epochs=${MAESTRO_NUMBER_EPOCHS}")
fi

RUN_DIRECTORY="${PROJECT_ROOT}/experiments/${PROJECT_NAME}/job"
GPU_METRICS_PATH="${RUN_DIRECTORY}/gpu.csv"
CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/${USER}/maestro-${SLURM_JOB_ID}"
SLURM_LOG_SOURCE="${PROJECT_ROOT}/experiments/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
SLURM_LOG_PATH="${RUN_DIRECTORY}/slurm.log"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_DIRECTORY}" "${CACHE_ROOT}/triton" "${CACHE_ROOT}/keops"

# Slurm opens its output file before this script can create the run directory.
# Add a hard link inside the run, then remove the temporary top-level name during
# cleanup; Slurm's open descriptor continues writing to the run-local file.
if [[ -f "${SLURM_LOG_SOURCE}" && ! -e "${SLURM_LOG_PATH}" ]]; then
    ln "${SLURM_LOG_SOURCE}" "${SLURM_LOG_PATH}"
fi

module load cuda/12.8.1

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export KEOPS_CACHE_FOLDER="${CACHE_ROOT}/keops"
CUDA_LIBRARY_DIRECTORY="${CUDA_HOME}/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="${CUDA_LIBRARY_DIRECTORY}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_LIBRARY_DIRECTORY}:${LIBRARY_PATH:-}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH" >&2
    exit 1
fi

# Resolve the locked environment once before the distributed workers start.
uv sync --frozen --no-dev --inexact

if [[ ! -x "${PROJECT_ROOT}/.venv/bin/maestro-run" ]]; then
    echo "The maestro-run entry point was not installed" >&2
    exit 1
fi

DEVICE_COUNT=$(
    .venv/bin/python -c "import torch; print(torch.cuda.device_count())"
)
if [[ "${DEVICE_COUNT}" -ne "${SLURM_NTASKS}" ]]; then
    echo "Expected ${SLURM_NTASKS} GPUs, but PyTorch found ${DEVICE_COUNT}" >&2
    exit 1
fi

if ! .venv/bin/python -c \
    "import pykeops; raise SystemExit(0 if pykeops.config.gpu_available else 1)"; then
    echo "PyKeOps could not initialize its CUDA backend" >&2
    exit 1
fi

git rev-parse HEAD > "${RUN_DIRECTORY}/git-commit.txt"
git status --short > "${RUN_DIRECTORY}/git-status.txt"
git diff --binary HEAD > "${RUN_DIRECTORY}/uncommitted-changes.patch"
sha256sum uv.lock > "${RUN_DIRECTORY}/uv-lock.sha256"
cp "$0" "${RUN_DIRECTORY}/submitted.slurm"
cp configs/train.yaml "${RUN_DIRECTORY}/base-config.yaml"
scontrol show job "${SLURM_JOB_ID}" --oneliner \
    > "${RUN_DIRECTORY}/slurm-allocation.txt"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Project: ${PROJECT_NAME}"
echo "Hydra overrides: ${HYDRA_OVERRIDES[*]}"
echo "CUDA devices: 0,1,2,3"
echo "CUDA_HOME: ${CUDA_HOME}"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader

{
    echo "timestamp,gpu_index,utilization_gpu_percent,utilization_memory_percent,memory_used_mib,memory_total_mib,power_draw_watts,temperature_c"
    while true; do
        TIMESTAMP=$(date +%s)
        nvidia-smi \
            --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
            --format=csv,noheader,nounits \
            | sed "s/^/${TIMESTAMP},/"
        sleep 5
    done
} > "${GPU_METRICS_PATH}" &
GPU_MONITOR_PID=$!

cleanup() {
    exit_status=$?
    trap - EXIT
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    printf '%s\n' "${exit_status}" > "${RUN_DIRECTORY}/exit-status.txt"
    sacct -j "${SLURM_JOB_ID}" --parsable2 \
        --format=JobIDRaw,JobName,Submit,Eligible,Start,ElapsedRaw,State,ExitCode,AllocTRES,MaxRSS \
        > "${RUN_DIRECTORY}/slurm-accounting.csv" || true
    echo "Finished: $(date --iso-8601=seconds) (exit status ${exit_status})"
    if [[ -e "${SLURM_LOG_SOURCE}" && "${SLURM_LOG_SOURCE}" -ef "${SLURM_LOG_PATH}" ]]; then
        unlink "${SLURM_LOG_SOURCE}"
    fi
    exit "${exit_status}"
}
trap cleanup EXIT

.venv/bin/maestro-run \
    --cfg job \
    --resolve \
    "${HYDRA_OVERRIDES[@]}" > "${RUN_DIRECTORY}/resolved-config.yaml"

/usr/bin/time -v srun \
    --kill-on-bad-exit=1 \
    --cpu-bind=cores \
    .venv/bin/maestro-run \
    "${HYDRA_OVERRIDES[@]}"
