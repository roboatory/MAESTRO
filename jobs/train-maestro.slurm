#!/bin/bash
#SBATCH --job-name=maestro-all-samples
#SBATCH --account=arjunraj-rohitproject
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=28
#SBATCH --mem=896G
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
#SBATCH --output=output/%x_%j.log
#SBATCH --error=output/%x_%j.log

set -euo pipefail

PROJECT_ROOT="/vast/projects/arjunraj/rohitproject/maestro"
DATA_ROOT="${PROJECT_ROOT}/data/raw/preprocessed-cyto-sources"
DATA_DIRECTORIES=(
    "${DATA_ROOT}/allof"
    "${DATA_ROOT}/prepro"
)

PROJECT_NAME="${PROJECT_NAME:-maestro-allof-prepro-${SLURM_JOB_ID}}"
NUMBER_EPOCHS="${NUMBER_EPOCHS:-500}"
NUMBER_OUTPUTS="${NUMBER_OUTPUTS:-5000}"
SINKHORN_START_EPOCH="${SINKHORN_START_EPOCH:-25}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

RUN_DIRECTORY="${PROJECT_ROOT}/output/${PROJECT_NAME}"
METRICS_PATH="${RUN_DIRECTORY}/gpu.csv"
CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/${USER}/maestro-${SLURM_JOB_ID}"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_DIRECTORY}" "${CACHE_ROOT}/triton" "${CACHE_ROOT}/keops"

module load cuda/12.8.1

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export KEOPS_CACHE_FOLDER="${CACHE_ROOT}/keops"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH" >&2
    exit 1
fi

for data_directory in "${DATA_DIRECTORIES[@]}"; do
    if [[ ! -d "${data_directory}" ]]; then
        echo "Missing data directory: ${data_directory}" >&2
        exit 1
    fi
done

ALLOF_FILE_COUNT=$(find "${DATA_DIRECTORIES[0]}" -maxdepth 1 -type f -name '*.h5' | wc -l)
PREPRO_FILE_COUNT=$(find "${DATA_DIRECTORIES[1]}" -maxdepth 1 -type f -name '*.h5' | wc -l)
TOTAL_FILE_COUNT=$((ALLOF_FILE_COUNT + PREPRO_FILE_COUNT))

if [[ "${TOTAL_FILE_COUNT}" -eq 0 ]]; then
    echo "No .h5 input files were found" >&2
    exit 1
fi

# Resolve the locked environment once before the distributed workers start.
uv sync --frozen --no-dev

if [[ ! -x "${PROJECT_ROOT}/.venv/bin/maestro-train" ]]; then
    echo "The maestro-train entry point was not installed" >&2
    exit 1
fi

DEVICE_COUNT=$(
    .venv/bin/python -c "import torch; print(torch.cuda.device_count())"
)
if [[ "${DEVICE_COUNT}" -ne "${SLURM_NTASKS}" ]]; then
    echo "Expected ${SLURM_NTASKS} GPUs, but PyTorch found ${DEVICE_COUNT}" >&2
    exit 1
fi

if [[ "${DEVICE_COUNT}" -eq 1 ]]; then
    DEVICE_IDS="0,"
else
    DEVICE_IDS=$(seq --separator=, 0 "$((DEVICE_COUNT - 1))")
fi

if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "Resume checkpoint does not exist: ${RESUME_CHECKPOINT}" >&2
    exit 1
fi

git rev-parse HEAD > "${RUN_DIRECTORY}/git-commit.txt"
git status --short > "${RUN_DIRECTORY}/git-status.txt"
git diff --binary HEAD > "${RUN_DIRECTORY}/uncommitted-changes.patch"
sha256sum uv.lock > "${RUN_DIRECTORY}/uv-lock.sha256"
cp "$0" "${RUN_DIRECTORY}/submitted.slurm"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Project: ${PROJECT_NAME}"
echo "Input files: ${TOTAL_FILE_COUNT} (${ALLOF_FILE_COUNT} allof + ${PREPRO_FILE_COUNT} prepro)"
echo "Epochs: ${NUMBER_EPOCHS}"
echo "Reconstruction outputs: ${NUMBER_OUTPUTS}"
echo "Sinkhorn start epoch: ${SINKHORN_START_EPOCH}"
echo "CUDA devices: ${DEVICE_IDS}"
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
} > "${METRICS_PATH}" &
GPU_MONITOR_PID=$!

cleanup() {
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

TRAINING_ARGUMENTS=(
    --project "${PROJECT_NAME}"
    --devices "${DEVICE_IDS}"
    --data_dirs "${DATA_DIRECTORIES[@]}"
    --number_cells_subset 40000
    --dim_input 30
    --num_outputs "${NUMBER_OUTPUTS}"
    --num_inds 16
    --dim_hidden 384
    --dim_latent 256
    --num_heads 1
    --ln
    --initial_lr 1e-4
    --min_lr 1e-12
    --epochs "${NUMBER_EPOCHS}"
    --sinkhorn_start "${SINKHORN_START_EPOCH}"
    --student_temperature 0.10
    --teacher_temperature 0.07
    --center_momentum 0.99
    --teacher_beta 0.99
    --mode Train
)

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    TRAINING_ARGUMENTS+=(--ckpt_resume "${RESUME_CHECKPOINT}")
fi

/usr/bin/time -v srun \
    --kill-on-bad-exit=1 \
    --cpu-bind=cores \
    .venv/bin/maestro-train \
    "${TRAINING_ARGUMENTS[@]}"

echo "Finished: $(date --iso-8601=seconds)"
