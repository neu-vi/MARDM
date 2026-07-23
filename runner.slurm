#!/bin/bash

#SBATCH -A shakeri-lab
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH -c 1
#SBATCH -t 72:00:00
#SBATCH -o ./slurm_logs/%x_%A_%a.out
#SBATCH -e ./slurm_logs/%x_%A_%a.err

set -euo pipefail

CONDA_ENV_PATH="/home/tsl2b/.conda/envs/MARDM"
PYTHON_EX="${CONDA_ENV_PATH}/bin/python"

: "${SHARD_SIZE:=3}"
NAME="MARDM_SiT_XL"

now=$(date -Iseconds)
wallclock=$(squeue -h -j "${SLURM_JOB_ID}" -o %l || true)
gpuname=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)

echo "======================================"
echo "[JOB ID]: ${SLURM_ARRAY_JOB_ID}"
echo "[ARRAY ID]: ${SLURM_ARRAY_TASK_ID}"
echo "[JOB NAME]: ${SLURM_JOB_NAME}"
echo "[WALLCLOCK]: ${wallclock}"
echo "[START TIME]: ${now}"
echo "[GPU]: ${gpuname}"
echo "======================================"

$PYTHON_EX sample.py --name "$NAME" --shard_size "$SHARD_SIZE" --shard_index "$SLURM_ARRAY_TASK_ID" --descriptions --batch_size 8