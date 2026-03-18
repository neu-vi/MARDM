#!/bin/bash
set -euo pipefail

mkdir -p slurm_logs

N_TASKS="${1:-200}"
SHARD_SIZE="${2:-437}"
MAX_CONCURRENT="${3:-200}"

export SHARD_SIZE="$SHARD_SIZE"

ARRAY_SPEC="0-$(($N_TASKS - 1))%${MAX_CONCURRENT}"

echo "Submitting array: ${ARRAY_SPEC}"
echo "    SHARD_SIZE=${SHARD_SIZE}"

sbatch --export=ALL,SHARD_SIZE="$SHARD_SIZE" --array="${ARRAY_SPEC}" runner.slurm
#awk -F, 'NR>1 {print int($1/437)}' missing_outputs.csv | sort -n | uniq
#sbatch --export=ALL,SHARD_SIZE=437 --array="38,89,136,144%4" runner.slurm
