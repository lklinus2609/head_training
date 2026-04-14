#!/bin/bash
# Usage:
#   sbatch -N 1 -t 12:00:00 --partition gpu-a100 -A IRI25030 slurm/run_stage2.sh
#   sbatch -N 1 -t 24:00:00 --partition gpu-a100 -A IRI25030 slurm/run_stage2.sh --resume auto
#   sbatch -N 1 -t 24:00:00 --partition gpu-a100 -A IRI25030 slurm/run_stage2.sh --config configs/stage2_pretrain.yaml --resume auto
#
# All args after the script name are forwarded to training/train_stage2.py.
#SBATCH --job-name=d4head-stage2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

eval "$($WORK/miniconda3/bin/conda shell.bash hook)"
conda activate d4head

cd $WORK/head_training

# Detect GPU count from Slurm allocation (falls back to 1)
NPROC=$(echo "${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}" | awk -F',' '{print $1}')

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export MASTER_ADDR=localhost
export MASTER_PORT=${MASTER_PORT:-29500}

# Default config if caller didn't pass one
if ! printf '%s\n' "$@" | grep -q -- '--config'; then
    set -- --config configs/stage2_pretrain.yaml "$@"
fi

torchrun \
    --nproc_per_node="$NPROC" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    training/train_stage2.py "$@"
