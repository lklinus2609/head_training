#!/bin/bash
#SBATCH --job-name=d4head-stage2
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL

eval "$($WORK/miniconda3/bin/conda shell.bash hook)"
conda activate d4head

cd $WORK/D4_head/train_head

# NCCL settings for TACC
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

torchrun \
    --nproc_per_node=3 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    training/train_stage2.py \
    --config configs/stage2_pretrain.yaml \
    --resume auto
