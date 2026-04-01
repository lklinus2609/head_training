#!/bin/bash
#SBATCH --job-name=d4head-preprocess
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL

eval "$($WORK/miniconda3/bin/conda shell.bash hook)"
conda activate d4head

cd $WORK/D4_head/train_head

# Step 1: Download (if not already done)
python data/download_beat2.py --output_dir $WORK/data/beat2_raw

# Step 2: Preprocess
python data/preprocess.py \
    --config configs/base.yaml \
    --raw_dir $WORK/data/beat2_raw \
    --output_dir $WORK/data/beat2_processed \
    --num_workers 32 \
    --audio_feature mel
