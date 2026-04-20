#!/bin/bash
# One-time environment setup for TACC (LS6 / Stampede3).
# Run this on the login node after cloning the repo.

set -euo pipefail

echo "=== D4 Head Training Environment Setup ==="

# Install Miniconda to $WORK if not already present
if [ ! -d "$WORK/miniconda3" ]; then
    echo "Installing Miniconda to \$WORK/miniconda3..."
    cd "$WORK"
    mkdir -p src && cd src
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$WORK/miniconda3"
    rm Miniconda3-latest-Linux-x86_64.sh
    echo "Miniconda installed."
else
    echo "Miniconda already installed at \$WORK/miniconda3"
fi

# Initialize conda
eval "$($WORK/miniconda3/bin/conda shell.bash hook)"

# Create conda environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"

if conda env list | awk '{print $1}' | grep -qx "d4head"; then
    echo "Conda environment 'd4head' already exists. Updating..."
    conda env update -f "$TRAIN_DIR/environment.yml" --prune
else
    echo "Creating conda environment 'd4head'..."
    conda env create -f "$TRAIN_DIR/environment.yml"
fi

conda activate d4head

# Install the training package in editable mode
pip install -e "$TRAIN_DIR"

# Expose bundled NVRTC so cuDNN v8 can JIT-compile conv kernels (eliminates
# "install nvrtc.so" UserWarning during training). Writes an activate.d hook
# that fires on every `conda activate d4head`.
NVRTC_LIB_DIR=$(ls -d "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/cuda_nvrtc/lib 2>/dev/null | head -n 1)
if [ -n "$NVRTC_LIB_DIR" ] && [ -d "$NVRTC_LIB_DIR" ]; then
    mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
    printf 'export LD_LIBRARY_PATH="%s:$LD_LIBRARY_PATH"\n' "$NVRTC_LIB_DIR" \
        > "$CONDA_PREFIX/etc/conda/activate.d/nvrtc.sh"
    echo "Registered NVRTC lib path in conda activate.d hook."
fi

# Create necessary directories
mkdir -p "$WORK/data/beat2_raw"
mkdir -p "$WORK/data/beat2_processed"
mkdir -p "$WORK/models/flame"
mkdir -p "$WORK/checkpoints/d4head"
mkdir -p "$WORK/wandb"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "IMPORTANT: Download the FLAME model manually:"
echo "  1. Register at https://flame.is.tue.mpg.de/"
echo "  2. Download generic_model.pkl"
echo "  3. Place it at: \$WORK/models/flame/generic_model.pkl"
echo ""
echo "To activate the environment: conda activate d4head"
