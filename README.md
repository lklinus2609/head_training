# Training Pipeline: Adversarial Full-Face Motion Generation

Training pipeline for generating realistic, speech-synchronized facial animation for humanoid robots. Learns natural facial motion from the BEAT2 dataset using an adversarial motion prior, operating entirely in FLAME expression parameter space (100 dimensions).

All training stages are hardware-independent and can be completed before the physical robot is built.

---

## Table of Contents

- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Stage 1: Data Preparation](#stage-1-data-preparation)
- [Stage 2: Generator Pretraining](#stage-2-generator-pretraining)
- [Stage 3: Adversarial Co-Training](#stage-3-adversarial-co-training)
- [Web Viewer](#web-viewer)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [TACC Cluster Usage](#tacc-cluster-usage)

---

## Requirements

- Python 3.10+
- PyTorch 2.1+ with CUDA
- FLAME model (register and download from https://flame.is.tue.mpg.de/)
- BEAT2 dataset (downloaded automatically from HuggingFace)
- wandb account for experiment tracking
- TACC access for GPU training (or any SLURM cluster with NVIDIA GPUs)

## Project Structure

```
train_head/
├── configs/                        # YAML configuration files
│   ├── base.yaml                   #   Shared defaults (paths, data, model architecture)
│   ├── stage2_pretrain.yaml        #   Generator pretraining hyperparameters
│   └── stage3_adversarial.yaml     #   Adversarial co-training hyperparameters
├── configs_schema.py               # Dataclass validation for all configs
├── data/                           # Data pipeline
│   ├── download_beat2.py           #   Download BEAT2 from HuggingFace
│   ├── preprocess.py               #   Extract FLAME params, audio features, write HDF5
│   ├── audio_features.py           #   Mel spectrogram and Wav2Vec 2.0 extraction
│   ├── flame_utils.py              #   Temporal derivatives, normalization, resampling
│   └── dataset.py                  #   PyTorch Dataset classes for HDF5 data
├── models/                         # Neural network architectures
│   ├── generator.py                #   Autoregressive transformer (audio+emotion -> FLAME)
│   ├── discriminator.py            #   Audio-conditioned temporal discriminator
│   ├── audio_encoder.py            #   1D convolutional audio feature encoder
│   ├── layers.py                   #   AdaptiveLayerNorm, sinusoidal PE, causal mask
│   └── flame_decoder.py            #   FLAME mesh decoder for LVE metric
├── training/                       # Training loops and entry points
│   ├── train_stage2.py             #   Entry point: generator pretraining
│   ├── train_stage3.py             #   Entry point: adversarial co-training
│   ├── trainer_stage2.py           #   Stage 2 training loop class
│   ├── trainer_stage3.py           #   Stage 3 training loop class
│   ├── losses.py                   #   L1 recon, BCE discriminator, gradient penalty
│   └── schedulers.py               #   Adversarial weight warmup, LR schedules
├── evaluation/                     # Metrics and visualization
│   ├── metrics.py                  #   LVE, expression MSE, Frechet Gesture Distance
│   └── visualize.py                #   Expression trajectory plots, FLAME rendering
├── viewer/                         # Web-based FLAME expression viewer
│   ├── export_flame_mesh.py        #   Export FLAME mesh + expression basis to binary
│   ├── generate_sequence.py        #   Run model inference, save expression sequences
│   ├── server.py                   #   FastAPI server (REST + WebSocket)
│   └── static/                     #   Frontend (Three.js viewer)
│       ├── index.html
│       ├── viewer.js
│       └── style.css
├── utils/                          # Shared utilities
│   ├── ddp.py                      #   DDP setup, rank helpers
│   ├── checkpoint.py               #   Save/load with auto-resume
│   ├── logging_utils.py            #   wandb initialization and logging
│   └── seed.py                     #   Deterministic seeding
├── slurm/                          # TACC job scripts
│   ├── setup_env.sh                #   One-time environment setup
│   ├── preprocess.sbatch           #   Data preprocessing job
│   ├── train_stage2.sbatch         #   Stage 2 GPU training job
│   └── train_stage3.sbatch         #   Stage 3 GPU training job
├── environment.yml                 # Conda environment specification
├── requirements.txt                # pip requirements
└── setup.py                        # Editable install
```

---

## Environment Setup

### Local development

```bash
cd train_head
conda env create -f environment.yml
conda activate d4head
pip install -e .
```

### TACC cluster

Run the one-time setup script on the login node:

```bash
bash slurm/setup_env.sh
```

This installs Miniconda to `$WORK/miniconda3`, creates the `d4head` conda environment, and sets up directory structure for data, checkpoints, and wandb logs.

You must also download the FLAME model manually:

1. Register at https://flame.is.tue.mpg.de/
2. Download `generic_model.pkl`
3. Place it at `$WORK/models/flame/generic_model.pkl`

---

## Stage 1: Data Preparation

### Download BEAT2

```bash
# Inspect the dataset structure first
python data/download_beat2.py --output_dir $WORK/data/beat2_raw --dry_run

# Full download (~60 GB)
python data/download_beat2.py --output_dir $WORK/data/beat2_raw
```

### Preprocess

Preprocessing extracts FLAME expression parameters, computes temporal derivatives (velocity, acceleration), extracts audio features, and writes everything to HDF5 files split by speaker.

```bash
# Inspect one utterance
python data/preprocess.py --config configs/base.yaml --dry_run

# Full preprocessing
python data/preprocess.py \
    --config configs/base.yaml \
    --raw_dir $WORK/data/beat2_raw \
    --output_dir $WORK/data/beat2_processed \
    --audio_feature mel \
    --num_workers 32
```

Audio feature options:
- `mel` -- 80-dimensional log-mel spectrograms (25ms window, 10ms hop). Lightweight, fast to compute.
- `wav2vec` -- 768-dimensional Wav2Vec 2.0 embeddings. Richer features, requires GPU for extraction.

The preprocessor writes:
- `train.h5` and `val.h5` -- HDF5 files with per-utterance expression, velocity, acceleration, audio features, and emotion labels.
- `metadata.json` -- Dataset statistics and split information.

Data is split by speaker: 4 English speakers go to validation, the rest to training. This prevents identity leakage.

On TACC, submit as a batch job:

```bash
sbatch slurm/preprocess.sbatch
```

---

## Stage 2: Generator Pretraining

The generator is a deterministic autoregressive transformer that maps audio features and emotion conditioning to FLAME expression parameter sequences. It is pretrained with L1 reconstruction loss before adversarial training begins.

### Architecture

- Audio encoder: 1D convolutional layers (input dim depends on mel vs wav2vec)
- Transformer decoder: 4 layers, 8 attention heads, d_model=256
- Emotion conditioning via Adaptive Layer Normalization (AdaLN)
- Autoregressive: uses P=2 previous expression frames as input
- Audio context: C=10 past frames, F=5 future frames

### Training

```bash
# Local (single GPU)
python training/train_stage2.py --config configs/stage2_pretrain.yaml

# Multi-GPU with DDP
torchrun --nproc_per_node=3 training/train_stage2.py \
    --config configs/stage2_pretrain.yaml

# Resume from checkpoint
torchrun --nproc_per_node=3 training/train_stage2.py \
    --config configs/stage2_pretrain.yaml \
    --resume auto
```

On TACC:

```bash
sbatch slurm/train_stage2.sbatch
```

The `--resume auto` flag finds the latest checkpoint in the configured checkpoint directory and resumes from it. This handles SLURM wall-time preemptions transparently.

### Hyperparameters

Configured in `configs/stage2_pretrain.yaml`:

| Parameter | Value |
|-----------|-------|
| Loss | L1 (mean absolute error) |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Betas | (0.9, 0.999) |
| Weight decay | 1e-5 |
| Batch size | 64 sequences |
| Sequence length | 60 frames (2 seconds at 30 fps) |
| Gradient clipping | max norm 1.0 |
| Epochs | 100 |

Teacher forcing is used during training: ground truth previous frames are provided as autoregressive input.

---

## Stage 3: Adversarial Co-Training

Introduces an audio-conditioned temporal discriminator that learns to distinguish real human facial motion (from BEAT2) from generated motion. The generator is then co-trained with both the L1 reconstruction loss and the adversarial signal.

### Discriminator

- Input: 5-frame windows of concatenated expression (100d) + velocity (100d) + acceleration (100d) + audio features
- Architecture: transformer encoder (3 layers, 4 heads) with global average pooling and a sigmoid classification head
- Audio-conditioned: judges whether motion matches the given speech, not just whether it looks natural in isolation

### Training

```bash
# Requires a pretrained Stage 2 generator checkpoint
torchrun --nproc_per_node=3 training/train_stage3.py \
    --config configs/stage3_adversarial.yaml \
    --pretrained_gen $WORK/checkpoints/d4head/stage2_best.pt

# Resume adversarial training
torchrun --nproc_per_node=3 training/train_stage3.py \
    --config configs/stage3_adversarial.yaml \
    --resume auto
```

On TACC:

```bash
sbatch slurm/train_stage3.sbatch
```

### Hyperparameters

Configured in `configs/stage3_adversarial.yaml`:

| Parameter | Value |
|-----------|-------|
| Generator loss | L1_recon + lambda_adv * L_adv |
| lambda_adv | Linear warmup 0.01 -> 0.1 over 10k steps |
| Generator optimizer | AdamW, lr=5e-5, betas=(0.5, 0.999) |
| Discriminator optimizer | AdamW, lr=2e-4, betas=(0.5, 0.999) |
| Gradient penalty weight | 10.0 |
| D steps per G step | 1 |
| Batch size | 64 |
| Discriminator window | 5 frames (~167ms) |

The discriminator uses a higher learning rate than the generator to ensure it stays ahead. Gradient penalty regularization keeps the discriminator's output surface smooth, preventing gradient collapse for the generator.

---

## Web Viewer

A browser-based 3D viewer for inspecting FLAME expression sequences produced by the trained model. Uses Three.js to render a FLAME mesh with expression weights applied in real-time.

### Setup

```bash
# 1. Export FLAME mesh and expression basis to binary (one-time)
python viewer/export_flame_mesh.py \
    --flame_path $WORK/models/flame/generic_model.pkl \
    --output viewer/static/flame_data.bin

# 2. Generate a demo sequence for testing (sinusoidal motion)
python viewer/generate_sequence.py --demo --output viewer/static/sequences/demo.npy

# 3. Generate a sequence from a trained model
python viewer/generate_sequence.py \
    --checkpoint $WORK/checkpoints/d4head/stage3_best.pt \
    --audio path/to/speech.wav \
    --emotion 0 \
    --output viewer/static/sequences/my_sequence.npy

# 4. Start the viewer server
python viewer/server.py
# Open http://localhost:8765
```

### Features

- Drag and drop `.bin` mesh files and `.npy` expression sequences
- Play/pause/scrub timeline with adjustable playback speed
- Real-time WebSocket streaming from model inference
- Per-dimension weight visualization bars for all 100 expression parameters
- Orbit camera with mouse controls
- Keyboard shortcuts: Space (play/pause), arrow keys (step frame), R (reset)
- Auto-loads data from the server on startup

### How it works

The export script reads the FLAME model and writes a binary file containing the template mesh vertices (5023 vertices), face indices (9976 triangles), and 100 expression basis vectors. The Three.js viewer loads this file and computes vertex positions on each frame:

```
positions = template + sum(weight_i * basis_vector_i)  for i in 0..99
```

This runs on the CPU in JavaScript, which avoids the WebGL morph target limit of ~8 active targets per draw call and correctly handles all 100 FLAME expression dimensions.

### WebSocket streaming

The server provides a WebSocket endpoint at `/ws/stream` for real-time expression weight streaming. The protocol:

1. Client connects and sends `{"sequence": "name", "fps": 30}`
2. Server streams frames: `{"type": "frame", "frame": 42, "weights": [0.1, -0.3, ...]}`
3. Client can send commands: `{"command": "play"}`, `{"command": "pause"}`, `{"command": "seek", "frame": 100}`

---

## Evaluation

Quantitative metrics are computed during validation in both training stages and can also be run standalone.

### Metrics

- **Lip Vertex Error (LVE)** -- L2 distance between lip vertices of predicted and ground truth FLAME meshes, reported in millimeters. Measures lip sync accuracy.
- **Expression Parameter MSE** -- Mean squared error between predicted and ground truth 100-dimensional expression vectors.
- **Frechet Gesture Distance (FGD)** -- Distributional metric comparing statistics of generated and real motion feature distributions, analogous to FID in image generation.

### Visualization

- **Expression trajectory plots** -- Per-dimension time series comparing predicted vs ground truth expression parameters. Useful for spotting jitter, smoothing, or drift.
- **FLAME mesh rendering** -- Offline rendering of predicted expression sequences on the FLAME mesh using pyrender. Videos can be logged to wandb.

---

## Configuration

All configuration is managed through YAML files with Python dataclass validation. Configs support inheritance via a `_base_` field.

### Config hierarchy

```
configs/base.yaml              # Shared defaults
configs/stage2_pretrain.yaml   # Inherits from base, overrides Stage 2 params
configs/stage3_adversarial.yaml # Inherits from base, overrides Stage 3 params
```

### Environment variables

Paths in YAML configs support environment variable expansion. Use `$WORK` for TACC paths:

```yaml
paths:
  beat2_raw_dir: "$WORK/data/beat2_raw"
  processed_dir: "$WORK/data/beat2_processed"
  checkpoint_dir: "$WORK/checkpoints/d4head"
```

### Key configuration sections

| Section | Purpose |
|---------|---------|
| `paths` | Data directories, FLAME model path, checkpoint and wandb directories |
| `data` | Frame rate, audio feature type (mel/wav2vec), sequence lengths, speaker splits |
| `generator` | Transformer architecture: layers, heads, d_model, feedforward dim, emotion embedding |
| `discriminator` | Discriminator architecture: layers, heads, gradient penalty weight |
| `stage2` | Pretraining: epochs, batch size, learning rate, gradient clipping |
| `stage3` | Adversarial: separate G/D learning rates, lambda warmup schedule |

See `configs_schema.py` for the full list of parameters and their defaults.

---

## TACC Cluster Usage

### Job submission

```bash
# 1. Preprocess data (CPU job, ~6 hours)
sbatch slurm/preprocess.sbatch

# 2. Train generator (GPU job, ~24 hours per submission)
sbatch slurm/train_stage2.sbatch

# 3. Adversarial co-training (GPU job, ~24 hours per submission)
sbatch slurm/train_stage3.sbatch
```

### Auto-resume

All training scripts support `--resume auto`, which finds the latest checkpoint and resumes from it. The SLURM scripts use this by default. If a job hits the 24-hour wall time, resubmit the same script and training continues from the last checkpoint.

### File storage

| Location | Use | Persistence |
|----------|-----|-------------|
| `$WORK` | Checkpoints, processed data, conda env | Persistent |
| `$SCRATCH` | Temporary files only | Purged after 10 days |
| `$HOME` | Small config files | Persistent, limited quota |

All training data and checkpoints should be stored on `$WORK`, not `$SCRATCH`.

### Multi-GPU

Training uses PyTorch DDP with `torchrun` as the launcher. The SLURM scripts request 3 GPUs per node (matching the TACC A100 node configuration). The effective batch size is `config.batch_size` split across GPUs.

For multi-node training (6+ GPUs across 2 nodes), modify the sbatch script:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)

torchrun --nproc_per_node=3 --nnodes=2 --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR --master_port=29500 \
    training/train_stage2.py --config configs/stage2_pretrain.yaml --resume auto
```

### wandb

Logging is active on rank 0 only. Set your wandb entity in the config or via environment variable:

```bash
export WANDB_ENTITY=your-username
```

The wandb dashboard shows training loss curves, validation metrics (L1, MSE, LVE), discriminator accuracy, adversarial weight schedule, and optionally rendered expression videos.
