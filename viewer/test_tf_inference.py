"""Quick test: run model with ground truth teacher forcing to verify model weights work."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torch
import numpy as np
import soundfile as sf
import h5py

from models.generator import Generator
from data.audio_features import extract_mel
from configs_schema import TrainConfig, _merge_dict_into_dataclass

WORK = os.environ.get("WORK", "")
ckpt_path = f"{WORK}/checkpoints/d4head/stage2_best.pt"
audio_path = f"{WORK}/data/beat2_raw/beat_english_v2.0.0/wave16k/1_wayne_0_1_1.wav"
npz_path = f"{WORK}/data/beat2_raw/beat_english_v2.0.0/smplxflame_30/1_wayne_0_1_1.npz"
stats_path = f"{WORK}/data/beat2_processed/train.h5"

# Load model
ckpt = torch.load(ckpt_path, map_location="cuda:0", weights_only=False)
config = TrainConfig()
_merge_dict_into_dataclass(config, ckpt["config"])

G = Generator(
    audio_dim=80, expr_dim=100, d_model=256, n_layers=4, n_heads=8,
    d_ff=1024, dropout=0.0, n_emotions=8, emotion_embed_dim=64,
    prev_frames=2, audio_conv_channels=[128, 256], audio_conv_kernel_sizes=[5, 3],
).cuda()
G.load_state_dict(ckpt["generator_model"])
G.eval()

# Load GT and audio
gt = np.load(npz_path, allow_pickle=True)["expressions"].astype(np.float32)
wav, sr = sf.read(audio_path)
mel = extract_mel(wav.astype(np.float32), sr, 30)

# Load normalization stats
with h5py.File(stats_path, "r") as f:
    em = f["stats"]["expr_mean"][:]
    es = f["stats"]["expr_std"][:]
    es[es < 1e-8] = 1.0

# Normalize GT
gt_norm = (gt - em) / es

# Prepare inputs for 60 frames
mel_padded = np.pad(mel, ((10, 5), (0, 0)), mode="edge")[:75]  # C=10, F=5
mel_t = torch.from_numpy(mel_padded).float().unsqueeze(0).cuda()
gt_t = torch.from_numpy(gt_norm[:60]).float().unsqueeze(0).cuda()
prev_t = torch.from_numpy(gt_norm[:2]).float().unsqueeze(0).cuda()
emo_t = torch.tensor([0]).cuda()

# Run with teacher forcing
with torch.no_grad():
    pred = G(mel_t, emo_t, prev_t, target_expression=gt_t)

# Denormalize
pred_np = pred[0].cpu().numpy() * es + em
gt_60 = gt[:60]

print(f"Teacher-forced prediction:")
print(f"  Pred std (first 5 dims): {pred_np.std(axis=0)[:5].round(3)}")
print(f"  GT std (first 5 dims):   {gt_60.std(axis=0)[:5].round(3)}")
print(f"  Pred range: [{pred_np.min():.3f}, {pred_np.max():.3f}]")
print(f"  GT range:   [{gt_60.min():.3f}, {gt_60.max():.3f}]")
print(f"  L1 error:   {np.abs(pred_np - gt_60).mean():.4f}")

np.save("viewer/static/sequences/wayne_tf.npy", pred_np.astype(np.float32))
print(f"Saved to viewer/static/sequences/wayne_tf.npy")
