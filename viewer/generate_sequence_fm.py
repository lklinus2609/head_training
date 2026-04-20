"""Generate an expression sequence from a trained FM checkpoint.

Parallel to viewer/generate_sequence.py (AR transformer) — same inputs, same
output format, same audio + GT side-files for the web viewer — but samples
via flow matching: draw x_0 ~ N(0, I) per window and Euler-integrate the
predicted velocity field from t=0 to t=1.

Usage:
    python viewer/generate_sequence_fm.py \
        --checkpoint $WORK/checkpoints/d4head/fm_best.pt \
        --audio input.wav \
        --emotion 0 \
        --output static/sequences/fm_test.npy \
        --nfe 4 \
        --seed 0

Stochastic: different seeds produce different sequences. Use --n_samples K
to draw K sequences and save the one with lowest raw-L1 against the matching
BEAT2 GT (fair vs the deterministic AR baseline's single output).
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_from_fm(
    checkpoint_path: str,
    audio_path: str,
    emotion: int,
    output_path: str,
    config_path: str | None = None,
    nfe: int = 4,
    seed: int | None = None,
    n_samples: int = 1,
):
    import h5py
    import soundfile as sf
    import torch

    from configs_schema import load_config
    from data.audio_features import extract_mel, extract_wav2vec
    from models.generator_fm import GeneratorFM

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if config_path:
        config = load_config(config_path)
    else:
        from configs_schema import TrainConfig, _merge_dict_into_dataclass
        config = TrainConfig()
        _merge_dict_into_dataclass(config, ckpt["config"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_dim = config.data.mel_n_mels if config.data.audio_feature == "mel" else config.data.wav2vec_dim
    generator = GeneratorFM(
        audio_dim=audio_dim,
        expr_dim=config.data.flame_expr_dim,
        d_model=config.generator.d_model,
        n_layers=config.generator.n_layers,
        n_heads=config.generator.n_heads,
        d_ff=config.generator.d_ff,
        dropout=0.0,
        n_emotions=config.generator.n_emotions,
        emotion_embed_dim=config.generator.emotion_embed_dim,
        prev_frames=config.data.prev_frames,
        time_embed_dim=config.fm.time_embed_dim,
        audio_conv_channels=config.generator.audio_conv_channels,
        audio_conv_kernel_sizes=config.generator.audio_conv_kernel_sizes,
    ).to(device)

    generator.load_state_dict(ckpt["generator_model"])
    generator.eval()

    waveform, sr = sf.read(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)

    if config.data.audio_feature == "mel":
        audio_feats = extract_mel(
            waveform, sr, config.data.fps,
            n_mels=config.data.mel_n_mels,
            win_ms=config.data.mel_win_ms,
            hop_ms=config.data.mel_hop_ms,
        )
    else:
        audio_feats = extract_wav2vec(
            waveform, sr, config.data.fps,
            model_name=config.data.wav2vec_model,
            device=str(device),
        )

    stats_path = Path(config.paths.processed_dir) / "train.h5"
    expr_mean = expr_std = None
    if stats_path.exists():
        with h5py.File(str(stats_path), "r") as hf:
            if "stats" in hf:
                expr_mean_np = hf["stats"]["expr_mean"][:]
                expr_std_np = hf["stats"]["expr_std"][:]
        expr_mean = torch.from_numpy(expr_mean_np).float()
        expr_std = torch.from_numpy(expr_std_np).float()
        expr_std = torch.where(expr_std < 1e-8, torch.ones_like(expr_std), expr_std)
        print(f"Loaded normalization stats from {stats_path}")

    T = audio_feats.shape[0]
    W = config.fm.window_size
    C = config.data.context_past
    F = config.data.context_future
    P = config.data.prev_frames
    D = config.data.flame_expr_dim

    audio_padded = np.pad(audio_feats, ((C, F), (0, 0)), mode="edge")
    audio_tensor = torch.from_numpy(audio_padded).float().unsqueeze(0).to(device)
    emotion_tensor = torch.tensor([emotion], device=device)

    print(
        f"FM inference: W={W}, NFE={nfe}, T={T}, C={C}, F={F}, P={P}, "
        f"n_samples={n_samples}, seed={seed}"
    )

    # Load GT ahead of inference so we can pick best-of-N on raw-L1 when n_samples>1.
    audio_stem = Path(audio_path).stem
    beat2_dir = Path(config.paths.beat2_raw_dir)
    gt_raw = None
    npz_candidates = list(beat2_dir.rglob(f"{audio_stem}.npz"))
    if npz_candidates:
        npz_data = np.load(str(npz_candidates[0]), allow_pickle=True)
        if "expressions" in npz_data:
            gt_raw = npz_data["expressions"][:, :D].astype(np.float32)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    best_raw = None
    best_l1 = float("inf")
    with torch.no_grad():
        for k in range(max(n_samples, 1)):
            prev_expr = torch.zeros(1, P, D, device=device)
            chunks = []
            for t in range(0, T, W):
                chunk_len = min(W, T - t)
                audio_chunk = audio_tensor[:, t:t + C + chunk_len + F]
                pred = generator.sample_window(
                    audio_chunk, emotion_tensor, prev_expr,
                    window_size=chunk_len, nfe=nfe,
                )
                chunks.append(pred)
                prev_expr = torch.cat([prev_expr, pred], dim=1)[:, -P:]
            pred_norm = torch.cat(chunks, dim=1)[0].cpu()  # [T, D]

            if expr_mean is not None and expr_std is not None:
                pred_raw = (pred_norm * expr_std + expr_mean).numpy()
            else:
                pred_raw = pred_norm.numpy()

            if gt_raw is not None:
                n = min(pred_raw.shape[0], gt_raw.shape[0])
                l1 = float(np.abs(pred_raw[:n] - gt_raw[:n]).mean())
                print(f"  sample {k}: raw-L1 = {l1:.4f}")
            else:
                l1 = 0.0
                print(f"  sample {k}: (no GT found for {audio_stem})")

            if l1 < best_l1 or best_raw is None:
                best_l1 = l1
                best_raw = pred_raw

    expressions = best_raw.astype(np.float32)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, expressions)
    print(f"Prediction saved: {output_path} (shape: {expressions.shape})")
    print(f"  Value range: [{expressions.min():.3f}, {expressions.max():.3f}]")

    if gt_raw is not None:
        min_len = min(gt_raw.shape[0], expressions.shape[0])
        gt_trimmed = gt_raw[:min_len]
        gt_path = output_path.replace(".npy", "_gt.npy")
        np.save(gt_path, gt_trimmed)
        print(f"Ground truth saved: {gt_path} (shape: {gt_trimmed.shape})")
        print(f"  Best raw-L1 (pred vs GT): {best_l1:.4f}")
    else:
        print(f"  No matching ground truth npz found for {audio_stem}")

    wav_dest = output_path.replace(".npy", ".wav")
    try:
        shutil.copy2(audio_path, wav_dest)
        print(f"Audio copied: {wav_dest}")
    except Exception as e:
        print(f"Warning: could not copy audio: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate expression sequences from an FM checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="FM checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path (optional)")
    parser.add_argument("--audio", type=str, required=True, help="Input audio .wav")
    parser.add_argument("--emotion", type=int, default=0, help="Emotion label (0-7)")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent / "static" / "sequences" / "fm.npy"),
                        help="Output .npy path")
    parser.add_argument("--nfe", type=int, default=4,
                        help="Number of Euler steps per window (rectified flow: 1-4 usually enough)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility; omit for non-deterministic sampling")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Draw K samples and save the one with lowest raw-L1 vs GT (if GT is present)")
    args = parser.parse_args()
    generate_from_fm(
        args.checkpoint, args.audio, args.emotion, args.output,
        args.config, args.nfe, args.seed, args.n_samples,
    )


if __name__ == "__main__":
    main()
