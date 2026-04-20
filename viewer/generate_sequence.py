"""Generate an expression parameter sequence from a trained model and save for the web viewer.

Usage:
    python generate_sequence.py \
        --checkpoint $WORK/checkpoints/d4head/stage3_best.pt \
        --audio input.wav \
        --emotion 0 \
        --output static/sequences/test_sequence.npy

    # Or generate a test sequence with random/sinusoidal weights for viewer testing:
    python generate_sequence.py --demo --output static/sequences/demo.npy
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_demo_sequence(output_path: str, duration: float = 5.0, fps: int = 30):
    """Generate a demo expression sequence with sinusoidal motion for testing the viewer."""
    T = int(duration * fps)
    n_expr = 100

    weights = np.zeros((T, n_expr), dtype=np.float32)

    # Animate first few dimensions with different frequencies for visible motion
    # Dim 0-2: jaw/mouth (low freq, large amplitude)
    for i in range(3):
        freq = 0.5 + i * 0.3
        weights[:, i] = 0.5 * np.sin(2 * np.pi * freq * np.arange(T) / fps + i)

    # Dim 3-8: lip shapes (medium freq)
    for i in range(3, 9):
        freq = 1.0 + i * 0.2
        weights[:, i] = 0.3 * np.sin(2 * np.pi * freq * np.arange(T) / fps + i * 0.5)

    # Dim 9-20: eyebrows, cheeks (slow freq)
    for i in range(9, 20):
        freq = 0.2 + i * 0.05
        weights[:, i] = 0.2 * np.sin(2 * np.pi * freq * np.arange(T) / fps + i * 0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, weights)
    print(f"Demo sequence saved: {output_path} (shape: {weights.shape}, {duration}s at {fps}fps)")


def generate_from_model(
    checkpoint_path: str,
    audio_path: str,
    emotion: int,
    output_path: str,
    config_path: str = None,
    mode: str = "sliding",
    fp16: bool = False,
    kv_cache: bool = False,
):
    """Run model inference on audio and save expression sequence."""
    import torch
    import soundfile as sf

    from configs_schema import load_config
    from data.audio_features import extract_mel, extract_wav2vec
    from models.generator import Generator
    from models.generator_inference import GeneratorInference

    # Load config from checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if config_path:
        config = load_config(config_path)
    else:
        from configs_schema import TrainConfig, _merge_dict_into_dataclass
        config = TrainConfig()
        _merge_dict_into_dataclass(config, ckpt["config"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    audio_dim = config.data.mel_n_mels if config.data.audio_feature == "mel" else config.data.wav2vec_dim
    generator = Generator(
        audio_dim=audio_dim,
        expr_dim=config.data.flame_expr_dim,
        d_model=config.generator.d_model,
        n_layers=config.generator.n_layers,
        n_heads=config.generator.n_heads,
        d_ff=config.generator.d_ff,
        dropout=0.0,  # No dropout at inference
        n_emotions=config.generator.n_emotions,
        emotion_embed_dim=config.generator.emotion_embed_dim,
        prev_frames=config.data.prev_frames,
        audio_conv_channels=config.generator.audio_conv_channels,
        audio_conv_kernel_sizes=config.generator.audio_conv_kernel_sizes,
    ).to(device)

    generator.load_state_dict(ckpt["generator_model"])
    generator.eval()

    # Optional KV-cache inference model (for AR modes; teacher mode still uses training model)
    gen_infer = None
    if kv_cache and mode in ("sliding", "autoregressive"):
        gen_infer = GeneratorInference.from_training_generator(generator).to(device)

    # Load and process audio
    waveform, sr = sf.read(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)

    if config.data.audio_feature == "mel":
        audio_feats = extract_mel(waveform, sr, config.data.fps,
                                   n_mels=config.data.mel_n_mels)
    else:
        audio_feats = extract_wav2vec(waveform, sr, config.data.fps,
                                       model_name=config.data.wav2vec_model,
                                       device=str(device))

    # Load normalization stats from the processed dataset
    import h5py
    stats_path = Path(config.paths.processed_dir) / "train.h5"
    expr_mean = expr_std = None
    if stats_path.exists():
        with h5py.File(str(stats_path), "r") as hf:
            if "stats" in hf:
                expr_mean = torch.from_numpy(hf["stats"]["expr_mean"][:]).float().to(device)
                expr_std = torch.from_numpy(hf["stats"]["expr_std"][:]).float().to(device)
                expr_std = torch.where(expr_std < 1e-8, torch.ones_like(expr_std), expr_std)
                print(f"Loaded normalization stats from {stats_path}")

    # Normalize audio features (the model was trained on normalized expression data,
    # but audio features were not normalized in the dataset -- they go in raw)

    T = audio_feats.shape[0]
    seq_len = config.data.seq_len
    C = config.data.context_past
    F = config.data.context_future
    P = config.data.prev_frames
    H = getattr(config.stage2, "gen_horizon", 4)

    # Pad audio for context
    audio_padded = np.pad(audio_feats, ((C, F), (0, 0)), mode="edge")
    audio_tensor = torch.from_numpy(audio_padded).float().unsqueeze(0).to(device)

    all_expressions = []
    prev_expr = torch.zeros(1, P, config.data.flame_expr_dim, device=device)
    emotion_tensor = torch.tensor([emotion], device=device)

    # Load GT for teacher forcing mode
    gt_norm = None
    if mode == "teacher":
        audio_stem = Path(audio_path).stem
        beat2_dir = Path(config.paths.beat2_raw_dir)
        npz_candidates = list(beat2_dir.rglob(f"{audio_stem}.npz"))
        if not npz_candidates:
            print(f"No GT found for {audio_stem}, falling back to sliding mode")
            mode = "sliding"
        else:
            npz_data = np.load(str(npz_candidates[0]), allow_pickle=True)
            gt_raw = torch.from_numpy(npz_data["expressions"][:T].astype(np.float32)).to(device)
            if expr_mean is not None and expr_std is not None:
                gt_norm = (gt_raw - expr_mean) / expr_std
            else:
                gt_norm = gt_raw
            gt_norm = gt_norm.unsqueeze(0)  # [1, T, 100]

    use_autocast = fp16 and device.type == "cuda"
    print(f"Inference mode: {mode} (H={H}, P={P}, C={C}, F={F}) fp16={use_autocast} kv_cache={gen_infer is not None}")

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_autocast
        else torch.autocast(device_type="cpu", enabled=False)
    )

    with torch.no_grad(), autocast_ctx:
        if mode == "sliding":
            # Sliding window: predict H frames at a time, use own predictions as context
            for t in range(0, T, H):
                chunk_len = min(H, T - t)
                audio_chunk = audio_tensor[:, t:t + C + chunk_len + F]
                if gen_infer is not None:
                    pred = gen_infer(audio_chunk, emotion_tensor, prev_expr, max_len=chunk_len)
                else:
                    pred = generator(audio_chunk, emotion_tensor, prev_expr, target_expression=None, max_len=chunk_len)
                all_expressions.append(pred.cpu())
                # Update prev_expr, keeping exactly P frames
                prev_expr = torch.cat([prev_expr, pred], dim=1)[:, -P:]

        elif mode == "teacher":
            # Teacher forcing: use GT as input, for evaluation only
            gt_T = gt_norm.shape[1]
            for start in range(0, gt_T, seq_len):
                end = min(start + seq_len, gt_T)
                if end <= start:
                    break
                gt_chunk = gt_norm[:, start:end]
                audio_chunk = audio_tensor[:, start:start + C + (end - start) + F]
                if start >= P:
                    prev_expr_gt = gt_norm[:, start - P:start]
                else:
                    pad = torch.zeros(1, P - start, config.data.flame_expr_dim, device=device)
                    prev_expr_gt = torch.cat([pad, gt_norm[:, :start]], dim=1) if start > 0 else torch.zeros(1, P, config.data.flame_expr_dim, device=device)
                pred = generator(audio_chunk, emotion_tensor, prev_expr_gt, target_expression=gt_chunk)
                all_expressions.append(pred.cpu())

        elif mode == "autoregressive":
            # Legacy frame-by-frame autoregressive
            for start in range(0, T, seq_len):
                end = min(start + seq_len, T)
                chunk_len = end - start
                audio_chunk = audio_tensor[:, start:start + C + chunk_len + F]
                if gen_infer is not None:
                    pred = gen_infer(audio_chunk, emotion_tensor, prev_expr, max_len=chunk_len)
                else:
                    pred = generator(audio_chunk, emotion_tensor, prev_expr, target_expression=None, max_len=chunk_len)
                all_expressions.append(pred.cpu())
                prev_expr = torch.cat([prev_expr, pred], dim=1)[:, -P:]

    expressions = torch.cat(all_expressions, dim=1)[0]  # [T, 100]

    # Denormalize if we have stats (model outputs normalized values)
    if expr_mean is not None and expr_std is not None:
        expressions = expressions.cpu() * expr_std.cpu() + expr_mean.cpu()

    expressions = expressions.numpy()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, expressions.astype(np.float32))
    print(f"Prediction saved: {output_path} (shape: {expressions.shape})")
    print(f"  Value range: [{expressions.min():.3f}, {expressions.max():.3f}]")

    # Save matching ground truth from BEAT2 npz
    audio_stem = Path(audio_path).stem  # e.g. "1_wayne_0_1_1"
    beat2_dir = Path(config.paths.beat2_raw_dir)
    npz_candidates = list(beat2_dir.rglob(f"{audio_stem}.npz"))
    if npz_candidates:
        npz_data = np.load(str(npz_candidates[0]), allow_pickle=True)
        if "expressions" in npz_data:
            gt = npz_data["expressions"].astype(np.float32)
            # Trim to same length as prediction
            min_len = min(gt.shape[0], expressions.shape[0])
            gt = gt[:min_len]

            gt_path = output_path.replace(".npy", "_gt.npy")
            np.save(gt_path, gt)
            print(f"Ground truth saved: {gt_path} (shape: {gt.shape})")
            print(f"  Value range: [{gt.min():.3f}, {gt.max():.3f}]")

            # Print comparison stats
            pred_trimmed = expressions[:min_len]
            l1 = np.abs(pred_trimmed - gt).mean()
            print(f"  L1 error (pred vs GT): {l1:.4f}")
    else:
        print(f"  No matching ground truth npz found for {audio_stem}")

    # Copy source audio alongside the sequence for the web viewer
    wav_dest = output_path.replace(".npy", ".wav")
    try:
        shutil.copy2(audio_path, wav_dest)
        print(f"Audio copied: {wav_dest}")
    except Exception as e:
        print(f"Warning: could not copy audio: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate expression sequences")
    parser.add_argument("--demo", action="store_true",
                        help="Generate a demo sinusoidal sequence for testing")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path. If omitted, auto-detects the best.pt "
                             "in the most recent run folder under --checkpoint_dir (falling "
                             "back to the highest-epoch checkpoint if no best.pt exists).")
    parser.add_argument("--checkpoint_dir", type=str,
                        default=os.path.expandvars("$WORK/checkpoints/d4head"),
                        help="Root directory scanned for the default checkpoint when "
                             "--checkpoint is not given.")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path (optional)")
    parser.add_argument("--audio", type=str, help="Input audio file path")
    parser.add_argument("--emotion", type=int, default=0, help="Emotion label (0-7)")
    parser.add_argument("--output", type=str, default=str(Path(__file__).parent / "static" / "sequences" / "demo.npy"),
                        help="Output .npy file path")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Demo sequence duration in seconds")
    parser.add_argument("--mode", type=str, default="sliding",
                        choices=["sliding", "teacher", "autoregressive"],
                        help="Inference mode: sliding (H-frame windows, default), "
                             "teacher (GT context, eval only), autoregressive (legacy frame-by-frame)")
    parser.add_argument("--fp16", action="store_true",
                        help="Run inference under torch.autocast(float16) on CUDA. Ignored on CPU.")
    parser.add_argument("--kv_cache", action="store_true",
                        help="Use KV-cached inference generator for AR modes (sliding, autoregressive).")
    args = parser.parse_args()

    if args.demo:
        generate_demo_sequence(args.output, args.duration)
    else:
        if not args.audio:
            parser.error("--audio is required when not using --demo")
        checkpoint = args.checkpoint
        if checkpoint is None:
            from utils.checkpoint import find_latest_best_checkpoint
            checkpoint = find_latest_best_checkpoint(args.checkpoint_dir)
            if checkpoint is None:
                parser.error(
                    f"--checkpoint not given and no checkpoint found under "
                    f"{args.checkpoint_dir}. Pass --checkpoint explicitly."
                )
            print(f"Using auto-detected checkpoint: {checkpoint}")
        generate_from_model(checkpoint, args.audio, args.emotion, args.output,
                            args.config, args.mode, args.fp16, args.kv_cache)


if __name__ == "__main__":
    main()
