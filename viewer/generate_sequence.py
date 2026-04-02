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
):
    """Run model inference on audio and save expression sequence."""
    import torch
    import soundfile as sf

    from configs_schema import load_config
    from data.audio_features import extract_mel, extract_wav2vec
    from models.generator import Generator

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

    # Pad audio for context
    audio_padded = np.pad(audio_feats, ((C, F), (0, 0)), mode="edge")
    audio_tensor = torch.from_numpy(audio_padded).float().unsqueeze(0).to(device)

    # Generate in chunks using teacher-forcing style (parallel within chunk)
    # This avoids autoregressive drift by generating each chunk independently
    # and only passing the last P frames as context to the next chunk.
    all_expressions = []
    prev_expr = torch.zeros(1, P, config.data.flame_expr_dim, device=device)
    emotion_tensor = torch.tensor([emotion], device=device)

    with torch.no_grad():
        for start in range(0, T, seq_len):
            end = min(start + seq_len, T)
            chunk_len = end - start

            # Audio context for this chunk
            audio_start = start
            audio_end = audio_start + chunk_len + C + F
            audio_chunk = audio_tensor[:, audio_start:audio_end]

            # Use a dummy target to trigger teacher-forcing path (parallel generation)
            # Feed zeros as target -- the model uses prev_expr + causal self-attention
            dummy_target = torch.zeros(1, chunk_len, config.data.flame_expr_dim, device=device)
            pred = generator(audio_chunk, emotion_tensor, prev_expr, target_expression=dummy_target)
            pred = pred[:, :chunk_len]

            # Clamp output to reasonable range (prevent any residual drift)
            pred = pred.clamp(-5.0, 5.0)

            all_expressions.append(pred.cpu())

            # Pass last P frames as context for next chunk
            if pred.shape[1] >= P:
                prev_expr = pred[:, -P:]
            else:
                prev_expr = pred

    expressions = torch.cat(all_expressions, dim=1)[0]  # [T, 100]

    # Denormalize if we have stats (model outputs normalized values)
    if expr_mean is not None and expr_std is not None:
        expressions = expressions.cpu() * expr_std.cpu() + expr_mean.cpu()

    expressions = expressions.numpy()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, expressions.astype(np.float32))
    print(f"Expression sequence saved: {output_path} (shape: {expressions.shape})")
    print(f"Value range: [{expressions.min():.3f}, {expressions.max():.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Generate expression sequences")
    parser.add_argument("--demo", action="store_true",
                        help="Generate a demo sinusoidal sequence for testing")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path (optional)")
    parser.add_argument("--audio", type=str, help="Input audio file path")
    parser.add_argument("--emotion", type=int, default=0, help="Emotion label (0-7)")
    parser.add_argument("--output", type=str, default="static/sequences/demo.npy",
                        help="Output .npy file path")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Demo sequence duration in seconds")
    args = parser.parse_args()

    if args.demo:
        generate_demo_sequence(args.output, args.duration)
    else:
        if not args.checkpoint or not args.audio:
            parser.error("--checkpoint and --audio are required when not using --demo")
        generate_from_model(args.checkpoint, args.audio, args.emotion, args.output, args.config)


if __name__ == "__main__":
    main()
