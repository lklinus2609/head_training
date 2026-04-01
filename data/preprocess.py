"""Preprocess BEAT2 dataset: extract FLAME params, audio features, compute derivatives, write HDF5."""

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf
from tqdm import tqdm

from data.audio_features import extract_mel, extract_wav2vec
from data.flame_utils import compute_acceleration, compute_velocity, resample_to_fps


# Emotion label mapping (BEAT2 convention)
EMOTION_MAP = {
    "neutral": 0,
    "happiness": 1,
    "anger": 2,
    "sadness": 3,
    "contempt": 4,
    "surprise": 5,
    "fear": 6,
    "disgust": 7,
}


def discover_utterances(raw_dir: str) -> list[dict]:
    """Walk the BEAT2 raw directory and build a manifest of all utterances.

    Returns a list of dicts with keys: speaker_id, utterance_id, npz_path, wav_path, emotion.
    """
    manifest = []
    raw_path = Path(raw_dir)

    for npz_file in sorted(raw_path.rglob("*.npz")):
        # BEAT2 naming: {id}_{speaker}_{emotion}_{seq}_{take}.npz
        stem = npz_file.stem
        parts = stem.split("_")
        if len(parts) < 3:
            continue

        speaker_id = parts[1] if len(parts) >= 2 else "unknown"

        # Look for matching wav file
        wav_file = npz_file.with_suffix(".wav")
        if not wav_file.exists():
            # Try looking in adjacent audio directory
            wav_file = npz_file.parent / "audio" / f"{stem}.wav"
            if not wav_file.exists():
                continue

        # Try to extract emotion from the npz data or filename
        # BEAT2 stores emotion in the npz metadata
        manifest.append({
            "speaker_id": speaker_id,
            "utterance_id": stem,
            "npz_path": str(npz_file),
            "wav_path": str(wav_file),
        })

    print(f"Discovered {len(manifest)} utterances from {raw_dir}")
    return manifest


def process_utterance(
    entry: dict,
    audio_feature: str,
    target_fps: int,
    flame_expr_dim: int,
    mel_config: dict,
    wav2vec_config: dict,
) -> dict | None:
    """Process a single utterance: extract FLAME params, audio features, derivatives.

    Returns a dict with processed arrays, or None on failure.
    """
    try:
        # Load npz
        data = np.load(entry["npz_path"], allow_pickle=True)

        # Extract FLAME expression parameters
        # BEAT2 stores these under various keys - try common ones
        expression = None
        for key in ["expression", "exp", "flame_expression", "face_expression"]:
            if key in data:
                expression = data[key].astype(np.float32)
                break

        if expression is None:
            # Try extracting from SMPLX params
            if "poses" in data:
                # Some BEAT2 formats embed expression in the full parameter vector
                # Expression is typically the last 100 dims of the SMPLX params
                poses = data["poses"]
                if poses.shape[-1] >= flame_expr_dim:
                    expression = poses[:, -flame_expr_dim:].astype(np.float32)

        if expression is None:
            return None

        # Truncate or pad to flame_expr_dim
        if expression.shape[-1] > flame_expr_dim:
            expression = expression[:, :flame_expr_dim]
        elif expression.shape[-1] < flame_expr_dim:
            pad_width = flame_expr_dim - expression.shape[-1]
            expression = np.pad(expression, ((0, 0), (0, pad_width)))

        # Determine source FPS (BEAT2 is typically 30 FPS, but check)
        source_fps = 30.0
        if "fps" in data:
            source_fps = float(data["fps"])

        # Resample motion to target FPS
        expression = resample_to_fps(expression, source_fps, target_fps)

        # Compute derivatives
        velocity = compute_velocity(expression)
        acceleration = compute_acceleration(expression)

        # Load audio
        waveform, sr = sf.read(entry["wav_path"])
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # Mono
        waveform = waveform.astype(np.float32)

        # Extract audio features
        if audio_feature == "mel":
            audio_feats = extract_mel(
                waveform, sr, target_fps,
                n_mels=mel_config["n_mels"],
                win_ms=mel_config["win_ms"],
                hop_ms=mel_config["hop_ms"],
            )
        else:
            audio_feats = extract_wav2vec(
                waveform, sr, target_fps,
                model_name=wav2vec_config["model"],
            )

        # Align lengths (audio features and motion may differ by a frame or two)
        min_len = min(expression.shape[0], audio_feats.shape[0])
        expression = expression[:min_len]
        velocity = velocity[:min_len]
        acceleration = acceleration[:min_len]
        audio_feats = audio_feats[:min_len]

        if min_len < 10:  # Skip very short utterances
            return None

        # Extract emotion label
        emotion = 0  # Default to neutral
        if "emotion" in data:
            emo = data["emotion"]
            if isinstance(emo, np.ndarray):
                emo = str(emo.item()) if emo.size == 1 else str(emo[0])
            else:
                emo = str(emo)
            emotion = EMOTION_MAP.get(emo.lower(), 0)

        return {
            "speaker_id": entry["speaker_id"],
            "utterance_id": entry["utterance_id"],
            "expression": expression,
            "velocity": velocity,
            "acceleration": acceleration,
            "audio_features": audio_feats,
            "emotion_label": emotion,
        }

    except Exception as e:
        print(f"Error processing {entry['utterance_id']}: {e}")
        return None


def compute_statistics(results: list[dict]) -> dict:
    """Compute mean and std of expression, velocity, acceleration across all utterances."""
    all_expr = np.concatenate([r["expression"] for r in results], axis=0)
    all_vel = np.concatenate([r["velocity"] for r in results], axis=0)
    all_accel = np.concatenate([r["acceleration"] for r in results], axis=0)

    return {
        "expr_mean": all_expr.mean(axis=0).astype(np.float32),
        "expr_std": all_expr.std(axis=0).astype(np.float32),
        "vel_mean": all_vel.mean(axis=0).astype(np.float32),
        "vel_std": all_vel.std(axis=0).astype(np.float32),
        "accel_mean": all_accel.mean(axis=0).astype(np.float32),
        "accel_std": all_accel.std(axis=0).astype(np.float32),
    }


def write_hdf5(results: list[dict], stats: dict, output_path: str):
    """Write processed data to an HDF5 file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Write statistics
        stats_grp = f.create_group("stats")
        for key, value in stats.items():
            stats_grp.create_dataset(key, data=value)

        # Write utterances grouped by speaker
        for result in results:
            speaker = result["speaker_id"]
            utt = result["utterance_id"]
            grp_name = f"speaker_{speaker}/{utt}"
            grp = f.create_group(grp_name)
            grp.create_dataset("expression", data=result["expression"], compression="gzip")
            grp.create_dataset("velocity", data=result["velocity"], compression="gzip")
            grp.create_dataset("acceleration", data=result["acceleration"], compression="gzip")
            grp.create_dataset("audio_features", data=result["audio_features"], compression="gzip")
            grp.create_dataset("emotion_label", data=result["emotion_label"])
            grp.attrs["speaker_id"] = speaker

    print(f"Wrote {len(results)} utterances to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess BEAT2 dataset")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--raw_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--audio_feature", type=str, default=None, choices=["mel", "wav2vec"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true", help="Process one utterance and print structure")
    args = parser.parse_args()

    # Load config
    from configs_schema import load_config
    config = load_config(args.config)

    raw_dir = args.raw_dir or config.paths.beat2_raw_dir
    output_dir = args.output_dir or config.paths.processed_dir
    audio_feature = args.audio_feature or config.data.audio_feature

    # Discover utterances
    manifest = discover_utterances(raw_dir)
    if not manifest:
        print("No utterances found. Check raw_dir path.")
        return

    if args.dry_run:
        print("=== Dry run: processing first utterance ===")
        result = process_utterance(
            manifest[0],
            audio_feature=audio_feature,
            target_fps=config.data.fps,
            flame_expr_dim=config.data.flame_expr_dim,
            mel_config={"n_mels": config.data.mel_n_mels, "win_ms": config.data.mel_win_ms, "hop_ms": config.data.mel_hop_ms},
            wav2vec_config={"model": config.data.wav2vec_model},
        )
        if result:
            print(f"Speaker: {result['speaker_id']}")
            print(f"Utterance: {result['utterance_id']}")
            print(f"Expression: {result['expression'].shape}")
            print(f"Velocity: {result['velocity'].shape}")
            print(f"Acceleration: {result['acceleration'].shape}")
            print(f"Audio features: {result['audio_features'].shape}")
            print(f"Emotion: {result['emotion_label']}")
        else:
            print("Failed to process utterance.")
        return

    # Split by speaker
    dev_speakers = set(config.data.dev_speakers)
    train_manifest = [e for e in manifest if e["speaker_id"] not in dev_speakers]
    val_manifest = [e for e in manifest if e["speaker_id"] in dev_speakers]

    print(f"Train: {len(train_manifest)} utterances, Val: {len(val_manifest)} utterances")

    # Process utterances
    process_fn = partial(
        process_utterance,
        audio_feature=audio_feature,
        target_fps=config.data.fps,
        flame_expr_dim=config.data.flame_expr_dim,
        mel_config={"n_mels": config.data.mel_n_mels, "win_ms": config.data.mel_win_ms, "hop_ms": config.data.mel_hop_ms},
        wav2vec_config={"model": config.data.wav2vec_model},
    )

    print("Processing training utterances...")
    with Pool(args.num_workers) as pool:
        train_results = list(tqdm(
            pool.imap(process_fn, train_manifest),
            total=len(train_manifest),
            desc="Train",
        ))
    train_results = [r for r in train_results if r is not None]

    print("Processing validation utterances...")
    with Pool(args.num_workers) as pool:
        val_results = list(tqdm(
            pool.imap(process_fn, val_manifest),
            total=len(val_manifest),
            desc="Val",
        ))
    val_results = [r for r in val_results if r is not None]

    # Compute statistics from training set
    print("Computing dataset statistics...")
    stats = compute_statistics(train_results)
    print(f"Expression mean range: [{stats['expr_mean'].min():.4f}, {stats['expr_mean'].max():.4f}]")
    print(f"Expression std range: [{stats['expr_std'].min():.4f}, {stats['expr_std'].max():.4f}]")

    # Write HDF5 files
    write_hdf5(train_results, stats, os.path.join(output_dir, "train.h5"))
    write_hdf5(val_results, stats, os.path.join(output_dir, "val.h5"))

    # Save metadata
    metadata = {
        "audio_feature": audio_feature,
        "fps": config.data.fps,
        "flame_expr_dim": config.data.flame_expr_dim,
        "train_utterances": len(train_results),
        "val_utterances": len(val_results),
        "dev_speakers": list(dev_speakers),
        "emotion_map": EMOTION_MAP,
        "train_total_frames": sum(r["expression"].shape[0] for r in train_results),
        "val_total_frames": sum(r["expression"].shape[0] for r in val_results),
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
