"""Preprocess BEAT2 dataset: extract FLAME params, audio features, compute derivatives, write HDF5.

BEAT2 directory structure (from HuggingFace H-Liu1997/BEAT2):
    beat_english_v2.0.0/
        smplxflame_30/    # NPZ motion files at 30fps
        wave16k/          # WAV audio files at 16kHz
        train_test_split.csv  # Official train/val/test split

NPZ keys: betas, poses, expressions [T,100], trans, model, gender, mocap_frame_rate
Filename convention: {speaker_id}_{speaker_name}_{emotion}_{seq}_{take}.npz
Emotion is encoded in the filename (3rd field): 0=neutral, 1=happiness, etc.
"""

import argparse
import csv
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


# BEAT2 emotion mapping (encoded as integer in filename)
EMOTION_LABELS = {
    0: "neutral",
    1: "happiness",
    2: "anger",
    3: "sadness",
    4: "contempt",
    5: "surprise",
    6: "fear",
    7: "disgust",
}


def load_split_csv(raw_dir: str) -> dict[str, str]:
    """Load the official BEAT2 train/val/test split.

    Returns dict mapping utterance_id -> split ("train", "val", "test").
    """
    csv_path = Path(raw_dir) / "beat_english_v2.0.0" / "train_test_split.csv"
    if not csv_path.exists():
        # Try one level up
        csv_path = Path(raw_dir) / "train_test_split.csv"
    if not csv_path.exists():
        return {}

    splits = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            splits[row["id"]] = row["type"]

    print(f"Loaded split CSV: {len(splits)} entries")
    return splits


def discover_utterances(raw_dir: str) -> list[dict]:
    """Walk the BEAT2 raw directory and build a manifest of all utterances.

    Looks for NPZ files in smplxflame_30/ and matching WAV files in wave16k/.
    """
    manifest = []
    raw_path = Path(raw_dir)

    # Find the smplxflame_30 directory
    npz_dirs = list(raw_path.rglob("smplxflame_30"))
    if not npz_dirs:
        # Fallback: search for npz files anywhere
        npz_dirs = [raw_path]

    # Find the wave16k directory
    wav_dirs = list(raw_path.rglob("wave16k"))
    wav_dir = wav_dirs[0] if wav_dirs else None

    for npz_dir in npz_dirs:
        for npz_file in sorted(npz_dir.glob("*.npz")):
            stem = npz_file.stem
            parts = stem.split("_")
            # Format: {speaker_id}_{speaker_name}_{emotion}_{seq}_{take}
            if len(parts) < 5:
                continue

            speaker_id = parts[0]
            speaker_name = parts[1]
            emotion = int(parts[2])

            # Find matching wav file
            wav_file = None
            if wav_dir and (wav_dir / f"{stem}.wav").exists():
                wav_file = wav_dir / f"{stem}.wav"
            else:
                # Try sibling directory
                for candidate in [
                    npz_file.parent.parent / "wave16k" / f"{stem}.wav",
                    npz_file.with_suffix(".wav"),
                ]:
                    if candidate.exists():
                        wav_file = candidate
                        break

            if wav_file is None:
                continue

            manifest.append({
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "utterance_id": stem,
                "npz_path": str(npz_file),
                "wav_path": str(wav_file),
                "emotion": emotion,
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
    """Process a single utterance: extract FLAME params, audio features, derivatives."""
    try:
        data = np.load(entry["npz_path"], allow_pickle=True)

        # BEAT2 stores expressions under the "expressions" key [T, 100] float64
        if "expressions" not in data:
            return None
        expression = data["expressions"].astype(np.float32)

        # Truncate or pad to flame_expr_dim
        if expression.shape[-1] > flame_expr_dim:
            expression = expression[:, :flame_expr_dim]
        elif expression.shape[-1] < flame_expr_dim:
            pad_width = flame_expr_dim - expression.shape[-1]
            expression = np.pad(expression, ((0, 0), (0, pad_width)))

        # Source FPS from the file
        source_fps = 30.0
        if "mocap_frame_rate" in data:
            source_fps = float(data["mocap_frame_rate"])

        # Resample motion to target FPS if needed
        expression = resample_to_fps(expression, source_fps, target_fps)

        # Compute derivatives
        velocity = compute_velocity(expression)
        acceleration = compute_acceleration(expression)

        # Load audio (BEAT2 wave16k is 16kHz mono)
        waveform, sr = sf.read(entry["wav_path"])
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
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

        # Align lengths (audio and motion may differ by a few frames)
        min_len = min(expression.shape[0], audio_feats.shape[0])
        expression = expression[:min_len]
        velocity = velocity[:min_len]
        acceleration = acceleration[:min_len]
        audio_feats = audio_feats[:min_len]

        if min_len < 10:
            return None

        # Emotion from filename
        emotion = entry.get("emotion", 0)

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
        stats_grp = f.create_group("stats")
        for key, value in stats.items():
            stats_grp.create_dataset(key, data=value)

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

    # Load official train/val/test split
    split_map = load_split_csv(raw_dir)

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
            print(f"Emotion: {result['emotion_label']} ({EMOTION_LABELS.get(result['emotion_label'], 'unknown')})")
        else:
            print("Failed to process utterance.")
        return

    # Split using official CSV if available, otherwise fall back to speaker-based split.
    # `additional` clips are folded into train: in the BEAT2 English subset they are
    # the only emotion=1 (happy) data, and dropping them left the model with zero
    # emo=1 training examples — making emo=1 conditioning at inference an
    # extrapolation from a randomly-initialized embedding. NOTE: this means held-out
    # emo=1 evaluation is no longer possible without an explicit per-clip exclusion
    # mechanism; if a held-out emo=1 set is needed, exclude specific utterance_ids
    # from `train_set_labels` below.
    if split_map:
        train_set_labels = {"train", "additional"}
        train_manifest = [e for e in manifest if split_map.get(e["utterance_id"]) in train_set_labels]
        val_manifest = [e for e in manifest if split_map.get(e["utterance_id"]) == "val"]
        test_manifest = [e for e in manifest if split_map.get(e["utterance_id"]) == "test"]
        # Include unmatched entries in training
        matched_ids = set(split_map.keys())
        unmatched = [e for e in manifest if e["utterance_id"] not in matched_ids]
        if unmatched:
            print(f"Warning: {len(unmatched)} utterances not in split CSV, adding to train")
            train_manifest.extend(unmatched)
    else:
        print("No split CSV found, falling back to speaker-based split")
        dev_speakers = set(config.data.dev_speakers)
        train_manifest = [e for e in manifest if e["speaker_id"] not in dev_speakers]
        val_manifest = [e for e in manifest if e["speaker_id"] in dev_speakers]

    print(f"Train: {len(train_manifest)}, Val: {len(val_manifest)}")

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
        "emotion_labels": EMOTION_LABELS,
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
