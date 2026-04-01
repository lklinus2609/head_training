"""PyTorch Dataset for preprocessed BEAT2 HDF5 data."""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FacialMotionDataset(Dataset):
    """Dataset that loads preprocessed FLAME expression sequences from HDF5.

    Each sample is a fixed-length window of expression parameters with
    corresponding audio context, derivatives, and emotion labels.
    """

    def __init__(
        self,
        h5_path: str,
        seq_len: int = 60,
        context_past: int = 10,
        context_future: int = 5,
        prev_frames: int = 2,
        normalize: bool = True,
    ):
        """
        Args:
            h5_path: Path to the HDF5 file (train.h5 or val.h5).
            seq_len: Number of frames per training sample.
            context_past: Past audio context frames (C).
            context_future: Future audio context frames (F).
            prev_frames: Number of previous expression frames for autoregressive input (P).
            normalize: Whether to apply Z-score normalization.
        """
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.context_past = context_past
        self.context_future = context_future
        self.prev_frames = prev_frames
        self.normalize = normalize

        # Build index: list of (group_path, start_frame, num_frames)
        self.index = []
        self.stats = {}
        self._h5_file = None

        with h5py.File(h5_path, "r") as f:
            # Load normalization statistics
            if "stats" in f:
                for key in f["stats"]:
                    self.stats[key] = f["stats"][key][:]

            # Build sample index
            for speaker_key in sorted(f.keys()):
                if speaker_key == "stats":
                    continue
                speaker_grp = f[speaker_key]
                for utt_key in sorted(speaker_grp.keys()):
                    grp_path = f"{speaker_key}/{utt_key}"
                    num_frames = f[grp_path]["expression"].shape[0]

                    # All valid start positions where a full seq_len window fits
                    for start in range(0, num_frames - seq_len + 1, seq_len // 2):
                        self.index.append((grp_path, start, num_frames))

    def __len__(self):
        return len(self.index)

    def _get_h5(self):
        """Lazily open HDF5 file (per-worker for multiprocessing safety)."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __getitem__(self, idx):
        grp_path, start, total_frames = self.index[idx]
        f = self._get_h5()
        grp = f[grp_path]

        end = start + self.seq_len

        # Expression, velocity, acceleration for the target window
        expression = grp["expression"][start:end].astype(np.float32)
        velocity = grp["velocity"][start:end].astype(np.float32)
        acceleration = grp["acceleration"][start:end].astype(np.float32)

        # Audio features with past/future context
        audio_start = max(0, start - self.context_past)
        audio_end = min(total_frames, end + self.context_future)
        audio_raw = grp["audio_features"][audio_start:audio_end].astype(np.float32)

        # Pad audio if needed (at boundaries)
        audio_dim = audio_raw.shape[-1]
        total_audio_len = self.seq_len + self.context_past + self.context_future
        audio = np.zeros((total_audio_len, audio_dim), dtype=np.float32)

        # Compute where the raw audio fits in the padded array
        pad_before = (start - self.context_past) - audio_start + (self.context_past - (start - audio_start))
        # Simpler approach: place raw audio in correct position
        offset = self.context_past - (start - audio_start)
        length = audio_raw.shape[0]
        audio[offset:offset + length] = audio_raw

        # Previous expression frames (autoregressive context)
        prev_start = max(0, start - self.prev_frames)
        prev_expr = grp["expression"][prev_start:start].astype(np.float32)

        # Pad if at the beginning of the utterance
        if prev_expr.shape[0] < self.prev_frames:
            pad = np.zeros((self.prev_frames - prev_expr.shape[0], expression.shape[-1]), dtype=np.float32)
            prev_expr = np.concatenate([pad, prev_expr], axis=0)

        # Emotion label
        emotion = int(grp["emotion_label"][()])

        # Normalize
        if self.normalize and self.stats:
            expression = self._normalize(expression, "expr")
            velocity = self._normalize(velocity, "vel")
            acceleration = self._normalize(acceleration, "accel")
            prev_expr = self._normalize(prev_expr, "expr")

        return {
            "expression": torch.from_numpy(expression),
            "velocity": torch.from_numpy(velocity),
            "acceleration": torch.from_numpy(acceleration),
            "audio": torch.from_numpy(audio),
            "prev_expression": torch.from_numpy(prev_expr),
            "emotion": torch.tensor(emotion, dtype=torch.long),
        }

    def _normalize(self, data: np.ndarray, prefix: str) -> np.ndarray:
        mean = self.stats.get(f"{prefix}_mean")
        std = self.stats.get(f"{prefix}_std")
        if mean is not None and std is not None:
            std_safe = np.where(std < 1e-8, 1.0, std)
            return (data - mean) / std_safe
        return data

    def get_audio_dim(self) -> int:
        """Return the dimensionality of audio features."""
        with h5py.File(self.h5_path, "r") as f:
            for speaker_key in f.keys():
                if speaker_key == "stats":
                    continue
                for utt_key in f[speaker_key].keys():
                    return f[f"{speaker_key}/{utt_key}"]["audio_features"].shape[-1]
        return 80  # Default mel dim


class DiscriminatorWindowDataset(Dataset):
    """Dataset that yields K-frame windows for discriminator training from real data."""

    def __init__(
        self,
        h5_path: str,
        window_size: int = 5,
        normalize: bool = True,
    ):
        self.h5_path = h5_path
        self.window_size = window_size
        self.normalize = normalize

        self.index = []
        self.stats = {}
        self._h5_file = None

        with h5py.File(h5_path, "r") as f:
            if "stats" in f:
                for key in f["stats"]:
                    self.stats[key] = f["stats"][key][:]

            for speaker_key in sorted(f.keys()):
                if speaker_key == "stats":
                    continue
                for utt_key in sorted(f[speaker_key].keys()):
                    grp_path = f"{speaker_key}/{utt_key}"
                    num_frames = f[grp_path]["expression"].shape[0]
                    for start in range(num_frames - window_size + 1):
                        self.index.append((grp_path, start))

    def __len__(self):
        return len(self.index)

    def _get_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __getitem__(self, idx):
        grp_path, start = self.index[idx]
        f = self._get_h5()
        grp = f[grp_path]

        end = start + self.window_size

        expression = grp["expression"][start:end].astype(np.float32)
        velocity = grp["velocity"][start:end].astype(np.float32)
        acceleration = grp["acceleration"][start:end].astype(np.float32)
        audio = grp["audio_features"][start:end].astype(np.float32)

        if self.normalize and self.stats:
            expression = self._normalize(expression, "expr")
            velocity = self._normalize(velocity, "vel")
            acceleration = self._normalize(acceleration, "accel")

        # Concatenate all features for discriminator input
        window = np.concatenate([expression, velocity, acceleration, audio], axis=-1)

        return {
            "window": torch.from_numpy(window),
            "audio": torch.from_numpy(audio),
        }

    def _normalize(self, data: np.ndarray, prefix: str) -> np.ndarray:
        mean = self.stats.get(f"{prefix}_mean")
        std = self.stats.get(f"{prefix}_std")
        if mean is not None and std is not None:
            std_safe = np.where(std < 1e-8, 1.0, std)
            return (data - mean) / std_safe
        return data
