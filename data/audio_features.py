"""Audio feature extraction: mel spectrograms and Wav2Vec 2.0 embeddings."""

import numpy as np
import torch
import torchaudio
from scipy.interpolate import interp1d


def extract_mel(
    waveform: np.ndarray,
    sr: int,
    target_fps: int = 30,
    n_mels: int = 80,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """Extract log-mel spectrogram features and resample to target FPS.

    Args:
        waveform: Audio waveform, shape [num_samples].
        sr: Sample rate of the waveform.
        target_fps: Target frame rate to align with motion data.
        n_mels: Number of mel frequency bins.
        win_ms: Window length in milliseconds.
        hop_ms: Hop length in milliseconds.

    Returns:
        Log-mel features of shape [T_target, n_mels].
    """
    win_length = int(sr * win_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)

    waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=max(512, win_length),
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    mel_spec = mel_transform(waveform_tensor)  # [1, n_mels, T_mel]
    log_mel = torch.log(mel_spec.clamp(min=1e-7))
    log_mel = log_mel.squeeze(0).transpose(0, 1).numpy()  # [T_mel, n_mels]

    # Resample to target FPS
    mel_fps = sr / hop_length
    return _resample_features(log_mel, mel_fps, target_fps)


def extract_wav2vec(
    waveform: np.ndarray,
    sr: int,
    target_fps: int = 30,
    model_name: str = "facebook/wav2vec2-base-960h",
    device: str = "cpu",
) -> np.ndarray:
    """Extract Wav2Vec 2.0 features and resample to target FPS.

    Args:
        waveform: Audio waveform, shape [num_samples].
        sr: Sample rate (must be 16000 for wav2vec).
        target_fps: Target frame rate.
        model_name: HuggingFace model identifier.
        device: Device for inference.

    Returns:
        Wav2Vec features of shape [T_target, 768].
    """
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    # Resample to 16kHz if needed
    if sr != 16000:
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform_tensor).squeeze(0).numpy()
        sr = 16000

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()

    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device != "cpu")):
        outputs = model(input_values)
        features = outputs.last_hidden_state  # [1, T_wav2vec, 768]

    features = features.squeeze(0).cpu().numpy()  # [T_wav2vec, 768]

    # Wav2Vec outputs at ~49.9 Hz for 16kHz audio (stride=320 samples)
    wav2vec_fps = sr / 320
    return _resample_features(features, wav2vec_fps, target_fps)


def _resample_features(
    features: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    """Resample feature sequence to target frame rate."""
    if abs(source_fps - target_fps) < 0.1:
        return features

    T_source = features.shape[0]
    duration = (T_source - 1) / source_fps
    T_target = int(round(duration * target_fps)) + 1

    if T_target < 1:
        T_target = 1

    source_times = np.linspace(0, duration, T_source)
    target_times = np.linspace(0, duration, T_target)

    interp_func = interp1d(
        source_times, features, axis=0, kind="linear", fill_value="extrapolate"
    )
    return interp_func(target_times)
