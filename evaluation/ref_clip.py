"""Reference-clip evaluation shared across training tracks.

Used by Stage 2, Stage 3, and the flow-matching trainer for best-checkpoint
selection and for side-by-side A/B comparison between training approaches.

Split into three steps so different trainers can plug in their own inference:
  1. `prepare_ref_clip` — one-time load of audio + GT + stats (rank-0 only).
  2. A trainer-specific inference call that produces normalized predictions.
     For the AR transformer track, use `ar_slide_inference` here.
     For the FM track, plug in a per-window FM sampler.
  3. `compute_ref_metrics` — raw-L1 + lag-tolerant L1 + per-dim std ratios.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from data.audio_features import extract_mel


# Perceptually-dominant jaw/mouth dims (for diagnosing motion-range collapse).
MOUTH_DIMS = list(range(11))
# Dims flagged in commit cb97777 as having the worst pred_std/gt_std gap.
PROBLEM_DIMS = [22, 76, 95]


def prepare_ref_clip(config, train_stats: dict, device: torch.device) -> dict | None:
    """Load reference audio + matching BEAT2 GT for per-epoch ref evaluation.

    Returns None (with a printed warning) if the audio file, matching npz,
    expression stats, or required config are missing. Only call this on
    rank 0 — the returned dict contains tensors on `device`.

    Args:
        config: The TrainConfig. Reads `data.audio_feature`, `data.fps`,
            `data.mel_*`, `data.context_past/future`, `data.flame_expr_dim`,
            `paths.beat2_raw_dir`, and the track-specific
            `stage2.ref_clip_audio_path` (stage 3 / FM reuse the same field).
        train_stats: The training dataset's `stats` dict (must contain
            `expr_mean` and `expr_std`).
        device: Device for the audio/emotion tensors.

    Returns:
        Dict with keys `audio, emotion, gt_raw, expr_mean, expr_std, T`,
        or None on any load failure.
    """
    audio_path_str = config.stage2.ref_clip_audio_path
    if not audio_path_str:
        return None

    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        print(f"  [ref_clip] audio not found: {audio_path} — skipping ref eval")
        return None

    if config.data.audio_feature != "mel":
        print(f"  [ref_clip] only mel audio_feature supported; got "
              f"{config.data.audio_feature} — skipping ref eval")
        return None

    waveform, sr = sf.read(str(audio_path))
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)

    audio_feats = extract_mel(
        waveform, sr, config.data.fps,
        n_mels=config.data.mel_n_mels,
        win_ms=config.data.mel_win_ms,
        hop_ms=config.data.mel_hop_ms,
    )

    C = config.data.context_past
    F = config.data.context_future
    audio_padded = np.pad(audio_feats, ((C, F), (0, 0)), mode="edge")
    audio_tensor = torch.from_numpy(audio_padded).float().unsqueeze(0).to(device)

    parts = audio_path.stem.split("_")
    try:
        emotion = int(parts[2])
    except (IndexError, ValueError):
        emotion = 0
    emotion_tensor = torch.tensor([emotion], device=device)

    beat2_dir = Path(config.paths.beat2_raw_dir)
    npz_candidates = list(beat2_dir.rglob(f"{audio_path.stem}.npz"))
    if not npz_candidates:
        print(f"  [ref_clip] no matching npz for {audio_path.stem} under "
              f"{beat2_dir} — skipping ref eval")
        return None
    npz_data = np.load(str(npz_candidates[0]), allow_pickle=True)
    if "expressions" not in npz_data:
        print(f"  [ref_clip] npz has no 'expressions' field — skipping ref eval")
        return None
    gt_raw = npz_data["expressions"][:, :config.data.flame_expr_dim].astype(np.float32)

    if "expr_mean" not in train_stats or "expr_std" not in train_stats:
        print(f"  [ref_clip] train dataset missing expr_mean/expr_std — skipping ref eval")
        return None
    expr_mean = torch.from_numpy(train_stats["expr_mean"]).float().to(device)
    expr_std = torch.from_numpy(train_stats["expr_std"]).float().to(device)
    expr_std = torch.where(expr_std < 1e-8, torch.ones_like(expr_std), expr_std)

    T = audio_feats.shape[0]
    print(f"  [ref_clip] loaded {audio_path.name}: T={T} frames, emotion={emotion}, "
          f"gt_frames={gt_raw.shape[0]}")

    return {
        "audio": audio_tensor,
        "emotion": emotion_tensor,
        "gt_raw": gt_raw,
        "expr_mean": expr_mean,
        "expr_std": expr_std,
        "T": T,
    }


@torch.no_grad()
def ar_slide_inference(
    generator,
    clip: dict,
    *,
    horizon: int,
    context_past: int,
    context_future: int,
    prev_frames: int,
    expr_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Sliding-window autoregressive inference for the AR transformer generator.

    Mirrors the loop previously inlined in `Stage2Trainer._run_ref_inference`.
    Returns normalized predictions as a CPU float tensor of shape [T, D].
    """
    T = clip["T"]
    audio = clip["audio"]
    emotion = clip["emotion"]

    gen = generator.module if hasattr(generator, "module") else generator

    prev_expr = torch.zeros(1, prev_frames, expr_dim, device=device)
    chunks = []
    for t in range(0, T, horizon):
        chunk_len = min(horizon, T - t)
        audio_chunk = audio[:, t:t + context_past + chunk_len + context_future]
        pred = gen(
            audio_chunk, emotion, prev_expr,
            target_expression=None, max_len=chunk_len,
        )
        chunks.append(pred)
        prev_expr = torch.cat([prev_expr, pred], dim=1)[:, -prev_frames:]

    pred_norm = torch.cat(chunks, dim=1)[0].float().cpu()
    return pred_norm


@torch.no_grad()
def fm_slide_inference(
    generator,
    clip: dict,
    *,
    window_size: int,
    nfe: int,
    context_past: int,
    context_future: int,
    prev_frames: int,
    expr_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Sliding-window FM sampling. Each window is drawn by Euler-integrating
    the FM velocity field 0 → 1 with `nfe` steps; last `prev_frames` frames
    feed the next window as conditioning.

    The FM generator is stochastic (x_0 drawn from N(0, I)), so calling this
    repeatedly on the same clip produces different sequences — pair it with
    `compute_ref_metrics` inside a best-of-K loop when comparing to the
    deterministic AR baseline's single metric.
    """
    T = clip["T"]
    audio = clip["audio"]
    emotion = clip["emotion"]

    gen = generator.module if hasattr(generator, "module") else generator

    prev_expr = torch.zeros(1, prev_frames, expr_dim, device=device)
    chunks = []
    for t in range(0, T, window_size):
        chunk_len = min(window_size, T - t)
        audio_chunk = audio[:, t:t + context_past + chunk_len + context_future]
        pred = gen.sample_window(
            audio_chunk, emotion, prev_expr,
            window_size=chunk_len, nfe=nfe,
        )
        chunks.append(pred)
        prev_expr = torch.cat([prev_expr, pred], dim=1)[:, -prev_frames:]

    pred_norm = torch.cat(chunks, dim=1)[0].float().cpu()
    return pred_norm


def _lag_tolerant_l1(pred_raw: np.ndarray, gt_raw: np.ndarray, max_shift: int = 2) -> float:
    """Minimum raw-L1 over integer frame shifts in [-max_shift, +max_shift].

    Motivation: a single reference-clip raw-L1 with zero tolerance conflates
    amplitude error with 1-2 frame timing offsets on expressive clips. Taking
    the minimum over small shifts isolates amplitude/dynamics accuracy.
    """
    T = min(pred_raw.shape[0], gt_raw.shape[0])
    pred = pred_raw[:T]
    gt = gt_raw[:T]

    best = float(np.abs(pred - gt).mean())
    for shift in range(1, max_shift + 1):
        # pred leads gt by `shift` frames: compare pred[shift:] vs gt[:-shift]
        lead = float(np.abs(pred[shift:] - gt[:-shift]).mean())
        # pred lags gt by `shift` frames: compare pred[:-shift] vs gt[shift:]
        lag = float(np.abs(pred[:-shift] - gt[shift:]).mean())
        best = min(best, lead, lag)
    return best


def compute_ref_metrics(pred_norm: torch.Tensor, clip: dict) -> dict:
    """Raw-space L1 + lag-tolerant L1 + per-dim std ratios.

    Args:
        pred_norm: Normalized predictions [T, D] (CPU or any device).
        clip: Dict from `prepare_ref_clip`.

    Returns:
        Dict of scalar metrics. Also contains `per_dim_std_ratio` as an
        ndarray for detailed wandb logging.
    """
    expr_std = clip["expr_std"].detach().cpu().numpy()
    expr_mean = clip["expr_mean"].detach().cpu().numpy()

    pred_raw = pred_norm.detach().cpu().numpy() * expr_std + expr_mean
    gt_raw = clip["gt_raw"]
    n = min(pred_raw.shape[0], gt_raw.shape[0])
    pred_raw = pred_raw[:n]
    gt_raw = gt_raw[:n]

    raw_l1 = float(np.abs(pred_raw - gt_raw).mean())
    raw_l1_lag = _lag_tolerant_l1(pred_raw, gt_raw, max_shift=2)

    pred_std_raw = pred_raw.std(axis=0)
    gt_std_raw = gt_raw.std(axis=0)
    eps = 1e-8
    ratio = pred_std_raw / (gt_std_raw + eps)

    D = ratio.shape[0]
    mouth_idx = [i for i in MOUTH_DIMS if i < D]
    problem_idx = [i for i in PROBLEM_DIMS if i < D]

    return {
        "ref_raw_l1": raw_l1,
        "ref_raw_l1_lag": raw_l1_lag,
        "ref_std_ratio_full": float(ratio.mean()),
        "ref_std_ratio_mouth": float(ratio[mouth_idx].mean()) if mouth_idx else float("nan"),
        "ref_std_ratio_problem": float(ratio[problem_idx].mean()) if problem_idx else float("nan"),
        "per_dim_std_ratio": ratio,
    }
