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

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from data.audio_features import extract_mel


# Perceptually-dominant jaw/mouth dims (for diagnosing motion-range collapse).
MOUTH_DIMS = list(range(11))
# Dims flagged in commit cb97777 as having the worst pred_std/gt_std gap.
PROBLEM_DIMS = [22, 76, 95]


def _prepare_one_clip(
    audio_path_str: str, config, train_stats: dict, device: torch.device
) -> dict | None:
    """Load one reference clip's audio + matching BEAT2 GT.

    Shared inner helper for `prepare_ref_clip` (singular) and
    `prepare_ref_clips` (plural). Returns None on any load failure with a
    printed warning.

    Returns:
        Dict with keys `audio, emotion, gt_raw, expr_mean, expr_std, T,
        label`, or None.
    """
    if not audio_path_str:
        return None

    # `_merge_dict_into_dataclass` only resolves env vars on scalar string
    # fields, not on list-of-strings (e.g. `ref_clip_audio_paths`). Expand
    # here so $WORK/... entries in the multi-clip list work the same as the
    # singular path. Idempotent on already-expanded values.
    audio_path_str = os.path.expandvars(audio_path_str)

    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        print(f"  [ref_clip] audio not found: {audio_path} — skipping")
        return None

    if config.data.audio_feature != "mel":
        print(f"  [ref_clip] only mel audio_feature supported; got "
              f"{config.data.audio_feature} — skipping")
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
              f"{beat2_dir} — skipping")
        return None
    npz_data = np.load(str(npz_candidates[0]), allow_pickle=True)
    if "expressions" not in npz_data:
        print(f"  [ref_clip] {audio_path.stem}.npz has no 'expressions' field — skipping")
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
        "label": audio_path.stem,
    }


def prepare_ref_clip(config, train_stats: dict, device: torch.device) -> dict | None:
    """Single-clip ref-eval preparation (legacy single-path API).

    Reads `config.stage2.ref_clip_audio_path`. Use `prepare_ref_clips` for
    the multi-clip path. Returns None on any load failure.
    """
    return _prepare_one_clip(
        config.stage2.ref_clip_audio_path, config, train_stats, device
    )


def prepare_ref_clips(config, train_stats: dict, device: torch.device) -> list[dict] | None:
    """Multi-clip ref-eval preparation.

    If `config.stage2.ref_clip_audio_paths` (plural) is non-empty, loads
    each path and returns the list (skipping any that fail to load).
    Otherwise falls back to `config.stage2.ref_clip_audio_path` (singular),
    returning a one-element list. Returns None if nothing loads — callers
    should treat None as "ref-eval disabled".

    Only call on rank 0.
    """
    paths = list(config.stage2.ref_clip_audio_paths)
    if paths:
        clips = []
        for p in paths:
            clip = _prepare_one_clip(p, config, train_stats, device)
            if clip is not None:
                clips.append(clip)
        if not clips:
            print("  [ref_clip] no clips loaded from ref_clip_audio_paths — disabling ref eval")
            return None
        print(f"  [ref_clip] {len(clips)} clip(s) loaded for multi-clip selection")
        return clips

    legacy = _prepare_one_clip(
        config.stage2.ref_clip_audio_path, config, train_stats, device
    )
    return [legacy] if legacy is not None else None


def aggregate_ref_metrics(per_clip_metrics: list[dict]) -> dict:
    """Average scalar metrics across clips.

    `per_dim_std_ratio` is averaged element-wise. NaN per-clip values are
    skipped per metric. Returns a dict with the same keys as
    `compute_ref_metrics` (averaged). Returns {} when the input list is
    empty.
    """
    if not per_clip_metrics:
        return {}
    scalar_keys = [
        "ref_raw_l1", "ref_raw_l1_lag",
        "ref_std_ratio_full", "ref_std_ratio_mouth", "ref_std_ratio_problem",
    ]
    out = {}
    for k in scalar_keys:
        vals = [m[k] for m in per_clip_metrics if k in m and not np.isnan(m[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")

    ratios = [m["per_dim_std_ratio"] for m in per_clip_metrics if "per_dim_std_ratio" in m]
    if ratios:
        out["per_dim_std_ratio"] = np.mean(np.stack(ratios, axis=0), axis=0)
    return out


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
