"""Configuration dataclasses with YAML loading and validation."""

import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

import yaml


@dataclass
class PathConfig:
    beat2_raw_dir: str = ""
    processed_dir: str = ""
    flame_model_path: str = ""
    checkpoint_dir: str = ""
    wandb_dir: str = ""


@dataclass
class DataConfig:
    fps: int = 30
    audio_feature: str = "mel"  # "mel" or "wav2vec"
    mel_n_mels: int = 80
    mel_win_ms: float = 25.0
    mel_hop_ms: float = 10.0
    wav2vec_model: str = "facebook/wav2vec2-base-960h"
    wav2vec_dim: int = 768
    flame_expr_dim: int = 100
    context_past: int = 10
    context_future: int = 5
    prev_frames: int = 2
    seq_len: int = 60
    disc_window: int = 5
    dev_speakers: list[str] = field(default_factory=lambda: [
        "2", "4", "6", "8"
    ])
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class GeneratorConfig:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    n_emotions: int = 8
    emotion_embed_dim: int = 64
    audio_conv_channels: list[int] = field(default_factory=lambda: [128, 256])
    audio_conv_kernel_sizes: list[int] = field(default_factory=lambda: [5, 3])


@dataclass
class DiscriminatorConfig:
    d_model: int = 256
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    fc_hidden: int = 128
    gp_weight: float = 10.0


@dataclass
class Stage2Config:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    gen_horizon: int = 1
    val_every: int = 1
    save_every: int = 5
    patience: int = 0
    seed: int = 42
    lambda_var: float = 0.0
    lambda_var_warmup_epochs: int = 0
    lambda_var_decay_epochs: int = 0
    ref_clip_audio_path: str = ""
    use_cosine_lr: bool = False
    cosine_lr_min_ratio: float = 0.1
    # Self-drift schedule (Track A2): probability of feeding the generator's
    # own prior output as prev-frames during short-horizon training. Linearly
    # ramped from start → end over `p_drift_warmup_epochs`. Default 0 = off.
    p_drift_start: float = 0.0
    p_drift_end: float = 0.0
    p_drift_warmup_epochs: int = 0
    # Longer-horizon variance matching (Track A3): run one full 60-frame AR
    # rollout per batch from `lambda_var_full_start` onward and compute
    # variance-matching loss on it with weight `lambda_var_full`. Faded to 0
    # over `lambda_var_full_decay_epochs` ending at `epochs`. Default 0 = off.
    lambda_var_full: float = 0.0
    lambda_var_full_start: int = 0
    lambda_var_full_decay_epochs: int = 0
    # Velocity and acceleration L1 losses on the short-horizon prediction.
    # Cheap (fully batched, no sequential cost); directly attack motion-range
    # damping (velocity) and frame-to-frame jitter (acceleration). Default 0.
    lambda_vel: float = 0.0
    lambda_accel: float = 0.0
    # Quality-neutral GPU-efficiency toggles.
    # use_bf16: wrap generator forward + loss computation in bfloat16 autocast.
    #   A100 native bf16 tensor cores; same range as fp32 so no GradScaler
    #   needed and well within the numerical budget of this model.
    # use_compile: torch.compile the generator before DDP wrapping. Fuses
    #   kernels, removes Python/autograd overhead. First iteration pays a
    #   ~30-60s compile cost; subsequent iterations are 1.3-1.5x faster.
    use_bf16: bool = False
    use_compile: bool = False


@dataclass
class Stage3Config:
    epochs: int = 100
    batch_size: int = 64
    gen_lr: float = 5e-5
    gen_beta1: float = 0.5
    gen_beta2: float = 0.999
    disc_lr: float = 2e-4
    disc_beta1: float = 0.5
    disc_beta2: float = 0.999
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    lambda_adv_start: float = 0.01
    lambda_adv_end: float = 0.1
    lambda_warmup_steps: int = 10000
    disc_steps_per_gen: int = 1
    gen_horizon: int = 1
    pretrained_gen_path: str = ""
    val_every: int = 1
    save_every: int = 5
    seed: int = 42
    # Self-drift schedule (Track A2). Default 0 = off (current behavior).
    p_drift_start: float = 0.0
    p_drift_end: float = 0.0
    p_drift_warmup_epochs: int = 0


@dataclass
class FMConfig:
    """Flow-matching training config (parallel track to Stage 2 / Stage 3).

    FM trains on `window_size`-frame windows from each training sample. At
    inference it slides windows and autoregresses between them, feeding the
    last `prev_frames` generated frames as conditioning for the next window.
    """
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    window_size: int = 30
    t_sampling: str = "uniform"   # "uniform" or "logit_normal"
    nfe_inference: int = 4
    time_embed_dim: int = 64
    val_every: int = 1
    save_every: int = 5
    patience: int = 0
    seed: int = 42
    use_cosine_lr: bool = False
    cosine_lr_min_ratio: float = 0.1
    # Best-of-K stochastic ref-clip evaluation. 0 or 1 → single-sample metric;
    # >1 → report min over K samples (fair comparison vs deterministic baseline).
    eval_n_samples: int = 4


@dataclass
class TrainConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    fm: FMConfig = field(default_factory=FMConfig)
    wandb_project: str = "d4-head-facial-motion"
    wandb_entity: str = ""


def _resolve_env_vars(value: str) -> str:
    """Expand environment variables like $WORK in string values."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _merge_dict_into_dataclass(dc, d: dict):
    """Recursively merge a dict into a dataclass instance."""
    for key, value in d.items():
        if not hasattr(dc, key):
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dict_into_dataclass(current, value)
        elif isinstance(value, str):
            setattr(dc, key, _resolve_env_vars(value))
        else:
            setattr(dc, key, value)


def _validate(config: TrainConfig):
    """Validate config constraints."""
    if config.data.audio_feature not in ("mel", "wav2vec"):
        raise ValueError(
            f"data.audio_feature must be 'mel' or 'wav2vec', got '{config.data.audio_feature}'"
        )
    if config.data.seq_len < config.data.disc_window:
        raise ValueError("data.seq_len must be >= data.disc_window")
    if config.data.seq_len < config.fm.window_size:
        raise ValueError("data.seq_len must be >= fm.window_size")
    if config.fm.t_sampling not in ("uniform", "logit_normal"):
        raise ValueError(
            f"fm.t_sampling must be 'uniform' or 'logit_normal', got '{config.fm.t_sampling}'"
        )
    if config.generator.n_layers < 1:
        raise ValueError("generator.n_layers must be >= 1")
    if config.discriminator.n_layers < 1:
        raise ValueError("discriminator.n_layers must be >= 1")


def load_config(yaml_path: str) -> TrainConfig:
    """Load a YAML config file into a validated TrainConfig.

    Supports a _base_ field for inheriting from a base config file.
    """
    yaml_path = Path(yaml_path)

    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    # Handle base config inheritance
    config = TrainConfig()
    if "_base_" in raw:
        base_path = yaml_path.parent / raw.pop("_base_")
        with open(base_path) as f:
            base_raw = yaml.safe_load(f) or {}
        _merge_dict_into_dataclass(config, base_raw)

    _merge_dict_into_dataclass(config, raw)
    _validate(config)
    return config
