"""Typed experiment configuration schema, loader, validator and override helper.

Mirrors `Hands-on VAE/src/configs/schema.py`. Callers keep working with the raw ``dict``; the
dataclasses are the single source of defaults and validation (data-model.md §3).
"""

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is malformed or violates a constraint."""


@dataclass
class ExperimentSectionConfig:
    name: str = "cifar10_baseline"
    seed: int = 42
    device: str = "auto"
    output_dir: str = "artifacts/runs/cifar10_baseline"


@dataclass
class DataSectionConfig:
    dataset: str = "cifar10"
    data_dir: str = "data"
    batch_size: int = 32
    num_workers: int = 4
    val_split: float = 0.1
    random_flip: bool = True


@dataclass
class ModelSectionConfig:
    name: str = "unet"
    in_channels: int = 3
    out_channels: int = 3
    image_size: int = 32
    channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    num_res_blocks: int = 2
    attention_resolutions: list[int] = field(default_factory=lambda: [16, 8])
    num_heads: int = 4
    num_groups: int = 32
    dropout: float = 0.1
    time_embed_dim: int = 256


@dataclass
class DiffusionSectionConfig:
    name: str = "gaussian"
    schedule: str = "linear"
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    cosine_s: float = 0.008
    variance_type: str = "fixed_large"


@dataclass
class TrainingSectionConfig:
    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 5000
    grad_accum_steps: int = 4
    gradient_clip_val: float = 1.0
    ema_decay: float = 0.9999
    mixed_precision: str = "none"
    save_every: int = 10
    sample_every: int = 5


@dataclass
class SamplingSectionConfig:
    batch_size: int = 256
    strip_steps: list[int] = field(default_factory=lambda: [1000, 800, 600, 400, 200, 100, 50, 0])


@dataclass
class EvaluationSectionConfig:
    num_samples: int = 5000
    batch_size: int = 64
    is_splits: int = 10


@dataclass
class VerifySectionConfig:
    memory_budget_mib: int = 3072


@dataclass
class ExperimentConfig:
    experiment: ExperimentSectionConfig = field(default_factory=ExperimentSectionConfig)
    data: DataSectionConfig = field(default_factory=DataSectionConfig)
    model: ModelSectionConfig = field(default_factory=ModelSectionConfig)
    diffusion: DiffusionSectionConfig = field(default_factory=DiffusionSectionConfig)
    training: TrainingSectionConfig = field(default_factory=TrainingSectionConfig)
    sampling: SamplingSectionConfig = field(default_factory=SamplingSectionConfig)
    evaluation: EvaluationSectionConfig = field(default_factory=EvaluationSectionConfig)
    verify: VerifySectionConfig = field(default_factory=VerifySectionConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS: dict[str, type] = {
    "experiment": ExperimentSectionConfig,
    "data": DataSectionConfig,
    "model": ModelSectionConfig,
    "diffusion": DiffusionSectionConfig,
    "training": TrainingSectionConfig,
    "sampling": SamplingSectionConfig,
    "evaluation": EvaluationSectionConfig,
    "verify": VerifySectionConfig,
}

MIXED_PRECISION_CHOICES = ("none", "bf16", "fp16")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration into a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def _build_section(name: str, raw: Any) -> Any:
    cls = _SECTIONS[name]
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Section '{name}' must be a mapping")
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in section '{name}': {unknown}. Allowed: {sorted(allowed)}")
    return cls(**raw)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_config(config: dict[str, Any]) -> ExperimentConfig:
    """Validate a raw config dict and return the typed ``ExperimentConfig``.

    Every constraint quoted in data-model.md §3 is enforced here. Unknown sections or keys raise
    ``ConfigError`` naming the offending key.
    """
    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a mapping")
    unknown_sections = sorted(set(config) - set(_SECTIONS))
    if unknown_sections:
        raise ConfigError(f"Unknown section(s): {unknown_sections}. Allowed: {sorted(_SECTIONS)}")

    cfg = ExperimentConfig(**{name: _build_section(name, config.get(name)) for name in _SECTIONS})
    e, d, m, df, t, s, ev, v = (
        cfg.experiment, cfg.data, cfg.model, cfg.diffusion, cfg.training, cfg.sampling, cfg.evaluation, cfg.verify,
    )

    # experiment
    _require(isinstance(e.name, str) and e.name != "", "experiment.name must be a non-empty string")
    _require(_is_int(e.seed) and e.seed >= 0, "experiment.seed must be an int ≥ 0")
    _require(e.device in ("auto", "cuda", "cpu"), "experiment.device must be one of 'auto' / 'cuda' / 'cpu'")

    # data
    _require(d.dataset == "cifar10", "data.dataset must be 'cifar10'")
    _require(_is_int(d.batch_size) and d.batch_size >= 1, "data.batch_size must be an int ≥ 1 (per-step micro-batch)")
    _require(_is_int(d.num_workers) and d.num_workers >= 0, "data.num_workers must be an int ≥ 0")
    _require(_is_num(d.val_split) and 0 < d.val_split < 1, "data.val_split must satisfy 0 < x < 1")
    _require(isinstance(d.random_flip, bool), "data.random_flip must be a bool")

    # model
    _require(isinstance(m.name, str) and m.name != "", "model.name must be a registered network name")
    _require(_is_int(m.in_channels) and m.in_channels >= 1, "model.in_channels must be ≥ 1")
    _require(m.in_channels == m.out_channels, "model.in_channels and model.out_channels must be equal")
    _require(isinstance(m.channels, list) and len(m.channels) >= 1 and all(_is_int(c) and c > 0 for c in m.channels),
             "model.channels must be a non-empty list of positive ints")
    _require(_is_int(m.num_groups) and m.num_groups >= 1, "model.num_groups must be ≥ 1")
    levels = len(m.channels)
    _require(_is_int(m.image_size) and m.image_size % (2 ** (levels - 1)) == 0,
             f"model.image_size must be divisible by 2^(len(channels)−1) = {2 ** (levels - 1)}")
    bad = [c for c in m.channels if c % m.num_groups != 0]
    _require(not bad, f"model.channels {bad} are not divisible by num_groups={m.num_groups}")
    _require(_is_int(m.num_res_blocks) and m.num_res_blocks >= 1, "model.num_res_blocks must be ≥ 1")
    resolutions = [m.image_size // (2**i) for i in range(levels)]
    attn = list(m.attention_resolutions)
    _require(all(r in resolutions for r in attn),
             f"model.attention_resolutions {attn} must be a subset of the U-Net resolutions {resolutions}")
    _require(m.image_size not in attn,
             f"model.attention_resolutions must not include image_size={m.image_size} (FR-010: no attention at full resolution)")
    _require(_is_int(m.num_heads) and m.num_heads >= 1, "model.num_heads must be ≥ 1")
    attended = [m.channels[resolutions.index(r)] for r in attn] + [m.channels[-1]]  # bottleneck attention
    bad = [c for c in attended if c % m.num_heads != 0]
    _require(not bad, f"model.num_heads={m.num_heads} must divide every attended channel width; fails for {bad}")
    _require(_is_num(m.dropout) and 0 <= m.dropout < 1, "model.dropout must satisfy 0 ≤ x < 1")
    _require(_is_int(m.time_embed_dim) and m.time_embed_dim >= 1, "model.time_embed_dim must be ≥ 1")

    # diffusion
    _require(isinstance(df.name, str) and df.name != "", "diffusion.name must be a registered diffusion name")
    _require(isinstance(df.schedule, str) and df.schedule != "", "diffusion.schedule must be a registered schedule name")
    _require(_is_int(df.timesteps) and df.timesteps >= 1, "diffusion.timesteps must be ≥ 1")
    if df.schedule == "linear":
        _require(_is_num(df.beta_start) and _is_num(df.beta_end) and 0 < df.beta_start < df.beta_end < 1,
                 "diffusion.beta_start/beta_end must satisfy 0 < start < end < 1 (linear)")
    if df.schedule == "cosine":
        _require(_is_num(df.cosine_s) and df.cosine_s > 0, "diffusion.cosine_s must be > 0 (cosine)")
    _require(df.variance_type in ("fixed_large", "fixed_small"),
             "diffusion.variance_type must be 'fixed_large' / 'fixed_small'")

    # training
    _require(_is_int(t.epochs) and t.epochs >= 1, "training.epochs must be ≥ 1")
    _require(_is_num(t.lr) and t.lr > 0, "training.lr must be > 0")
    _require(_is_num(t.weight_decay) and t.weight_decay >= 0, "training.weight_decay must be ≥ 0")
    _require(_is_int(t.warmup_steps) and t.warmup_steps >= 0, "training.warmup_steps must be ≥ 0 (optimizer steps)")
    _require(_is_int(t.grad_accum_steps) and t.grad_accum_steps >= 1, "training.grad_accum_steps must be ≥ 1")
    _require(_is_num(t.gradient_clip_val) and t.gradient_clip_val > 0, "training.gradient_clip_val must be > 0")
    _require(_is_num(t.ema_decay) and 0 < t.ema_decay < 1, "training.ema_decay must satisfy 0 < x < 1")
    _require(t.mixed_precision in MIXED_PRECISION_CHOICES,
             f"training.mixed_precision must be one of {' / '.join(MIXED_PRECISION_CHOICES)}")
    _require(_is_int(t.save_every) and t.save_every >= 1, "training.save_every must be ≥ 1 epochs")
    _require(_is_int(t.sample_every) and t.sample_every >= 1, "training.sample_every must be ≥ 1 epochs")

    # sampling
    _require(_is_int(s.batch_size) and s.batch_size >= 1, "sampling.batch_size must be ≥ 1")
    steps = list(s.strip_steps)
    _require(len(steps) >= 1 and all(_is_int(x) and 0 <= x <= df.timesteps for x in steps),
             f"sampling.strip_steps must lie within [0, {df.timesteps}]")
    _require(all(a > b for a, b in zip(steps, steps[1:])), "sampling.strip_steps must be strictly descending")

    # evaluation
    _require(_is_int(ev.is_splits) and ev.is_splits >= 1, "evaluation.is_splits must be ≥ 1")
    _require(_is_int(ev.num_samples) and ev.num_samples >= 10 and ev.num_samples % ev.is_splits == 0,
             "evaluation.num_samples must be ≥ 10 and divisible by is_splits")
    _require(_is_int(ev.batch_size) and ev.batch_size >= 1, "evaluation.batch_size must be ≥ 1")

    # verify
    _require(_is_num(v.memory_budget_mib) and v.memory_budget_mib > 0, "verify.memory_budget_mib must be > 0")

    return cfg


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the full config dict with every default filled in."""
    return validate_config(config).to_dict()


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``config`` with dotted-key overrides applied.

    ``None`` values are skipped so CLI options that were not given leave the config unchanged.

    Example:
        ``apply_overrides(cfg, {"data.batch_size": 16, "training.epochs": None})``
    """
    result = copy.deepcopy(config)
    for dotted, value in overrides.items():
        if value is None:
            continue
        keys = dotted.split(".")
        node = result
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Cannot override '{dotted}': '{key}' is not a section")
        node[keys[-1]] = value
    return result
