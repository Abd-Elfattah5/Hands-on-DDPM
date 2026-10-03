"""Foundations: typed containers, abstract base classes, config validation and overrides."""

import copy

import pytest
import torch

from src.configs.schema import ConfigError, apply_overrides, validate_config
from src.models.base import BaseDenoiser, BaseDiffusion, BaseNoiseSchedule
from src.models.types import DiffusionOutput, LossOutput, SamplingOutput


def test_diffusion_output_fields():
    x = torch.zeros(2, 3, 32, 32)
    out = DiffusionOutput(x_t=x, t=torch.zeros(2, dtype=torch.long), noise=x, predicted_noise=x)
    assert out.extra == {}
    assert out.t.dtype == torch.long


def test_loss_and_sampling_output_fields():
    loss = LossOutput(loss=torch.tensor(1.0))
    assert loss.metrics == {}
    s = SamplingOutput(samples=torch.zeros(1, 3, 32, 32), trajectory=None, trajectory_steps=[], seconds=0.0)
    assert s.trajectory is None


@pytest.mark.parametrize("cls", [BaseNoiseSchedule, BaseDenoiser, BaseDiffusion])
def test_base_classes_are_abstract(cls):
    with pytest.raises(TypeError):
        cls()


def test_baseline_and_tiny_configs_validate(baseline_config, tiny_config):
    assert validate_config(baseline_config).model.channels == [64, 128, 256]
    assert validate_config(tiny_config).diffusion.timesteps == 50


def _expect_error(cfg, match):
    with pytest.raises(ConfigError, match=match):
        validate_config(cfg)


def test_unknown_key_rejected(baseline_config):
    cfg = copy.deepcopy(baseline_config)
    cfg["model"]["not_a_key"] = 1
    _expect_error(cfg, "not_a_key")


def test_unknown_section_rejected(baseline_config):
    cfg = copy.deepcopy(baseline_config)
    cfg["bogus"] = {}
    _expect_error(cfg, "bogus")


def test_attention_at_full_resolution_rejected(baseline_config):
    cfg = copy.deepcopy(baseline_config)
    cfg["model"]["attention_resolutions"] = [32]
    _expect_error(cfg, "FR-010")


def test_channels_must_divide_groups(baseline_config):
    cfg = copy.deepcopy(baseline_config)
    cfg["model"]["channels"] = [64, 100]
    _expect_error(cfg, "num_groups")


def test_beta_order_enforced(baseline_config):
    cfg = copy.deepcopy(baseline_config)
    cfg["diffusion"]["beta_start"] = 0.05
    _expect_error(cfg, "beta_start")


def test_num_samples_divisible_by_splits(baseline_config):
    cfg = copy.deepcopy(baseline_config)
    cfg["evaluation"]["num_samples"] = 5001
    _expect_error(cfg, "divisible")


def test_mixed_precision_choices(baseline_config):
    cfg = copy.deepcopy(baseline_config)
    cfg["training"]["mixed_precision"] = False
    _expect_error(cfg, "mixed_precision")


def test_apply_overrides_does_not_mutate(baseline_config):
    original = copy.deepcopy(baseline_config)
    out = apply_overrides(baseline_config, {"data.batch_size": 16, "training.epochs": None})
    assert out["data"]["batch_size"] == 16
    assert out["training"]["epochs"] == original["training"]["epochs"]
    assert baseline_config == original
