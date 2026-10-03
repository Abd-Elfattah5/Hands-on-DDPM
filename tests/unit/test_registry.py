"""Component registries: registration rules, lookup errors and the config-driven factory."""

import pytest

from src.models import registry
from src.models.base import BaseNoiseSchedule


def test_register_non_subclass_raises():
    with pytest.raises(TypeError):
        registry.register_schedule("bad")(object)


def test_missing_name_lists_available():
    with pytest.raises(KeyError, match="Available schedules"):
        registry.get_schedule("missing")


def test_builtin_schedules_resolve():
    assert issubclass(registry.get_schedule("linear"), BaseNoiseSchedule)
    assert issubclass(registry.get_schedule("cosine"), BaseNoiseSchedule)


def test_builtin_denoiser_and_diffusion_resolve():
    assert registry.get_denoiser("unet").__name__ == "UNet"
    assert registry.get_diffusion("gaussian").__name__ == "GaussianDiffusion"


def test_build_diffusion_from_config(tiny_config):
    model = registry.build_diffusion_from_config(tiny_config)
    assert model.timesteps == 50
    assert model.config["diffusion"]["timesteps"] == 50
