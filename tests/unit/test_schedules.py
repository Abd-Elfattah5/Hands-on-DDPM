"""Noise schedules: exact linear endpoints, cosine clipping and cumulative signal level behaviour."""

import pytest
import torch

from src.models.schedules import CosineSchedule, LinearSchedule


def test_linear_endpoints():
    betas = LinearSchedule(timesteps=1000).betas()
    assert betas.dtype == torch.float64
    assert betas.shape == (1000,)
    assert abs(betas[0].item() - 1e-4) < 1e-12
    assert abs(betas[-1].item() - 0.02) < 1e-12


def test_cosine_bounds():
    betas = CosineSchedule(timesteps=1000).betas()
    assert betas.shape == (1000,)
    assert (betas > 0).all() and (betas <= 0.999).all()


@pytest.mark.parametrize("schedule,limit", [(LinearSchedule(timesteps=1000), 1e-3), (CosineSchedule(timesteps=1000), 1e-2)])
def test_alphas_cumprod_monotonic_to_zero(schedule, limit):
    abar = torch.cumprod(1 - schedule.betas(), dim=0)
    assert (abar[1:] < abar[:-1]).all()
    assert abar[-1] < limit
    assert abar[0] > 0.99
