"""Forward diffusion: buffer identities, closed-form q(x_t|x_0), statistics at t=T−1, posterior terms."""

import pytest
import torch

from src.models.registry import build_diffusion_from_config


@pytest.fixture(params=["tiny", "baseline"])
def model(request, tiny_config, baseline_config):
    cfg = tiny_config if request.param == "tiny" else baseline_config
    return build_diffusion_from_config(cfg)


def test_signal_noise_identity(model):
    total = model.sqrt_alphas_cumprod**2 + model.sqrt_one_minus_alphas_cumprod**2
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


def test_q_sample_closed_form(model, synthetic_image_batch):
    x0 = synthetic_image_batch
    t = torch.tensor([0, 5, 10, model.timesteps - 1])
    noise = torch.randn_like(x0)
    expected = model.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x0 + \
        model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise
    assert torch.equal(model.q_sample(x0, t, noise), expected)


def test_final_step_is_standard_normal(baseline_config):
    model = build_diffusion_from_config(baseline_config)
    x0 = torch.full((512, 3, 32, 32), 0.7)
    t = torch.full((512,), model.timesteps - 1)
    xt = model.q_sample(x0, t, torch.randn(x0.shape, generator=torch.Generator().manual_seed(0)))
    assert abs(xt.mean().item()) < 0.05
    assert abs(xt.std().item() - 1) < 0.05


def test_posterior_terms(model):
    b = model.betas.double()
    a = 1 - b
    abar = torch.cumprod(a, 0)
    abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])
    var = b * (1 - abar_prev) / (1 - abar)
    assert torch.allclose(model.posterior_variance.double(), var, atol=1e-6)
    assert model.posterior_variance[0].item() == 0.0
    assert torch.isfinite(model.posterior_log_variance_clipped).all()
    assert torch.allclose(model.posterior_log_variance_clipped[0], model.posterior_log_variance_clipped[1])
    assert torch.allclose(model.posterior_mean_coef1.double(), b * abar_prev.sqrt() / (1 - abar), atol=1e-6)
    assert torch.allclose(model.posterior_mean_coef2.double(), (1 - abar_prev) * a.sqrt() / (1 - abar), atol=1e-6)
