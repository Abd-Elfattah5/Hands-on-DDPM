"""L_simple: forward-pass contract, MSE value, gradient flow to every parameter, seeded determinism."""

import torch
import torch.nn.functional as F

from src.models.registry import build_diffusion_from_config
from src.models.gaussian_diffusion import perturb_zero_init_
from src.models.types import DiffusionOutput


def test_forward_contract(tiny_config, synthetic_image_batch):
    model = build_diffusion_from_config(tiny_config)
    out = model(synthetic_image_batch, generator=torch.Generator().manual_seed(0))
    assert isinstance(out, DiffusionOutput)
    assert out.t.dtype == torch.long
    assert out.t.min() >= 0 and out.t.max() <= model.timesteps - 1
    assert out.noise.shape == synthetic_image_batch.shape


def test_loss_value_and_gradients(tiny_config, synthetic_image_batch):
    model = build_diffusion_from_config(tiny_config)
    out = model(synthetic_image_batch, generator=torch.Generator().manual_seed(0))
    loss = model.compute_loss(out)
    assert torch.isfinite(loss.loss)
    assert torch.allclose(loss.loss, F.mse_loss(out.predicted_noise, out.noise))
    # Zero-initialized layers (research §4) block upstream gradients until they are trained, so the
    # full gradient path is checked on a copy whose all-zero tensors are perturbed (same as verify).
    perturb_zero_init_(model)
    model.zero_grad()
    model.compute_loss(model(synthetic_image_batch, generator=torch.Generator().manual_seed(1))).loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, name
            norm = p.grad.norm().item()
            assert norm > 0 and torch.isfinite(p.grad).all(), name


def test_same_seed_same_timesteps_and_noise(tiny_config, synthetic_image_batch):
    model = build_diffusion_from_config(tiny_config)
    a = model(synthetic_image_batch, generator=torch.Generator().manual_seed(3))
    b = model(synthetic_image_batch, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a.t, b.t) and torch.equal(a.noise, b.noise)
