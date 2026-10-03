"""Algorithm 2 sampling: shapes, clamping, last-step noise, variance options, determinism, trajectory."""

import pytest
import torch

from src.models.registry import build_diffusion_from_config
from src.models.types import SamplingOutput
from src.utils.seeding import make_generator


@pytest.fixture
def model(tiny_config):
    torch.manual_seed(0)
    m = build_diffusion_from_config(tiny_config).eval()
    for p in m.parameters():  # non-trivial outputs despite zero-init
        torch.nn.init.normal_(p, std=0.02)
    return m


def test_sample_shape_and_range(model):
    out = model.sample(4, torch.device("cpu"), generator=make_generator(0))
    assert isinstance(out, SamplingOutput)
    assert out.samples.shape == (4, 3, 32, 32)
    assert torch.isfinite(out.samples).all()
    assert out.samples.min() >= -1 and out.samples.max() <= 1


def test_seed_determinism(model):
    a = model.sample(3, torch.device("cpu"), generator=make_generator(7)).samples
    b = model.sample(3, torch.device("cpu"), generator=make_generator(7)).samples
    c = model.sample(3, torch.device("cpu"), generator=make_generator(8)).samples
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_last_step_adds_no_noise(model, monkeypatch):
    x = torch.randn(2, 3, 32, 32)
    expected = model.p_sample(x, 0)
    monkeypatch.setattr(torch, "randn", lambda *a, **k: torch.full(a[0] if a else k["size"], 100.0))
    assert torch.equal(model.p_sample(x, 0), expected)


@pytest.mark.parametrize("variance_type,buffer", [("fixed_large", "betas"), ("fixed_small", "posterior_variance")])
def test_variance_options(tiny_config, variance_type, buffer):
    tiny_config["diffusion"]["variance_type"] = variance_type
    m = build_diffusion_from_config(tiny_config).eval()  # zero-init output head → ε_θ = 0
    x = torch.zeros(64, 3, 32, 32)  # 196,608 noise draws → std estimate within ~0.3%
    t = 10
    step = m.p_sample(x, t, generator=make_generator(0))
    mean = m.sqrt_recip_alphas[t] * x
    std = (step - mean).std().item()
    assert abs(std - getattr(m, buffer)[t].sqrt().item()) < 0.01 * getattr(m, buffer)[t].sqrt().item() + 1e-4


def test_trajectory_labels(model):
    out = model.sample(4, torch.device("cpu"), generator=make_generator(1), trajectory_steps=[50, 25, 0])
    assert out.trajectory.shape == (3, 4, 3, 32, 32)
    assert out.trajectory_steps == [50, 25, 0]
    x_T = torch.randn((4, 3, 32, 32), generator=make_generator(1))
    assert torch.equal(out.trajectory[0], x_T)  # label T = initial noise
    assert torch.equal(out.trajectory[-1], out.samples)  # label 0 = final output


def test_batching_matches_single_batch(model):
    a = model.sample(4, torch.device("cpu"), generator=make_generator(3), batch_size=2).samples
    g = make_generator(3)
    b1 = model.sample(2, torch.device("cpu"), generator=g).samples
    b2 = model.sample(2, torch.device("cpu"), generator=g).samples
    assert torch.equal(a, torch.cat([b1, b2]))


def test_on_batch_callback(model):
    seen = []
    model.sample(5, torch.device("cpu"), generator=make_generator(0), batch_size=2,
                 on_batch=lambda i, x: seen.append((i, x.shape[0])))
    assert seen == [(0, 2), (1, 2), (2, 1)]
