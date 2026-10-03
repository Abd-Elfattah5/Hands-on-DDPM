"""Unit tests for quantitative benchmarking metrics (Fréchet distance, Inception Score)."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    calculate_frechet_distance,
    calculate_inception_score,
)


def test_frechet_distance_identical_distributions():
    """Verify Fréchet distance is zero when distributions are identical."""
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([
        [1.0, 0.1, 0.0],
        [0.1, 2.0, 0.2],
        [0.0, 0.2, 1.5],
    ])

    fd = calculate_frechet_distance(mu, sigma, mu, sigma)
    assert pytest.approx(fd, abs=1e-5) == 0.0


def test_frechet_distance_known_shift():
    """Verify Fréchet distance equals squared Euclidean distance when covariances are identical."""
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([3.0, 4.0])
    sigma = np.eye(2)

    # Expected: ||mu1 - mu2||^2 + Tr(sigma + sigma - 2 * sigma) = 3^2 + 4^2 + 0 = 25.0
    fd = calculate_frechet_distance(mu1, sigma, mu2, sigma)
    assert pytest.approx(fd, rel=1e-4) == 25.0


def test_inception_score_uniform_predictions():
    """Verify Inception Score equals 1.0 when predictions are uniform across classes."""
    # 100 samples, 10 classes, uniform probabilities -> KL(p(y|x) || p(y)) = 0 -> exp(0) = 1.0
    probs = np.full((100, 10), 0.1)
    is_mean, is_std = calculate_inception_score(probs, splits=5)

    assert pytest.approx(is_mean, rel=1e-4) == 1.0
    assert pytest.approx(is_std, abs=1e-5) == 0.0


def test_inception_score_diverse_confident_predictions():
    """Verify Inception Score is high when predictions are confident and uniformly distributed across classes."""
    # Each sample predicts one class with probability 1.0, and all 10 classes are equally represented
    num_samples = 100
    probs = np.zeros((num_samples, 10))
    for i in range(num_samples):
        probs[i, i % 10] = 1.0

    is_mean, is_std = calculate_inception_score(probs, splits=5)
    # Theoretical IS for 10 distinct uniform classes is exp(log(10)) = 10.0
    assert pytest.approx(is_mean, rel=1e-2) == 10.0


# ---------------------------------------------------------------------------------- DDPM additions

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import metrics as M
from src.models.registry import build_diffusion_from_config


def test_inception_score_split_size():
    probs = np.random.default_rng(0).dirichlet(np.ones(10), size=5000)
    seen = []
    original = np.mean

    def spy(a, *args, **kwargs):
        if getattr(a, "ndim", 0) == 2 and kwargs.get("axis") == 0:
            seen.append(a.shape[0])
        return original(a, *args, **kwargs)

    np.mean = spy
    try:
        calculate_inception_score(probs, splits=10)
    finally:
        np.mean = original
    assert seen == [500] * 10


class _StubExtractor(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.g = torch.Generator().manual_seed(0)

    def forward(self, x):
        b = x.shape[0]
        feats = x.reshape(b, -1)[:, :16].cpu().double() + torch.randn(b, 16, generator=self.g, dtype=torch.float64)
        probs = torch.softmax(torch.randn(b, 1000, generator=self.g), dim=1)
        return feats.float(), probs


def test_compute_fid_and_is_with_stub(tiny_config, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "InceptionFeatureExtractor", _StubExtractor)
    tiny_config["diffusion"]["timesteps"] = 5
    tiny_config["sampling"]["strip_steps"] = [5, 0]
    model = build_diffusion_from_config(tiny_config).eval()
    real = DataLoader(TensorDataset(torch.rand(20, 3, 32, 32) * 2 - 1, torch.zeros(20)), batch_size=8)
    sample_dir = tmp_path / "samples"
    out = M.compute_fid_and_is(model, real, num_samples=20, batch_size=8, device=torch.device("cpu"),
                               sample_batch_size=8, seed=1, sample_dir=sample_dir, is_splits=2)
    for key in ("fid", "inception_score_mean", "inception_score_std", "benchmark_samples",
                "sampling_seconds_total", "sampling_seconds_per_image", "class_coverage"):
        assert key in out
    assert out["benchmark_samples"] == 20
    cov = out["class_coverage"]
    assert 1 <= cov["distinct_top1_classes"] <= 1000 and cov["marginal_entropy_nats"] >= 0
    assert len(list(sample_dir.glob("batch_*.pt"))) == 3

    calls = []
    monkeypatch.setattr(model, "sample", lambda *a, **k: calls.append(1))
    again = M.compute_fid_and_is(model, real, num_samples=20, batch_size=8, device=torch.device("cpu"),
                                 sample_batch_size=8, seed=1, sample_dir=sample_dir, reuse_samples=True, is_splits=2)
    assert calls == []
    assert again["benchmark_samples"] == 20
