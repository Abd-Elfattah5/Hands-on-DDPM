"""Fixed-seed, repeatable test noise-prediction loss."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluator import evaluate_model
from src.models.registry import build_diffusion_from_config


def test_evaluate_is_repeatable(tiny_config):
    model = build_diffusion_from_config(tiny_config).eval()
    loader = DataLoader(TensorDataset(torch.rand(10, 3, 32, 32) * 2 - 1, torch.zeros(10)), batch_size=4)
    a = evaluate_model(model, loader, torch.device("cpu"), seed=42)
    b = evaluate_model(model, loader, torch.device("cpu"), seed=42)
    assert a["total_samples"] == 10
    assert a["test_loss"] == b["test_loss"] and a["test_loss"] > 0
