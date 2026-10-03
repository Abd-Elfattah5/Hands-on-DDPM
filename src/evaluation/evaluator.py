"""Test-split noise-prediction loss with fixed-seed timesteps and noise (repeatable across runs)."""

from typing import Any

import torch
from torch.utils.data import DataLoader

from src.models.base import BaseDiffusion
from src.utils.seeding import make_generator


@torch.no_grad()
def evaluate_model(model: BaseDiffusion, data_loader: DataLoader, device: torch.device, seed: int = 42) -> dict[str, Any]:
    """Sample-weighted mean L_simple over ``data_loader``.

    A dedicated generator seeded with ``seed`` draws every timestep and noise tensor, so two calls with
    the same model and data return the same value.
    """
    model.eval()
    generator = make_generator(seed, device)
    total, n = 0.0, 0
    for batch in data_loader:
        x0 = batch[0].to(device)
        loss = model.compute_loss(model(x0, generator=generator)).loss
        total += loss.item() * x0.shape[0]
        n += x0.shape[0]
    return {"test_loss": round(total / max(n, 1), 6), "total_samples": n}
