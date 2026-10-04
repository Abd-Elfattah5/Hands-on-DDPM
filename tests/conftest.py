"""Shared pytest fixtures: devices, a tiny valid config, synthetic batches and in-memory loaders."""

import copy
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.configs.schema import load_config

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = ROOT / "configs" / "cifar10_baseline.yaml"


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def baseline_config() -> dict:
    return load_config(BASELINE_CONFIG)


def make_tiny_config(output_dir: Path) -> dict:
    """A small, fast, valid config used by CPU unit tests."""
    cfg = copy.deepcopy(load_config(BASELINE_CONFIG))
    cfg["experiment"]["output_dir"] = str(output_dir)
    cfg["experiment"]["device"] = "cpu"
    cfg["data"]["batch_size"] = 4
    cfg["data"]["num_workers"] = 0
    cfg["model"].update(
        channels=[32, 64],
        attention_resolutions=[16],
        num_res_blocks=1,
        num_groups=8,
        time_embed_dim=64,
    )
    cfg["diffusion"]["timesteps"] = 50
    cfg["training"].update(grad_accum_steps=2, warmup_steps=2, epochs=2, save_every=1, sample_every=100)
    cfg["sampling"].update(batch_size=8, strip_steps=[50, 25, 0])
    cfg["evaluation"].update(num_samples=20, batch_size=8, is_splits=2)
    return cfg


@pytest.fixture
def tiny_config(tmp_path: Path) -> dict:
    return make_tiny_config(tmp_path / "run")


@pytest.fixture
def synthetic_image_batch() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.rand(4, 3, 32, 32, generator=g) * 2 - 1


def _loader(n: int, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, 32, 32, generator=g) * 2 - 1
    labels = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None, drop_last=shuffle)


def make_fake_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """In-memory train/val/test loaders (16/8/8 images) so no test downloads CIFAR-10."""
    return _loader(16, 4, 1, True), _loader(8, 4, 2, False), _loader(8, 4, 3, False)


@pytest.fixture
def fake_cifar_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    return make_fake_loaders()


@pytest.fixture
def fake_loader_factory():
    """Factory returning fresh, identically seeded loaders (for determinism tests)."""
    return make_fake_loaders
