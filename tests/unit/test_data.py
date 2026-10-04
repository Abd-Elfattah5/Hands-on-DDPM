"""CIFAR-10 pipeline with a fake dataset: split sizes, VAE split parity, transforms and bounds."""

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, random_split

from src.data import cifar10


class FakeCIFAR10(Dataset):
    def __init__(self, root, train=True, download=True, transform=None):
        rng = np.random.default_rng(0 if train else 1)
        self.data = rng.integers(0, 256, size=(100 if train else 20, 32, 32, 3), dtype=np.uint8)
        self.targets = list(rng.integers(0, 10, size=len(self.data)))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.fromarray(self.data[i])
        return (self.transform(img) if self.transform else img), int(self.targets[i])


@pytest.fixture(autouse=True)
def fake_cifar(monkeypatch):
    monkeypatch.setattr(cifar10.torchvision.datasets, "CIFAR10", FakeCIFAR10)


def test_split_sizes_and_disjoint(tmp_path):
    train, val, test = cifar10.get_cifar10_datasets(tmp_path, val_split=0.1, seed=42, random_flip=True)
    assert len(train) == 90 and len(val) == 10 and len(test) == 20
    assert set(train.indices).isdisjoint(val.indices)


def test_split_is_seeded(tmp_path):
    a, _, _ = cifar10.get_cifar10_datasets(tmp_path, val_split=0.1, seed=42, random_flip=False)
    b, _, _ = cifar10.get_cifar10_datasets(tmp_path, val_split=0.1, seed=42, random_flip=False)
    c, _, _ = cifar10.get_cifar10_datasets(tmp_path, val_split=0.1, seed=7, random_flip=False)
    assert list(a.indices) == list(b.indices)
    assert list(a.indices) != list(c.indices)


def test_split_matches_vae_random_split(tmp_path):
    """ADR 0001 D7: the partition must equal Hands-on VAE's random_split for the same seed."""
    train, val, _ = cifar10.get_cifar10_datasets(tmp_path, val_split=0.1, seed=42, random_flip=True)
    ref_train, ref_val = random_split(range(100), [90, 10], generator=torch.Generator().manual_seed(42))
    assert list(train.indices) == list(ref_train.indices)
    assert list(val.indices) == list(ref_val.indices)


def test_no_flip_on_val_and_test(tmp_path):
    train, val, test = cifar10.get_cifar10_datasets(tmp_path, val_split=0.1, seed=42, random_flip=True)
    def has_flip(ds):
        return any(isinstance(t, T.RandomHorizontalFlip) for t in ds.dataset.transform.transforms)
    assert has_flip(train)
    assert not has_flip(val)
    assert not any(isinstance(t, T.RandomHorizontalFlip) for t in test.transform.transforms)


def test_batches_in_range(tmp_path):
    loaders = cifar10.get_cifar10_dataloaders(tmp_path, batch_size=8, val_split=0.1, num_workers=0, seed=42)
    for loader in loaders:
        images, _ = next(iter(loader))
        assert images.min() >= -1.0 and images.max() <= 1.0


def test_unnormalize():
    out = cifar10.unnormalize(torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.allclose(out, torch.tensor([0.0, 0.5, 1.0]))
