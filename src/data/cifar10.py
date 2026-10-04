"""CIFAR-10 data pipeline, transforms and dataloaders.

Ported from `Hands-on VAE/src/data/cifar10.py`. Difference (ADR 0001 D7): training and validation
use two dataset objects (flip vs. no flip) split by the same seeded permutation that
`torch.utils.data.random_split` draws, so the partition is identical to the VAE's while the training
augmentation never reaches validation.
"""

from pathlib import Path
from typing import Any

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, default_collate


def get_cifar10_transforms() -> transforms.Compose:
    """Return image normalization transform mapping [0, 255] pixels to [-1, 1]."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def get_cifar10_train_transforms(random_flip: bool = True) -> transforms.Compose:
    """Training transform: optional random horizontal flip, then the [-1, 1] normalization."""
    ops: list[Any] = [transforms.RandomHorizontalFlip()] if random_flip else []
    return transforms.Compose([*ops, *get_cifar10_transforms().transforms])


def unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Inverse transform converting [-1, 1] normalized tensors back to [0, 1] image range."""
    return (tensor * 0.5 + 0.5).clamp(0.0, 1.0)


def ddpm_collate_fn(batch: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate batch and verify image values reside within [-1, 1]."""
    collated = default_collate(batch)
    images, targets = collated[0], collated[1]
    if images.min() < -1.01 or images.max() > 1.01:
        raise ValueError(
            f"Image batch pixel values outside [-1, 1] bounds: "
            f"min={images.min().item():.4f}, max={images.max().item():.4f}"
        )
    return images, targets


def split_indices(total: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    """Seeded train/val index split identical to ``random_split(range(total), [train, val], seed)``."""
    val_len = int(total * val_split)
    perm = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[: total - val_len], perm[total - val_len :]


def _cifar(root: Path, train: bool, transform: transforms.Compose, download: bool) -> Dataset:
    try:
        return torchvision.datasets.CIFAR10(root=str(root), train=train, download=download, transform=transform)
    except Exception as exc:  # network errors surface as several exception types
        raise RuntimeError(
            f"CIFAR-10 download/load failed under '{root}' ({exc}). Are you offline? "
            "Place 'cifar-10-batches-py' in the data directory or retry with network access."
        ) from exc


def get_cifar10_datasets(
    data_dir: str | Path = "data",
    val_split: float = 0.1,
    seed: int = 42,
    random_flip: bool = True,
    download: bool = True,
) -> tuple[Subset, Subset, Dataset]:
    """Load CIFAR-10 and return (train, val, test) with a deterministic train/val split."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_full = _cifar(data_path, True, get_cifar10_train_transforms(random_flip), download)
    val_full = _cifar(data_path, True, get_cifar10_transforms(), download)
    test_dataset = _cifar(data_path, False, get_cifar10_transforms(), download)

    train_idx, val_idx = split_indices(len(train_full), val_split, seed)
    return Subset(train_full, train_idx), Subset(val_full, val_idx), test_dataset


def get_cifar10_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    random_flip: bool = True,
    download: bool = True,
    pin_memory: bool | None = None,
    eval_batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for the CIFAR-10 train, validation and test splits."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    eval_batch_size = eval_batch_size or batch_size

    train_ds, val_ds, test_ds = get_cifar10_datasets(data_dir, val_split, seed, random_flip, download)
    common = dict(num_workers=num_workers, pin_memory=pin_memory, collate_fn=ddpm_collate_fn,
                  persistent_workers=num_workers > 0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(seed), **common)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader


def get_dataloaders_from_config(
    config: dict[str, Any],
    batch_size: int | None = None,
    data_dir: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build loaders from an experiment config (optionally overriding batch size / data dir)."""
    data = config.get("data", {})
    return get_cifar10_dataloaders(
        data_dir=data_dir or data.get("data_dir", "data"),
        batch_size=batch_size or data.get("batch_size", 32),
        val_split=data.get("val_split", 0.1),
        num_workers=data.get("num_workers", 4),
        seed=config.get("experiment", {}).get("seed", 42),
        random_flip=data.get("random_flip", True),
    )
