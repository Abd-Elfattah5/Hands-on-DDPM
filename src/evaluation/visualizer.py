"""Visual artifacts: sample grids, denoising strips and the nearest-neighbor memorization panel.

`upscale_tensor` and `generate_sample_grid` are ported from `Hands-on VAE/src/evaluation/visualizer.py`;
sampling uses a dedicated seeded generator instead of the global RNG (research §7).
"""

import json
import math
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image, ImageDraw

from src.data.cifar10 import unnormalize
from src.models.base import BaseDiffusion
from src.utils.seeding import make_generator


def upscale_tensor(tensor: torch.Tensor, factor: int = 4) -> torch.Tensor:
    """Bicubic upscaling of ``[N, C, H, W]`` images in [0, 1], clamped back to [0, 1]."""
    if factor <= 1:
        return tensor
    return F.interpolate(tensor, scale_factor=float(factor), mode="bicubic", align_corners=False).clamp(0.0, 1.0)


def _grid_nrow(n: int) -> int:
    root = math.isqrt(n)
    return root if root * root == n else min(8, n)


def save_grid(images01: torch.Tensor, out_path: str | Path, nrow: int, upscale: int = 4) -> Path:
    """Save ``[N, C, H, W]`` images in [0, 1] as a padded grid PNG."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(upscale_tensor(images01.cpu(), upscale), str(out), nrow=nrow, normalize=False,
                      padding=2 * max(upscale, 1))
    return out


def generate_sample_grid(
    model: BaseDiffusion,
    num_samples: int = 64,
    device: torch.device | None = None,
    out_path: str | Path = "artifacts/samples/sample_grid.png",
    seed: int | None = None,
    upscale: int = 4,
    batch_size: int = 256,
) -> Path:
    """Sample ``num_samples`` images with Algorithm 2 and save them as a square grid."""
    if device is None:
        device = next(model.parameters()).device
    generator = make_generator(seed if seed is not None else 0, device)
    was_training = model.training
    model.eval()
    samples = model.sample(num_samples, device, generator=generator, batch_size=batch_size).samples
    model.train(was_training)
    return save_grid(unnormalize(samples), out_path, _grid_nrow(num_samples), upscale)


def load_train_images(data_dir: str | Path = "data") -> torch.Tensor:
    """All 50,000 CIFAR-10 training images as float32 in [0, 1], ``[50000, 3, 32, 32]``, no augmentation."""
    import numpy as np
    import torchvision

    ds = torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=True)
    return torch.from_numpy(np.asarray(ds.data)).permute(0, 3, 1, 2).float() / 255.0


@torch.no_grad()
def find_nearest_neighbors(
    samples01: torch.Tensor,
    train01: torch.Tensor,
    k: int = 3,
    chunk_size: int = 5000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each sample, the ``k`` training images with the smallest pixel-space L2 distance.

    Training images are moved to the samples' device in chunks (≈ 60 MB per 5,000 images), so the full
    search stays well inside the 3 GB budget. Returns ``(indices [N, k], distances [N, k])`` sorted by
    increasing distance.
    """
    device = samples01.device
    flat = samples01.reshape(samples01.shape[0], -1).float()
    best_d = torch.full((flat.shape[0], k), float("inf"), device=device)
    best_i = torch.zeros((flat.shape[0], k), dtype=torch.long, device=device)
    for start in range(0, train01.shape[0], chunk_size):
        chunk = train01[start : start + chunk_size].to(device).reshape(-1, flat.shape[1]).float()
        d2 = torch.cdist(flat, chunk).pow(2)
        cand_d = torch.cat([best_d, d2], dim=1)
        cand_i = torch.cat([best_i, torch.arange(start, start + chunk.shape[0], device=device).expand(flat.shape[0], -1)], dim=1)
        best_d, pos = cand_d.topk(k, dim=1, largest=False)
        best_i = cand_i.gather(1, pos)
    return best_i.cpu(), best_d.clamp_min(0).sqrt().cpu()


def render_nearest_neighbors(
    samples01: torch.Tensor,
    train01: torch.Tensor,
    indices: torch.Tensor,
    distances: torch.Tensor,
    out_path: str | Path,
    upscale: int = 2,
) -> Path:
    """One row per generated image: the sample, a white separator column, then its neighbors.

    Also writes ``<stem>.json`` with the matched training indices and L2 distances per row.
    """
    n, k = indices.shape
    blank = torch.ones_like(samples01[:1].cpu())
    tiles = []
    for r in range(n):
        tiles += [samples01[r].cpu(), blank[0]] + [train01[i] for i in indices[r].tolist()]
    out = save_grid(torch.stack(tiles), out_path, nrow=k + 2, upscale=upscale)
    meta = {"rows": [{"sample": r, "train_indices": indices[r].tolist(),
                      "distances": [round(float(d), 4) for d in distances[r]]} for r in range(n)]}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def render_denoise_strip(
    trajectory: torch.Tensor,
    steps: Sequence[int],
    out_path: str | Path,
    upscale: int = 4,
) -> Path:
    """Render ``[F, N, C, H, W]`` frames as rows = images, columns = timesteps, with a ``t=…`` label band."""
    frames, n = trajectory.shape[0], trajectory.shape[1]
    images = unnormalize(trajectory.permute(1, 0, 2, 3, 4).reshape(frames * n, *trajectory.shape[2:]))
    tmp = Path(out_path).with_suffix(".grid.png")
    save_grid(images, tmp, nrow=frames, upscale=upscale)
    grid = Image.open(tmp).convert("RGB")
    tmp.unlink()

    pad = 2 * max(upscale, 1)
    tile = trajectory.shape[-1] * max(upscale, 1)
    band = max(14, 4 * upscale + 6)
    canvas = Image.new("RGB", (grid.width, grid.height + band), "white")
    canvas.paste(grid, (0, band))
    draw = ImageDraw.Draw(canvas)
    for j, step in enumerate(steps):
        draw.text((pad + j * (tile + pad) + 2, 2), f"t={step}", fill="black")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out
