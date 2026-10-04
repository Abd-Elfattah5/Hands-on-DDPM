"""Sinusoidal timestep embedding and its projection MLP (research §4, deep dive §5.8)."""

import math

import torch
from torch import Tensor, nn


class SinusoidalTimestepEmbedding(nn.Module):
    """emb(t)_{2i} = sin(t·ω_i), emb(t)_{2i+1} = cos(t·ω_i), ω_i = 10000^(−i/half)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dim must be even, got {dim}")
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class TimestepMLP(nn.Module):
    """Linear → SiLU → Linear projection of the sinusoidal embedding."""

    def __init__(self, in_dim: int = 64, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, emb: Tensor) -> Tensor:
        return self.net(emb)
