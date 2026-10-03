"""Registered noise schedules β_1..β_T (research §2), computed in float64."""

import math

import torch

from src.models.base import BaseNoiseSchedule
from src.models.registry import register_schedule


@register_schedule("linear")
class LinearSchedule(BaseNoiseSchedule):
    """Ho et al. (2020) linear schedule: β_t evenly spaced from ``beta_start`` to ``beta_end``."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, **_: object) -> None:
        self.timesteps = int(timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

    def betas(self) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps, dtype=torch.float64)


@register_schedule("cosine")
class CosineSchedule(BaseNoiseSchedule):
    """Nichol & Dhariwal (2021) cosine schedule with offset ``s`` and β_t clipped at 0.999.

    ᾱ_t = f(t) / f(0), f(t) = cos²(((t/T) + s) / (1 + s) · π/2), β_t = min(1 − ᾱ_t/ᾱ_{t−1}, 0.999).
    """

    max_beta = 0.999

    def __init__(self, timesteps: int = 1000, cosine_s: float = 0.008, **_: object) -> None:
        self.timesteps = int(timesteps)
        self.s = float(cosine_s)

    def betas(self) -> torch.Tensor:
        steps = torch.arange(self.timesteps + 1, dtype=torch.float64)
        f = torch.cos(((steps / self.timesteps) + self.s) / (1 + self.s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return betas.clamp(min=1e-12, max=self.max_beta)
