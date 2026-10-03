"""Gaussian diffusion (Ho et al. 2020): schedule buffers, q(x_t|x_0), L_simple and Algorithm 2.

Timesteps are zero-based indices ``t ∈ [0, T-1]`` (index 0 = paper step 1); see research §3.
"""

import time
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseDenoiser, BaseDiffusion, BaseNoiseSchedule
from src.models.registry import register_diffusion
from src.models.types import DiffusionOutput, LossOutput, SamplingOutput

VARIANCE_TYPES = ("fixed_large", "fixed_small")


@torch.no_grad()
def perturb_zero_init_(module: torch.nn.Module, std: float = 1e-3, seed: int = 0) -> None:
    """Fill every all-zero parameter with small Gaussian noise (in place).

    Zero-initialized output convolutions make an untrained network start as an identity mapping, which
    also blocks gradients to upstream layers until those convolutions are trained. Gradient-flow checks
    (unit tests and ``ddpm verify``) therefore run on a *copy* perturbed with this helper.
    """
    g = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        if p.numel() > 0 and not p.any():
            p.copy_(torch.randn(p.shape, generator=g) * std)


@register_diffusion("gaussian")
class GaussianDiffusion(BaseDiffusion):
    """Wraps a denoiser with the fixed forward process and the learned reverse process."""

    def __init__(self, schedule: BaseNoiseSchedule, denoiser: BaseDenoiser, variance_type: str = "fixed_large") -> None:
        super().__init__()
        if variance_type not in VARIANCE_TYPES:
            raise ValueError(f"variance_type must be one of {VARIANCE_TYPES}, got {variance_type!r}")
        self.denoiser = denoiser
        self.variance_type = variance_type
        self.config: dict = {}

        betas = schedule.betas().to(torch.float64)
        self.timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        log_var = torch.log(torch.cat([posterior_variance[1:2], posterior_variance[1:]]))

        buffers = {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "sqrt_alphas_cumprod": alphas_cumprod.sqrt(),
            "sqrt_one_minus_alphas_cumprod": (1.0 - alphas_cumprod).sqrt(),
            "sqrt_recip_alphas": (1.0 / alphas).sqrt(),
            "posterior_variance": posterior_variance,
            "posterior_log_variance_clipped": log_var,
            "posterior_mean_coef1": betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
            "posterior_mean_coef2": (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod),
        }
        for name, value in buffers.items():
            self.register_buffer(name, value.to(torch.float32), persistent=False)

    @staticmethod
    def _extract(buf: Tensor, t: Tensor, shape: torch.Size) -> Tensor:
        """Gather per-sample coefficients and reshape to broadcast over ``[B, C, H, W]``."""
        return buf.gather(0, t).reshape(t.shape[0], *([1] * (len(shape) - 1)))

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε (FR-004)."""
        if noise is None:
            noise = torch.randn_like(x0)
        return (self._extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
                + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def forward(self, x0: Tensor, generator: torch.Generator | None = None) -> DiffusionOutput:
        """Sample t ~ U{0..T−1} and ε ~ N(0, I) per image, noise x_0 and predict ε."""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device, generator=generator)
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        x_t = self.q_sample(x0, t, noise)
        return DiffusionOutput(x_t=x_t, t=t, noise=noise, predicted_noise=self.denoiser(x_t, t))

    def compute_loss(self, output: DiffusionOutput) -> LossOutput:
        """L_simple = ‖ε − ε_θ(x_t, t)‖² averaged over batch and pixels (FR-005)."""
        loss = F.mse_loss(output.predicted_noise.float(), output.noise.float())
        return LossOutput(loss=loss, metrics={"mse": float(loss.detach())})

    @torch.no_grad()
    def p_sample(self, x_t: Tensor, t: int, generator: torch.Generator | None = None) -> Tensor:
        """One step of Algorithm 2 from zero-based index ``t`` (FR-006, FR-007).

        mean = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε_θ(x_t, t)); σ_t² = β_t (fixed_large) or β̃_t
        (fixed_small); fresh noise σ_t·z is added for t > 0 and omitted at t = 0.
        """
        b = x_t.shape[0]
        eps = self.denoiser(x_t, torch.full((b,), t, device=x_t.device, dtype=torch.long))
        mean = self.sqrt_recip_alphas[t] * (x_t - self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t] * eps)
        if t == 0:
            return mean
        var = self.betas[t] if self.variance_type == "fixed_large" else self.posterior_variance[t]
        z = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
        return mean + var.sqrt() * z

    def sample(
        self,
        num_samples: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        trajectory_steps: Sequence[int] | None = None,
        batch_size: int | None = None,
        on_batch: Callable[[int, Tensor], None] | None = None,
    ) -> SamplingOutput:
        raise NotImplementedError("Reverse sampling is implemented in T046")
