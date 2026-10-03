"""Abstract base classes for registered diffusion components (contracts/component-interfaces.md §1)."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from src.models.types import DiffusionOutput, LossOutput, SamplingOutput


class BaseNoiseSchedule(ABC):
    """A variance schedule β_1..β_T."""

    timesteps: int

    @abstractmethod
    def betas(self) -> Tensor:
        """Return ``[T]`` float64 per-step variances, each in (0, 1)."""


class BaseDenoiser(nn.Module, ABC):
    """A time-conditioned network predicting the noise added to ``x_t``."""

    @abstractmethod
    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """Map ``[B, C, H, W]`` noisy images and ``[B]`` int64 timesteps to ``[B, C, H, W]``."""


class BaseDiffusion(nn.Module, ABC):
    """A diffusion process wrapping a denoiser: forward noising, loss and reverse sampling."""

    @abstractmethod
    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """Closed-form forward marginal q(x_t | x_0)."""

    @abstractmethod
    def forward(self, x0: Tensor, generator: torch.Generator | None = None) -> DiffusionOutput:
        """Training forward pass on clean images."""

    @abstractmethod
    def compute_loss(self, output: DiffusionOutput) -> LossOutput:
        """Training objective for a forward-pass output."""

    @abstractmethod
    def p_sample(self, x_t: Tensor, t: int, generator: torch.Generator | None = None) -> Tensor:
        """One reverse step from zero-based index ``t`` to ``t-1``."""

    @abstractmethod
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        trajectory_steps: Sequence[int] | None = None,
        batch_size: int | None = None,
        on_batch: Callable[[int, Tensor], None] | None = None,
    ) -> SamplingOutput:
        """Generate images by running the full reverse chain."""
