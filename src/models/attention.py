"""Multi-head spatial self-attention built from primitives (research §5).

The softmax(QKᵀ/√d)V product uses PyTorch's fused ``scaled_dot_product_attention`` kernel
(measured ~170 MiB less memory and ~6% faster); ``reference_attention`` is the explicit formula used
by the unit tests to document and check the maths.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def reference_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Explicit attention: softmax(Q Kᵀ / √d) V for ``[..., N, d]`` tensors."""
    weights = torch.softmax(torch.einsum("...nd,...md->...nm", q, k) / math.sqrt(q.shape[-1]), dim=-1)
    return torch.einsum("...nm,...md->...nd", weights, v)


class SpatialSelfAttention(nn.Module):
    """GroupNorm → 1×1 conv to Q/K/V → multi-head attention → 1×1 output conv → residual add."""

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 32) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = (t.transpose(-1, -2) for t in qkv.unbind(1))  # [B, heads, HW, d]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)
