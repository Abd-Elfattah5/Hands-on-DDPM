"""Time-conditioned U-Net denoiser ε_θ(x_t, t) (research §4, deep dive §5).

Layout matches the GPU memory probe (`specs/001-create-ddpm/research/gpu_memory_probe.py`): a skip
connection is stored after the input conv, after every down-path residual block and after every
downsample; the up path uses ``num_res_blocks + 1`` residual blocks per level, one per skip.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from src.models.attention import SpatialSelfAttention
from src.models.base import BaseDenoiser
from src.models.embeddings import SinusoidalTimestepEmbedding, TimestepMLP
from src.models.registry import register_denoiser
from src.models.resnet import Downsample, ResBlock, Upsample


class _Level(nn.Module):
    """A residual block optionally followed by self-attention."""

    def __init__(self, res: ResBlock, attn: nn.Module | None) -> None:
        super().__init__()
        self.res = res
        self.attn = attn if attn is not None else nn.Identity()

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        return self.attn(self.res(x, time_emb))


@register_denoiser("unet")
class UNet(BaseDenoiser):
    """Encoder–decoder with skip connections, timestep injection and attention at low resolutions."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        image_size: int = 32,
        channels: Sequence[int] = (64, 128, 256),
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (16, 8),
        num_heads: int = 4,
        num_groups: int = 32,
        dropout: float = 0.1,
        time_embed_dim: int = 256,
        **_: object,
    ) -> None:
        super().__init__()
        channels = list(channels)
        attn_res = set(attention_resolutions)

        def block(cin: int, cout: int, res: int) -> _Level:
            attn = SpatialSelfAttention(cout, num_heads, num_groups) if res in attn_res else None
            return _Level(ResBlock(cin, cout, time_embed_dim, dropout, num_groups), attn)

        self.time_embed = SinusoidalTimestepEmbedding(channels[0])
        self.time_mlp = TimestepMLP(channels[0], time_embed_dim)
        self.input_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.down = nn.ModuleList()
        skip_channels = [channels[0]]
        ch, res = channels[0], image_size
        for i, out_ch in enumerate(channels):
            for _ in range(num_res_blocks):
                self.down.append(block(ch, out_ch, res))
                ch = out_ch
                skip_channels.append(ch)
            if i < len(channels) - 1:
                self.down.append(Downsample(ch))
                skip_channels.append(ch)
                res //= 2

        self.mid_res1 = ResBlock(ch, ch, time_embed_dim, dropout, num_groups)
        self.mid_attn = SpatialSelfAttention(ch, num_heads, num_groups)
        self.mid_res2 = ResBlock(ch, ch, time_embed_dim, dropout, num_groups)

        self.up = nn.ModuleList()
        for i, out_ch in reversed(list(enumerate(channels))):
            for _ in range(num_res_blocks + 1):
                self.up.append(block(ch + skip_channels.pop(), out_ch, res))
                ch = out_ch
            if i > 0:
                self.up.append(Upsample(ch))
                res *= 2

        self.out_norm = nn.GroupNorm(num_groups, ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        time_emb = self.time_mlp(self.time_embed(t))
        h = self.input_conv(x_t)
        skips = [h]
        for layer in self.down:
            h = layer(h, time_emb) if isinstance(layer, _Level) else layer(h)
            skips.append(h)
        h = self.mid_res2(self.mid_attn(self.mid_res1(h, time_emb)), time_emb)
        for layer in self.up:
            if isinstance(layer, _Level):
                h = layer(torch.cat([h, skips.pop()], dim=1), time_emb)
            else:
                h = layer(h)
        return self.out_conv(self.out_act(self.out_norm(h)))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
