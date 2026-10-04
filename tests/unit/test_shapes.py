"""Shape contracts for embeddings, residual blocks, attention and the full U-Net."""

import copy

import pytest
import torch

from src.configs.schema import ConfigError, validate_config
from src.models.attention import SpatialSelfAttention, reference_attention
from src.models.embeddings import SinusoidalTimestepEmbedding, TimestepMLP
from src.models.resnet import Downsample, ResBlock, Upsample
from src.models.unet import UNet


def test_sinusoidal_embedding_shape_and_uniqueness():
    emb = SinusoidalTimestepEmbedding(64)(torch.tensor([0, 1, 500, 999]))
    assert emb.shape == (4, 64)
    assert torch.unique(emb, dim=0).shape[0] == 4


def test_timestep_mlp_shape():
    out = TimestepMLP(64, 256)(torch.randn(3, 64))
    assert out.shape == (3, 256)


def test_resblock_shapes():
    block = ResBlock(64, 128, time_dim=256)
    out = block(torch.randn(2, 64, 16, 16), torch.randn(2, 256))
    assert out.shape == (2, 128, 16, 16)


def test_down_and_upsample():
    x = torch.randn(1, 64, 16, 16)
    assert Downsample(64)(x).shape == (1, 64, 8, 8)
    assert Upsample(64)(x).shape == (1, 64, 32, 32)


def test_attention_preserves_shape_and_matches_reference():
    torch.manual_seed(0)
    attn = SpatialSelfAttention(128, num_heads=4)
    x = torch.randn(2, 128, 8, 8)
    assert attn(x).shape == x.shape
    q, k, v = torch.randn(3, 2, 4, 64, 32).unbind(0)
    fused = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(fused, reference_attention(q, k, v), atol=1e-5)


def test_unet_baseline_io_and_param_count(baseline_config):
    m = dict(baseline_config["model"])
    m.pop("name")
    net = UNet(**m)
    out = net(torch.randn(2, 3, 32, 32), torch.tensor([0, 999]))
    assert out.shape == (2, 3, 32, 32)
    assert abs(net.num_parameters() - 16.06e6) < 0.05e6


def test_attention_at_32_rejected(baseline_config):
    cfg = copy.deepcopy(baseline_config)
    cfg["model"]["attention_resolutions"] = [32, 16]
    with pytest.raises(ConfigError):
        validate_config(cfg)
