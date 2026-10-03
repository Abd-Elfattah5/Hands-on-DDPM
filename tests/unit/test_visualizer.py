"""Visual artifacts: sample grid, denoising strip, upscaling and the nearest-neighbor panel."""

import json

import torch
from PIL import Image

from src.evaluation import visualizer as vz
from src.models.registry import build_diffusion_from_config


def test_upscale_range():
    out = vz.upscale_tensor(torch.rand(2, 3, 8, 8), factor=2)
    assert out.shape == (2, 3, 16, 16)
    assert out.min() >= 0 and out.max() <= 1


def test_sample_grid_size(tiny_config, tmp_path):
    model = build_diffusion_from_config(tiny_config)
    path = vz.generate_sample_grid(model, num_samples=16, device=torch.device("cpu"), out_path=tmp_path / "g.png",
                                   seed=0, upscale=2)
    w, h = Image.open(path).size
    tile, pad = 32 * 2, 2 * 2
    assert w == h == 4 * (tile + pad) + pad


def test_denoise_strip_layout(tmp_path):
    frames = torch.rand(8, 3, 3, 32, 32) * 2 - 1
    steps = [1000, 800, 600, 400, 200, 100, 50, 0]
    path = vz.render_denoise_strip(frames, steps, tmp_path / "s.png", upscale=1)
    w, h = Image.open(path).size
    assert w == 8 * (32 + 2) + 2
    assert h > 3 * (32 + 2)  # three rows plus a label band


def test_nearest_neighbors_find_copy(tmp_path):
    g = torch.Generator().manual_seed(0)
    train = torch.rand(50, 3, 32, 32, generator=g)
    samples = torch.rand(4, 3, 32, 32, generator=g)
    samples[2] = train[7]
    idx, dist = vz.find_nearest_neighbors(samples, train, k=3, chunk_size=16)
    assert idx.shape == (4, 3) and dist.shape == (4, 3)
    assert idx[2, 0].item() == 7 and dist[2, 0].item() < 1e-4
    assert (dist[:, 1:] >= dist[:, :-1]).all()

    path = vz.render_nearest_neighbors(samples, train, idx, dist, tmp_path / "nn.png", upscale=1)
    w, h = Image.open(path).size
    assert h == 4 * (32 + 2) + 2
    assert w > 4 * (32 + 2)  # 1 sample + 3 neighbors (+ separator)
    meta = json.loads(path.with_suffix(".json").read_text())
    assert len(meta["rows"]) == 4 and meta["rows"][2]["train_indices"][0] == 7
