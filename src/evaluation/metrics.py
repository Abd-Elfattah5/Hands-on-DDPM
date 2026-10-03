"""Quantitative generative evaluation metrics: Inception Score (IS) and Fréchet Inception Distance (FID).

`InceptionFeatureExtractor`, `calculate_frechet_distance` and `calculate_inception_score` are ported
from `Hands-on VAE/src/evaluation/metrics.py` with unchanged maths, so results follow the shared
benchmark protocol (research §9). `compute_fid_and_is` is adapted to batched, resumable DDPM sampling
and also reports a `class_coverage` block for the report's distribution-coverage section (FR-029).
"""

import math
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm

from src.data.cifar10 import unnormalize
from src.models.base import BaseDiffusion
from src.utils.seeding import make_generator


class InceptionFeatureExtractor(nn.Module):
    """Wraps pre-trained Inception-v3 to extract 2048-dim penultimate features and softmax probabilities."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        inception = inception_v3(weights=weights, transform_input=False).to(device)
        inception.eval()
        self.inception = inception
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process images [B, 3, 32, 32] in [0, 1] range.

        Returns:
            Tuple of (features [B, 2048], probs [B, 1000]).
        """
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Resize from 32x32 to 299x299 as expected by Inception-v3
        x_299 = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        # Normalize to Inception's expected [-1, 1] input range
        x_norm = x_299 * 2.0 - 1.0

        inc = self.inception
        x = inc.Conv2d_1a_3x3(x_norm)
        x = inc.Conv2d_2a_3x3(x)
        x = inc.Conv2d_2b_3x3(x)
        x = inc.maxpool1(x)
        x = inc.Conv2d_3b_1x1(x)
        x = inc.Conv2d_4a_3x3(x)
        x = inc.maxpool2(x)
        x = inc.Mixed_5b(x)
        x = inc.Mixed_5c(x)
        x = inc.Mixed_5d(x)
        x = inc.Mixed_6a(x)
        x = inc.Mixed_6b(x)
        x = inc.Mixed_6c(x)
        x = inc.Mixed_6d(x)
        x = inc.Mixed_6e(x)
        x = inc.Mixed_7a(x)
        x = inc.Mixed_7b(x)
        x = inc.Mixed_7c(x)
        x = inc.avgpool(x)
        x = inc.dropout(x)
        features = torch.flatten(x, 1)  # [B, 2048]

        logits = inc.fc(features)
        probs = F.softmax(logits, dim=1)  # [B, 1000]

        return features, probs


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute the Fréchet distance between two multivariate Gaussians N(mu1, sigma1) and N(mu2, sigma2).

    Formula:
        d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * (sigma1 * sigma2)^(1/2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical imaginary error correction
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_inception_score(
    probs: np.ndarray,
    splits: int = 10,
) -> Tuple[float, float]:
    """Calculate Inception Score (IS) from class probability predictions.

    Formula:
        IS = exp( E_x [ KL( p(y|x) || p(y) ) ] )
    """
    num_samples = probs.shape[0]
    split_size = num_samples // splits
    scores: list[float] = []

    for i in range(splits):
        part = probs[i * split_size : (i + 1) * split_size]
        p_y = np.expand_dims(np.mean(part, axis=0), 0)
        kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        kl_div = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl_div))

    return float(np.mean(scores)), float(np.std(scores))


def calculate_class_coverage(probs: np.ndarray, top: int = 20) -> dict[str, Any]:
    """Distribution coverage from Inception probabilities of generated images (Constitution report gate).

    Returns the number of distinct top-1 ImageNet classes, the entropy (nats) of the marginal class
    distribution p(y) and the ``top`` most frequent top-1 classes with their counts.
    """
    top1 = probs.argmax(axis=1)
    classes, counts = np.unique(top1, return_counts=True)
    order = np.argsort(-counts)[:top]
    p_y = probs.mean(axis=0)
    entropy = float(-(p_y * np.log(p_y + 1e-12)).sum())
    return {
        "distinct_top1_classes": int(len(classes)),
        "marginal_entropy_nats": round(entropy, 4),
        "top20": [[int(classes[i]), int(counts[i])] for i in order],
    }


def _to_uint8(images01: torch.Tensor) -> torch.Tensor:
    return (images01.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


@torch.no_grad()
def compute_fid_and_is(
    model: BaseDiffusion,
    real_loader: DataLoader,
    num_samples: int = 5000,
    batch_size: int = 64,
    device: torch.device | None = None,
    sample_batch_size: int = 256,
    seed: int = 42,
    sample_dir: str | Path | None = None,
    reuse_samples: bool = False,
    is_splits: int = 10,
) -> dict[str, Any]:
    """FID (first ``num_samples`` real test images vs ``num_samples`` generated) and IS over ``is_splits``.

    Generated batches are saved as ``sample_dir/batch_{i:04d}.pt`` (uint8). With ``reuse_samples`` the
    existing batches are loaded instead of regenerated, so an interrupted benchmark can resume.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = InceptionFeatureExtractor(device=device)

    def extract(images01: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        feats, probs = [], []
        for i in range(0, images01.shape[0], batch_size):
            f, p = extractor(images01[i : i + batch_size].to(device).float())
            feats.append(f.double().cpu().numpy())
            probs.append(p.double().cpu().numpy())
        return np.concatenate(feats), np.concatenate(probs)

    # 1. real features
    real_list, collected = [], 0
    for batch in tqdm(real_loader, desc="Extracting Real Features", leave=False):
        images = unnormalize(batch[0])
        real_list.append(extract(images)[0])
        collected += images.shape[0]
        if collected >= num_samples:
            break
    real_features = np.concatenate(real_list)[:num_samples]

    # 2. generated samples (batched, written to disk as they finish)
    sample_path = Path(sample_dir) if sample_dir else None
    if sample_path:
        sample_path.mkdir(parents=True, exist_ok=True)
    n_batches = math.ceil(num_samples / sample_batch_size)
    generator = make_generator(seed, device)
    fake_u8: list[torch.Tensor] = []
    sampling_seconds = 0.0
    model.eval()
    for i in tqdm(range(n_batches), desc="Generating Samples", leave=False):
        n = min(sample_batch_size, num_samples - i * sample_batch_size)
        file = sample_path / f"batch_{i:04d}.pt" if sample_path else None
        if reuse_samples and file is not None and file.exists():
            chunk = torch.load(file)
            if chunk.shape[0] == n:
                fake_u8.append(chunk)
                continue
        t0 = time.time()
        out = model.sample(n, device, generator=generator, batch_size=n)
        sampling_seconds += time.time() - t0
        chunk = _to_uint8(unnormalize(out.samples))
        if file is not None:
            torch.save(chunk, file)
        fake_u8.append(chunk)
    fake01 = torch.cat(fake_u8)[:num_samples].float() / 255.0

    # 3. features, FID, IS, coverage
    fake_features, fake_probs = extract(fake01)
    mu_r, sigma_r = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_f, sigma_f = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    fid = calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
    is_mean, is_std = calculate_inception_score(fake_probs, splits=is_splits)
    return {
        "fid": round(fid, 2),
        "inception_score_mean": round(is_mean, 2),
        "inception_score_std": round(is_std, 2),
        "benchmark_samples": int(fake01.shape[0]),
        "sampling_seconds_total": round(sampling_seconds, 2),
        "sampling_seconds_per_image": round(sampling_seconds / num_samples, 4),
        "class_coverage": calculate_class_coverage(fake_probs),
    }
