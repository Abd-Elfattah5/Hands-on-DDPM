"""CUDA-only: every heavy workload stays within the 3 GB (3072 MiB) budget (SC-002).

Each scenario runs in a fresh subprocess so allocator state from one test cannot hide another.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

ROOT = Path(__file__).resolve().parents[2]
BUDGET_MIB = 3072

PRELUDE = """
import json, copy, torch
from src.configs.schema import load_config
from src.models.registry import build_diffusion_from_config
from src.utils.memory import reset_peak, measured_total_mib
dev = torch.device("cuda")
cfg = load_config("configs/cifar10_baseline.yaml")
"""

SCENARIOS = {
    "train_32x4": """
model = build_diffusion_from_config(cfg).to(dev)
ema = copy.deepcopy(model.denoiser).requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
reset_peak(dev)
for _ in range(4):
    x = torch.rand(32, 3, 32, 32, device=dev) * 2 - 1
    (model.compute_loss(model(x)).loss / 4).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step(); opt.zero_grad(set_to_none=True)
with torch.no_grad():
    for e, p in zip(ema.parameters(), model.denoiser.parameters()):
        e.lerp_(p, 1e-4)
""",
    "sample_256": """
model = build_diffusion_from_config(cfg).to(dev).eval()
reset_peak(dev)
with torch.no_grad():
    x = torch.randn(256, 3, 32, 32, device=dev)
    for t in range(999, 994, -1):
        x = model.denoiser(x, torch.full((256,), t, device=dev))
""",
    "inception_64": """
from torchvision.models import inception_v3
import torch.nn.functional as F
net = inception_v3(weights=None, aux_logits=True, init_weights=False).to(dev).eval()
reset_peak(dev)
with torch.no_grad():
    net(F.interpolate(torch.rand(64, 3, 32, 32, device=dev), size=(299, 299), mode="bilinear", align_corners=False))
""",
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_within_budget(name):
    code = PRELUDE + SCENARIOS[name] + "\nprint(json.dumps({'mib': measured_total_mib(dev)}))\n"
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    mib = json.loads(proc.stdout.strip().splitlines()[-1])["mib"]
    print(f"{name}: {mib} MiB")
    assert mib <= BUDGET_MIB, f"{name} used {mib} MiB > {BUDGET_MIB} MiB"
