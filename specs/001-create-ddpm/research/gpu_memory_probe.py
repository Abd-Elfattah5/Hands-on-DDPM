"""Empirical GPU memory probe for the spec'd DDPM U-Net (throwaway, not project code).

Builds the architecture described in spec FR-008..FR-011 / HANDOFF.md section 3:
channels [64,128,256], 2 ResBlocks per stage, attention (4 heads) at 16x16 and 8x8 and in
the bottleneck, none at 32x32, GroupNorm(32), dropout 0.1, sinusoidal time embedding -> 256-d MLP.
Uses the conservative Ho et al. layout (a skip per ResBlock, attention after every ResBlock at
attention resolutions) so measurements are an upper bound for the blueprint.
"""
import copy, json, math, subprocess, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

DEV = torch.device("cuda")
MiB = 1024 ** 2


def gn(c): return nn.GroupNorm(32, c)


class TimeEmb(nn.Module):
    def __init__(self, base=64, dim=256):
        super().__init__(); self.base = base
        self.mlp = nn.Sequential(nn.Linear(base, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, t):
        half = self.base // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], 1))


class Res(nn.Module):
    def __init__(self, cin, cout, tdim, p=0.1):
        super().__init__()
        self.n1, self.c1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.d, self.c2 = gn(cout), nn.Dropout(p), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x))) + self.t(F.silu(te))[:, :, None, None]
        h = self.c2(self.d(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attn(nn.Module):
    """Explicit (non-fused) attention: materializes the full N x N matrix -> worst case memory."""
    def __init__(self, c, heads=4):
        super().__init__(); self.h = heads
        self.n, self.qkv, self.o = gn(c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.h, C // self.h, H * W).unbind(1)
        w = torch.softmax(torch.einsum("bhcn,bhcm->bhnm", q, k) / math.sqrt(C // self.h), -1)
        out = torch.einsum("bhnm,bhcm->bhcn", w, v).reshape(B, C, H, W)
        return x + self.o(out)


class UNet(nn.Module):
    def __init__(self, ch=(64, 128, 256), nres=2, attn_res=(16, 8), tdim=256):
        super().__init__()
        self.temb = TimeEmb(64, tdim)
        self.inc = nn.Conv2d(3, ch[0], 3, padding=1)
        self.down, skips, c, res = nn.ModuleList(), [ch[0]], ch[0], 32
        for i, co in enumerate(ch):
            for _ in range(nres):
                self.down.append(nn.ModuleList([Res(c, co, tdim), Attn(co) if res in attn_res else nn.Identity()]))
                c = co; skips.append(c)
            if i < len(ch) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1)); skips.append(c); res //= 2
        self.mid = nn.ModuleList([Res(c, c, tdim), Attn(c), Res(c, c, tdim)])
        self.up = nn.ModuleList()
        for i, co in reversed(list(enumerate(ch))):
            for _ in range(nres + 1):
                self.up.append(nn.ModuleList([Res(c + skips.pop(), co, tdim), Attn(co) if res in attn_res else nn.Identity()]))
                c = co
            if i > 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1))); res *= 2
        self.out = nn.Sequential(gn(c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))

    def forward(self, x, t):
        te = self.temb(t); h = self.inc(x); hs = [h]
        for m in self.down:
            if isinstance(m, nn.ModuleList): h = m[1](m[0](h, te))
            else: h = m(h)
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, te)), te)
        for m in self.up:
            if isinstance(m, nn.ModuleList): h = m[1](m[0](torch.cat([h, hs.pop()], 1), te))
            else: h = m(h)
        return self.out(h)


def nvsmi_used():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return int(out.decode().strip().splitlines()[0])


def reset():
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def report(name, extra=None):
    torch.cuda.synchronize()
    r = {"scenario": name,
         "peak_allocated_MiB": round(torch.cuda.max_memory_allocated() / MiB),
         "peak_reserved_MiB": round(torch.cuda.max_memory_reserved() / MiB),
         "nvidia_smi_used_MiB": nvsmi_used()}
    if extra: r.update(extra)
    print(json.dumps(r), flush=True); return r


def train_probe(bs, accum, amp, steps=6):
    reset()
    model = UNet().to(DEV); ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    betas = torch.linspace(1e-4, 0.02, 1000, device=DEV); abar = torch.cumprod(1 - betas, 0)
    t0 = time.time()
    for s in range(steps):
        for _ in range(accum):
            x0 = torch.rand(bs, 3, 32, 32, device=DEV) * 2 - 1
            t = torch.randint(0, 1000, (bs,), device=DEV); eps = torch.randn_like(x0)
            a = abar[t][:, None, None, None]
            xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = F.mse_loss(model(xt, t), eps) / accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - 0.9999)
    torch.cuda.synchronize()
    params = sum(p.numel() for p in model.parameters())
    r = report(f"train bs={bs} accum={accum} amp={amp} (+AdamW +EMA on GPU)",
               {"params_M": round(params / 1e6, 2), "sec_per_optim_step": round((time.time() - t0) / steps, 3),
                "loss_finite": bool(torch.isfinite(loss).item())})
    del model, ema, opt; return r


@torch.no_grad()
def sample_probe(bs, amp):
    reset()
    model = UNet().to(DEV).eval()
    x = torch.randn(bs, 3, 32, 32, device=DEV)
    t0 = time.time()
    for step in range(20):  # memory is flat across steps; time extrapolated to 1000 steps
        t = torch.full((bs,), 999 - step, device=DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            eps = model(x, t)
        x = x - 0.01 * eps.float() + 0.01 * torch.randn_like(x)
    torch.cuda.synchronize(); dt = (time.time() - t0) / 20
    r = report(f"sample bs={bs} amp={amp} (EMA model, no grad)",
               {"est_sec_1000_steps_per_batch": round(dt * 1000, 1),
                "est_hours_5000_images": round(dt * 1000 * math.ceil(5000 / bs) / 3600, 2)})
    del model; return r


@torch.no_grad()
def inception_probe(bs):
    from torchvision.models import inception_v3
    reset()
    net = inception_v3(weights=None, aux_logits=True, init_weights=False).to(DEV).eval()  # same arch/size as pretrained
    x = torch.rand(bs, 3, 32, 32, device=DEV)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    net(x)
    r = report(f"FID/IS Inception-v3 feature extraction bs={bs}")
    del net; return r


if __name__ == "__main__":
    torch.manual_seed(0)
    print(json.dumps({"gpu": torch.cuda.get_device_name(0),
                      "total_MiB": torch.cuda.get_device_properties(0).total_memory // MiB,
                      "torch": torch.__version__}), flush=True)
    scenario = sys.argv[1]
    if scenario == "train": train_probe(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1")
    elif scenario == "sample": sample_probe(int(sys.argv[2]), sys.argv[3] == "1")
    elif scenario == "inception": inception_probe(int(sys.argv[2]))
