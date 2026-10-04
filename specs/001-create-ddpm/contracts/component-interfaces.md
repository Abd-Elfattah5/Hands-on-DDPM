# Python Interface Contracts: Component Registries & Base Classes

**Feature**: `001-create-ddpm`
**Status**: Planned

Mirrors `Hands-on VAE/specs/001-create-vae/contracts/component-interfaces.md`. Signatures only;
bodies are implementation work.

---

## 1. Abstract Base Classes (`src/models/base.py`)

```python
class BaseNoiseSchedule(ABC):
    timesteps: int
    @abstractmethod
    def betas(self) -> torch.Tensor: ...            # [T], float64, values in (0, 1)

class BaseDenoiser(nn.Module, ABC):
    @abstractmethod
    def forward(self, x_t: Tensor, t: Tensor) -> Tensor: ...   # [B,C,H,W], [B] int64 -> [B,C,H,W]

class BaseDiffusion(nn.Module, ABC):
    @abstractmethod
    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor: ...
    @abstractmethod
    def forward(self, x0: Tensor, generator: torch.Generator | None = None) -> DiffusionOutput: ...
    @abstractmethod
    def compute_loss(self, output: DiffusionOutput) -> LossOutput: ...
    @abstractmethod
    def p_sample(self, x_t: Tensor, t: int, generator: torch.Generator | None = None) -> Tensor: ...
    @abstractmethod
    def sample(self, num_samples: int, device: torch.device, generator: torch.Generator | None = None,
               trajectory_steps: Sequence[int] | None = None, batch_size: int | None = None) -> SamplingOutput: ...
```

Concrete components:
- `src/models/schedules.py`: `@register_schedule("linear") LinearSchedule`,
  `@register_schedule("cosine") CosineSchedule`
- `src/models/unet.py`: `@register_denoiser("unet") UNet` (built from `embeddings.py`, `resnet.py`,
  `attention.py`)
- `src/models/gaussian_diffusion.py`: `@register_diffusion("gaussian") GaussianDiffusion`
- `src/training/ema.py`: `EMA(model, decay)` with `update(model)`, `copy_to(model)`, `state_dict()`,
  `load_state_dict()`

## 2. Registry Decorators & Factory API (`src/models/registry.py`)

```python
SCHEDULE_REGISTRY: dict[str, type[BaseNoiseSchedule]]
DENOISER_REGISTRY: dict[str, type[BaseDenoiser]]
DIFFUSION_REGISTRY: dict[str, type[BaseDiffusion]]

def register_schedule(name: str) -> Callable[[T], T]: ...   # TypeError if not a BaseNoiseSchedule subclass
def register_denoiser(name: str) -> Callable[[T], T]: ...
def register_diffusion(name: str) -> Callable[[T], T]: ...

def get_schedule(name: str) -> type[BaseNoiseSchedule]: ...  # KeyError lists the available names
def get_denoiser(name: str) -> type[BaseDenoiser]: ...
def get_diffusion(name: str) -> type[BaseDiffusion]: ...

def build_diffusion_from_config(config: dict) -> BaseDiffusion: ...
# validate_config -> schedule(diffusion.*) -> denoiser(model.*) -> diffusion(schedule, denoiser, variance_type)
```

## 3. Training & Evaluation Entry Points

```python
class DDPMTrainer:
    def __init__(self, config: dict, model: BaseDiffusion | None = None, train_loader=None,
                 val_loader=None, device: torch.device | None = None): ...
    def train_epoch(self, epoch: int) -> dict[str, float]: ...
    @torch.no_grad()
    def validate(self, epoch: int, use_ema: bool = True) -> dict[str, float]: ...
    def resume_from_checkpoint(self, path: Path | str) -> int: ...   # returns the next epoch
    def train(self, start_epoch: int = 1) -> dict[str, Any]: ...

def save_checkpoint(path, model, ema, optimizer, scheduler, scaler, epoch, global_step,
                    best_val_loss, history, config, metrics, seed) -> Path: ...
def load_checkpoint(path, map_location="cpu") -> dict: ...
def restore_model_from_checkpoint(path, map_location="cpu", weights: str = "ema") -> tuple[BaseDiffusion, dict]: ...

def evaluate_model(model, data_loader, device, seed: int = 42) -> dict: ...          # evaluator.py
def compute_fid_and_is(model, real_loader, num_samples=5000, batch_size=64, device=None,
                       sample_batch_size=256, seed=42, sample_dir=None) -> dict: ...  # metrics.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float: ...       # ported unchanged
def calculate_inception_score(probs, splits=10) -> tuple[float, float]: ...             # ported unchanged
def generate_sample_grid(...) -> Path: ...
def render_denoise_strip(...) -> Path: ...
def render_nearest_neighbors(samples, train_dataset, k=3, out_path=...) -> Path: ...     # visualizer.py
```
