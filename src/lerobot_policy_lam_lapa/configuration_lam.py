from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

DEFAULT_DINO_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"


@PreTrainedConfig.register_subclass("lam_lapa")
@dataclass
class LAMConfig(PreTrainedConfig):
    """Plain LAPA-style latent action model policy.

    This is a training-only policy. It consumes frame pairs `(t, t + future_frames)`
    from one camera stream and ignores dataset actions during the loss computation.
    """

    n_obs_steps: int = 1
    future_frames: int = 10
    future_seconds: float | None = None
    camera_key: str | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
        }
    )

    dim: int = 1024
    quant_dim: int = 32
    codebook_size: int = 8
    code_seq_len: int = 4
    image_size: tuple[int, int] = (256, 256)
    spatial_depth: int = 8
    temporal_depth: int = 8
    dim_head: int = 64
    heads: int = 16
    channels: int = 3
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    latent_ablation: str = "none"

    dino_model_name: str = DEFAULT_DINO_MODEL_NAME
    dino_freeze: bool = True

    vq_discarding_threshold: float = 0.02
    vq_discarding_threshold_schedule: list[tuple[float, int]] = field(
        default_factory=lambda: [
            (0.05, 1_000),
            (0.01, 5_000),
            (0.005, 50_000),
            (0.0002, 500_000),
        ]
    )
    codebook_replace_schedule: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (10, 100),
            (100, 1_000),
            (500, 5_000),
            (1_000, 10_000),
            (5_000, 100_000),
            (10_000, 500_000),
        ]
    )
    metrics_num_unique_codes_every_n_steps: int = 50

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 200_000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.future_frames < 1:
            raise ValueError(f"future_frames must be >= 1, got {self.future_frames}.")
        if self.future_seconds is not None and self.future_seconds <= 0:
            raise ValueError(
                f"future_seconds must be > 0 when set, got {self.future_seconds}."
            )
        if self.metrics_num_unique_codes_every_n_steps < 1:
            raise ValueError("metrics_num_unique_codes_every_n_steps must be >= 1.")
        if tuple(self.image_size) != (256, 256):
            raise ValueError(
                f"lam_lapa currently supports image_size=(256, 256) only, got {self.image_size}."
            )

    def validate_features(self) -> None:
        if len(self.image_features) == 0:
            raise ValueError("LAM requires at least one visual input feature.")

        if self.camera_key is None:
            self.camera_key = next(iter(self.image_features))
        elif self.camera_key not in self.image_features:
            raise ValueError(
                f"camera_key={self.camera_key!r} is not in the available image features "
                f"{tuple(self.image_features.keys())}."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0, self.future_frames]

    def get_observation_delta_indices_for_fps(self, fps: float | int) -> list[int]:
        if self.future_seconds is None:
            return [0, self.future_frames]
        return [0, max(1, round(float(fps) * float(self.future_seconds)))]

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
