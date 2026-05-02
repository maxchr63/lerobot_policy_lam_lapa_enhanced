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

    # Single-camera mode: set one camera key (legacy, still supported)
    camera_key: str | None = None
    # Multi-camera mode: list of camera keys (1–3 cameras supported).
    # When len > 1 the bottleneck fusion path is activated automatically.
    # camera_key is ignored when camera_keys is set.
    camera_keys: list[str] | None = None

    # ── Multi-camera bottleneck fusion ──────────────────────────────────────
    # n_bottleneck_tokens: number of MBT bottleneck query tokens.
    #   MBT paper ablation finds 4 optimal; use 8 for more complex scenes.
    n_bottleneck_tokens: int = 4
    # bottleneck_heads: attention heads in the cross-attention fusion layer.
    bottleneck_heads: int = 8
    # view_dropout_prob: probability of masking a non-primary camera view per
    #   training step.  Set to 0.0 to disable.  Primary camera is never dropped.
    view_dropout_prob: float = 0.2
    # max_camera_slots: size of the view-ID embedding table.  Fixed upper bound
    #   independent of the current camera count so that e.g. a 2-camera checkpoint
    #   can be resumed with 3 cameras without a shape mismatch on load.
    max_camera_slots: int = 8
    # camera_key_to_slot: explicit, stable mapping from camera key string to
    #   embedding table slot index (0..max_camera_slots-1).  Once set for a run,
    #   never change it — checkpoint slot assignments must be consistent.
    #   If None when camera_keys is set, slots are auto-assigned in list order.
    #   Example: {"observation.images.top": 0, "observation.images.wrist": 1}
    camera_key_to_slot: dict[str, int] | None = None
    # fusion_mode: controls how cross-camera information is fused into the
    #   primary camera's spatial token grid before the temporal transformer.
    #   In ALL modes, every camera first goes through a shared
    #   motion-saliency weighted pool (per-patch L2 of tokens_t1 - tokens_t,
    #   softmaxed across 64 patches, applied to both frames).  This pre-step
    #   is parameter-free and emphasises moving foreground over static
    #   background — crucial for orthogonal cameras with different scenery.
    #   The mode then chooses how those weighted patches feed cross-camera
    #   fusion.  Primary camera (v=0) is excluded from keys in all modes —
    #   fusion injects extra-camera context into primary, never primary into
    #   itself.
    #
    #   "spatial_64" (DEFAULT — spatially grounded) —
    #       Primary's 64 weighted patches act as queries; extra cameras'
    #       (N-1)*64 weighted patches are keys/values.  Each primary spatial
    #       position gets its own spatially-specific cross-camera enrichment
    #       (a primary patch viewing the gripper attends most strongly to
    #       extra-camera patches that also see the gripper).  Output
    #       [B, 64, D] is reshaped and added to primary spatial tokens via
    #       a LayerScale-style learned gate (init 0) for training stability.
    #       Highest fidelity; recommended default.
    #
    #   "spatial_4" (ablation — intermediate) —
    #       4 learnable bottleneck queries attend to (N-1)*64 extra patches.
    #       Output [B, D] is broadcast-added to all 64 primary positions.
    #       Cheaper than spatial_64; loses per-position specificity.
    #
    #   "pool_4" (ablation — cheapest baseline) —
    #       Each extra camera mean-pools its 64 weighted patches → 1
    #       summary.  4 learnable bottleneck queries attend to N-1 extra
    #       summaries.  Output [B, D] is broadcast-added to primary.
    #       Uniform broadcast over spatial positions.
    #
    #   References: MBT (Nagrani et al., NeurIPS 2021) for bottleneck queries;
    #   MotionBERT (Zhu et al., ICCV 2023) and TokenLearner (Ryoo et al.,
    #   NeurIPS 2021) for the motion-weighted pre-step; LayerScale (Touvron
    #   et al., 2021) and ReZero (Bachlechner et al., 2021) for the learned
    #   residual gate in spatial_64.
    fusion_mode: str = "spatial_64"
    # fusion_keys_include_primary: when True, the primary camera's tokens are
    #   included in the cross-attention KEY sequence alongside extras.  Useful
    #   for pool_4 (otherwise keys=[B, 1, D] for N=2 — degenerate softmax) and
    #   spatial_4 (richer context).  For spatial_64 the primary patches are
    #   already the QUERIES — including them as keys creates a redundant
    #   self-cross-attention on top of the spatial transformer.
    #
    #   Default (None) auto-resolves based on fusion_mode:
    #       pool_4     → True   (avoids 1-key degenerate softmax)
    #       spatial_4  → True   (richer key context across all cameras)
    #       spatial_64 → False  (avoid redundant self-attention; primary is
    #                            already query, residual preserves its stream)
    #
    #   Explicitly set True/False to override.
    fusion_keys_include_primary: bool | None = None

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
        if self.camera_keys is not None:
            n = len(self.camera_keys)
            if n < 1 or n > 3:
                raise ValueError(
                    f"camera_keys must contain 1–3 entries, got {n}. "
                    "Only 1, 2, and 3 cameras are currently supported."
                )
        if self.n_bottleneck_tokens < 1:
            raise ValueError(f"n_bottleneck_tokens must be >= 1, got {self.n_bottleneck_tokens}.")
        if self.fusion_mode not in {"pool_4", "spatial_4", "spatial_64"}:
            raise ValueError(
                f"fusion_mode must be 'pool_4', 'spatial_4', or 'spatial_64', "
                f"got {self.fusion_mode!r}."
            )
        # Resolve fusion_keys_include_primary default based on fusion_mode.
        # spatial_64 → False (primary already queries; including as keys would
        # be redundant self-attention).  pool_4 / spatial_4 → True.
        if self.fusion_keys_include_primary is None:
            self.fusion_keys_include_primary = self.fusion_mode != "spatial_64"
        if not (0.0 <= self.view_dropout_prob < 1.0):
            raise ValueError(
                f"view_dropout_prob must be in [0, 1), got {self.view_dropout_prob}."
            )
        if self.max_camera_slots < 1:
            raise ValueError(f"max_camera_slots must be >= 1, got {self.max_camera_slots}.")
        if self.camera_key_to_slot is not None:
            bad_slots = [
                (k, s) for k, s in self.camera_key_to_slot.items()
                if not (0 <= s < self.max_camera_slots)
            ]
            if bad_slots:
                raise ValueError(
                    f"camera_key_to_slot contains slot indices outside "
                    f"[0, max_camera_slots={self.max_camera_slots}): {bad_slots}."
                )

    # ── Camera key helpers ───────────────────────────────────────────────────

    @property
    def active_camera_keys(self) -> list[str]:
        """Ordered list of camera keys used for training (1–3 entries)."""
        if self.camera_keys:
            return list(self.camera_keys)
        if self.camera_key:
            return [self.camera_key]
        return []

    @property
    def multi_camera_enabled(self) -> bool:
        """True when more than one camera stream is active."""
        return len(self.active_camera_keys) > 1

    @property
    def resolved_slot_map(self) -> dict[str, int]:
        """Stable camera-key → embedding-slot mapping.

        Uses ``camera_key_to_slot`` when explicitly set; otherwise auto-assigns
        slots in the order of ``active_camera_keys``.  Always call this rather
        than reading ``camera_key_to_slot`` directly so the auto-assignment
        logic is applied consistently.
        """
        keys = self.active_camera_keys
        if self.camera_key_to_slot is not None:
            return dict(self.camera_key_to_slot)
        return {k: i for i, k in enumerate(keys)}

    def validate_features(self) -> None:
        if len(self.image_features) == 0:
            raise ValueError("LAM requires at least one visual input feature.")

        if self.camera_keys is not None:
            # Multi-camera mode: validate every key
            missing = [k for k in self.camera_keys if k not in self.image_features]
            if missing:
                raise ValueError(
                    f"camera_keys contains keys not in image_features: {missing}. "
                    f"Available: {tuple(self.image_features.keys())}."
                )
        else:
            # Single-camera mode: legacy behaviour
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
