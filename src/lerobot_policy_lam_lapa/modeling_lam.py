from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

logger = logging.getLogger(__name__)

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot_policy_lam_lapa.configuration_lam import LAMConfig
from lerobot_policy_lam_lapa.core_model import PlainLAMModel

LATENT_FORMAT_CONTINUOUS = "continuous"
REPRESENTATION_CONTINUOUS_VECTORS = "continuous_vector_latents"


def _separate_weight_decayable_params(
    params: list[torch.nn.Parameter],
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    wd_params = []
    no_wd_params = []
    for param in params:
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            wd_params.append(param)
        else:
            no_wd_params.append(param)
    return wd_params, no_wd_params


class LAMPolicy(PreTrainedPolicy):
    config_class = LAMConfig
    name = "LAMDino"

    def __init__(self, config: LAMConfig, **_: Any):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.lam = PlainLAMModel(
            dim=config.dim,
            quant_dim=config.quant_dim,
            image_size=config.image_size,
            spatial_depth=config.spatial_depth,
            temporal_depth=config.temporal_depth,
            dim_head=config.dim_head,
            heads=config.heads,
            channels=config.channels,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.ff_dropout,
            code_seq_len=config.code_seq_len,
            latent_ablation=config.latent_ablation,
            dino_model_name=config.dino_model_name,
            dino_freeze=config.dino_freeze,
            n_cameras=len(config.active_camera_keys) or 1,
            n_bottleneck_tokens=config.n_bottleneck_tokens,
            bottleneck_heads=config.bottleneck_heads,
            view_dropout_prob=config.view_dropout_prob,
            max_camera_slots=config.max_camera_slots,
            camera_slot_ids=[
                config.resolved_slot_map[k] for k in config.active_camera_keys
            ] if config.active_camera_keys else None,
            fusion_mode=config.fusion_mode,
            fusion_keys_include_primary=config.fusion_keys_include_primary,
        )
        self._train_step = 0
        self.reset()

    def get_optim_params(self) -> list[dict[str, Any]]:
        params = [param for param in self.parameters() if param.requires_grad]
        wd_params, no_wd_params = _separate_weight_decayable_params(params)
        param_groups: list[dict[str, Any]] = []
        if wd_params:
            param_groups.append({"params": wd_params})
        if no_wd_params:
            param_groups.append({"params": no_wd_params, "weight_decay": 0.0})
        return param_groups

    def reset(self) -> None:
        pass

    def update(self) -> None:
        self._train_step += 1

    @classmethod
    def load_from_single_camera_checkpoint(
        cls,
        checkpoint_path: str,
        new_config: "LAMConfig",
        *,
        freeze_shared_encoder_steps: int = 0,
    ) -> "LAMPolicy":
        """Load a single-camera Stage-1 checkpoint and migrate it for multi-camera training.

        Single-camera checkpoints lack ``view_id_embedding`` and ``bottleneck_fusion``
        parameters.  This method loads them with ``strict=False``, logs which
        parameters are newly initialised (they will NOT be in the checkpoint), and
        optionally freezes the shared encoder for the first N steps so the newly
        added fusion module can initialise stably before joint optimisation begins.

        Usage
        -----
        ::

            new_config = LAMConfig(
                camera_keys=["observation.images.top", "observation.images.wrist"],
                camera_key_to_slot={"observation.images.top": 0, "observation.images.wrist": 1},
                n_bottleneck_tokens=4,
                view_dropout_prob=0.2,
            )
            policy = LAMPolicy.load_from_single_camera_checkpoint(
                "/path/to/single_cam_checkpoint/model.safetensors",
                new_config,
                freeze_shared_encoder_steps=2000,
            )

        After migration the ``view_id_embedding`` and ``bottleneck_fusion`` weights
        start from their random initialisations (N(0, 0.02) for bottleneck tokens,
        zeros for the mask token — per MBT paper §3).  All other weights are copied
        from the checkpoint.

        Data-config note
        ----------------
        The LeRobot data pipeline must be configured to load all camera streams
        listed in ``new_config.camera_keys``.  In your dataset source YAML add each
        secondary camera key under ``camera_role_to_key`` alongside the existing
        primary.  Example diff::

            camera_role_to_key:
              top: observation.images.top        # existing primary
        +   wrist: observation.images.wrist      # new secondary camera

        Args:
            checkpoint_path: Path to a ``model.safetensors`` or ``pytorch_model.bin``
                file from a single-camera ``LAMPolicy`` checkpoint.
            new_config: ``LAMConfig`` instance with ``camera_keys`` (or
                ``camera_key_to_slot``) configured for multi-camera training.
            freeze_shared_encoder_steps: If > 0, freezes the DINOv3 backbone,
                downsampler, and spatial/temporal transformers for this many steps
                after loading so the fusion module can warm up first.  Set to 0 to
                train everything jointly from the start.

        Returns:
            ``LAMPolicy`` with migrated weights and optional encoder freeze.
        """
        policy = cls(new_config)

        # Load checkpoint with strict=False — new multi-camera params are absent
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Handle both raw state-dicts and HuggingFace-style {"model": {...}} dicts
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]

        missing, unexpected = policy.load_state_dict(state, strict=False)

        expected_new = {"lam.view_id_embedding", "lam.bottleneck_fusion"}
        truly_missing = [k for k in missing if not any(k.startswith(p) for p in expected_new)]
        new_params    = [k for k in missing if any(k.startswith(p) for p in expected_new)]

        if new_params:
            logger.info(
                "Checkpoint migration: %d new multi-camera parameters initialised "
                "from scratch (expected): %s",
                len(new_params), new_params,
            )
        if truly_missing:
            logger.warning(
                "Checkpoint migration: %d parameters missing from checkpoint "
                "(unexpected — verify checkpoint compatibility): %s",
                len(truly_missing), truly_missing,
            )
        if unexpected:
            logger.warning(
                "Checkpoint migration: %d parameters in checkpoint not in new "
                "model (ignored): %s", len(unexpected), unexpected,
            )

        if freeze_shared_encoder_steps > 0 and new_config.multi_camera_enabled:
            # Freeze shared encoder modules so the newly added bottleneck_fusion
            # can stabilise before full joint training.
            modules_to_freeze = [
                policy.lam.encoder_tokenizer,
                policy.lam.enc_spatial_transformer,
                policy.lam.enc_temporal_transformer,
            ]
            for mod in modules_to_freeze:
                for param in mod.parameters():
                    param.requires_grad_(False)
            logger.info(
                "Encoder frozen for first %d steps (freeze_shared_encoder_steps). "
                "Call policy.unfreeze_encoder() when ready.",
                freeze_shared_encoder_steps,
            )
            policy._freeze_encoder_until_step = freeze_shared_encoder_steps

        return policy

    def unfreeze_encoder(self) -> None:
        """Unfreeze the shared encoder after the warm-up period.

        Call this manually (or hook it to a training callback) after
        ``freeze_shared_encoder_steps`` have elapsed.
        """
        for mod in [
            self.lam.encoder_tokenizer,
            self.lam.enc_spatial_transformer,
            self.lam.enc_temporal_transformer,
        ]:
            for param in mod.parameters():
                param.requires_grad_(True)
        logger.info("Shared encoder unfrozen — full joint training active.")

    def _resolve_representation(self, representation: str) -> tuple[str, tuple[int, ...], str, int | float]:
        if representation == REPRESENTATION_CONTINUOUS_VECTORS:
            return (
                LATENT_FORMAT_CONTINUOUS,
                (self.config.code_seq_len, self.config.quant_dim),
                "float32",
                0.0,
            )
        raise ValueError(f"Unsupported representation={representation!r}.")

    def _representation_specs(self) -> dict[str, dict[str, Any]]:
        return {
            representation: {
                "shape": shape,
                "dtype": dtype,
                "invalid_fill_value": invalid_fill_value,
            }
            for representation, (_, shape, dtype, invalid_fill_value) in {
                REPRESENTATION_CONTINUOUS_VECTORS: self._resolve_representation(REPRESENTATION_CONTINUOUS_VECTORS),
            }.items()
        }

    def prepare_latent_export(self, dataset_meta: Any) -> dict[str, Any]:
        active_keys = self.config.active_camera_keys
        if not active_keys:
            active_keys = [next(iter(self.config.image_features))]
        delta_ts = [delta_idx / dataset_meta.fps for delta_idx in self.config.observation_delta_indices]
        delta_timestamps = {k: delta_ts for k in active_keys}
        return {
            "delta_timestamps": delta_timestamps,
            "representations": self._representation_specs(),
        }

    @torch.inference_mode()
    def export_latent_labels(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Export latent labels, using multi-camera fusion when configured."""
        camera_keys = self.config.active_camera_keys
        if not camera_keys:
            camera_keys = [next(iter(self.config.image_features))]

        video, valid_mask, _ = self._extract_frame_pair(batch)  # primary camera
        if not bool(valid_mask.any().item()):
            return {
                "labels_by_name": {
                    name: torch.empty((0, *spec["shape"]), device=video.device)
                    for name, spec in self._representation_specs().items()
                },
                "valid_mask": valid_mask,
            }

        if len(camera_keys) > 1:
            # Multi-camera: run fusion to get the fused latent
            all_videos = [video]
            for ck in camera_keys[1:]:
                vid_ck, _ = self._extract_frame_pair_for_key(batch, ck)
                all_videos.append(vid_ck)
            valid_videos = [v[valid_mask] for v in all_videos]
            labels = self._extract_all_latents_from_video_multi(valid_videos)
        else:
            valid_video = video[valid_mask]
            labels = self._extract_all_latents_from_video(valid_video)

        return {"labels_by_name": labels, "valid_mask": valid_mask}

    def _extract_all_latents_from_video(self, video: Tensor) -> dict[str, Tensor]:
        model = self.lam
        video = model._normalize_video_input(video)
        first_frame = video[:, :, :1]
        last_frame = video[:, :, 1:]
        _, _, first_tokens_flat, last_tokens_flat = model._encode_frames(first_frame, last_frame)

        batch_size = first_tokens_flat.shape[0]
        first = model.bottleneck.encode(first_tokens_flat.contiguous(), batch_size)
        last = model.bottleneck.encode(last_tokens_flat.contiguous(), batch_size)
        delta = last - first
        continuous = delta.reshape(batch_size, model.code_seq_len, model.bottleneck.embedding_dim)

        return {REPRESENTATION_CONTINUOUS_VECTORS: continuous}

    def _extract_all_latents_from_video_multi(self, videos: list[Tensor]) -> dict[str, Tensor]:
        """Multi-camera variant of _extract_all_latents_from_video.

        Runs the MBT bottleneck fusion encoder to produce the fused latents
        that downstream Stage-2/Stage-3 will consume.  Output shapes are
        identical to the single-camera version — only N=1 produces the same
        result as the single-camera path.
        """
        model = self.lam
        primary = model._normalize_video_input(videos[0])
        first_frame = primary[:, :, :1]
        last_frame  = primary[:, :, 1:]

        extra_pairs = []
        for v in videos[1:]:
            v = model._normalize_video_input(v)
            extra_pairs.append((v[:, :, :1], v[:, :, 1:]))

        all_pairs = [(first_frame, last_frame)] + extra_pairs
        _, _, first_tokens_flat, last_tokens_flat, _ = model._encode_frames_multi(all_pairs)

        batch_size = first_tokens_flat.shape[0]
        first = model.bottleneck.encode(first_tokens_flat.contiguous(), batch_size)
        last  = model.bottleneck.encode(last_tokens_flat.contiguous(), batch_size)
        delta = last - first
        continuous = delta.reshape(batch_size, model.code_seq_len, model.bottleneck.embedding_dim)

        return {REPRESENTATION_CONTINUOUS_VECTORS: continuous}

    def extract_latents_from_video(self, video: Tensor, *, latent_format: str = LATENT_FORMAT_CONTINUOUS) -> Tensor:
        if latent_format != LATENT_FORMAT_CONTINUOUS:
            raise ValueError(f"Unsupported latent_format={latent_format!r}.")
        return self._extract_all_latents_from_video(video)[REPRESENTATION_CONTINUOUS_VECTORS]

    @torch.inference_mode()
    def extract_latents(
        self,
        batch: dict[str, Tensor],
        *,
        latent_format: str = LATENT_FORMAT_CONTINUOUS,
    ) -> tuple[Tensor, Tensor, str]:
        video, valid_pair, camera_key = self._extract_frame_pair(batch)
        if not bool(valid_pair.any().item()):
            empty = torch.empty(
                (0, self.config.code_seq_len, self.config.quant_dim), device=video.device
            )
            return empty, valid_pair, camera_key
        latents = self.extract_latents_from_video(video[valid_pair], latent_format=latent_format)
        return latents, valid_pair, camera_key

    def _extract_frame_pair(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, str]:
        """Extract and preprocess a frame pair for the primary camera key.

        Delegates to ``_extract_frame_pair_for_key`` which contains the full
        preprocessing logic (layout normalisation, float conversion, resize,
        is_pad masking).  For multi-camera training the caller invokes
        ``_extract_frame_pair_for_key`` directly for each camera key.
        """
        active = self.config.active_camera_keys
        camera_key = active[0] if active else next(iter(self.config.image_features))
        video, valid_pair = self._extract_frame_pair_for_key(batch, camera_key)
        return video, valid_pair, camera_key

    def _extract_frame_pair_for_key(self, batch: dict[str, Tensor], camera_key: str) -> tuple[Tensor, Tensor]:
        """Extract and preprocess a frame pair for the given camera key.

        Returns:
            (video [B, C, 2, H, W], valid_pair [B] bool)
        """
        frames = batch[camera_key]
        if not torch.is_tensor(frames):
            frames = torch.as_tensor(frames)

        if frames.ndim != 5:
            raise ValueError(
                f"Expected batched frame pairs for {camera_key!r}, got shape {tuple(frames.shape)}."
            )

        if frames.shape[1] == 2 and frames.shape[2] == 3:
            pair = frames
        elif frames.shape[1] == 2 and frames.shape[-1] == 3:
            pair = frames.permute(0, 1, 4, 2, 3)
        elif frames.shape[2] == 2 and frames.shape[1] == 3:
            pair = frames.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(
                f"Unsupported frame pair layout for {camera_key!r}: expected [B,2,C,H,W], "
                f"[B,2,H,W,C], or [B,C,2,H,W], got {tuple(frames.shape)}."
            )

        pair = pair.to(device=self.config.device)
        if pair.dtype == torch.uint8:
            pair = pair.to(torch.float32) / 255.0
        else:
            pair = pair.to(torch.float32)
            if pair.max().item() > 1.5:
                pair = pair / 255.0

        if tuple(pair.shape[-2:]) != tuple(self.config.image_size):
            b, t, c, h, w = pair.shape
            pair = F.interpolate(
                pair.reshape(b * t, c, h, w),
                size=self.config.image_size,
                mode="bilinear",
                align_corners=False,
            ).reshape(b, t, c, self.config.image_size[0], self.config.image_size[1])

        is_pad_key = f"{camera_key}_is_pad"
        if is_pad_key in batch:
            is_pad = batch[is_pad_key]
            if not torch.is_tensor(is_pad):
                is_pad = torch.as_tensor(is_pad)
            is_pad = is_pad.to(device=pair.device, dtype=torch.bool)
            if is_pad.ndim != 2 or is_pad.shape[1] != 2:
                raise ValueError(
                    f"Expected {is_pad_key!r} to have shape [B,2], got {tuple(is_pad.shape)}."
                )
            valid_pair = (~is_pad[:, 0]) & (~is_pad[:, 1])
        else:
            valid_pair = torch.ones(pair.shape[0], device=pair.device, dtype=torch.bool)

        # Return [B, C, 2, H, W]
        return pair.permute(0, 2, 1, 3, 4), valid_pair

    def _zero_loss(self) -> torch.Tensor:
        return next(self.parameters()).sum() * 0.0

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict[str, Any]]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction={reduction!r}.")

        camera_keys = self.config.active_camera_keys
        if not camera_keys:
            camera_keys = [next(iter(self.config.image_features))]

        # ── Single-camera path (original behaviour) ──────────────────────────
        if len(camera_keys) == 1:
            video, valid_pair, camera_key = self._extract_frame_pair(batch)
            batch_size = video.shape[0]
            if not bool(valid_pair.any().item()):
                zero = self._zero_loss()
                output_dict = {"loss": 0.0, "valid_pairs": 0, "camera_key": camera_key}
                if reduction == "none":
                    return torch.zeros(batch_size, device=video.device, dtype=zero.dtype), output_dict
                return zero, output_dict

            valid_video = video[valid_pair]
            loss_or_losses, metrics = self.lam(valid_video, step=self._train_step + 1, reduction=reduction)
            output_dict = dict(metrics)
            output_dict["valid_pairs"] = int(valid_pair.sum().item())
            output_dict["camera_key"] = camera_key

            if reduction == "none":
                per_sample = torch.zeros(batch_size, device=video.device, dtype=loss_or_losses.dtype)
                per_sample[valid_pair] = loss_or_losses
                output_dict["loss"] = float(loss_or_losses.mean().detach().item())
                return per_sample, output_dict

            output_dict["loss"] = float(loss_or_losses.detach().item())
            return loss_or_losses, output_dict

        # ── Multi-camera path ─────────────────────────────────────────────────
        # Extract frame pairs for every camera; a sample is valid only if the
        # primary camera (index 0) has a valid pair.
        all_videos: list[Tensor] = []
        all_valid: list[Tensor] = []
        for ck in camera_keys:
            vid_ck, valid_ck = self._extract_frame_pair_for_key(batch, ck)
            all_videos.append(vid_ck)
            all_valid.append(valid_ck)

        primary_valid = all_valid[0]  # primary camera determines sample validity
        batch_size = all_videos[0].shape[0]

        if not bool(primary_valid.any().item()):
            zero = self._zero_loss()
            output_dict = {
                "loss": 0.0,
                "valid_pairs": 0,
                "camera_keys": camera_keys,
            }
            if reduction == "none":
                return torch.zeros(batch_size, device=all_videos[0].device, dtype=zero.dtype), output_dict
            return zero, output_dict

        # Keep only samples where the primary camera is valid
        valid_primary_videos = [v[primary_valid] for v in all_videos]
        primary_video = valid_primary_videos[0]
        extra_videos  = valid_primary_videos[1:]

        loss_or_losses, metrics = self.lam(
            primary_video,
            extra_videos=extra_videos,
            step=self._train_step + 1,
            reduction=reduction,
        )
        output_dict = dict(metrics)
        output_dict["valid_pairs"] = int(primary_valid.sum().item())
        output_dict["camera_keys"] = camera_keys

        if reduction == "none":
            per_sample = torch.zeros(batch_size, device=primary_video.device, dtype=loss_or_losses.dtype)
            per_sample[primary_valid] = loss_or_losses
            output_dict["loss"] = float(loss_or_losses.mean().detach().item())
            return per_sample, output_dict

        output_dict["loss"] = float(loss_or_losses.detach().item())
        return loss_or_losses, output_dict

    def predict_action_chunk(self, batch: dict[str, Tensor], **_: Any) -> Tensor:
        raise NotImplementedError("LAM is a training-only policy and does not produce environment actions.")

    def select_action(self, batch: dict[str, Tensor], **_: Any) -> Tensor:
        raise NotImplementedError("LAM is a training-only policy and cannot be rolled out in an environment.")
