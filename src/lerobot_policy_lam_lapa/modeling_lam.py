from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot_policy_lam_lapa.configuration_lam import LAMConfig
from lerobot_policy_lam_lapa.core_model import PlainLAMModel

LATENT_FORMAT_IDS = "ids"
LATENT_FORMAT_CONTINUOUS = "continuous"
LATENT_FORMAT_CODEBOOK_VECTORS = "codebook_vectors"
REPRESENTATION_CODEBOOK_IDS = "codebook_id_latents"
REPRESENTATION_CODEBOOK_VECTORS = "codebook_vector_latents"
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
            codebook_size=config.codebook_size,
            image_size=config.image_size,
            spatial_depth=config.spatial_depth,
            temporal_depth=config.temporal_depth,
            dim_head=config.dim_head,
            heads=config.heads,
            channels=config.channels,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.ff_dropout,
            code_seq_len=config.code_seq_len,
            vq_discarding_threshold=config.vq_discarding_threshold,
            vq_discarding_threshold_schedule=config.vq_discarding_threshold_schedule,
            codebook_replace_schedule=config.codebook_replace_schedule,
            latent_ablation=config.latent_ablation,
            metrics_num_unique_codes_every_n_steps=config.metrics_num_unique_codes_every_n_steps,
            dino_model_name=config.dino_model_name,
            dino_freeze=config.dino_freeze,
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

    def _resolve_representation(self, representation: str) -> tuple[str, tuple[int, ...], str, int | float]:
        if representation == REPRESENTATION_CODEBOOK_IDS:
            return LATENT_FORMAT_IDS, (self.config.code_seq_len,), "int64", -100
        if representation == REPRESENTATION_CODEBOOK_VECTORS:
            return (
                LATENT_FORMAT_CODEBOOK_VECTORS,
                (self.config.code_seq_len, self.config.quant_dim),
                "float32",
                0.0,
            )
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
                REPRESENTATION_CODEBOOK_IDS: self._resolve_representation(REPRESENTATION_CODEBOOK_IDS),
                REPRESENTATION_CODEBOOK_VECTORS: self._resolve_representation(REPRESENTATION_CODEBOOK_VECTORS),
                REPRESENTATION_CONTINUOUS_VECTORS: self._resolve_representation(REPRESENTATION_CONTINUOUS_VECTORS),
            }.items()
        }

    def prepare_latent_export(self, dataset_meta: Any) -> dict[str, Any]:
        camera_key = self.config.camera_key or next(iter(self.config.image_features))
        delta_timestamps = {
            camera_key: [delta_idx / dataset_meta.fps for delta_idx in self.config.observation_delta_indices]
        }
        return {
            "delta_timestamps": delta_timestamps,
            "representations": self._representation_specs(),
        }

    @torch.inference_mode()
    def export_latent_labels(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        video, valid_mask, _ = self._extract_frame_pair(batch)
        if not bool(valid_mask.any().item()):
            return {
                "labels_by_name": {
                    name: torch.empty((0, *spec["shape"]), device=video.device)
                    for name, spec in self._representation_specs().items()
                },
                "valid_mask": valid_mask,
            }

        valid_video = video[valid_mask]
        return {
            "labels_by_name": self._extract_all_latents_from_video(valid_video),
            "valid_mask": valid_mask,
        }

    def _extract_all_latents_from_video(self, video: Tensor) -> dict[str, Tensor]:
        model = self.lam
        video = model._normalize_video_input(video)
        first_frame = video[:, :, :1]
        last_frame = video[:, :, 1:]
        _, _, first_tokens_flat, last_tokens_flat = model._encode_frames(first_frame, last_frame)

        batch_size = first_tokens_flat.shape[0]
        first = model.vq.encode(first_tokens_flat.contiguous(), batch_size)
        last = model.vq.encode(last_tokens_flat.contiguous(), batch_size)
        delta = last - first
        continuous = delta.reshape(batch_size, model.code_seq_len, model.vq.embedding_dim)

        distances = (
            torch.sum(delta**2, dim=1, keepdim=True)
            - 2 * torch.matmul(delta, model.vq.codebooks.t())
            + torch.sum(model.vq.codebooks.t() ** 2, dim=0, keepdim=True)
        )
        min_indices = torch.argmin(distances, dim=1)
        codebook_vectors = model.vq.codebooks[min_indices].reshape(
            batch_size, model.code_seq_len, model.vq.embedding_dim
        )

        return {
            REPRESENTATION_CODEBOOK_IDS: min_indices.reshape(batch_size, model.code_seq_len),
            REPRESENTATION_CODEBOOK_VECTORS: codebook_vectors,
            REPRESENTATION_CONTINUOUS_VECTORS: continuous,
        }

    def extract_latents_from_video(self, video: Tensor, *, latent_format: str = LATENT_FORMAT_IDS) -> Tensor:
        latents_by_representation = self._extract_all_latents_from_video(video)
        if latent_format == LATENT_FORMAT_IDS:
            return latents_by_representation[REPRESENTATION_CODEBOOK_IDS]
        if latent_format == LATENT_FORMAT_CONTINUOUS:
            return latents_by_representation[REPRESENTATION_CONTINUOUS_VECTORS]
        if latent_format == LATENT_FORMAT_CODEBOOK_VECTORS:
            return latents_by_representation[REPRESENTATION_CODEBOOK_VECTORS]
        raise ValueError(f"Unsupported latent_format={latent_format!r}.")

    @torch.inference_mode()
    def extract_latents(
        self,
        batch: dict[str, Tensor],
        *,
        latent_format: str = LATENT_FORMAT_IDS,
    ) -> tuple[Tensor, Tensor, str]:
        video, valid_pair, camera_key = self._extract_frame_pair(batch)
        if not bool(valid_pair.any().item()):
            empty_shape = (0, self.config.code_seq_len)
            if latent_format in {LATENT_FORMAT_CONTINUOUS, LATENT_FORMAT_CODEBOOK_VECTORS}:
                empty_shape = (0, self.config.code_seq_len, self.config.quant_dim)
            empty = torch.empty(empty_shape, device=video.device)
            return empty, valid_pair, camera_key
        latents = self.extract_latents_from_video(video[valid_pair], latent_format=latent_format)
        return latents, valid_pair, camera_key

    def _extract_frame_pair(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, str]:
        camera_key = self.config.camera_key or next(iter(self.config.image_features))
        frames = batch[camera_key]
        if not torch.is_tensor(frames):
            frames = torch.as_tensor(frames)

        if frames.ndim != 5:
            raise ValueError(f"Expected batched frame pairs for {camera_key!r}, got shape {tuple(frames.shape)}.")

        if frames.shape[1] == 2 and frames.shape[2] == 3:
            pair = frames
        elif frames.shape[1] == 2 and frames.shape[-1] == 3:
            pair = frames.permute(0, 1, 4, 2, 3)
        elif frames.shape[2] == 2 and frames.shape[1] == 3:
            pair = frames.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(
                f"Unsupported frame pair layout for {camera_key!r}: expected [B,2,C,H,W], [B,2,H,W,C], "
                f"or [B,C,2,H,W], got {tuple(frames.shape)}."
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
                raise ValueError(f"Expected {is_pad_key!r} to have shape [B,2], got {tuple(is_pad.shape)}.")
            valid_pair = (~is_pad[:, 0]) & (~is_pad[:, 1])
        else:
            valid_pair = torch.ones(pair.shape[0], device=pair.device, dtype=torch.bool)

        return pair.permute(0, 2, 1, 3, 4), valid_pair, camera_key

    def _zero_loss(self) -> torch.Tensor:
        return next(self.parameters()).sum() * 0.0

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict[str, Any]]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction={reduction!r}.")

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

    def predict_action_chunk(self, batch: dict[str, Tensor], **_: Any) -> Tensor:
        raise NotImplementedError("LAM is a training-only policy and does not produce environment actions.")

    def select_action(self, batch: dict[str, Tensor], **_: Any) -> Tensor:
        raise NotImplementedError("LAM is a training-only policy and cannot be rolled out in an environment.")
