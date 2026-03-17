from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from lerobot_policy_lam.core_attention import ContinuousPositionBias, Transformer
from lerobot_policy_lam.core_nsvq import NSVQ


def pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, value) if not isinstance(value, tuple) else value


@dataclass
class CodebookStats:
    current_threshold: float
    codebook_replaced: float
    replaced_count: float
    used_count: float
    min_count: float


class PlainLAMModel(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        quant_dim: int,
        codebook_size: int,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        spatial_depth: int,
        temporal_depth: int,
        dim_head: int,
        heads: int,
        channels: int,
        attn_dropout: float,
        ff_dropout: float,
        code_seq_len: int,
        vq_discarding_threshold: float,
        vq_discarding_threshold_schedule: list[tuple[float, int]] | None,
        codebook_replace_schedule: list[tuple[int, int]] | None,
        latent_ablation: str,
        metrics_num_unique_codes_every_n_steps: int,
    ):
        super().__init__()
        if latent_ablation not in {"none", "permute_batch"}:
            raise ValueError(f"Unsupported latent_ablation={latent_ablation!r}.")

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        self.code_seq_len = code_seq_len
        self.latent_ablation = latent_ablation
        self.vq_discarding_threshold = float(vq_discarding_threshold)
        self.vq_discarding_threshold_schedule = list(vq_discarding_threshold_schedule or [])
        self.codebook_replace_schedule = list(codebook_replace_schedule or [])
        self.metrics_num_unique_codes_every_n_steps = int(metrics_num_unique_codes_every_n_steps)

        image_height, image_width = self.image_size
        patch_height, patch_width = self.patch_size
        if image_height % patch_height != 0 or image_width % patch_width != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by patch_size={self.patch_size}."
            )

        self.grid_h = image_height // patch_height
        self.grid_w = image_width // patch_width
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        self.pixel_projection = nn.Sequential(
            Rearrange("b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(channels * patch_height * patch_width),
            nn.Linear(channels * patch_height * patch_width, dim),
            nn.LayerNorm(dim),
        )
        self.enc_spatial_transformer = Transformer(
            depth=spatial_depth,
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
        )
        self.enc_temporal_transformer = Transformer(
            depth=temporal_depth,
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
        )
        self.vq = NSVQ(
            dim=dim,
            num_embeddings=codebook_size,
            embedding_dim=quant_dim,
            discarding_threshold=vq_discarding_threshold,
            code_seq_len=code_seq_len,
            patch_size=self.patch_size,
            image_size=self.image_size,
            grid_size=(self.grid_h, self.grid_w),
        )
        self.pixel_decoder = Transformer(
            depth=spatial_depth,
            dim=dim,
            dim_context=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
            has_cross_attn=True,
        )
        self.pixel_to_pixels = nn.Sequential(
            nn.Linear(dim, channels * patch_height * patch_width),
            Rearrange("b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)", p1=patch_height, p2=patch_width),
        )

    @property
    def action_shape(self) -> tuple[int, int]:
        if self.code_seq_len == 2:
            return (2, 1)
        side = int(self.code_seq_len**0.5)
        if side * side != self.code_seq_len:
            raise ValueError("code_seq_len must be a square number or 2.")
        return (side, side)

    def _normalize_video_input(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 4:
            video = rearrange(video, "b c h w -> b c 1 h w")
        if video.ndim != 5:
            raise ValueError(f"Expected 5D video input [B,C,T,H,W], got {tuple(video.shape)}.")
        if video.shape[2] != 2:
            raise ValueError(f"Expected exactly 2 frames, got T={int(video.shape[2])}.")
        if tuple(video.shape[-2:]) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}, got {tuple(video.shape[-2:])}.")
        return video

    def _encode_frames(self, first_frame: torch.Tensor, last_frame: torch.Tensor) -> tuple[torch.Tensor, ...]:
        first_tokens = self.pixel_projection(first_frame)
        last_tokens = self.pixel_projection(last_frame)
        tokens = torch.cat((first_tokens, last_tokens), dim=1)
        b = tokens.shape[0]
        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        attn_bias = self.spatial_rel_pos_bias(self.grid_h, self.grid_w, device=tokens.device)
        tokens = self.enc_spatial_transformer(tokens, video_shape=video_shape, attn_bias=attn_bias)
        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=b, h=self.grid_h, w=self.grid_w)

        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")
        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)
        tokens = rearrange(tokens, "(b h w) t d -> b t h w d", b=b, h=self.grid_h, w=self.grid_w)

        first_tokens = tokens[:, :1]
        last_tokens = tokens[:, 1:]
        first_tokens_flat = rearrange(first_tokens, "b t h w d -> b (t h w) d")
        last_tokens_flat = rearrange(last_tokens, "b t h w d -> b (t h w) d")
        return first_tokens, last_tokens, first_tokens_flat, last_tokens_flat

    def _get_vq_discarding_threshold(self, step: int) -> float:
        threshold = self.vq_discarding_threshold
        for scheduled_threshold, until_step in self.vq_discarding_threshold_schedule:
            if step <= int(until_step):
                return float(scheduled_threshold)
            threshold = float(scheduled_threshold)
        return threshold

    def _should_replace_codebook(self, step: int) -> bool:
        for interval, until_step in self.codebook_replace_schedule:
            if step <= int(until_step):
                return step % int(interval) == 0
        return False

    def _update_codebook_stats(self, step: int) -> CodebookStats:
        current_threshold = self._get_vq_discarding_threshold(step)
        unused_indices, used_indices, min_count = self.vq._get_replacement_indices_from_counts(
            self.vq.codebooks_used, discarding_threshold=current_threshold
        )
        codebook_replaced = 0.0
        replaced_count = float(unused_indices.shape[0])
        used_count = float(used_indices.shape[0])
        if step != 0 and self._should_replace_codebook(step):
            codebook_replaced = 1.0
            replaced_count, used_count, _, min_count = self.vq.replace_unused_codebooks(
                discarding_threshold=current_threshold
            )
        return CodebookStats(
            current_threshold=current_threshold,
            codebook_replaced=codebook_replaced,
            replaced_count=float(replaced_count),
            used_count=float(used_count),
            min_count=float(min_count),
        )

    def _build_metrics(self, *, per_sample_loss: torch.Tensor, perplexity: torch.Tensor, indices: torch.Tensor, step: int, codebook: CodebookStats) -> dict[str, float]:
        metrics = {
            "loss": float(per_sample_loss.mean().detach().item()),
            "pixel_loss": float(per_sample_loss.mean().detach().item()),
            "perplexity": float(perplexity.detach().item()),
            "codebook_replaced": float(codebook.codebook_replaced),
            "codebook_unused_count": float(codebook.replaced_count),
            "codebook_used_count": float(codebook.used_count),
            "vq_discarding_threshold": float(codebook.current_threshold),
            "vq_min_count": float(codebook.min_count),
        }
        if step == 0 or step % self.metrics_num_unique_codes_every_n_steps == 0:
            metrics["num_unique_codes"] = float(indices.unique().numel())
        return metrics

    def _prepare_action_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        action_h, action_w = self.action_shape
        action_tokens = rearrange(tokens, "b (t h w) d -> b t h w d", h=action_h, w=action_w)
        if self.latent_ablation == "permute_batch" and action_tokens.shape[0] > 1:
            perm = torch.randperm(action_tokens.shape[0], device=action_tokens.device)
            action_tokens = action_tokens[perm]
        return action_tokens

    def forward(
        self,
        video: torch.Tensor,
        *,
        step: int = 0,
        reduction: str = "mean",
        return_only_codebook_ids: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]] | torch.Tensor:
        video = self._normalize_video_input(video)
        first_frame = video[:, :, :1]
        last_frame = video[:, :, 1:]
        first_tokens, _, first_tokens_flat, last_tokens_flat = self._encode_frames(first_frame, last_frame)

        if return_only_codebook_ids:
            return self.vq.get_indices(first_tokens_flat, last_tokens_flat)

        quantized_tokens, perplexity, _, indices = self.vq(first_tokens_flat, last_tokens_flat)
        codebook = self._update_codebook_stats(int(step))
        action_tokens = self._prepare_action_tokens(quantized_tokens)

        attn_bias = self.spatial_rel_pos_bias(self.grid_h, self.grid_w, device=video.device)
        video_shape = tuple(first_tokens.shape[:-1])
        pixel_context = rearrange(first_tokens, "b t h w d -> (b t) (h w) d")
        action_context = rearrange(action_tokens, "b t h w d -> (b t) (h w) d")
        decoded = self.pixel_decoder(pixel_context, video_shape=video_shape, attn_bias=attn_bias, context=action_context)
        decoded = rearrange(decoded, "(b t) (h w) d -> b t h w d", b=video.shape[0], h=self.grid_h, w=self.grid_w)
        recon = self.pixel_to_pixels(decoded)

        per_sample_loss = F.mse_loss(recon, last_frame, reduction="none").mean(dim=(1, 2, 3, 4))
        metrics = self._build_metrics(
            per_sample_loss=per_sample_loss,
            perplexity=perplexity,
            indices=indices,
            step=int(step),
            codebook=codebook,
        )
        if reduction == "none":
            return per_sample_loss, metrics
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction={reduction!r}.")
        return per_sample_loss.mean(), metrics
