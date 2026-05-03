from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from lerobot_policy_lam_lapa.core_attention import ContinuousPositionBias, Transformer
from lerobot_policy_lam_lapa.core_bottleneck import ContinuousLatentBottleneck
from lerobot_policy_lam_lapa.core_bottleneck_fusion import (
    BottleneckCameraFusion,
    SpatialCrossCamera,
)
from lerobot_policy_lam_lapa.core_dino import DinoTokenEncoder


def pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, value) if not isinstance(value, tuple) else value


class PlainLAMModel(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        quant_dim: int,
        image_size: int | tuple[int, int],
        spatial_depth: int,
        temporal_depth: int,
        dim_head: int,
        heads: int,
        channels: int,
        attn_dropout: float,
        ff_dropout: float,
        code_seq_len: int,
        latent_ablation: str,
        dino_model_name: str,
        dino_freeze: bool,
        # ── Multi-camera fusion (optional) ──────────────────────────────────
        # Set n_cameras > 1 to activate the MBT bottleneck fusion path.
        # n_cameras == 1 is fully identical to the original single-camera model.
        n_cameras: int = 1,
        n_bottleneck_tokens: int = 4,
        bottleneck_heads: int = 8,
        view_dropout_prob: float = 0.2,
        # max_camera_slots: embedding table size (fixed upper bound, see LAMConfig).
        max_camera_slots: int = 8,
        # camera_slot_ids: ordered list of embedding slot indices for cameras
        # [0..n_cameras-1].  len must equal n_cameras when n_cameras > 1.
        camera_slot_ids: list[int] | None = None,
        # fusion_mode: which cross-camera fusion architecture.
        # "spatial_64" (default), "spatial_4", or "pool_4".  See LAMConfig docs.
        fusion_mode: str = "spatial_64",
        # fusion_keys_include_primary: include primary in cross-attn keys.
        # True (default) → all cameras participate as keys, primary preserved
        # via residual.  False → strict extras-only ablation.
        fusion_keys_include_primary: bool = True,
    ):
        super().__init__()
        if latent_ablation not in {"none", "permute_batch"}:
            raise ValueError(f"Unsupported latent_ablation={latent_ablation!r}.")
        if n_cameras < 1 or n_cameras > 3:
            raise ValueError(f"n_cameras must be 1, 2, or 3; got {n_cameras}.")

        self.image_size = pair(image_size)
        self.code_seq_len = code_seq_len
        self.latent_ablation = latent_ablation
        self.n_cameras = n_cameras
        self.view_dropout_prob = float(view_dropout_prob)
        # Slot IDs: e.g. [0, 1] for 2-cam or [0, 2] if camera 1 was mapped to slot 2
        if n_cameras > 1:
            if camera_slot_ids is None:
                camera_slot_ids = list(range(n_cameras))
            if len(camera_slot_ids) != n_cameras:
                raise ValueError(
                    f"camera_slot_ids length {len(camera_slot_ids)} != n_cameras {n_cameras}."
                )
        self.camera_slot_ids: list[int] = camera_slot_ids or list(range(n_cameras))
        if fusion_mode not in {"pool_4", "spatial_4", "spatial_64"}:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode!r}.")
        self.fusion_mode = fusion_mode
        self.fusion_keys_include_primary = bool(fusion_keys_include_primary)

        image_height, image_width = self.image_size
        self.encoder_tokenizer = DinoTokenEncoder(
            model_name=dino_model_name,
            image_size=self.image_size,
            output_dim=dim,
            freeze=dino_freeze,
        )
        self.grid_h, self.grid_w = self.encoder_tokenizer.output_grid_size
        self.patch_size = (image_height // self.grid_h, image_width // self.grid_w)
        patch_height, patch_width = self.patch_size
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        self.decoder_context_projection = self._build_pixel_projection(
            dim=dim,
            channels=channels,
            patch_height=patch_height,
            patch_width=patch_width,
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
        self.bottleneck = ContinuousLatentBottleneck(
            dim=dim,
            embedding_dim=quant_dim,
            code_seq_len=code_seq_len,
            grid_size=(self.grid_h, self.grid_w),
        )
        # ── Multi-camera modules (only built when n_cameras > 1) ────────────
        # ``bottleneck_fusion`` is built for pool_4 / spatial_4 (learnable
        # bottleneck queries → broadcast residual).  ``spatial_cross`` is built
        # for spatial_64 (primary patches as queries → spatially-grounded
        # residual with LayerScale-style learned gate).  Only the one matching
        # ``fusion_mode`` is built so the parameter count and checkpoint state
        # dict stay clean.
        if n_cameras > 1:
            self.view_id_embedding = nn.Embedding(max_camera_slots, dim)
            nn.init.normal_(self.view_id_embedding.weight, std=0.02)
            if fusion_mode in ("pool_4", "spatial_4"):
                self.bottleneck_fusion = BottleneckCameraFusion(
                    model_dim=dim,
                    n_bottleneck_tokens=n_bottleneck_tokens,
                    num_heads=bottleneck_heads,
                )
                self.spatial_cross = None
            else:  # spatial_64
                self.bottleneck_fusion = None
                self.spatial_cross = SpatialCrossCamera(
                    model_dim=dim,
                    num_heads=bottleneck_heads,
                )
        else:
            self.view_id_embedding = None
            self.bottleneck_fusion = None
            self.spatial_cross = None

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

    @staticmethod
    def _build_pixel_projection(
        *,
        dim: int,
        channels: int,
        patch_height: int,
        patch_width: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            Rearrange("b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(channels * patch_height * patch_width),
            nn.Linear(channels * patch_height * patch_width, dim),
            nn.LayerNorm(dim),
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
        first_tokens = self.encoder_tokenizer(first_frame)
        last_tokens = self.encoder_tokenizer(last_frame)
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

    @staticmethod
    def _motion_weighted_patches(
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-patch motion-saliency weighted patches for both frames.

        Shared pre-step for all three fusion modes.  Computes per-patch L2 of
        ``tokens_t1 - tokens_t``, softmaxes the resulting [B, 64] motion scores
        across the 64 patches, and applies the same weights to both frames so
        moving foreground patches dominate while static background contributes
        negligibly.  Output is *weighted*, not pooled — pooling is the caller's
        choice (mean for pool_4, no further reduction for spatial_4/spatial_64).

        Args:
            tokens: ``[B, 2, grid_h, grid_w, D]`` post-spatial-transformer
                tokens for one camera.

        Returns:
            weighted_t:  ``[B, 64, D]`` frame-t patches scaled by motion weights.
            weighted_t1: ``[B, 64, D]`` frame-t+1 patches scaled by same weights.
        """
        B = tokens.shape[0]
        D = tokens.shape[-1]
        tokens_t_flat  = tokens[:, 0].reshape(B, -1, D)   # [B, 64, D]
        tokens_t1_flat = tokens[:, 1].reshape(B, -1, D)
        diff = tokens_t1_flat - tokens_t_flat
        magnitude = diff.norm(dim=-1)                      # [B, 64]
        weights = magnitude.softmax(dim=1).unsqueeze(-1)   # [B, 64, 1]
        return tokens_t_flat * weights, tokens_t1_flat * weights

    def _encode_frames_multi(
        self,
        frame_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        present_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Multi-camera encoder with motion-weighted cross-camera fusion.

        Pipeline
        --------
        1. Each camera independently runs through shared DINOv3 + spatial
           transformer with a view-ID embedding stamp.
        2. ``_motion_weighted_patches`` produces ``[B, 64, D]`` motion-weighted
           patches per camera per frame.
        3. ``fusion_mode`` chooses how extra-camera weighted patches enrich the
           primary camera's spatial grid.  Primary (v=0) is excluded from keys
           in all modes — fusion injects extra context into primary, never
           primary into itself.

           * ``pool_4``    — extras mean-pooled to [B, N-1, D], 4 bottleneck
                             queries → [B, D] → broadcast residual add.
           * ``spatial_4`` — extras as [B, (N-1)*64, D], 4 bottleneck queries
                             → [B, D] → broadcast residual add.
           * ``spatial_64``— primary's 64 patches as queries, extras as
                             [B, (N-1)*64, D] keys, → [B, 64, D] enriched
                             primary patches, gated residual add (LayerScale).

        4. View dropout: extra-camera positions are masked via attention
           ``key_padding_mask`` so absent cameras contribute nothing rather
           than being replaced by a learnable token.  Primary is never dropped.

        5. Temporal transformer runs on the (possibly enriched) primary grid.

        Single-camera fallback: when N==1, fusion is skipped entirely and
        primary tokens flow straight into the temporal transformer.

        Args:
            frame_pairs: list of N ``(first_frame, last_frame)`` tuples, each
                ``[B, C, 1, H, W]``.  Index 0 is the primary camera.
            present_mask: ``[B, N]`` bool — True = camera present for sample b.
                Used to build the attention key-padding mask for extras.
                Primary slot (index 0) is always treated as True.
                If None, all cameras are present.

        Returns:
            first_tokens, last_tokens, first_tokens_flat, last_tokens_flat
                — same shapes as ``_encode_frames``.
            per_cam_attn: ``[B, 2, N]`` attention weight per camera, split per
                frame (index 0 = frame t, index 1 = frame t+1).  Logged as
                ``fusion_attn_cam{v}_frame_t`` and ``fusion_attn_cam{v}_frame_t1``
                so temporal variation in attention is visible (e.g. wrist
                camera highly attended near goal frame but not start frame).
                When primary is excluded from keys, slot 0 is hard-zero.
        """
        N = len(frame_pairs)
        B = frame_pairs[0][0].shape[0]
        device = frame_pairs[0][0].device

        # T=2 guard: all cameras must supply exactly 2 frames
        for v, (ff, lf) in enumerate(frame_pairs):
            if ff.shape[2] != 1 or lf.shape[2] != 1:
                raise ValueError(
                    f"Camera {v}: expected single-frame tensors [B,C,1,H,W], "
                    f"got first={tuple(ff.shape)}, last={tuple(lf.shape)}."
                )

        attn_bias = self.spatial_rel_pos_bias(self.grid_h, self.grid_w, device=device)

        # ── 1. Per-camera spatial encoding + motion-weighted patches ─────────
        primary_spatial_tokens: torch.Tensor | None = None
        # patches_t[v]/patches_t1[v]: [B, 64, dim] motion-weighted patches
        patches_t:  list[torch.Tensor] = []
        patches_t1: list[torch.Tensor] = []

        for v, (first_frame, last_frame) in enumerate(frame_pairs):
            first_toks = self.encoder_tokenizer(first_frame)  # [B, 1, grid_h, grid_w, dim]
            last_toks  = self.encoder_tokenizer(last_frame)
            tokens = torch.cat((first_toks, last_toks), dim=1)  # [B, 2, grid_h, grid_w, dim]

            slot = torch.tensor(self.camera_slot_ids[v], device=device, dtype=torch.long)
            tokens = tokens + self.view_id_embedding(slot)  # broadcast over [B, 2, H, W]

            video_shape = tuple(tokens.shape[:-1])
            tokens_2d = rearrange(tokens, "b t h w d -> (b t) (h w) d")
            tokens_2d = self.enc_spatial_transformer(
                tokens_2d, video_shape=video_shape, attn_bias=attn_bias
            )
            tokens = rearrange(
                tokens_2d, "(b t) (h w) d -> b t h w d",
                b=B, t=2, h=self.grid_h, w=self.grid_w,
            )

            if v == 0:
                primary_spatial_tokens = tokens  # kept un-weighted for residual base

            wt, wt1 = self._motion_weighted_patches(tokens)  # [B, 64, D] each
            patches_t.append(wt)
            patches_t1.append(wt1)

        # ── 2. Single-camera fallback ────────────────────────────────────────
        # When only one camera is present there is no extra-camera information
        # to fuse — pass primary tokens straight to the temporal transformer.
        # This branch is also defensive: the policy normally routes N==1
        # through ``_encode_frames`` directly.
        if N == 1:
            tokens = primary_spatial_tokens
            video_shape = tuple(tokens.shape[:-1])
            tokens_t = rearrange(tokens, "b t h w d -> (b h w) t d")
            tokens_t = self.enc_temporal_transformer(tokens_t, video_shape=video_shape)
            tokens = rearrange(
                tokens_t, "(b h w) t d -> b t h w d",
                b=B, h=self.grid_h, w=self.grid_w,
            )
            first_tokens = tokens[:, :1]
            last_tokens  = tokens[:, 1:]
            first_tokens_flat = rearrange(first_tokens, "b t h w d -> b (t h w) d")
            last_tokens_flat  = rearrange(last_tokens,  "b t h w d -> b (t h w) d")
            per_cam_attn = torch.zeros(B, 2, N, device=device)  # [B, 2 frames, N cams]
            return first_tokens, last_tokens, first_tokens_flat, last_tokens_flat, per_cam_attn

        # ── 3. Build cross-attention key sequence + key_padding_mask ─────────
        # When fusion_keys_include_primary=True (default), primary (v=0) joins
        # extras in the key sequence, giving cross-attention all cameras to
        # choose from.  When False, only extras (v≥1) are keys (legacy
        # extras-only baseline).  The primary stream is always preserved
        # through the residual path either way.
        mode = self.fusion_mode
        extras_start = 0 if self.fusion_keys_include_primary else 1
        n_keys = N - extras_start  # number of cameras represented in keys

        if mode == "pool_4":
            # Mean-pool each contributing camera's 64 weighted patches → 1 summary
            keys_list_t  = [p.mean(dim=1, keepdim=True) for p in patches_t[extras_start:]]
            keys_list_t1 = [p.mean(dim=1, keepdim=True) for p in patches_t1[extras_start:]]
            keys_t  = torch.cat(keys_list_t,  dim=1)  # [B, n_keys, D]
            keys_t1 = torch.cat(keys_list_t1, dim=1)
            tokens_per_cam = 1
        else:  # spatial_4 or spatial_64 — full 64 weighted patches per camera
            keys_t  = torch.cat(patches_t[extras_start:],  dim=1)  # [B, n_keys*64, D]
            keys_t1 = torch.cat(patches_t1[extras_start:], dim=1)
            tokens_per_cam = self.grid_h * self.grid_w  # 64

        # Build key_padding_mask from present_mask.
        # nn.MultiheadAttention convention: True = position is masked / ignored.
        # Primary (slot 0 of present_mask) is always True (never dropped),
        # so when included it is automatically never masked.
        if present_mask is not None:
            keys_present = (
                present_mask
                if self.fusion_keys_include_primary
                else present_mask[:, 1:]
            )  # [B, n_keys]
            # Safety: ensure each sample has ≥1 active key.  When primary is
            # included this is already guaranteed by primary always being
            # present; kept as defence-in-depth for the extras-only branch.
            none_active = ~keys_present.any(dim=1)
            if none_active.any():
                keys_present = keys_present.clone()
                keys_present[none_active, 0] = True
            absent = ~keys_present
            if tokens_per_cam > 1:
                key_padding_mask = (
                    absent.unsqueeze(2)
                    .expand(B, n_keys, tokens_per_cam)
                    .reshape(B, n_keys * tokens_per_cam)
                )
            else:
                key_padding_mask = absent
        else:
            key_padding_mask = None

        # ── 4. Cross-camera fusion → residual injection ──────────────────────
        if mode in ("pool_4", "spatial_4"):
            # Bottleneck queries → fused [B, D] → broadcast residual to primary
            fused_first, attn_first = self.bottleneck_fusion(keys_t,  key_padding_mask)
            fused_last,  attn_last  = self.bottleneck_fusion(keys_t1, key_padding_mask)
            # fused_*: [B, D],  attn_*: [B, n_bt, (N-1)*K]
            fused_pair = (
                torch.stack([fused_first, fused_last], dim=1)  # [B, 2, D]
                .unsqueeze(2).unsqueeze(3)                      # [B, 2, 1, 1, D]
            )
            tokens = primary_spatial_tokens + fused_pair  # broadcast → [B, 2, H, W, D]
        else:  # spatial_64
            # Primary patches as queries → enriched [B, 64, D] → reshape + add.
            # When fusion_keys_include_primary=True, primary patches as queries
            # also see primary patches as keys (a self-cross-attention).  This
            # is safe because (a) the LayerScale gate starts at 0 so the
            # contribution is zero at init, and (b) the spatial transformer
            # already gave primary self-attention; the gate lets the model
            # learn whether the redundant self-reference is useful.
            primary_q_t  = rearrange(primary_spatial_tokens[:, 0], "b h w d -> b (h w) d")
            primary_q_t1 = rearrange(primary_spatial_tokens[:, 1], "b h w d -> b (h w) d")
            enriched_t,  attn_first = self.spatial_cross(primary_q_t,  keys_t,  key_padding_mask)
            enriched_t1, attn_last  = self.spatial_cross(primary_q_t1, keys_t1, key_padding_mask)
            # enriched_*: [B, 64, D],  attn_*: [B, 64, (N-1)*64]
            # The SpatialCrossCamera module already includes its residual+gate
            # internally (enriched = norm(Q + gate * attn_out)).  We then add
            # this back to the un-weighted primary spatial grid so the temporal
            # transformer sees BOTH the original primary tokens and the
            # cross-camera-enriched contribution.  This double-residual is
            # intentional: SpatialCrossCamera's internal residual stabilises
            # gradient flow during cross-attention; the outer add keeps the
            # primary stream consistent with the pool_4 / spatial_4 paths
            # (which all "+= fused_*" into primary_spatial_tokens).
            enriched_t  = rearrange(
                enriched_t,  "b (h w) d -> b h w d", h=self.grid_h, w=self.grid_w,
            )
            enriched_t1 = rearrange(
                enriched_t1, "b (h w) d -> b h w d", h=self.grid_h, w=self.grid_w,
            )
            enriched_pair = torch.stack([enriched_t, enriched_t1], dim=1)  # [B, 2, H, W, D]
            tokens = primary_spatial_tokens + enriched_pair

        # ── 5. Temporal transformer — identical to single-camera path ────────
        video_shape = tuple(tokens.shape[:-1])
        tokens_t = rearrange(tokens, "b t h w d -> (b h w) t d")
        tokens_t = self.enc_temporal_transformer(tokens_t, video_shape=video_shape)
        tokens = rearrange(
            tokens_t, "(b h w) t d -> b t h w d",
            b=B, h=self.grid_h, w=self.grid_w,
        )

        first_tokens      = tokens[:, :1]
        last_tokens       = tokens[:, 1:]
        first_tokens_flat = rearrange(first_tokens, "b t h w d -> b (t h w) d")
        last_tokens_flat  = rearrange(last_tokens,  "b t h w d -> b (t h w) d")

        # ── 6. Per-camera, per-frame attention diagnostic ────────────────────
        # Return attention split per frame ([B, 2, N]) — averaging frames
        # would erase real temporal variation (e.g. a wrist camera attended
        # heavily near the goal frame but ignored in the start frame).
        # Different modes produce different attn shapes:
        #   pool_4 / spatial_4 : [B, n_bt, n_keys * tokens_per_cam]
        #   spatial_64         : [B, 64,    n_keys * tokens_per_cam]
        def _reduce_per_cam(attn: torch.Tensor) -> torch.Tensor:
            tot = attn.shape[-1]
            per = tot // n_keys
            return attn.reshape(B, attn.shape[1], n_keys, per).mean(dim=(1, 3))  # [B, n_keys]

        per_first = _reduce_per_cam(attn_first)  # [B, n_keys]
        per_last  = _reduce_per_cam(attn_last)
        if self.fusion_keys_include_primary:
            per_cam_attn_t  = per_first  # [B, N]
            per_cam_attn_t1 = per_last
        else:
            zeros = torch.zeros(B, 1, device=device)
            per_cam_attn_t  = torch.cat([zeros, per_first], dim=1)  # [B, N], slot 0 zero
            per_cam_attn_t1 = torch.cat([zeros, per_last],  dim=1)
        # Stack into [B, 2, N] — frame index then camera index.
        per_cam_attn = torch.stack([per_cam_attn_t, per_cam_attn_t1], dim=1)
        return first_tokens, last_tokens, first_tokens_flat, last_tokens_flat, per_cam_attn

    def _build_metrics(
        self,
        *,
        per_sample_loss: torch.Tensor,
        latent_flat: torch.Tensor,
    ) -> dict[str, float]:
        return {
            "loss": float(per_sample_loss.mean().detach().item()),
            "pixel_loss": float(per_sample_loss.mean().detach().item()),
            "latent_std": float(latent_flat.detach().std().item()),
            "latent_abs_mean": float(latent_flat.detach().abs().mean().item()),
        }

    def _prepare_action_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        action_h, action_w = self.action_shape
        action_tokens = rearrange(tokens, "b (t h w) d -> b t h w d", h=action_h, w=action_w)
        if self.latent_ablation == "permute_batch" and action_tokens.shape[0] > 1:
            perm = torch.randperm(action_tokens.shape[0], device=action_tokens.device)
            action_tokens = action_tokens[perm]
        return action_tokens

    def _decode_and_loss(
        self,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        action_tokens: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Pixel reconstruction loss for one (first_frame, last_frame) pair.

        Args:
            first_frame: ``[B, C, 1, H, W]`` conditioning frame.
            last_frame:  ``[B, C, 1, H, W]`` reconstruction target.
            action_tokens: ``[B, t_a, h_a, w_a, dim]`` quantised action tokens.
            attn_bias: spatial relative position bias for the pixel decoder.

        Returns:
            ``[B]`` per-sample MSE loss.
        """
        B = first_frame.shape[0]
        decoder_context = self.decoder_context_projection(first_frame).detach()
        video_shape = tuple(decoder_context.shape[:-1])
        pixel_context = rearrange(decoder_context, "b t h w d -> (b t) (h w) d")
        action_context = rearrange(action_tokens, "b t h w d -> (b t) (h w) d")
        decoded = self.pixel_decoder(
            pixel_context, video_shape=video_shape, attn_bias=attn_bias, context=action_context
        )
        decoded = rearrange(
            decoded, "(b t) (h w) d -> b t h w d", b=B, h=self.grid_h, w=self.grid_w
        )
        recon = self.pixel_to_pixels(decoded)
        return F.mse_loss(recon, last_frame, reduction="none").mean(dim=(1, 2, 3, 4))

    def forward(
        self,
        video: torch.Tensor,
        *,
        extra_videos: list[torch.Tensor] | None = None,
        step: int = 0,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, dict[str, float]] | torch.Tensor:
        """Forward pass supporting both single-camera and multi-camera modes.

        Args:
            video: Primary camera frames ``[B, C, 2, H, W]`` (always required).
            extra_videos: Additional camera frames, each ``[B, C, 2, H, W]``.
                When provided (and ``self.n_cameras > 1``), activates MBT
                bottleneck fusion.  Must have ``len(extra_videos) == n_cameras-1``.
            step: Global training step (kept for API parity; no schedules wired).
            reduction: ``"mean"`` or ``"none"``.
        """
        video = self._normalize_video_input(video)
        first_frame = video[:, :, :1]
        last_frame  = video[:, :, 1:]

        multi_cam = (
            extra_videos is not None
            and len(extra_videos) > 0
            and self.n_cameras > 1
            and (self.bottleneck_fusion is not None or self.spatial_cross is not None)
        )

        if multi_cam:
            # ── Multi-camera path ──────────────────────────────────────────
            B = video.shape[0]
            N = 1 + len(extra_videos)

            # Normalise extra cameras
            extra_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
            for ev in extra_videos:
                ev = self._normalize_video_input(ev)
                extra_pairs.append((ev[:, :, :1], ev[:, :, 1:]))

            all_pairs = [(first_frame, last_frame)] + extra_pairs

            # Stochastic view dropout (primary camera 0 is never dropped)
            present_mask: torch.Tensor | None = None
            if self.training and self.view_dropout_prob > 0.0:
                present_mask = torch.ones(B, N, dtype=torch.bool, device=video.device)
                for v in range(1, N):
                    keep = torch.rand(B, device=video.device) >= self.view_dropout_prob
                    present_mask[:, v] = keep

            _, _, first_tokens_flat, last_tokens_flat, per_cam_attn = self._encode_frames_multi(
                all_pairs, present_mask=present_mask
            )

            decoded_tokens, latent_flat = self.bottleneck(first_tokens_flat, last_tokens_flat)
            action_tokens = self._prepare_action_tokens(decoded_tokens)
            attn_bias = self.spatial_rel_pos_bias(self.grid_h, self.grid_w, device=video.device)

            # Decoder loss per camera — absent-camera samples are zeroed.
            # primary camera (v=0) is never dropped.
            view_losses: list[torch.Tensor] = []
            # active_counts[b] = number of cameras contributing loss for sample b
            active_counts = torch.ones(B, device=video.device)  # primary always active
            for v, (ff_v, lf_v) in enumerate(all_pairs):
                loss_v = self._decode_and_loss(ff_v, lf_v, action_tokens, attn_bias)
                if v > 0:
                    if present_mask is not None:
                        active = present_mask[:, v].float()  # [B]
                        loss_v = loss_v * active
                        active_counts = active_counts + active
                    else:
                        active_counts = active_counts + 1.0
                view_losses.append(loss_v)

            # Normalise by number of active cameras per sample so gradient
            # magnitude stays ~equal to single-camera training regardless of N.
            per_sample_loss = torch.stack(view_losses, dim=0).sum(dim=0) / active_counts

            metrics = self._build_metrics(
                per_sample_loss=per_sample_loss,
                latent_flat=latent_flat,
            )
            # Per-camera pixel losses (unnormalised) for monitoring
            for v, lv in enumerate(view_losses):
                metrics[f"pixel_loss_cam{v}"] = float(lv.mean().detach().item())
            # Bottleneck attention weights per camera, split per frame.
            # per_cam_attn shape: [B, 2, N]  (frame_t, frame_t1).
            # Logging both frames separately exposes temporal variation —
            # e.g. a wrist camera near the goal in frame_t1 should attract
            # higher attention than in frame_t.  Averaging would hide this.
            # Collapse (all weight on cam0) on either frame still indicates
            # fusion not learning.
            for v in range(N):
                metrics[f"fusion_attn_cam{v}_frame_t"]  = float(
                    per_cam_attn[:, 0, v].mean().detach().item()
                )
                metrics[f"fusion_attn_cam{v}_frame_t1"] = float(
                    per_cam_attn[:, 1, v].mean().detach().item()
                )

        else:
            # ── Single-camera path (original behaviour, unchanged) ─────────
            _, _, first_tokens_flat, last_tokens_flat = self._encode_frames(first_frame, last_frame)

            decoded_tokens, latent_flat = self.bottleneck(first_tokens_flat, last_tokens_flat)
            action_tokens = self._prepare_action_tokens(decoded_tokens)
            attn_bias = self.spatial_rel_pos_bias(self.grid_h, self.grid_w, device=video.device)
            per_sample_loss = self._decode_and_loss(first_frame, last_frame, action_tokens, attn_bias)
            metrics = self._build_metrics(
                per_sample_loss=per_sample_loss,
                latent_flat=latent_flat,
            )

        if reduction == "none":
            return per_sample_loss, metrics
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction={reduction!r}.")
        return per_sample_loss.mean(), metrics
