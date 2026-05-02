"""Cross-camera fusion modules for the multi-camera Stage-1 LAM.

Two complementary fusion modules:

* ``BottleneckCameraFusion`` — Multimodal Bottleneck Transformer (MBT) style
  (Nagrani et al., NeurIPS 2021).  A small set of learnable bottleneck queries
  attends to extra-camera tokens and returns a single ``[B, D]`` fused vector,
  which the caller broadcast-adds to the primary camera spatial grid.
  Used by ``fusion_mode="pool_4"`` and ``"spatial_4"``.

* ``SpatialCrossCamera`` — primary patches as queries cross-attend to extra-
  camera patches.  Returns ``[B, 64, D]`` enriched primary tokens, one per
  spatial position.  Used by ``fusion_mode="spatial_64"``.

Both modules use ``nn.MultiheadAttention`` with ``key_padding_mask`` to handle
view dropout (per-sample dropping of extra cameras).  The primary camera
(view index 0) is never dropped.

Key training diagnostic
-----------------------
The caller logs ``fusion_attn_cam{v}`` per extra camera v ≥ 1.  Collapse to
near-zero attention on a particular camera means the fusion is not using that
view (MBT paper §5.1).
"""

from __future__ import annotations

import torch
from torch import nn


class BottleneckCameraFusion(nn.Module):
    """MBT bottleneck fusion. Learnable queries attend to extra-camera tokens.

    Note: an earlier revision used a learnable ``mask_token`` parameter to
    replace dropped-camera summaries before attention.  That has been removed
    in favour of ``key_padding_mask`` — the standard ``nn.MultiheadAttention``
    mechanism for ignoring positions.  Functionally equivalent but cleaner
    (no extra parameter, no phantom mask-token signal mixed into attention).

    Used by ``fusion_mode="pool_4"`` (keys = N-1 mean-pooled extra summaries)
    and ``fusion_mode="spatial_4"`` (keys = (N-1)*64 full extra-camera patches).
    The number of bottleneck queries is fixed by ``n_bottleneck_tokens`` (4 by
    default — the "4" in pool_4/spatial_4).

    Output is a single ``[B, D]`` vector per timestep that the caller broadcast-
    adds to the primary camera's ``[B, 8, 8, D]`` spatial grid.

    Args:
        model_dim: Feature dimension D.
        n_bottleneck_tokens: Number of learnable query tokens.
        num_heads: Attention heads for cross-attention.
    """

    def __init__(self, model_dim: int, n_bottleneck_tokens: int, num_heads: int) -> None:
        super().__init__()
        self.n_bottleneck_tokens = n_bottleneck_tokens
        self.model_dim = model_dim

        # Learnable bottleneck queries — init matches ViT positional embedding (MBT §3)
        self.bottleneck_tokens = nn.Parameter(torch.empty(n_bottleneck_tokens, model_dim))
        nn.init.normal_(self.bottleneck_tokens, mean=0.0, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        keys: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse extra-camera tokens into a single ``[B, D]`` vector.

        Args:
            keys: ``[B, M, D]`` extra-camera tokens.  Layout is camera-major:
                ``[cam1_tok0, …, cam1_tokK, cam2_tok0, …]`` (primary excluded).
                For ``pool_4``: ``M = N-1`` (one mean-pooled summary per extra cam).
                For ``spatial_4``: ``M = (N-1)*64``.
            key_padding_mask: ``[B, M]`` bool — True = position is a dropped
                camera and should be ignored by attention.  None = no dropout.

        Returns:
            fused: ``[B, D]`` cross-camera fused representation.
            attn_weights: ``[B, n_bt, M]`` raw attention weights (averaged over
                heads inside MultiheadAttention).  Caller reshapes per-camera
                for diagnostics.
        """
        B = keys.shape[0]
        q = self.bottleneck_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, n_bt, D]
        fused, attn_weights = self.cross_attn(
            q, keys, keys, key_padding_mask=key_padding_mask, need_weights=True,
        )
        fused = fused.mean(dim=1)  # [B, D]
        fused = self.norm(fused)
        return fused, attn_weights


class SpatialCrossCamera(nn.Module):
    """Spatial cross-camera enrichment. Primary patches are queries.

    Used by ``fusion_mode="spatial_64"``.  Each of the 64 primary-camera patches
    issues its own attention query against the concatenated extra-camera patches
    ``[B, (N-1)*64, D]``, so the enrichment is spatially specific (each primary
    patch picks up the cross-camera context most relevant to its own location).

    Output ``[B, 64, D]`` is reshaped to ``[B, 8, 8, D]`` and added to the
    primary camera's spatial grid by the caller.

    Learned residual gate (LayerScale-style)
    ----------------------------------------
    A scalar parameter ``gate`` is initialised to 0 so at step 0:
    ``enriched = Q + gate * cross_attn_out ≡ Q``.  This ensures the cross-
    attention output starts as a zero contribution and grows as training
    learns it is useful.  Crucial here because the enrichment goes directly
    into the 64 spatial tokens that feed NSVQ — random cross-attn output
    added at init can destabilise codebook learning (perplexity collapse,
    high reconstruction loss in the first few hundred steps).  References:
    LayerScale in CaiT (Touvron et al., 2021); ReZero (Bachlechner et al., 2021).
    Gradient flows to the cross_attn weights through the residual path even
    when gate ≈ 0, so the module still learns from step 0.

    Args:
        model_dim: Feature dimension D.
        num_heads: Attention heads for cross-attention.
    """

    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Spatially-grounded cross-camera enrichment of primary patches.

        Args:
            query: ``[B, 64, D]`` primary camera patches.
            keys: ``[B, (N-1)*64, D]`` extra-camera patches (camera-major).
            key_padding_mask: ``[B, (N-1)*64]`` bool, True = ignored.

        Returns:
            enriched: ``[B, 64, D]`` query + gated cross-attention output, LayerNormed.
            attn_weights: ``[B, 64, (N-1)*64]`` raw attention weights.
        """
        attn_out, attn_weights = self.cross_attn(
            query, keys, keys,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        enriched = query + self.gate * attn_out
        enriched = self.norm(enriched)
        return enriched, attn_weights
