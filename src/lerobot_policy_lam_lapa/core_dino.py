from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

DEFAULT_OUTPUT_GRID_SIZE = (8, 8)


def pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, value) if not isinstance(value, tuple) else value


class DINOv3FeatureExtractor(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        image_size: int | tuple[int, int],
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.image_size = pair(image_size)
        if self.image_size[0] != self.image_size[1]:
            raise ValueError(f"DINOv3FeatureExtractor expects square image_size, got {self.image_size}.")

        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(getattr(self.backbone.config, "hidden_size"))
        self.patch_size = int(getattr(self.backbone.config, "patch_size", 16))
        if self.patch_size != 16:
            raise ValueError(f"Expected DINOv3 patch_size=16, got {self.patch_size}.")

        self.freeze = bool(freeze)
        if self.freeze:
            self.backbone.eval()
            self.backbone.requires_grad_(False)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def native_grid_size(self) -> tuple[int, int]:
        height, width = self.image_size
        return height // self.patch_size, width // self.patch_size

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 5:
            if frames.shape[2] != 1:
                raise ValueError(f"Expected a single frame in time dimension, got shape {tuple(frames.shape)}.")
            frames = frames.squeeze(2)
        if frames.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] or [B,C,1,H,W], got {tuple(frames.shape)}.")
        if tuple(frames.shape[-2:]) != self.image_size:
            frames = F.interpolate(
                frames,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )

        pixel_values = (frames.to(torch.float32) - self.mean) / self.std
        context_manager = torch.no_grad if self.freeze else nullcontext
        with context_manager():
            outputs = self.backbone(pixel_values=pixel_values, return_dict=True)

        grid_h, grid_w = self.native_grid_size
        num_spatial_tokens = grid_h * grid_w
        spatial_tokens = outputs.last_hidden_state[:, -num_spatial_tokens:, :]
        return spatial_tokens.reshape(frames.shape[0], grid_h, grid_w, self.hidden_size)


class LearnedTokenDownsampler(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        input_grid_size: tuple[int, int],
        output_grid_size: tuple[int, int] = DEFAULT_OUTPUT_GRID_SIZE,
    ) -> None:
        super().__init__()
        if input_grid_size != (16, 16):
            raise ValueError(
                f"LearnedTokenDownsampler expects a 16x16 input grid from DINOv3-S at 256x256, got {input_grid_size}."
            )
        if output_grid_size != (8, 8):
            raise ValueError(
                f"LearnedTokenDownsampler expects output_grid_size=(8, 8), got {output_grid_size}."
            )

        self.input_norm = nn.LayerNorm(input_dim)
        self.downsample = nn.Conv2d(input_dim, output_dim, kernel_size=2, stride=2, bias=False)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.input_norm(tokens)
        tokens = tokens.permute(0, 3, 1, 2)
        tokens = self.downsample(tokens)
        tokens = tokens.permute(0, 2, 3, 1)
        return self.output_norm(tokens)


class DinoTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        image_size: int | tuple[int, int],
        output_dim: int,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.feature_extractor = DINOv3FeatureExtractor(
            model_name=model_name,
            image_size=image_size,
            freeze=freeze,
        )
        self.downsampler = LearnedTokenDownsampler(
            input_dim=self.feature_extractor.hidden_size,
            output_dim=output_dim,
            input_grid_size=self.feature_extractor.native_grid_size,
            output_grid_size=DEFAULT_OUTPUT_GRID_SIZE,
        )
        self.output_grid_size = DEFAULT_OUTPUT_GRID_SIZE

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        tokens = self.feature_extractor(frames)
        tokens = self.downsampler(tokens)
        return tokens.unsqueeze(1)
