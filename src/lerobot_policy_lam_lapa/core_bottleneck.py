from __future__ import annotations

import torch
from torch import nn


class ContinuousLatentBottleneck(nn.Module):
    """Continuous latent bottleneck — replaces NSVQ with no quantisation.

    project_in + CNN compresses [B, gh*gw, dim] down to [B*code_seq_len,
    embedding_dim].  The encoded delta (last - first) is used directly as the
    action latent.
    """

    def __init__(
        self,
        *,
        dim: int,
        embedding_dim: int,
        code_seq_len: int,
        grid_size: tuple[int, int],
    ):
        super().__init__()
        self.dim = dim
        self.embedding_dim = embedding_dim
        self.code_seq_len = code_seq_len
        self.grid_h, self.grid_w = grid_size

        self.project_in = nn.Linear(dim, embedding_dim)
        self.project_out = nn.Linear(embedding_dim, dim)
        self.cnn_encoder = self._build_cnn_encoder(code_seq_len=code_seq_len)

    def _build_cnn_encoder(self, code_seq_len: int) -> nn.Sequential:
        input_size = self.grid_h
        ed = self.embedding_dim
        if input_size == 8:
            if code_seq_len == 1:
                return nn.Sequential(
                    nn.Conv2d(ed, ed, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(ed, ed, kernel_size=4, stride=1, padding=0),
                )
            if code_seq_len == 2:
                return nn.Sequential(
                    nn.Conv2d(ed, ed, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(ed, ed, kernel_size=(3, 4), stride=1, padding=0),
                )
            if code_seq_len == 4:
                return nn.Sequential(
                    nn.Conv2d(ed, ed, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(ed, ed, kernel_size=3, stride=1, padding=0),
                )
            raise ValueError(f"code_seq_len={code_seq_len} not supported for 8x8 grid")
        if input_size == 16:
            if code_seq_len == 1:
                return nn.Sequential(
                    nn.Conv2d(ed, ed, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(ed, ed, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(ed, ed, kernel_size=4, stride=1, padding=0),
                )
            if code_seq_len == 4:
                return nn.Sequential(
                    nn.Conv2d(ed, ed, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(ed, ed, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(ed, ed, kernel_size=3, stride=1, padding=0),
                )
            if code_seq_len == 16:
                return nn.Sequential(
                    nn.Conv2d(ed, ed, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(ed, ed, kernel_size=3, stride=2, padding=1),
                )
            if code_seq_len == 64:
                return nn.Sequential(
                    nn.Conv2d(ed, ed, kernel_size=3, stride=2, padding=1),
                )
            raise ValueError(f"code_seq_len={code_seq_len} not supported for 16x16 grid")
        raise ValueError(f"Grid size {input_size}x{input_size} not supported. Use 8 or 16.")

    def encode(self, input_data: torch.Tensor, batch_size: int) -> torch.Tensor:
        # input: [B, gh*gw, dim] -> [B*code_seq_len, embedding_dim]
        x = self.project_in(input_data)
        x = x.permute(0, 2, 1).contiguous()
        x = x.reshape(batch_size, self.embedding_dim, self.grid_h, self.grid_w)
        x = self.cnn_encoder(x)
        x = x.reshape(batch_size, self.embedding_dim, -1)
        x = x.permute(0, 2, 1).contiguous()
        return x.reshape(-1, self.embedding_dim)

    def decode(self, latent: torch.Tensor, batch_size: int) -> torch.Tensor:
        latent = latent.reshape(batch_size, self.embedding_dim, -1)
        latent = latent.permute(0, 2, 1).contiguous()
        return self.project_out(latent)

    def forward(
        self,
        input_data_first: torch.Tensor,
        input_data_last: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns (decoded_tokens [B, code_seq_len, dim], delta_flat [B*code_seq_len, embedding_dim]).
        batch_size = input_data_first.shape[0]
        first = self.encode(input_data_first.contiguous(), batch_size)
        last = self.encode(input_data_last.contiguous(), batch_size)
        delta = last - first
        decoded = self.decode(delta, batch_size)
        return decoded, delta
