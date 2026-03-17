import logging

import torch
import torch.distributions.uniform as uniform_dist

logger = logging.getLogger(__name__)


class NSVQ(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device = torch.device("cpu"),
        discarding_threshold: float = 0.1,
        initialization: str = "normal",
        code_seq_len: int = 1,
        patch_size: int | tuple[int, int] = 32,
        image_size: int | tuple[int, int] = 256,
        grid_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-12
        self.dim = dim
        self.patch_size = patch_size

        if grid_size is not None:
            self.grid_h, self.grid_w = grid_size
        else:
            if isinstance(image_size, tuple) or isinstance(patch_size, tuple):
                raise ValueError("grid_size must be provided when image_size or patch_size are tuples.")
            self.grid_h = self.grid_w = int(image_size / patch_size)

        if initialization == "normal":
            codebooks = torch.randn(self.num_embeddings, self.embedding_dim, device=device)
        elif initialization == "uniform":
            codebooks = uniform_dist.Uniform(-1 / self.num_embeddings, 1 / self.num_embeddings).sample(
                [self.num_embeddings, self.embedding_dim]
            ).to(device)
        else:
            raise ValueError("initialization must be 'normal' or 'uniform'.")

        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)
        self.register_buffer(
            "codebooks_used",
            torch.zeros(self.num_embeddings, dtype=torch.int32, device=device),
            persistent=False,
        )

        self.project_in = torch.nn.Linear(dim, embedding_dim)
        self.project_out = torch.nn.Linear(embedding_dim, dim)
        self.cnn_encoder = self._build_cnn_encoder(code_seq_len=code_seq_len)

    def _build_cnn_encoder(self, code_seq_len: int) -> torch.nn.Sequential:
        input_size = self.grid_h
        if input_size == 8:
            if code_seq_len == 1:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=1, padding=0),
                )
            if code_seq_len == 2:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        self.embedding_dim, self.embedding_dim, kernel_size=(3, 4), stride=1, padding=0
                    ),
                )
            if code_seq_len == 4:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=1, padding=0),
                )
            raise ValueError(f"code_seq_len={code_seq_len} not supported for 8x8 grid")
        if input_size == 16:
            if code_seq_len == 1:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=1, padding=0),
                )
            if code_seq_len == 4:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=1, padding=0),
                )
            if code_seq_len == 16:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=2, padding=1),
                )
            if code_seq_len == 64:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=2, padding=1),
                )
            raise ValueError(f"code_seq_len={code_seq_len} not supported for 16x16 grid")
        raise ValueError(f"Grid size {input_size}x{input_size} not supported. Use 8 or 16.")

    def encode(self, input_data: torch.Tensor, batch_size: int) -> torch.Tensor:
        input_data = self.project_in(input_data)
        input_data = input_data.permute(0, 2, 1).contiguous()
        input_data = input_data.reshape(batch_size, self.embedding_dim, self.grid_h, self.grid_w)
        input_data = self.cnn_encoder(input_data)
        input_data = input_data.reshape(batch_size, self.embedding_dim, -1)
        input_data = input_data.permute(0, 2, 1).contiguous()
        return input_data.reshape(-1, self.embedding_dim)

    def decode(self, quantized_input: torch.Tensor, batch_size: int) -> torch.Tensor:
        quantized_input = quantized_input.reshape(batch_size, self.embedding_dim, -1)
        quantized_input = quantized_input.permute(0, 2, 1).contiguous()
        return self.project_out(quantized_input)

    def forward(
        self,
        input_data_first: torch.Tensor,
        input_data_last: torch.Tensor,
        codebook_training_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = input_data_first.shape[0]
        first = self.encode(input_data_first.contiguous(), batch_size)
        last = self.encode(input_data_last.contiguous(), batch_size)
        input_data = last - first

        distances = (
            torch.sum(input_data**2, dim=1, keepdim=True)
            - 2 * torch.matmul(input_data, self.codebooks.t())
            + torch.sum(self.codebooks.t() ** 2, dim=0, keepdim=True)
        )
        min_indices = torch.argmin(distances, dim=1)
        hard_quantized_input = self.codebooks[min_indices]
        random_vector = torch.randn_like(input_data)
        norm_quantization_residual = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()
        vq_error = (norm_quantization_residual / norm_random_vector + self.eps) * random_vector
        quantized_input = hard_quantized_input if codebook_training_only else input_data + vq_error

        encodings = torch.zeros(input_data.shape[0], self.num_embeddings, device=input_data.device)
        encodings.scatter_(1, min_indices.reshape([-1, 1]), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        with torch.no_grad():
            self.codebooks_used.scatter_add_(
                0,
                min_indices,
                torch.ones_like(min_indices, dtype=self.codebooks_used.dtype),
            )

        quantized_input = self.decode(quantized_input, batch_size)
        return quantized_input, perplexity, self.codebooks_used, min_indices.reshape(batch_size, -1)

    def _is_distributed(self) -> bool:
        try:
            import torch.distributed as dist
        except Exception:
            return False
        return dist.is_available() and dist.is_initialized()

    def _get_replacement_indices_from_counts(
        self, counts: torch.Tensor, discarding_threshold: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        total = int(counts.sum().item())
        if total <= 0 or self.num_embeddings <= 0:
            all_idx = torch.arange(self.num_embeddings, dtype=torch.long, device=counts.device)
            empty = torch.empty((0,), dtype=torch.long, device=counts.device)
            return all_idx, empty, 0.0

        expected = total / float(self.num_embeddings)
        threshold = self.discarding_threshold if discarding_threshold is None else float(discarding_threshold)
        min_count = threshold * expected

        counts_f = counts.to(dtype=torch.float32)
        used_indices = torch.where(counts_f >= min_count)[0].to(dtype=torch.long)
        unused_indices = torch.where(counts_f < min_count)[0].to(dtype=torch.long)
        return unused_indices, used_indices, min_count

    def replace_unused_codebooks(self, discarding_threshold: float | None = None) -> tuple[int, int, int, float]:
        with torch.no_grad():
            counts = self.codebooks_used.to(dtype=torch.int64)
            if self._is_distributed():
                import torch.distributed as dist

                counts = counts.clone()
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)

            threshold = self.discarding_threshold if discarding_threshold is None else float(discarding_threshold)
            unused_indices, used_indices, min_count = self._get_replacement_indices_from_counts(
                counts, discarding_threshold=threshold
            )
            unused_count = unused_indices.shape[0]
            used_count = used_indices.shape[0]
            total_assignments = int(counts.sum().item())

            is_dist = self._is_distributed()
            dist_rank = 0
            if is_dist:
                import torch.distributed as dist

                dist_rank = int(dist.get_rank())

            if used_count == 0:
                if dist_rank == 0:
                    self.codebooks += self.eps * torch.randn(self.codebooks.size(), device=self.codebooks.device)
            elif unused_count > 0 and dist_rank == 0:
                used = self.codebooks[used_indices].clone()
                if used_count < unused_count:
                    used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                    used_codebooks = used_codebooks[
                        torch.randperm(used_codebooks.shape[0], device=used_codebooks.device)
                    ]
                else:
                    used_codebooks = used
                self.codebooks[unused_indices] = used_codebooks[:unused_count] + 0.02 * torch.randn(
                    (unused_count, self.embedding_dim), device=self.codebooks.device
                )

            if is_dist:
                import torch.distributed as dist

                dist.broadcast(self.codebooks.data, src=0)

            logger.info(
                "Replaced %d codebooks (used=%d, total=%d, min_count=%.4f, threshold=%.6f)",
                unused_count,
                used_count,
                total_assignments,
                min_count,
                threshold,
            )
            self.codebooks_used.zero_()
            return int(unused_count), int(used_count), int(total_assignments), float(min_count)

    def get_indices(self, input_data_first: torch.Tensor, input_data_last: torch.Tensor) -> torch.Tensor:
        batch_size = input_data_first.shape[0]
        first = self.encode(input_data_first.contiguous(), batch_size)
        last = self.encode(input_data_last.contiguous(), batch_size)
        delta = last - first
        distances = (
            torch.sum(delta**2, dim=1, keepdim=True)
            - 2 * torch.matmul(delta, self.codebooks.t())
            + torch.sum(self.codebooks.t() ** 2, dim=0, keepdim=True)
        )
        return torch.argmin(distances, dim=1).reshape(batch_size, -1)
