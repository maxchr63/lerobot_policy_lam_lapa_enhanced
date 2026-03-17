import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn


def exists(value):
    return value is not None


def default(value, default_value):
    return value if exists(value) else default_value


def l2norm(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor, dim=-1)


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def feed_forward(dim: int, mult: int = 4, dropout: float = 0.0) -> nn.Sequential:
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PEG(nn.Module):
    def __init__(self, dim: int, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    def forward(self, x: torch.Tensor, shape: tuple[int, int, int, int] | None = None) -> torch.Tensor:
        needs_shape = x.ndim == 3
        if needs_shape and shape is None:
            raise ValueError("shape is required when PEG receives a flattened sequence.")

        original_shape = x.shape
        if needs_shape and shape is not None:
            x = x.reshape(*shape, -1)

        x = rearrange(x, "b ... d -> b d ...")
        frame_padding = (2, 0) if self.causal else (1, 1)
        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value=0.0)
        x = self.dsconv(x)
        x = rearrange(x, "b d ... -> b ... d")

        if needs_shape:
            x = rearrange(x, "b ... d -> b (...) d")
        return x.reshape(original_shape)


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.heads = heads
        slopes = torch.tensor(self._get_slopes(heads))
        self.register_buffer("slopes", rearrange(slopes, "h -> h 1 1"), persistent=False)
        self.register_buffer("bias", None, persistent=False)

    @staticmethod
    def _get_slopes(heads: int) -> list[float]:
        def get_slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(
            2 * closest_power_of_2
        )[0::2][: heads - closest_power_of_2]

    def get_bias(self, i: int, j: int, device: torch.device) -> torch.Tensor:
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        return -torch.abs(rearrange(j_arange, "j -> 1 1 j") - rearrange(i_arange, "i -> 1 i 1"))

    def forward(self, sim: torch.Tensor) -> torch.Tensor:
        h, i, j, device = *sim.shape[-3:], sim.device
        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device) * self.slopes
        num_heads_unbiased = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unbiased))
        self.register_buffer("bias", bias, persistent=False)
        return self.bias


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_context: int | None = None,
        dim_head: int = 64,
        heads: int = 8,
        causal: bool = False,
        num_null_kv: int = 0,
        norm_context: bool = True,
        dropout: float = 0.0,
        scale: int = 8,
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads=heads)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()
        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, device = x.shape[0], x.device
        if exists(context):
            context = self.context_norm(context)
        kv_input = default(context, x)
        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        nk, nv = repeat(self.null_kv, "h (n r) d -> b h n r d", b=batch, r=2).unbind(dim=-2)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        i, j = sim.shape[-2:]
        if exists(attn_bias):
            sim = sim + F.pad(attn_bias, (self.num_null_kv, 0), value=0.0)
        if exists(mask):
            mask = rearrange(F.pad(mask, (self.num_null_kv, 0), value=True), "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        if self.causal:
            sim = sim + self.rel_pos_bias(sim)
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = self.attn_dropout(sim.softmax(dim=-1))
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ContinuousPositionBias(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        num_dims: int = 2,
        layers: int = 2,
        log_dist: bool = True,
        cache_rel_pos: bool = False,
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist
        self.cache_rel_pos = cache_rel_pos
        self.register_buffer("rel_pos", None, persistent=False)

        self.net = nn.ModuleList([nn.Sequential(nn.Linear(num_dims, dim), nn.LeakyReLU(0.1))])
        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(0.1)))
        self.net.append(nn.Linear(dim, heads))

    def forward(self, *dimensions: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device=device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing="ij"))
            grid = rearrange(grid, "c ... -> (...) c")
            rel_pos = rearrange(grid, "i c -> i 1 c") - rearrange(grid, "j c -> 1 j c")
            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)
            self.register_buffer("rel_pos", rel_pos, persistent=False)

        rel_pos = self.rel_pos.float()
        for layer in self.net:
            rel_pos = layer(rel_pos)
        return rearrange(rel_pos, "i j h -> h i j")


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        dim_context: int | None = None,
        causal: bool = False,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        peg: bool = False,
        peg_causal: bool = False,
        attn_num_null_kv: int = 2,
        has_cross_attn: bool = False,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PEG(dim=dim, causal=peg_causal) if peg else None,
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            causal=causal,
                            dropout=attn_dropout,
                        ),
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            dim_context=dim_context,
                            heads=heads,
                            causal=False,
                            num_null_kv=attn_num_null_kv,
                            dropout=attn_dropout,
                        )
                        if has_cross_attn
                        else None,
                        feed_forward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )
        self.norm_out = LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        video_shape: tuple[int, int, int, int] | None = None,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for peg, self_attn, cross_attn, ff in self.layers:
            if exists(peg):
                x = peg(x, shape=video_shape) + x
            x = self_attn(x, attn_bias=attn_bias, mask=self_attn_mask) + x
            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context=context, mask=cross_attn_context_mask) + x
            x = ff(x) + x
        return self.norm_out(x)
