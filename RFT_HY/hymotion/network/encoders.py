from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.misc import to_2tuple
from .bricks import get_activation_layer, get_norm_layer
from .modulate_layers import ModulateDiT, modulate


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        feat_dim: int,
        out_dim: Optional[int] = None,
        act_type: str = "gelu",
        norm_type: Optional[str] = None,
        bias: bool = True,
        drop: float = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_dim = out_dim or in_dim
        feat_dim = feat_dim or in_dim
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_dim, feat_dim, bias=bias[0] if isinstance(bias, (list, tuple)) else bias)
        self.act = get_activation_layer(act_type)()
        self.drop1 = nn.Dropout(drop_probs[0] if isinstance(drop_probs, (list, tuple)) else drop_probs)
        self.norm = get_norm_layer(norm_type)(feat_dim) if norm_type else nn.Identity()
        self.fc2 = linear_layer(feat_dim, out_dim, bias=bias[1] if isinstance(bias, (list, tuple)) else bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int, num_layers: int, act_type: str = "silu") -> None:
        super(MLPEncoder, self).__init__()
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        linears = []
        linears.append(nn.Linear(in_features=in_dim, out_features=self.feat_dim))
        for i in range(num_layers - 1):
            linears.append(get_activation_layer(act_type)())
            linears.append(nn.Linear(self.feat_dim, self.feat_dim))
        self.linears = nn.Sequential(*linears)

    def forward(self, x: Tensor) -> Tensor:
        return self.linears(x)


class FinalLayer(nn.Module):
    def __init__(self, feat_dim: int, out_dim: int, act_type: str = "gelu", zero_init=False, **kwargs):
        super().__init__()

        self.norm_final = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = ModulateDiT(feat_dim, factor=2, act_type=act_type)
        self.linear = nn.Linear(feat_dim, out_dim, bias=True)
        if zero_init:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, adapter: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(adapter).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x


class TimestepEmbeddingEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feat_dim: int,
        act_type: str = "silu",
        time_factor: float = 1.0,
    ) -> None:
        super(TimestepEmbeddingEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.feat_dim = feat_dim
        self.time_factor = time_factor
        blocks = [
            nn.Linear(embedding_dim, self.feat_dim),
            get_activation_layer(act_type)(),
            nn.Linear(self.feat_dim, self.feat_dim),
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, t: Tensor) -> Tensor:
        x = self.blocks(self.sinusodial_embedding(t, self.embedding_dim, time_factor=self.time_factor)).unsqueeze(1)
        return x

    @staticmethod
    def sinusodial_embedding(
        timesteps: Tensor, embedding_dim: int, temperature: float = 10000.0, time_factor: float = 1.0
    ) -> Tensor:
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        timesteps = timesteps * time_factor
        half = embedding_dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(temperature)) * torch.arange(start=0, end=half, dtype=torch.float) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if embedding_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
