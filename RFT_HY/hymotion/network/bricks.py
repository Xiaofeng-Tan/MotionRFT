from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor


def get_activation_layer(act_type: str) -> Callable[[], nn.Module]:
    if act_type == "gelu":
        return lambda: nn.GELU()
    elif act_type == "gelu_tanh":
        return lambda: nn.GELU(approximate="tanh")
    elif act_type == "relu":
        return nn.ReLU
    elif act_type == "silu":
        return nn.SiLU
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


def get_norm_layer(norm_type: Optional[str]):
    if norm_type == "layer":
        return nn.LayerNorm
    elif norm_type == "rms":
        return RMSNorm
    elif norm_type == "none" or norm_type is None:
        return nn.Identity
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine=True, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output
