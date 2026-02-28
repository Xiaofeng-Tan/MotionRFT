from typing import Optional

import torch.nn as nn
from torch import Tensor

from .bricks import get_activation_layer


class ModulateDiT(nn.Module):
    def __init__(self, feat_dim: int, factor: int, act_type: str = "silu"):
        super().__init__()
        self.act = get_activation_layer(act_type)()
        self.linear = nn.Linear(feat_dim, factor * feat_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.act(x))


def modulate(x: Tensor, shift: Optional[Tensor] = None, scale: Optional[Tensor] = None) -> Tensor:
    if shift is not None and scale is not None:
        assert len(x.shape) == len(shift.shape) == len(scale.shape), (
            "x, shift, scale must have the same number of dimensions, "
            f"but got x.shape: {x.shape}, "
            f"shift.shape: {shift.shape} "
            f"and scale.shape: {scale.shape}"
        )
    if shift is not None and scale is not None:
        return x * (1 + scale) + shift
    elif shift is not None:
        return x + shift
    elif scale is not None:
        return x * (1 + scale)
    else:
        return x


def apply_gate(x: Tensor, gate: Optional[Tensor] = None, tanh: bool = False) -> Tensor:
    if gate is not None:
        assert len(x.shape) == len(
            gate.shape
        ), f"x, gate must have the same number of dimensions, but got {x.shape} and {gate.shape}"
    if gate is None:
        return x
    if tanh:
        return x * gate.tanh()
    else:
        return x * gate
