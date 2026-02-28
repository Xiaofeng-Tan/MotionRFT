from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        num_feats: int,
        max_seq_len: Union[Tensor, int],
        temperature: int = 10000,
        use_real: bool = False,
        theta_rescale_factor: float = 1.0,
        interpolation_factor: float = 1.0,
    ) -> None:
        super(RotaryEmbedding, self).__init__()
        assert num_feats % 2 == 0, "num_feats (head_dim) must be even for RoPE."
        self.num_feats = num_feats
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.use_real = use_real
        self.theta_rescale_factor = theta_rescale_factor
        self.interpolation_factor = interpolation_factor

        if isinstance(max_seq_len, int):
            max_seq_len = torch.arange(max_seq_len).float()

        if theta_rescale_factor != 1.0:
            temperature *= theta_rescale_factor ** (self.num_feats / (self.num_feats - 2))
        dim_t = torch.arange(0, self.num_feats, 2, dtype=torch.float32)
        freqs = 1.0 / (temperature ** (2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_feats))  # [D/2]
        freqs = torch.outer(max_seq_len.float() * interpolation_factor, freqs)  # [S, D/2]
        if use_real:
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
            self.freqs_cis = (freqs_cos, freqs_sin)
        else:
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [S, D/2]
            self.freqs_cis = freqs_cis

    def reshape_for_broadcast(
        self, freqs_cis: Union[Tensor, Tuple[Tensor, Tensor]], x: Tensor
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        ndim = x.ndim
        assert 0 <= 1 < ndim

        if isinstance(freqs_cis, tuple):
            # freqs_cis: (cos, sin) in real space
            assert (
                freqs_cis[0].shape[-1] == x.shape[-1]
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape} on the head_dim dimension"
            assert freqs_cis[0].shape[0] >= x.shape[1], (
                f"freqs_cis shape {freqs_cis[0].shape} should be larger than or equal to "
                f"x shape {x.shape} on the time dimension"
            )
            shape = []
            for i, d in enumerate(x.shape):
                if i == 1:
                    shape.append(-1)
                elif i == ndim - 1:
                    shape.append(d)
                else:
                    shape.append(1)
            return (
                freqs_cis[0].view(*shape)[:, : x.shape[1], ...],
                freqs_cis[1].view(*shape)[:, : x.shape[1], ...],
            )
        else:
            # freqs_cis: values in complex space
            assert (
                freqs_cis.shape[-1] == x.shape[-1]
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape} on the head_dim dimension"
            assert freqs_cis.shape[0] >= x.shape[1], (
                f"freqs_cis shape {freqs_cis.shape} should be larger than or equal to "
                f"x shape {x.shape} on the time dimension"
            )
            shape = []
            for i, d in enumerate(x.shape):
                if i == 1:
                    shape.append(-1)
                elif i == ndim - 1:
                    shape.append(d)
                else:
                    shape.append(1)
            return freqs_cis.view(*shape)[:, : x.shape[1], ...]

    def rotate_half(self, x: Tensor) -> Tensor:
        x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        return torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    def apply_rotary_emb(self, xq: Tensor, xk: Tensor) -> Tuple[Tensor, Tensor]:
        xk_out = None
        if isinstance(self.freqs_cis, tuple):
            cos, sin = self.reshape_for_broadcast(self.freqs_cis, xq)  # [B, L, H, D]
            cos, sin = cos.to(xq.device), sin.to(xq.device)
            # real * cos - imag * sin
            # imag * cos + real * sin
            xq_out = (xq.float() * cos + self.rotate_half(xq.float()) * sin).type_as(xq)
            xk_out = (xk.float() * cos + self.rotate_half(xk.float()) * sin).type_as(xk)
        else:
            # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [B, S, H, D//2]
            freqs_cis = self.reshape_for_broadcast(self.freqs_cis, xq_)
            # Handle device transfer based on return type
            if isinstance(freqs_cis, tuple):
                freqs_cis = (freqs_cis[0].to(xq.device), freqs_cis[1].to(xq.device))
            else:
                freqs_cis = freqs_cis.to(xq.device)  # [S, D//2] --> [1, S, 1, D//2]
            # (real, imag) * (cos, sin) = (real * cos - imag * sin, imag * cos + real * sin)
            # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # [B, S, H, D//2]
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

        return xq_out, xk_out

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(num_feats={self.num_feats}, "
        repr_str += f"max_seq_len={self.max_seq_len}, "
        repr_str += f"temperature={self.temperature}, "
        repr_str += f"use_real={self.use_real}, "
        repr_str += f"theta_rescale_factor={self.theta_rescale_factor}, "
        repr_str += f"interpolation_factor={self.interpolation_factor})"
        return repr_str


class PositionalEncoding(nn.Module):
    def __init__(self, num_feats: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, num_feats)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_feats, 2).float() * (-np.log(10000.0) / num_feats))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape of [1, L, D]
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.shape[1], :]  # shape of [B, L, D]
        return self.dropout(x)


if __name__ == "__main__":
    # python -m hymotion.network.positional_encoding
    num_feats = 32
    rope = RotaryEmbedding(num_feats=num_feats, max_seq_len=5000, use_real=True)
    x = torch.ones(1, 360, 1, num_feats)
    text = torch.ones(1, 256, 1, num_feats)
    q1, k1 = x.clone(), x.clone()
    q2, k2 = text.clone(), text.clone()
    print(x.shape)
    # q1, k1 = rope.apply_rotary_emb(q1, k1)
    # q2, k2 = rope.apply_rotary_emb(q2, k2)
    q = torch.cat([q1, q2], dim=1)
    k = torch.cat([k1, k2], dim=1)
    q, k = rope.apply_rotary_emb(q, k)
    q, k = q[0, :, 0, :], k[0, :, 0, :]
    attn = (q[:, None] * k[None, :]).sum(dim=-1)
    # softmax
    # attn = torch.softmax(attn, dim=-1)
    attn = attn.cpu().numpy()

    import matplotlib.pyplot as plt

    plt.imshow(attn, cmap="hot")
    plt.colorbar()
    plt.savefig("attn.png")
    breakpoint()
