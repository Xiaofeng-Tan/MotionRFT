import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward, flash_attn_varlen_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None


MEMORY_LAYOUT = {
    "flash": (lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]), lambda x: x),
    "torch": (lambda x: x.transpose(1, 2), lambda x: x.transpose(1, 2)),
    "vanilla": (lambda x: x.transpose(1, 2), lambda x: x.transpose(1, 2)),
}


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mode: str = "flash",
    drop_rate: float = 0.0,
    attn_mask: Optional[Tensor] = None,
    causal: bool = False,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    batch_size: int = 1,
    training: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Perform QKV self attention.

    Args:
        q (Tensor): Query tensor with shape [b, s, h, d], where h is the number of heads.
        k (Tensor): Key tensor with shape [b, s1, h, d]
        v (Tensor): Value tensor with shape [b, s1, h, d]
        mode (str): Attention mode. Choose from 'self_flash', 'cross_flash', 'torch', and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, h, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        cu_seqlens_q (Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        cu_seqlens_kv (Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_q (int): The maximum sequence length in the batch of q.
        max_seqlen_kv (int): The maximum sequence length in the batch of k and v.

    Returns:
        Tensor: Output tensor after self attention with shape [b, s, hd]
    """
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)
    elif mode == "flash":
        assert flash_attn_varlen_func is not None, "flash_attn is not installed or not supported"
        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        # x with shape [(bxs), a, d]
        x = x.view(batch_size, max_seqlen_q, x.shape[-2], x.shape[-1])  # reshape x to [b, s, a, d]
    elif mode == "vanilla":
        scale_factor = 1.0 / math.sqrt(q.size(-1))
        b, a, s_q, _ = q.shape
        s_k = k.size(2)
        attn_bias = torch.zeros(b, a, s_q, s_k, dtype=q.dtype, device=q.device)
        if causal:
            # Only applied to self attention
            assert attn_mask is None, "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(b, a, s_q, s_q, dtype=torch.bool, device=q.device).tril(diagonal=0)
            attn_bias.masked_fill_(~temp_mask, float("-inf"))
            attn_bias = attn_bias.to(q.dtype)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(~attn_mask, float("-inf"))
            else:
                attn_bias = attn_bias + attn_mask

        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn = attn + attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=training)
        x = attn @ v
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)
    b, s, h, d = x.shape
    out = x.reshape(b, s, -1)
    return out
