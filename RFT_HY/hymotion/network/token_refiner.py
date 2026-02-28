from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .attention import attention
from .bricks import get_norm_layer
from .encoders import MLP, MLPEncoder, TimestepEmbeddingEncoder
from .modulate_layers import ModulateDiT, apply_gate


class IndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        mlp_act_type: str = "silu",
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        assert self.feat_dim % num_heads == 0, f"feat_dim {self.feat_dim} must be divisible by num_heads {num_heads}"
        self.head_dim = feat_dim // num_heads

        self.mlp_hidden_dim = int(feat_dim * mlp_ratio)

        self.norm1 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=True, eps=1e-6)
        self.self_attn_qkv = nn.Linear(feat_dim, feat_dim * 3, bias=qkv_bias)
        self.self_attn_q_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.self_attn_k_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.self_attn_proj = nn.Linear(feat_dim, feat_dim, bias=qkv_bias)

        self.norm2 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(
            in_dim=feat_dim,
            feat_dim=self.mlp_hidden_dim,
            act_type=mlp_act_type,
            drop=dropout,
        )

        self.adaLN_modulation = ModulateDiT(
            feat_dim=feat_dim,
            factor=2,
            act_type="silu",
        )

    def forward(self, x: Tensor, c: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=-1)
        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        # Apply QK-Norm if needed
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)
        # Self-Attention
        attn = attention(q, k, v, mode="torch", attn_mask=attn_mask)
        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)
        # FFN Layer
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)
        return x


class IndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        mlp_act_type: str = "silu",
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                IndividualTokenRefinerBlock(
                    feat_dim=feat_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    mlp_act_type=mlp_act_type,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, c: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        self_attn_mask = None
        if mask is not None:
            batch_size = mask.shape[0]
            seq_len = mask.shape[1]
            mask = mask.to(x.device)
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_1 = mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            # batch_size x 1 x seq_len x seq_len, 1 for broadcasting of num_heads
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            # avoids self-attention weight being NaN for padding tokens
            # assume the shape of self_attn_mask is [B, H, Q, K] and this is self-attention (Q==K==L)
            L = self_attn_mask.size(-1)
            diag = torch.eye(L, dtype=torch.bool, device=self_attn_mask.device).view(1, 1, L, L)  # [1,1,L,L]
            # mark which query row is "all False" (no visible key)
            all_false = ~self_attn_mask.any(dim=-1, keepdim=False)  # [B, H, Q]
            # expand to [B, H, Q, K], only for these rows, back to diagonal visible
            all_false = all_false.unsqueeze(-1).expand(-1, -1, -1, L)
            self_attn_mask = torch.where(all_false, diag.expand_as(self_attn_mask), self_attn_mask)

        if self_attn_mask is not None:
            self_attn_mask = torch.where(
                self_attn_mask,
                torch.zeros_like(self_attn_mask, dtype=torch.float),
                torch.full_like(self_attn_mask, float("-inf"), dtype=torch.float),
            )
        for block in self.blocks:
            x = block(x, c, self_attn_mask)
        return x


class SingleTokenRefiner(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feat_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        mlp_act_type: str = "silu",
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        attn_mode: str = "torch",
        **kwargs,
    ) -> None:
        super().__init__()
        self.attn_mode = attn_mode
        assert self.attn_mode == "torch", "Only support 'torch' mode for token refiner."

        self.input_embedder = nn.Linear(input_dim, feat_dim, bias=True)
        self.context_encoder = MLPEncoder(
            in_dim=feat_dim,
            feat_dim=feat_dim,
            num_layers=2,
            act_type=mlp_act_type,
        )
        self.timestep_encoder = TimestepEmbeddingEncoder(
            embedding_dim=feat_dim,
            feat_dim=feat_dim,
            act_type=mlp_act_type,
        )

        self.individual_token_refiner = IndividualTokenRefiner(
            feat_dim=feat_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            mlp_act_type=mlp_act_type,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
        )

    def forward(self, x: Tensor, t: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        timestep_aware_representations = self.timestep_encoder(t)

        if mask is None:
            context_aware_representations = x.mean(dim=1)
        else:
            mask_float = mask.float().unsqueeze(-1)
            denom = mask_float.sum(dim=1).clamp_min(1e-6)
            context_aware_representations = (x * mask_float).sum(dim=1) / denom
        context_aware_representations = self.context_encoder(context_aware_representations).unsqueeze(1)
        c = timestep_aware_representations + context_aware_representations

        x = self.input_embedder(x)

        x = self.individual_token_refiner(x, c, mask)

        return x
