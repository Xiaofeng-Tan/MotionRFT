import math
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ..utils.loaders import load_object
from ..utils.type_converter import get_module_device
from .attention import attention
from .bricks import get_activation_layer, get_norm_layer
from .encoders import MLP, MLPEncoder, TimestepEmbeddingEncoder
from .modulate_layers import ModulateDiT, apply_gate, modulate
from .positional_encoding import RotaryEmbedding


class MMBaseBlock(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        positional_encoding_cfg: dict,
        apply_rope_to_single_branch: bool,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        assert self.feat_dim % num_heads == 0, f"feat_dim {self.feat_dim} must be divisible by num_heads {num_heads}"
        self.head_dim = self.feat_dim // num_heads

        self.mlp_hidden_dim = int(self.feat_dim * mlp_ratio)

        self._positional_encoding_cfg = positional_encoding_cfg.copy()
        self.rotary_emb = RotaryEmbedding(num_feats=self.head_dim, **self._positional_encoding_cfg)
        self.apply_rope_to_single_branch = apply_rope_to_single_branch


class MMDoubleStreamBlock(MMBaseBlock):
    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        mlp_act_type: str,
        qk_norm_type: Optional[str] = None,
        qkv_bias: bool = False,
        positional_encoding_cfg: dict = {
            "max_seq_len": 5000,
            "use_real": True,
        },
        apply_rope_to_single_branch: bool = True,
    ):
        super().__init__(feat_dim, num_heads, mlp_ratio, dropout, positional_encoding_cfg, apply_rope_to_single_branch)

        self.motion_mod = ModulateDiT(
            self.feat_dim,
            factor=6,
            act_type="silu",
        )
        self.motion_norm1 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)

        motion_qkv_out_dim = self.feat_dim * 3
        self.motion_qkv = nn.Linear(self.feat_dim, motion_qkv_out_dim, bias=qkv_bias)

        self.motion_q_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.motion_k_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.motion_out_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=qkv_bias)
        self.motion_norm2 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)
        self.motion_mlp = MLP(
            self.feat_dim,
            self.mlp_hidden_dim,
            act_type=mlp_act_type,
            bias=True,
        )

        self.text_mod = ModulateDiT(
            self.feat_dim,
            factor=6,
            act_type="silu",
        )
        self.text_norm1 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)

        text_qkv_out_dim = self.feat_dim * 3
        self.text_qkv = nn.Linear(self.feat_dim, text_qkv_out_dim, bias=qkv_bias)

        self.text_q_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.text_k_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.text_out_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=qkv_bias)
        self.text_norm2 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)
        self.text_mlp = MLP(
            self.feat_dim,
            self.mlp_hidden_dim,
            act_type=mlp_act_type,
            bias=True,
        )

    def forward(
        self,
        motion_feat: Tensor,
        text_feat: Tensor,
        adapter: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        (
            motion_shift_msa,
            motion_scale_msa,
            motion_gate_msa,
            motion_shift_mlp,
            motion_scale_mlp,
            motion_gate_mlp,
        ) = self.motion_mod(adapter).chunk(6, dim=-1)
        (
            text_shift_msa,
            text_scale_msa,
            text_gate_msa,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = self.text_mod(
            adapter
        ).chunk(6, dim=-1)

        motion_modulated = self.motion_norm1(motion_feat)
        motion_modulated = modulate(motion_modulated, shift=motion_shift_msa, scale=motion_scale_msa)
        motion_qkv = self.motion_qkv(motion_modulated)

        motion_q, motion_k, motion_v = rearrange(motion_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        motion_q = self.motion_q_norm(motion_q).to(motion_v)
        motion_k = self.motion_k_norm(motion_k).to(motion_v)

        if self.apply_rope_to_single_branch:
            # NOTE: we don't apply RoPE to text_branch_two here
            motion_q, motion_k = self.rotary_emb.apply_rotary_emb(motion_q, motion_k)

        text_modulated = self.text_norm1(text_feat)
        text_modulated = modulate(text_modulated, shift=text_shift_msa, scale=text_scale_msa)
        text_qkv = self.text_qkv(text_modulated)

        text_q, text_k, text_v = rearrange(
            text_qkv,
            "B L (K H D) -> K B L H D",
            K=3,
            H=self.num_heads,
        )
        text_q = self.text_q_norm(text_q).to(text_v)
        text_k = self.text_k_norm(text_k).to(text_v)

        q = torch.cat((motion_q, text_q), dim=1)
        k = torch.cat((motion_k, text_k), dim=1)
        v = torch.cat((motion_v, text_v), dim=1)

        if not self.apply_rope_to_single_branch:
            q, k = self.rotary_emb.apply_rotary_emb(q, k)

        bsz, total_len, _, _ = q.shape
        motion_len = motion_feat.shape[1]
        text_len = text_feat.shape[1]
        dropout_p = 0.0 if not self.training else self.dropout

        attn_output = attention(
            q,
            k,
            v,
            mode="torch",
            drop_rate=dropout_p,
            attn_mask=attn_mask,
            causal=False,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            max_seqlen_q=None,
            max_seqlen_kv=None,
            batch_size=bsz,
            training=self.training,
        )

        motion_attn_output, text_attn_output = (
            attn_output[:, :motion_len, ...],
            attn_output[:, motion_len:, ...],
        )

        motion_feat = motion_feat + apply_gate(self.motion_out_proj(motion_attn_output), gate=motion_gate_msa)
        motion_feat = motion_feat + apply_gate(
            self.motion_mlp(
                modulate(
                    self.motion_norm2(motion_feat),
                    shift=motion_shift_mlp,
                    scale=motion_scale_mlp,
                )
            ),
            gate=motion_gate_mlp,
        )

        text_feat = text_feat + apply_gate(self.text_out_proj(text_attn_output), gate=text_gate_msa)
        text_feat = text_feat + apply_gate(
            self.text_mlp(
                modulate(
                    self.text_norm2(text_feat),
                    shift=text_shift_mlp,
                    scale=text_scale_mlp,
                )
            ),
            gate=text_gate_mlp,
        )

        return motion_feat, text_feat


class MMSingleStreamBlock(MMBaseBlock):
    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        mlp_act_type: str,
        qk_norm_type: Optional[str] = None,
        qkv_bias: bool = False,
        positional_encoding_cfg: dict = {
            "max_seq_len": 5000,
            "use_real": True,
        },
        apply_rope_to_single_branch: bool = True,
    ):
        super().__init__(feat_dim, num_heads, mlp_ratio, dropout, positional_encoding_cfg, apply_rope_to_single_branch)

        self.modulation = ModulateDiT(self.feat_dim, factor=3, act_type="silu")
        self.norm = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)

        # qkv and mlp_in
        qkv_factor = 3
        self.linear1 = nn.Linear(self.feat_dim, self.feat_dim * qkv_factor + self.mlp_hidden_dim, bias=qkv_bias)
        # proj and mlp_out
        self.linear2 = nn.Linear(self.feat_dim + self.mlp_hidden_dim, self.feat_dim, bias=qkv_bias)

        self.q_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.k_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)

        self.mlp_act = get_activation_layer(mlp_act_type)()

    def forward(
        self,
        x: Tensor,
        split_len: int,
        adapter: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
        ) = self.modulation(
            adapter
        ).chunk(3, dim=-1)
        x_modulated = modulate(self.norm(x), shift_msa, scale_msa)

        qkv, mlp_hidden = torch.split(self.linear1(x_modulated), [3 * self.feat_dim, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)

        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        q1, q2 = q[:, :split_len, ...], q[:, split_len:, ...]
        k1, k2 = k[:, :split_len, ...], k[:, split_len:, ...]
        # apply rotary position embedding
        if self.apply_rope_to_single_branch:
            q1, k1 = self.rotary_emb.apply_rotary_emb(q1, k1)
        q = torch.cat((q1, q2), dim=1)
        k = torch.cat((k1, k2), dim=1)
        if not self.apply_rope_to_single_branch:
            q, k = self.rotary_emb.apply_rotary_emb(q, k)

        bsz, total_len = x_modulated.shape[:2]
        dropout_p = 0.0 if not self.training else self.dropout

        attn_output = attention(
            q,
            k,
            v,
            mode="torch",
            drop_rate=dropout_p,
            attn_mask=attn_mask,
            causal=False,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            max_seqlen_q=None,
            max_seqlen_kv=None,
            batch_size=bsz,
            training=self.training,
        )
        output = self.linear2(torch.cat((attn_output, self.mlp_act(mlp_hidden)), 2))

        return x + apply_gate(output, gate=gate_msa)


class HunyuanMotionMMDiT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feat_dim: int,
        output_dim: Optional[int] = None,
        ctxt_input_dim: int = 4096,
        vtxt_input_dim: int = 256,
        text_refiner_module: str = "hymotion/network/token_refiner.SingleTokenRefiner",
        text_refiner_cfg: dict = {
            "num_layers": 2,
        },
        num_layers: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        norm_type: str = "layer",
        qk_norm_type: str = "rms",
        qkv_bias: bool = True,
        dropout: float = 0.0,
        final_layer_module: str = "hymotion/network/encoders.FinalLayer",
        final_layer_cfg: dict = {
            "act_type": "silu",
        },
        mask_mode: Optional[str] = None,
        apply_rope_to_single_branch: bool = True,
        insert_start_token: bool = False,
        with_long_skip_connection: bool = False,
        time_factor: float = 1.0,
        narrowband_length: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        self.motion_input_dim = input_dim
        self.ctxt_input_dim = ctxt_input_dim
        self.vtxt_input_dim = vtxt_input_dim
        self.feat_dim = feat_dim
        self.output_dim = output_dim or input_dim
        self.mask_mode = mask_mode
        self.insert_start_token = insert_start_token
        self.time_factor = time_factor
        self.narrowband_length = narrowband_length * 30.0
        if self.insert_start_token:
            self.start_token = nn.Parameter(torch.randn(1, feat_dim))
        self.with_long_skip_connection = with_long_skip_connection
        if self.with_long_skip_connection:
            from .encoders import FinalLayer

            self.long_skip_net = FinalLayer(feat_dim=feat_dim, out_dim=feat_dim, act_type="silu")

        self.input_encoder = nn.Linear(in_features=input_dim, out_features=feat_dim)
        self.ctxt_encoder = nn.Linear(in_features=ctxt_input_dim, out_features=feat_dim)
        self.vtxt_encoder = MLPEncoder(in_dim=vtxt_input_dim, feat_dim=feat_dim, num_layers=2, act_type="silu")
        self.timestep_encoder = TimestepEmbeddingEncoder(
            embedding_dim=feat_dim,
            feat_dim=feat_dim,
            time_factor=time_factor,
        )

        if text_refiner_module != "" and text_refiner_module is not None:
            text_refiner_cfg.update(input_dim=feat_dim, feat_dim=feat_dim, num_heads=num_heads)
            self._text_refiner_cfg = text_refiner_cfg.copy()
            self.text_refiner = load_object(text_refiner_module, text_refiner_cfg)

        self.num_layers = num_layers
        assert num_layers % 3 == 0, f"num_layers must be divisible by 3, but got {num_layers}"
        self.mm_double_blocks_layers = int(num_layers // 3)
        self.mm_single_blocks_layers = int(num_layers - num_layers // 3)

        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    feat_dim=feat_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    mlp_act_type=mlp_act_type,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    apply_rope_to_single_branch=apply_rope_to_single_branch,
                )
                for _ in range(self.mm_double_blocks_layers)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    feat_dim=feat_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    mlp_act_type=mlp_act_type,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    apply_rope_to_single_branch=apply_rope_to_single_branch,
                )
                for _ in range(self.mm_single_blocks_layers)
            ]
        )

        final_layer_cfg.update(feat_dim=feat_dim, out_dim=self.output_dim)
        self._final_layer_cfg = final_layer_cfg.copy()
        self.final_layer = load_object(final_layer_module, final_layer_cfg)

    def forward(
        self,
        x: Tensor,
        ctxt_input: Tensor,
        vtxt_input: Tensor,
        timesteps: Tensor,
        x_mask_temporal: Tensor,
        ctxt_mask_temporal: Tensor,
        **kwargs,
    ) -> Tensor:
        device = get_module_device(self)

        motion_feat = self.input_encoder(x)
        if self.with_long_skip_connection:
            origin_feat = motion_feat
        if self.insert_start_token:
            # (B, 1, D) + (B, L, D) -> (B, L+1, D)
            start_token = self.start_token[None].repeat(motion_feat.shape[0], 1, 1)
            motion_feat = torch.cat((start_token, motion_feat), dim=1)
            x_mask_temporal = torch.cat(
                [
                    torch.ones_like(x_mask_temporal[:, :1], dtype=torch.bool),
                    x_mask_temporal,
                ],
                dim=1,
            )

        timestep_feat = self.timestep_encoder(timesteps)
        vtxt_feat = self.vtxt_encoder(vtxt_input.float())
        adapter = timestep_feat + vtxt_feat

        motion_key_padding_mask = self._canonical_mask(x_mask_temporal).to(device)
        ctxt_key_padding_mask = self._canonical_mask(ctxt_mask_temporal).to(device)
        seq_key_padding_mask = torch.cat((motion_key_padding_mask, ctxt_key_padding_mask), dim=1)
        if self.mask_mode is None:
            seq_mask = None
        elif self.mask_mode == "causal":
            motion_len = motion_feat.shape[1]
            seq_mask = torch.triu(
                torch.full((motion_len, motion_len), float("-inf"), device=device),
                diagonal=1,
            )
        elif self.mask_mode == "narrowband":
            window = int(round(self.narrowband_length))
            motion_len = motion_feat.shape[1]
            idx = torch.arange(motion_len, device=device)
            dist = (idx[None, :] - idx[:, None]).abs()
            band = dist <= window
            seq_mask = torch.full((motion_len, motion_len), float("-inf"), device=device)
            seq_mask = seq_mask.masked_fill(band, 0.0)
        else:
            raise ValueError(f"Unsupported mask mode: {self.mask_mode}")

        ctxt_feat = self.ctxt_encoder(ctxt_input.float())
        if hasattr(self, "text_refiner"):
            ctxt_feat = self.text_refiner(x=ctxt_feat, t=timesteps, mask=(ctxt_key_padding_mask == 0).to(device))

        # precompute shared attention masks (broadcastable over heads)
        bsz = x.shape[0]
        motion_len = motion_feat.shape[1]
        text_len = ctxt_feat.shape[1]
        total_len = motion_len + text_len
        mask_dtype = motion_feat.dtype
        attn_mask_double = self._build_dmm_attn_mask_shared(
            bsz=bsz,
            motion_len=motion_len,
            text_len=text_len,
            dtype=mask_dtype,
            key_padding_mask=seq_key_padding_mask,
            attn_mask=seq_mask,
            device=device,
        )
        for i_layer, mod in enumerate(self.double_blocks):
            motion_feat, ctxt_feat = mod(
                motion_feat=motion_feat,
                text_feat=ctxt_feat,
                adapter=adapter,
                attn_mask=attn_mask_double,
            )

        # precompute shared attention masks for single stream blocks too
        split_len = motion_feat.shape[1]
        x = torch.cat((motion_feat, ctxt_feat), 1)
        attn_mask_single = self._build_smm_attn_mask_shared(
            bsz=bsz,
            split_len=split_len,
            total_len=total_len,
            dtype=mask_dtype,
            key_padding_mask=seq_key_padding_mask,
            attn_mask=seq_mask,
            device=device,
        )
        for i_layer, mod in enumerate(self.single_blocks):
            x = mod(
                x=x,
                split_len=split_len,
                adapter=adapter,
                attn_mask=attn_mask_single,
            )

        x = x[:, :split_len, ...]
        if self.insert_start_token:
            x = x[:, 1:, ...]

        if self.with_long_skip_connection:
            # long skip only consider timestep_feat
            x = self.long_skip_net(origin_feat, timestep_feat) + x

        predicted_res = self.final_layer(x, adapter)
        return predicted_res

    @staticmethod
    def _canonical_mask(input_mask: Tensor) -> Tensor:
        if input_mask.ndim == 1:
            input_mask = input_mask.unsqueeze(1)
        key_padding_mask = torch.where(
            input_mask,
            torch.zeros_like(input_mask, dtype=torch.float),
            torch.full_like(input_mask, float("-inf"), dtype=torch.float),
        )
        return key_padding_mask

    def _build_dmm_attn_mask_shared(
        self,
        bsz: int,
        motion_len: int,
        text_len: int,
        dtype: torch.dtype,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor],
        device: torch.device,
    ) -> Tensor:
        """
        NOTE:
                motion_k  text_k
        motion_q [M→M]   [M→T]
        text_q   [T→M]   [T→T]
        only [M→M] contains given mask
        """
        total_len = motion_len + text_len
        base = torch.zeros((bsz, 1, total_len, total_len), dtype=dtype, device=device)
        if attn_mask is not None:
            if attn_mask.dim() != 2 or attn_mask.shape != (motion_len, motion_len):
                raise RuntimeError(
                    f"attn_mask should be 2D with shape {(motion_len, motion_len)}, got {attn_mask.shape}"
                )
            base[:, :, :motion_len, :motion_len] += attn_mask.view(1, 1, motion_len, motion_len)
        if key_padding_mask is not None:
            mask_total_len = key_padding_mask.shape[1]
            if mask_total_len == motion_len:
                pad = torch.zeros((bsz, text_len), dtype=key_padding_mask.dtype, device=device)
                key_padding_mask = torch.cat((key_padding_mask, pad), dim=-1)
            base = base + key_padding_mask.view(bsz, 1, 1, total_len)
        # disable T→M
        base[:, :, motion_len:, :motion_len] = float("-inf")
        return base

    def _build_smm_attn_mask_shared(
        self,
        bsz: int,
        split_len: int,
        total_len: int,
        dtype: torch.dtype,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor],
        device: torch.device,
    ) -> Tensor:
        """
        NOTE:
                motion_k  text_k
        motion_q [M→M]   [M→T]
        text_q   [T→M]   [T→T]
        only [M→M] contains given mask
        """
        base = torch.zeros((bsz, 1, total_len, total_len), dtype=dtype, device=device)
        if attn_mask is not None:
            if attn_mask.dim() != 2 or attn_mask.shape != (split_len, split_len):
                raise RuntimeError(f"attn_mask should be 2D with shape {(split_len, split_len)}, got {attn_mask.shape}")
            base[:, :, :split_len, :split_len] += attn_mask.view(1, 1, split_len, split_len)
        if key_padding_mask is not None:
            mask_total_len = key_padding_mask.shape[1]
            if mask_total_len == split_len:
                pad = torch.zeros(
                    (bsz, total_len - split_len),
                    dtype=key_padding_mask.dtype,
                    device=device,
                )
                key_padding_mask = torch.cat((key_padding_mask, pad), dim=-1)
            base = base + key_padding_mask.view(bsz, 1, 1, total_len)
        # disable T→M
        base[:, :, split_len:, :split_len] = float("-inf")
        return base


if __name__ == "__main__":
    # python -m hymotion.network.hymotion_mmdit

    from configs._base_.model_network_base import MOTION_MODEL_CONFIG  # pyright: ignore

    network_module_cfg = MOTION_MODEL_CONFIG["1.04B_narrowband"]["network_module_args"]
    network_module_cfg = dict(network_module_cfg)  # convert to normal dict

    bsz, seq_len, text_seq_len, input_dim = 1, 360, 128, 201
    network_module_cfg["input_dim"] = input_dim
    MMDiT = HunyuanMotionMMDiT(**network_module_cfg)

    x = torch.randn(bsz, seq_len, input_dim)
    ctxt_condition = torch.randn(bsz, text_seq_len, 4096)
    vtxt_condition = torch.randn(bsz, 1, 768)
    timesteps = torch.randint(0, 1000, (bsz,))
    length = torch.arange(seq_len).unsqueeze(0).repeat(bsz, 1)
    ctxt_length = torch.arange(text_seq_len).unsqueeze(0).repeat(bsz, 1)
    x_mask_temporal = length < 100
    ctxt_mask_temporal = ctxt_length < 50
    x = MMDiT(
        x=x,
        ctxt_input=ctxt_condition,
        vtxt_input=vtxt_condition,
        timesteps=timesteps,
        x_mask_temporal=x_mask_temporal,
        ctxt_mask_temporal=ctxt_mask_temporal,
    )
    assert x.shape == (
        bsz,
        seq_len,
        input_dim,
    ), f"unexpected output shape: {x.shape}, which should be ({bsz}, {seq_len}, {input_dim})"
    print(x.shape)
