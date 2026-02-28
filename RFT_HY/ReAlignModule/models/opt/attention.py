from typing import Optional

import torch
import torch.nn as nn

from ..utils import get_clone, get_clones, get_activation_fn


class SkipTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None,
                 act: Optional[str] = None) -> None:
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm
        self.act = get_activation_fn(act)

        assert num_layers % 2 == 1

        num_block = (num_layers - 1) // 2
        self.input_blocks = get_clones(encoder_layer, num_block)
        self.middle_block = get_clone(encoder_layer)
        self.output_blocks = get_clones(encoder_layer, num_block)
        self.linear_blocks = get_clones(nn.Linear(2 * self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None) -> tuple:
        x = src
        xs = []

        for module in self.input_blocks:
            x = module(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)

        x = self.middle_block(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)


        if self.norm:
            x = self.act(self.norm(x))

        return x


class SkipTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None,
                 act: Optional[str] = None) -> None:
        super().__init__()
        self.d_model = decoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm
        self.act = get_activation_fn(act)

        assert num_layers % 2 == 1

        num_block = (num_layers - 1) // 2
        self.input_blocks = get_clones(decoder_layer, num_block)
        self.middle_block = get_clone(decoder_layer)
        self.output_blocks = get_clones(decoder_layer, num_block)
        self.linear_blocks = get_clones(nn.Linear(2 * self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None) -> tuple:
        x = tgt
        xs = []

        for module in self.input_blocks:
            x = module(x, memory, tgt_mask=tgt_mask,
                       memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                       memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
            xs.append(x)

        x = self.middle_block(x, memory, tgt_mask=tgt_mask,
                              memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, memory, tgt_mask=tgt_mask,
                       memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                       memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)

        if self.norm:
            x = self.act(self.norm(x))
        return x



class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[torch.Tensor] = None,
                     src_key_padding_mask: Optional[torch.Tensor] = None,
                     pos: Optional[torch.Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[torch.Tensor] = None,
                    src_key_padding_mask: Optional[torch.Tensor] = None,
                    pos: Optional[torch.Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[torch.Tensor] = None,
                     memory_mask: Optional[torch.Tensor] = None,
                     tgt_key_padding_mask: Optional[torch.Tensor] = None,
                     memory_key_padding_mask: Optional[torch.Tensor] = None,
                     pos: Optional[torch.Tensor] = None,
                     query_pos: Optional[torch.Tensor] = None):
                     
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[torch.Tensor] = None,
                    memory_mask: Optional[torch.Tensor] = None,
                    tgt_key_padding_mask: Optional[torch.Tensor] = None,
                    memory_key_padding_mask: Optional[torch.Tensor] = None,
                    pos: Optional[torch.Tensor] = None,
                    query_pos: Optional[torch.Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
