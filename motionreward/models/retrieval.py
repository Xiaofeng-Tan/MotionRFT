import torch, pickle, random
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
from typing import List, Optional, Union
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.distribution import Distribution
from sentence_transformers import SentenceTransformer



from .utils import lengths_to_mask, mld_collate_motion_only
from .opt.attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from .opt.position_encoding import build_position_encoding
from .opt.embeddings import TimestepEmbedding, Timesteps


class KLLoss:
    def __call__(self, q, p):
        mu_q, logvar_q = q
        mu_p, logvar_p = p

        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"

class InfoNCE_with_filtering:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim
        self.a, self.b = 0, 1
    def get_sim_matrix(self, x, y):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix

    def __call__(self, x, y, sent_emb=None):
        bs, device = len(x), x.device
        sim_matrix = self.get_sim_matrix(x, y) / self.temperature

        if sent_emb is not None and self.threshold_selfsim:
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            sent_emb_normalized = torch.nn.functional.normalize(sent_emb, dim=-1)
            selfsim = sent_emb_normalized @ sent_emb_normalized.T
            
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            sim_matrix[idx] = -torch.inf

        labels = torch.arange(bs, device=device)

        total_loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2

        return total_loss

    def __repr__(self):
        return f"Constrastive(temp={self.temperature})"



def process_T5_outputs(raw_texts, T5_model):
    with torch.no_grad():
        T5_outputs = T5_model.encode(raw_texts, show_progress_bar=False, batch_size=len(raw_texts), output_value=None)
        
        attn_masks = torch.stack([item['attention_mask'] for item in T5_outputs])
        token_embeddings = torch.stack([item['token_embeddings'] for item in T5_outputs])
        sentence_embeddings = torch.stack([item['sentence_embedding'] for item in T5_outputs])
        t_length = attn_masks.sum(1).tolist()
        return t_length, token_embeddings.float(), sentence_embeddings.unsqueeze(1).float()
    
    
    
    
def load_Retrieval(ckpt_path, model):
    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    from collections import OrderedDict
    retrieval_dict, retrieval_sum = OrderedDict(), 0.0
    for k, v in state_dict.items():
        if k.split(".")[0] == "xlm" or k.split(".")[0] == "clip":
            continue
        retrieval_sum += torch.sum(v).item()
        retrieval_dict[k] = v
    model.load_state_dict(retrieval_dict, strict=False)
    print(f'Retrieval Loading from {ckpt_path}, Note that Model exclude (CLIP, T5, BERT) ! ! !')

class Retrieval(nn.Module):

    def __init__(self,
                 nfeats: int = 263,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "encoder_decoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 temp: float = 0.1,
                 thr: float = 0.9,
                 t5_path: str = None,
                 pretrained_path: str = None,
                 lambda_t2m=0, 
                 lambda_m2m=0,
                 retrieval_mode=None,
                 **kwargs) -> None:

        super().__init__()
        if t5_path is not None:
            self.clip = SentenceTransformer(t5_path)
        else:
            self.clip = None
        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]

        self.arch = arch
        self.mlp_dist = 'mld'
        self.pe_type = 'mld'

        
        self.query_motion_pos_encoder = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.query_text_pos_encoder = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.query_pos_decoder = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)


        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.motion_encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)
        self.text_encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)
        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                                  decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)
        else:
            raise ValueError("Not support architecture!")

        if self.mlp_dist:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim))
            self.dist_motion_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)

            self.global_text_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim))
            self.dist_text_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)

        else:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))
            self.global_text_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))
        self.skel_embedding = nn.Linear(nfeats, self.latent_dim)
        self.text_embedding = nn.Linear(1024, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, nfeats)
        self.time_proj = Timesteps(512, True, 0)
        self.time_embedding = TimestepEmbedding(512, 256)
        
        
        self.retrieval_mode = retrieval_mode
        if pretrained_path is not None:
            print('================Construct Retrieval Start================')
            self.pretrained_path = pretrained_path.split('/')[-1].replace('.pth', '')
            load_Retrieval(pretrained_path, self)
            if 'M1T1' in pretrained_path:
                self.retrieval_mode = 'M1T1'
            elif 'M1T0' in pretrained_path:
                self.retrieval_mode = 'M1T0'
            elif 'M0T0' in pretrained_path:
                self.retrieval_mode = 'M0T0'
            elif 'M0T1' in pretrained_path:
                self.retrieval_mode = 'M0T1'
            
            self.lambda_t2m = lambda_t2m
            self.lambda_m2m = lambda_m2m
            print(f'Init Retrieval with [{pretrained_path}], Retrieval Mode = [{self.retrieval_mode}]\n', \
                f'(LAMBDA_T2M, LAMDA_M2M) = ({self.lambda_t2m}, {self.lambda_m2m})\n', \
                f'================Construct Retrieval Eendd================')
        if pretrained_path is None:
            self.recons_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
            self.latent_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
            self.kl_loss_fn = KLLoss()
            self.cl_loss_fn = InfoNCE_with_filtering(temperature=temp, threshold_selfsim=thr)
            print(f'self.cl_loss_fn = InfoNCE_with_filtering(temperature={temp}, threshold_selfsim={thr})')

    def encode_motion(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None,
            timestep=None,
            return_time=False
    ) -> Union[Tensor, Distribution]:

        if lengths is None:
            lengths = [len(feature) for feature in features]
        
        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device, max_len=nframes)

        x = features
        x = self.skel_embedding(x)
        x = x.permute(1, 0, 2)

        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        if timestep is not None:
            timesteps = timestep.expand(x.shape[1]).clone()
            time_emb = self.time_proj(timesteps)
            time_emb = time_emb.to(dtype=x.dtype)
            time_emb = self.time_embedding(time_emb).unsqueeze(0)
            xseq = torch.cat((dist, time_emb, x), 0)
            time_masks = torch.ones((bs,time_emb.shape[0]),
                                dtype=bool,
                                device=x.device)
            aug_mask = torch.cat((dist_masks, time_masks, mask), 1)
        else:
            xseq = torch.cat((dist, x), 0)
            aug_mask = torch.cat((dist_masks, mask), 1)

        xseq = self.query_motion_pos_encoder(xseq)
        dist = self.motion_encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]

        if self.mlp_dist:
            tokens_dist = self.dist_motion_layer(dist)
            mu = tokens_dist[:, :, :self.latent_dim]
            logvar = tokens_dist[:, :, self.latent_dim:]
        else:
            mu = dist[0:self.latent_size, ...]
            logvar = dist[self.latent_size:, ...]

        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        if return_time:
            return latent, dist
        return latent, dist

    def decode(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device, requires_grad=True)

        if self.arch == "all_encoder":
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones((bs, self.latent_size), dtype=bool, device=z.device)
            augmask = torch.cat((z_mask, mask), axis=1)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(xseq, src_key_padding_mask=~augmask)[z.shape[0]:]

        elif self.arch == "encoder_decoder":
            queries = self.query_pos_decoder(queries)
            output = self.decoder(tgt=queries, memory=z, tgt_key_padding_mask=~mask).squeeze(0)

        output = self.final_layer(output)
        output[~mask.T] = 0
        feats = output.permute(1, 0, 2)
        return feats

    def encode_text(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None,
            timestep = None,
            return_time=False
    ) -> Union[Tensor, Distribution]:

        device = features.device

        bs, nbtokens, nfeats = features.shape
        mask = lengths_to_mask(lengths, device, max_len=nbtokens)

        x = features
        x = self.text_embedding(x)
        x = x.permute(1, 0, 2)

        dist = torch.tile(self.global_text_token[:, None, :], (1, bs, 1))

        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        
        if timestep is not None:
            timesteps = timestep.expand(x.shape[1]).clone()
            time_emb = self.time_proj(timesteps)
            time_emb = time_emb.to(dtype=x.dtype)
            time_emb = self.time_embedding(time_emb).unsqueeze(0)
            xseq = torch.cat((dist, time_emb, x), 0)
            time_masks = torch.ones((bs,time_emb.shape[0]),
                                dtype=bool,
                                device=x.device)
            aug_mask = torch.cat((dist_masks, time_masks, mask), 1)
        else:
            xseq = torch.cat((dist, x), 0)
            aug_mask = torch.cat((dist_masks, mask), 1)

        xseq = self.query_text_pos_encoder(xseq)
        dist = self.text_encoder(xseq,
                            src_key_padding_mask=~aug_mask)[:dist.shape[0]]

        if self.mlp_dist:
            tokens_dist = self.dist_text_layer(dist)
            mu = tokens_dist[:, :, :self.latent_dim]
            logvar = tokens_dist[:, :, self.latent_dim:]
        else:
            mu = dist[0:self.latent_size, ...]
            logvar = dist[self.latent_size:, ...]

        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        if return_time:
            return latent, dist
        return latent, dist

    def compute_loss(self, m_rst, t_rst, m_ref, m_dist, t_dist, m_latent, t_latent, sent_emb, joints_ref=None, joints_rst_m=None, joints_rst_t=None, t_time_embed=None, m_time_embed=None):
        recons_loss = self.recons_loss_fn(t_rst, m_ref) + self.recons_loss_fn(m_rst, m_ref) + self.recons_loss_fn(m_rst, t_rst) 
        
        if joints_ref is not None and joints_rst_m is not None and joints_rst_t is not None:
            recons_loss += self.recons_loss_fn(joints_rst_m, joints_ref) + self.recons_loss_fn(joints_rst_t, joints_ref) + self.recons_loss_fn(joints_rst_t, joints_rst_m)
        
        if t_time_embed is not None and  m_time_embed is not None:
            recons_loss += self.recons_loss_fn(t_time_embed, m_time_embed) * 0.1
            
        m_dist = [m_dist.loc, 2 * torch.log(m_dist.scale)]
        t_dist = [t_dist.loc, 2 * torch.log(t_dist.scale)]
        ref_mus = torch.zeros_like(m_dist[0])
        ref_logvar = torch.zeros_like(t_dist[1])
        ref_dist = (ref_mus, ref_logvar)
        kl_loss = (
            self.kl_loss_fn(t_dist, m_dist)
            + self.kl_loss_fn(m_dist, t_dist)
            + self.kl_loss_fn(m_dist, ref_dist)
            + self.kl_loss_fn(t_dist, ref_dist)
        )
        latent_loss = self.latent_loss_fn(m_latent, t_latent)
        cl_loss = self.cl_loss_fn(t_latent.squeeze(), m_latent.squeeze(), sent_emb)
        
        return recons_loss*1e0 + kl_loss*1e-5 + latent_loss*1e-5 + cl_loss*1e-1
    
    
    def forward(self, motion_feature, text: Tensor, m_len: Optional[List[int]] = None, timestep=None, mode='M0T0', eval_tmr=False):
        with torch.no_grad():
            t_len, sent_emb, cls_token = process_T5_outputs(text, self.clip)

        if mode == 'M1T0':
            t_latent, t_dist = self.encode_text(sent_emb, t_len)
            m_latent, m_dist = self.encode_motion(motion_feature, m_len, timestep=timestep)
        elif mode == 'M0T1':
            t_latent, t_dist = self.encode_text(sent_emb, t_len, timestep=timestep)
            m_latent, m_dist = self.encode_motion(motion_feature, m_len)
        elif mode == 'M1T1':
            t_latent, t_dist = self.encode_text(sent_emb, t_len, timestep=timestep)
            m_latent, m_dist = self.encode_motion(motion_feature, m_len, timestep=timestep)
        elif mode == 'M0T0':
            t_latent, t_dist = self.encode_text(sent_emb, t_len)
            m_latent, m_dist = self.encode_motion(motion_feature, m_len)
        m_rst = self.decode(m_latent, m_len)
        t_rst = self.decode(t_latent, m_len)
        joints_ref = 0
        joints_rst_m = 0
        joints_rst_t = 0
        loss = self.compute_loss(m_rst, t_rst, motion_feature, m_dist, t_dist, m_latent, t_latent, cls_token.squeeze())

        
        if eval_tmr: 
            return m_latent, t_latent
        return loss
