"""
多表征 Retrieval 模型

包含:
- MultiReprRetrieval: 支持多种 motion 表征的 Retrieval 模型
- process_T5_outputs: T5 文本编码处理函数

支持的表征类型:
- 263: HumanML3D 263维表征
- 22x3: 关节位置表征 (22 joints × 3)
- 135: FlowMDM/BABEL rot6d 表征 (1 + 2 + 22×6)
- 201: FlowMDM xyz joints + velocities 表征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import CriticMLP, AIDetectionHead
from .projections import Repr263Projection, Repr22x3Projection, Repr135Projection, Repr201Projection


def process_T5_outputs(raw_texts, T5_model):
    """处理 T5 模型输出
    
    Args:
        raw_texts: 原始文本列表
        T5_model: SentenceTransformer T5 模型
    
    Returns:
        t_length: 文本长度列表
        token_embeddings: token 级别嵌入 [B, T, 1024]
        sentence_embeddings: 句子级别嵌入 [B, 1, 1024]
    """
    with torch.no_grad():
        T5_outputs = T5_model.encode(raw_texts, show_progress_bar=False, batch_size=len(raw_texts), output_value=None)
        attn_masks = torch.stack([item['attention_mask'] for item in T5_outputs])
        token_embeddings = torch.stack([item['token_embeddings'] for item in T5_outputs])
        sentence_embeddings = torch.stack([item['sentence_embedding'] for item in T5_outputs])
        t_length = attn_masks.sum(1).tolist()
        return t_length, token_embeddings.float(), sentence_embeddings.unsqueeze(1).float()


class MultiReprRetrieval(nn.Module):
    """多表征 Retrieval 模型 - 支持可配置模型规模
    
    支持多种 motion 表征:
    - 263 维: HumanML3D 表征
    - 22x3 (关节位置): 22 joints × 3
    - 135 维: FlowMDM/BABEL rot6d 表征 (1 + 2 + 22×6)
    - 201 维: FlowMDM xyz joints + velocities 表征
    """
    def __init__(self, t5_path, temp=0.1, thr=0.9, 
                 latent_dim=256, unified_dim=512,
                 encoder_num_layers=9, encoder_num_heads=4, encoder_ff_size=1024,
                 text_num_layers=9, text_num_heads=4, text_ff_size=1024,
                 proj_hidden_dim=512, proj_num_layers=3, proj_dropout=0.1,
                 use_unified_dim=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.unified_dim = unified_dim
        self.use_unified_dim = use_unified_dim
        self.temp = temp
        self.thr = thr
        
        encoder_dim = unified_dim if use_unified_dim else latent_dim
        
        self.proj_263 = Repr263Projection(
            input_dim=263, unified_dim=encoder_dim,
            hidden_dim=proj_hidden_dim, num_layers=proj_num_layers, dropout=proj_dropout
        )
        self.proj_22x3 = Repr22x3Projection(
            num_joints=22, unified_dim=encoder_dim,
            hidden_dim=proj_hidden_dim, num_layers=proj_num_layers, dropout=proj_dropout
        )
        self.proj_135 = Repr135Projection(
            input_dim=135, unified_dim=encoder_dim,
            hidden_dim=proj_hidden_dim, num_layers=proj_num_layers, dropout=proj_dropout
        )
        self.proj_201 = Repr201Projection(
            input_dim=201, unified_dim=encoder_dim,
            hidden_dim=proj_hidden_dim, num_layers=proj_num_layers, dropout=proj_dropout
        )
        
        from sentence_transformers import SentenceTransformer
        self.clip = SentenceTransformer(t5_path)
        
        from .opt.attention import SkipTransformerEncoder, TransformerEncoderLayer
        from .opt.position_encoding import build_position_encoding
        from .opt.embeddings import TimestepEmbedding, Timesteps
        
        self.query_pos_encoder = build_position_encoding(encoder_dim, position_embedding='learned')
        encoder_layer = TransformerEncoderLayer(
            encoder_dim, encoder_num_heads, encoder_ff_size, 0.1, 'gelu', False
        )
        encoder_norm = nn.LayerNorm(encoder_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, encoder_num_layers, encoder_norm)
        self.global_motion_token = nn.Parameter(torch.randn(1, encoder_dim))
        
        self.dist_layer = nn.Linear(encoder_dim, 2 * latent_dim)
        
        self.text_embedding = nn.Linear(1024, latent_dim)
        self.query_text_pos_encoder = build_position_encoding(latent_dim, position_embedding='learned')
        text_encoder_layer = TransformerEncoderLayer(
            latent_dim, text_num_heads, text_ff_size, 0.1, 'gelu', False
        )
        text_encoder_norm = nn.LayerNorm(latent_dim)
        self.text_encoder = SkipTransformerEncoder(text_encoder_layer, text_num_layers, text_encoder_norm)
        self.global_text_token = nn.Parameter(torch.randn(1, latent_dim))
        self.dist_text_layer = nn.Linear(latent_dim, 2 * latent_dim)
        
        self.time_proj = Timesteps(512, True, 0)
        self.time_embedding = TimestepEmbedding(512, encoder_dim)
        self.time_embedding_text = TimestepEmbedding(512, latent_dim)
        
        from .opt.attention import SkipTransformerDecoder, TransformerDecoderLayer
        self.query_pos_decoder = build_position_encoding(latent_dim, position_embedding='learned')
        decoder_layer = TransformerDecoderLayer(
            latent_dim, encoder_num_heads, encoder_ff_size, 0.1, 'gelu', False
        )
        decoder_norm = nn.LayerNorm(latent_dim)
        self.decoder = SkipTransformerDecoder(decoder_layer, encoder_num_layers, decoder_norm)
        
        self.final_layer_263 = nn.Linear(latent_dim, 263)
        self.final_layer_22x3 = nn.Linear(latent_dim, 22 * 3)
        self.final_layer_135 = nn.Linear(latent_dim, 135)
        self.final_layer_201 = nn.Linear(latent_dim, 201)
        
        self.recons_loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.latent_loss_fn = nn.SmoothL1Loss(reduction='mean')
        
        self.critic_head = None
        self.ai_detection_head = None
    
    def init_critic_head(self, hidden_dim=None):
        hidden_dim = hidden_dim or self.latent_dim
        self.critic_head = CriticMLP(
            in_features=self.latent_dim, hidden_features=hidden_dim, out_features=1, drop=0.1
        )
        return self.critic_head
    
    def init_ai_detection_head(self, hidden_dim=None, num_classes=2):
        hidden_dim = hidden_dim or self.latent_dim
        self.ai_detection_head = AIDetectionHead(
            in_features=self.latent_dim, hidden_features=hidden_dim, num_classes=num_classes, drop=0.1
        )
        return self.ai_detection_head
    
    def project_motion(self, motion, repr_type):
        if repr_type == '263':
            return self.proj_263(motion)
        elif repr_type == '22x3':
            return self.proj_22x3(motion)
        elif repr_type == '135':
            return self.proj_135(motion)
        elif repr_type == '201':
            return self.proj_201(motion)
        else:
            raise ValueError(f"Unknown repr_type: {repr_type}")
    
    def encode_motion(self, motion, lengths, repr_type='263', timestep=None):
        device = motion.device
        bs = motion.shape[0]
        if lengths is None:
            lengths = [motion.shape[1]] * bs
        x = self.project_motion(motion, repr_type)
        mask = self._lengths_to_mask(lengths, device, max_len=x.shape[1])
        x = x.permute(1, 0, 2)
        dist = self.global_motion_token.unsqueeze(1).expand(-1, bs, -1)
        dist_masks = torch.ones((bs, 1), dtype=bool, device=device)
        if timestep is not None:
            timesteps = timestep.expand(bs).clone()
            time_emb = self.time_proj(timesteps)
            time_emb = time_emb.to(dtype=x.dtype)
            time_emb = self.time_embedding(time_emb).unsqueeze(0)
            xseq = torch.cat((dist, time_emb, x), 0)
            time_masks = torch.ones((bs, 1), dtype=bool, device=device)
            aug_mask = torch.cat((dist_masks, time_masks, mask), 1)
        else:
            xseq = torch.cat((dist, x), 0)
            aug_mask = torch.cat((dist_masks, mask), 1)
        xseq = self.query_pos_encoder(xseq)
        dist_out = self.encoder(xseq, src_key_padding_mask=~aug_mask)[:1]
        tokens_dist = self.dist_layer(dist_out)
        mu = tokens_dist[:, :, :self.latent_dim]
        logvar = tokens_dist[:, :, self.latent_dim:]
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        return latent, dist
    
    def encode_text(self, features, lengths=None, timestep=None):
        device = features.device
        bs, nbtokens, nfeats = features.shape
        if lengths is None:
            lengths = [nbtokens] * bs
        mask = self._lengths_to_mask(lengths, device, max_len=nbtokens)
        x = self.text_embedding(features)
        x = x.permute(1, 0, 2)
        dist = self.global_text_token.unsqueeze(1).expand(-1, bs, -1)
        dist_masks = torch.ones((bs, 1), dtype=bool, device=device)
        if timestep is not None:
            timesteps = timestep.expand(bs).clone()
            time_emb = self.time_proj(timesteps)
            time_emb = time_emb.to(dtype=x.dtype)
            time_emb = self.time_embedding_text(time_emb).unsqueeze(0)
            xseq = torch.cat((dist, time_emb, x), 0)
            time_masks = torch.ones((bs, 1), dtype=bool, device=device)
            aug_mask = torch.cat((dist_masks, time_masks, mask), 1)
        else:
            xseq = torch.cat((dist, x), 0)
            aug_mask = torch.cat((dist_masks, mask), 1)
        xseq = self.query_text_pos_encoder(xseq)
        dist_out = self.text_encoder(xseq, src_key_padding_mask=~aug_mask)[:1]
        tokens_dist = self.dist_text_layer(dist_out)
        mu = tokens_dist[:, :, :self.latent_dim]
        logvar = tokens_dist[:, :, self.latent_dim:]
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        return latent, dist
    
    def decode(self, z, lengths, max_motion_length, repr_type='263'):
        device = z.device
        bs = z.shape[1]
        max_len = max_motion_length
        mask = self._lengths_to_mask(lengths, device, max_len=max_len)
        queries = torch.zeros(max_len, bs, self.latent_dim, device=device, requires_grad=True)
        queries = self.query_pos_decoder(queries)
        output = self.decoder(tgt=queries, memory=z, tgt_key_padding_mask=~mask)
        if repr_type == '263':
            output = self.final_layer_263(output)
            output = output.permute(1, 0, 2)
        elif repr_type == '135':
            output = self.final_layer_135(output)
            output = output.permute(1, 0, 2)
        elif repr_type == '201':
            output = self.final_layer_201(output)
            output = output.permute(1, 0, 2)
        else:
            output = self.final_layer_22x3(output)
            output = output.permute(1, 0, 2)
            output = output.view(output.shape[0], output.shape[1], -1, 3)
        for i, length in enumerate(lengths):
            output[i, length:] = 0
        return output
    
    def forward(self, text, motion_feature, m_len, repr_type='263', timestep=None, mode='M1T0'):
        with torch.no_grad():
            t_len, sent_emb, cls_token = process_T5_outputs(text, self.clip)
        if mode == 'M1T0':
            t_latent, t_dist = self.encode_text(sent_emb, t_len)
            m_latent, m_dist = self.encode_motion(motion_feature, m_len, repr_type=repr_type, timestep=timestep)
        elif mode == 'M0T1':
            t_latent, t_dist = self.encode_text(sent_emb, t_len, timestep=timestep)
            m_latent, m_dist = self.encode_motion(motion_feature, m_len, repr_type=repr_type)
        elif mode == 'M1T1':
            t_latent, t_dist = self.encode_text(sent_emb, t_len, timestep=timestep)
            m_latent, m_dist = self.encode_motion(motion_feature, m_len, repr_type=repr_type, timestep=timestep)
        else:
            t_latent, t_dist = self.encode_text(sent_emb, t_len)
            m_latent, m_dist = self.encode_motion(motion_feature, m_len, repr_type=repr_type)
        max_motion_length = motion_feature.shape[1]
        m_rst = self.decode(m_latent, m_len, max_motion_length, repr_type=repr_type)
        t_rst = self.decode(t_latent, m_len, max_motion_length, repr_type=repr_type)
        recons_loss = self.recons_loss_fn(t_rst, motion_feature) + self.recons_loss_fn(m_rst, motion_feature) + self.recons_loss_fn(m_rst, t_rst)
        m_dist_params = [m_dist.loc, 2 * torch.log(m_dist.scale)]
        t_dist_params = [t_dist.loc, 2 * torch.log(t_dist.scale)]
        ref_mus = torch.zeros_like(m_dist_params[0])
        ref_logvar = torch.zeros_like(t_dist_params[1])
        ref_dist = (ref_mus, ref_logvar)
        kl_loss = (
            self._kl_loss(t_dist_params, m_dist_params) +
            self._kl_loss(m_dist_params, t_dist_params) +
            self._kl_loss(m_dist_params, ref_dist) +
            self._kl_loss(t_dist_params, ref_dist)
        )
        latent_loss = self.latent_loss_fn(m_latent, t_latent)
        cl_loss = self._infonce_loss(t_latent.squeeze(0), m_latent.squeeze(0), cls_token.squeeze(1))
        total_loss = recons_loss * 1e0 + kl_loss * 1e-5 + latent_loss * 1e-5 + cl_loss * 1e-1
        return total_loss
    
    def forward_cross_repr(self, motion_263, motion_22x3, m_len, timestep=None):
        m_latent_263, m_dist_263 = self.encode_motion(motion_263, m_len, repr_type='263', timestep=timestep)
        m_latent_22x3, m_dist_22x3 = self.encode_motion(motion_22x3, m_len, repr_type='22x3', timestep=timestep)
        latent_align_loss = self.latent_loss_fn(m_latent_263, m_latent_22x3)
        dist_263_params = [m_dist_263.loc, 2 * torch.log(m_dist_263.scale)]
        dist_22x3_params = [m_dist_22x3.loc, 2 * torch.log(m_dist_22x3.scale)]
        kl_align_loss = self._kl_loss(dist_263_params, dist_22x3_params) + self._kl_loss(dist_22x3_params, dist_263_params)
        cl_align_loss = self._infonce_loss(m_latent_263.squeeze(0), m_latent_22x3.squeeze(0))
        cross_repr_loss = latent_align_loss * 1e-1 + kl_align_loss * 1e-5 + cl_align_loss * 1e-1
        return cross_repr_loss, {'latent_align': latent_align_loss.item(), 'kl_align': kl_align_loss.item(), 'cl_align': cl_align_loss.item()}
    
    def forward_critic(self, batch_data):
        if self.critic_head is None:
            return None
        motion_better = batch_data['motion_better']
        motion_worse = batch_data['motion_worse']
        repr_type = batch_data['repr_type']
        bs = motion_better.shape[0]
        lengths = [motion_better.shape[1]] * bs
        latent_better, _ = self.encode_motion(motion_better, lengths, repr_type=repr_type)
        latent_worse, _ = self.encode_motion(motion_worse, lengths, repr_type=repr_type)
        latent_better = latent_better.squeeze(0)
        latent_worse = latent_worse.squeeze(0)
        score_better = self.critic_head(latent_better)
        score_worse = self.critic_head(latent_worse)
        critic = torch.cat([score_better, score_worse], dim=1)
        return critic
    
    def forward_ai_detection(self, batch_data):
        if self.ai_detection_head is None:
            return None, None
        motion = batch_data['motion']
        labels = batch_data['label']
        repr_type = batch_data['repr_type']
        lengths = batch_data['length']
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.tolist()
        latent, _ = self.encode_motion(motion, lengths, repr_type=repr_type)
        latent = latent.squeeze(0)
        logits = self.ai_detection_head(latent)
        return logits, labels
    
    def get_critic_score(self, motion, lengths, repr_type='263'):
        if self.critic_head is None:
            raise RuntimeError("Critic head not initialized")
        with torch.no_grad():
            latent, _ = self.encode_motion(motion, lengths, repr_type=repr_type)
        latent = latent.squeeze(0)
        score = self.critic_head(latent)
        return score
    
    def get_ai_detection_score(self, motion, lengths, repr_type='263'):
        if self.ai_detection_head is None:
            raise RuntimeError("AI detection head not initialized")
        with torch.no_grad():
            latent, _ = self.encode_motion(motion, lengths, repr_type=repr_type)
        latent = latent.squeeze(0)
        logits = self.ai_detection_head(latent)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]
    
    def get_reward_t2m(self, raw_texts, motion_feats, t_len=None, token_emb=None, 
                       m_len=None, timestep=None, repr_type='135', mode='M0T0'):
        """计算 text-to-motion reward (cosine similarity)
        
        Args:
            raw_texts: 原始文本列表
            motion_feats: motion 特征 [B, T, D]
            t_len: 文本长度列表 (可选，如果 token_emb 为 None 则自动计算)
            token_emb: token embeddings [B, T, 1024] (可选，如果为 None 则自动计算)
            m_len: motion 长度列表
            timestep: 时间步 (可选)
            repr_type: motion 表征类型
            mode: step-aware 模式 ('M0T0', 'M1T0', 'M0T1', 'M1T1')
        
        Returns:
            reward: [B] cosine similarity
        """
        if token_emb is None:
            with torch.no_grad():
                t_len, token_emb, _ = process_T5_outputs(raw_texts, self.clip)
        if mode in ['M0T1', 'M1T1']:
            t_latent, _ = self.encode_text(token_emb, t_len, timestep=timestep)
        else:
            t_latent, _ = self.encode_text(token_emb, t_len)
        if mode in ['M1T0', 'M1T1']:
            m_latent, _ = self.encode_motion(motion_feats, m_len, repr_type=repr_type, timestep=timestep)
        else:
            m_latent, _ = self.encode_motion(motion_feats, m_len, repr_type=repr_type)
        reward = F.cosine_similarity(t_latent.squeeze(0), m_latent.squeeze(0), dim=-1)
        return reward
    
    def _lengths_to_mask(self, lengths, device, max_len=None):
        # 兼容 list 和 tensor 输入
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths, device=device)
        else:
            lengths = lengths.to(device)
        max_len = max_len if max_len else max(lengths)
        mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        return mask
    
    def _kl_loss(self, q, p):
        mu_q, logvar_q = q
        mu_p, logvar_p = p
        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()
    
    def _infonce_loss(self, x, y, sent_emb=None):
        bs, device = len(x), x.device
        x_logits = F.normalize(x, dim=-1)
        y_logits = F.normalize(y, dim=-1)
        sim_matrix = (x_logits @ y_logits.transpose(-2, -1)) / self.temp
        if sent_emb is not None and self.thr:
            real_threshold_selfsim = 2 * self.thr - 1
            sent_emb_normalized = F.normalize(sent_emb, dim=-1)
            selfsim = sent_emb_normalized @ sent_emb_normalized.transpose(-2, -1)
            selfsim_nodiag = selfsim - torch.diag(torch.diag(selfsim))
            idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            sim_matrix[idx] = -torch.inf
        labels = torch.arange(bs, device=device)
        total_loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.transpose(-2, -1), labels)) / 2
        return total_loss
