"""
使用 Critic Reward 微调 MLD 模型

基于 ft_mld.py，支持 Stage 2 Critic reward 作为训练信号
Critic reward 使用 Critic Head 输出的分数，而非 text-motion 相似度

Checkpoint 文件:
- backbone: *_spm_backbone_*.pth 或 *_retrieval_backbone_*.pth
- critic_lora: *_critic_lora_*.pth
- critic_head: *_critic_head_*.pth
"""
import os
import inspect
import sys
import logging
import datetime
import os.path as osp
import glob
import random

from tqdm.auto import tqdm
from omegaconf import OmegaConf

import torch
import swanlab
import diffusers
import transformers
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_scheduler

from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.utils.utils import print_table, set_seed, move_batch_to_device
from mld.data.humanml.utils.plot_script import plot_3d_motion

from ft_config import get_ft_config

# 添加 motionreward 到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motionreward.models import MultiReprRetrievalWithLoRA
from motionreward.utils.config_utils import get_model_config
from motionreward.models.lora_modules import load_lora_state_dict, enable_lora
from motionreward.evaluation.retrieval_metrics import calculate_retrieval_metrics, calculate_retrieval_metrics_small_batches
from motionreward.evaluation.critic_eval import eval_critic
from motionreward.datasets.critic_datasets import CriticPairDataset, CriticReprTypeBatchSampler, critic_collate_fn
from motionreward.utils.data_utils import load_normalization_stats
from motionreward.datasets.retrieval_datasets import Text2MotionDataset263, Text2MotionDataset135, JointLevelText2MotionDataset, retrieval_collate_fn

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CriticRewardAdapter(nn.Module):
    """
    Critic Reward 适配器 - 使用多阶段模型计算 reward
    
    支持四种 reward:
    - Critic reward: 使用 Critic head 输出的分数 (Stage 2)
    - Retrieval reward: 使用 text-motion 相似度 (Stage 1)
    - M2M reward: 使用 gt motion 和 generated motion 的余弦相似度
    - AI Detection reward: 使用 AI Detection head 输出的分数 (Stage 3)
    
    Checkpoint 加载顺序:
    1. 加载 Stage 1 backbone (spm_backbone 或 retrieval_backbone)
    2. 注入 Critic LoRA + 加载权重
    3. 注入 AI Detection LoRA + 加载权重 (如果需要)
    """
    
    def __init__(
        self,
        backbone_ckpt: str = None,      # Stage 1 backbone checkpoint
        critic_lora_ckpt: str = None,   # Stage 2 Critic LoRA checkpoint
        critic_head_ckpt: str = None,   # Stage 2 Critic Head checkpoint
        ai_detection_lora_ckpt: str = None,   # Stage 3 AI Detection LoRA checkpoint
        ai_detection_head_ckpt: str = None,   # Stage 3 AI Detection Head checkpoint
        t5_path: str = '../deps/sentence-t5-large',
        repr_type: str = '263',
        model_size: str = 'small',
        lora_rank: int = 128,
        lora_alpha: int = 256,
        lora_dropout: float = 0.1,
        device: str = 'cuda',
        lambda_critic: float = 1.0,     # Critic reward 权重
        lambda_retrieval: float = 0.0,  # Retrieval reward 权重
        lambda_m2m: float = 0.0,        # M2M reward 权重
        lambda_ai_detection: float = 0.0,  # AI Detection reward 权重
        **kwargs
    ):
        super().__init__()
        
        self.repr_type = repr_type
        self.device = device
        self.model_size = model_size
        self.lambda_critic = lambda_critic
        self.lambda_retrieval = lambda_retrieval
        self.lambda_m2m = lambda_m2m
        self.lambda_ai_detection = lambda_ai_detection
        
        # 1. 通过 model_size 获取模型配置
        model_cfg = get_model_config(model_size)
        print(f"[CriticRewardAdapter] Using model_size='{model_size}' config:")
        print(f"  latent_dim={model_cfg['latent_dim']}, unified_dim={model_cfg['unified_dim']}")
        print(f"  encoder_num_layers={model_cfg['encoder_num_layers']}, text_num_layers={model_cfg['text_num_layers']}")
        print(f"  lambda_critic={lambda_critic}, lambda_retrieval={lambda_retrieval}, lambda_m2m={lambda_m2m}, lambda_ai_detection={lambda_ai_detection}")
        
        # 2. 创建 MultiReprRetrievalWithLoRA 模型
        self.model = MultiReprRetrievalWithLoRA(
            t5_path=t5_path,
            temp=0.1,
            thr=0.8,
            latent_dim=model_cfg['latent_dim'],
            unified_dim=model_cfg['unified_dim'],
            encoder_num_layers=model_cfg['encoder_num_layers'],
            encoder_num_heads=model_cfg['encoder_num_heads'],
            encoder_ff_size=model_cfg['encoder_ff_size'],
            text_num_layers=model_cfg['text_num_layers'],
            text_num_heads=model_cfg['text_num_heads'],
            text_ff_size=model_cfg['text_ff_size'],
            proj_hidden_dim=model_cfg['proj_hidden_dim'],
            proj_num_layers=model_cfg['proj_num_layers'],
            use_unified_dim=True,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        # 3. 加载 Stage 1 backbone
        if backbone_ckpt and os.path.exists(backbone_ckpt):
            print(f"[CriticRewardAdapter] Loading backbone from: {backbone_ckpt}")
            checkpoint = torch.load(backbone_ckpt, map_location='cpu', weights_only=False)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            print(f"[CriticRewardAdapter] Backbone loaded successfully")
        else:
            print(f"[CriticRewardAdapter] Warning: Backbone checkpoint not found: {backbone_ckpt}")
        
        # 4. 注入 Critic LoRA
        print(f"[CriticRewardAdapter] Injecting Critic LoRA (rank={lora_rank}, alpha={lora_alpha})...")
        self.model.inject_critic_lora()
        
        # 5. 加载 Critic LoRA 权重
        if critic_lora_ckpt and os.path.exists(critic_lora_ckpt):
            print(f"[CriticRewardAdapter] Loading Critic LoRA from: {critic_lora_ckpt}")
            lora_ckpt = torch.load(critic_lora_ckpt, map_location='cpu', weights_only=False)
            if 'lora_state_dict' in lora_ckpt:
                load_lora_state_dict(self.model.critic_lora_modules, lora_ckpt['lora_state_dict'])
            else:
                load_lora_state_dict(self.model.critic_lora_modules, lora_ckpt)
            print(f"[CriticRewardAdapter] Critic LoRA loaded successfully")
        else:
            print(f"[CriticRewardAdapter] Warning: Critic LoRA checkpoint not found: {critic_lora_ckpt}")
        
        # 6. 加载 Critic Head 权重
        if critic_head_ckpt and os.path.exists(critic_head_ckpt):
            print(f"[CriticRewardAdapter] Loading Critic Head from: {critic_head_ckpt}")
            head_ckpt = torch.load(critic_head_ckpt, map_location='cpu', weights_only=False)
            if 'head_state_dict' in head_ckpt:
                self.model.critic_head.load_state_dict(head_ckpt['head_state_dict'])
            elif 'state_dict' in head_ckpt:
                self.model.critic_head.load_state_dict(head_ckpt['state_dict'])
            else:
                self.model.critic_head.load_state_dict(head_ckpt)
            print(f"[CriticRewardAdapter] Critic Head loaded successfully")
        else:
            print(f"[CriticRewardAdapter] Warning: Critic Head checkpoint not found: {critic_head_ckpt}")
        
        # 确保 Critic LoRA 启用
        enable_lora(self.model.critic_lora_modules)
        
        # 7. 注入 AI Detection LoRA (如果需要)
        if lambda_ai_detection > 0:
            print(f"[CriticRewardAdapter] Injecting AI Detection LoRA...")
            self.model.inject_ai_detection_lora(backbone_ckpt=backbone_ckpt)
            
            # 8. 加载 AI Detection LoRA 权重
            if ai_detection_lora_ckpt and os.path.exists(ai_detection_lora_ckpt):
                print(f"[CriticRewardAdapter] Loading AI Detection LoRA from: {ai_detection_lora_ckpt}")
                ai_lora_ckpt = torch.load(ai_detection_lora_ckpt, map_location='cpu', weights_only=False)
                if 'lora_state_dict' in ai_lora_ckpt:
                    load_lora_state_dict(self.model.ai_detection_lora_modules, ai_lora_ckpt['lora_state_dict'])
                else:
                    load_lora_state_dict(self.model.ai_detection_lora_modules, ai_lora_ckpt)
                print(f"[CriticRewardAdapter] AI Detection LoRA loaded successfully")
            else:
                print(f"[CriticRewardAdapter] Warning: AI Detection LoRA checkpoint not found: {ai_detection_lora_ckpt}")
            
            # 9. 加载 AI Detection Head 权重
            if ai_detection_head_ckpt and os.path.exists(ai_detection_head_ckpt):
                print(f"[CriticRewardAdapter] Loading AI Detection Head from: {ai_detection_head_ckpt}")
                ai_head_ckpt = torch.load(ai_detection_head_ckpt, map_location='cpu', weights_only=False)
                if 'head_state_dict' in ai_head_ckpt:
                    self.model.ai_detection_head.load_state_dict(ai_head_ckpt['head_state_dict'])
                elif 'state_dict' in ai_head_ckpt:
                    self.model.ai_detection_head.load_state_dict(ai_head_ckpt['state_dict'])
                else:
                    self.model.ai_detection_head.load_state_dict(ai_head_ckpt)
                print(f"[CriticRewardAdapter] AI Detection Head loaded successfully")
            else:
                print(f"[CriticRewardAdapter] Warning: AI Detection Head checkpoint not found: {ai_detection_head_ckpt}")
            
            # 启用 AI Detection LoRA
            enable_lora(self.model.ai_detection_lora_modules)
        
        self.model.to(device)
        self.model.eval()
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
    
    def get_critic_reward(
        self,
        motion_feats: torch.Tensor,
        m_len: list,
        timestep: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        获取 Critic reward（Stage 2 Critic Head 输出）
        
        Args:
            motion_feats: motion 特征 [B, T, D]
            m_len: motion 长度列表
            timestep: 当前扩散步的 timestep（可选，用于噪声感知编码）
            
        Returns:
            reward: Critic 分数 [B]
        """
        # 获取 motion embedding（使用 Critic LoRA）
        latent, _ = self.model.encode_motion(
            motion_feats, 
            m_len, 
            repr_type=self.repr_type,
            timestep=timestep
        )
        
        # 通过 Critic Head 获取分数
        latent_squeezed = latent.squeeze(0)  # [B, latent_dim]
        score = self.model.critic_head(latent_squeezed)  # [B, 1]
        reward = score.squeeze(-1)  # [B]
        
        return reward
    
    def get_retrieval_reward(
        self,
        raw_texts: list,
        motion_feats: torch.Tensor,
        m_len: list,
        timestep: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        获取 Retrieval reward（Stage 1 text-motion 相似度）
        
        Args:
            raw_texts: 文本列表
            motion_feats: motion 特征 [B, T, D]
            m_len: motion 长度列表
            timestep: 当前扩散步的 timestep（可选，用于噪声感知编码）
            
        Returns:
            reward: cosine similarity [B]
        """
        # 获取 text embedding
        t_latent = self.model.get_text_embedding(raw_texts)
        
        # 获取 motion embedding（不使用 Critic LoRA，使用原始 backbone）
        # 临时禁用 Critic LoRA
        from motionreward.models.lora_modules import disable_lora
        disable_lora(self.model.critic_lora_modules)
        
        m_latent = self.model.get_motion_embedding(
            motion_feats, 
            m_len, 
            repr_type=self.repr_type,
            timestep=timestep
        )
        
        # 重新启用 Critic LoRA
        enable_lora(self.model.critic_lora_modules)
        
        # 计算 cosine similarity 作为 reward
        reward = F.cosine_similarity(
            t_latent.squeeze(), 
            m_latent.squeeze(), 
            dim=-1
        )
        
        return reward
    
    def get_m2m_reward(
        self,
        gt_motion_feats: torch.Tensor,
        gen_motion_feats: torch.Tensor,
        gt_len: list,
        gen_len: list,
        timestep: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        获取 M2M reward（gt motion 和 generated motion 的余弦相似度）
        
        Args:
            gt_motion_feats: GT motion 特征 [B, T, D]
            gen_motion_feats: Generated motion 特征 [B, T, D]
            gt_len: GT motion 长度列表
            gen_len: Generated motion 长度列表
            timestep: 当前扩散步的 timestep（可选，仅用于 generated motion 的噪声感知编码）
            
        Returns:
            reward: cosine similarity [B]
        """
        # 临时禁用 Critic LoRA，使用原始 backbone
        from motionreward.models.lora_modules import disable_lora
        disable_lora(self.model.critic_lora_modules)
        
        # 获取 GT motion embedding（GT 是干净的，不传 timestep）
        gt_latent = self.model.get_motion_embedding(
            gt_motion_feats, 
            gt_len, 
            repr_type=self.repr_type
        )
        
        # 获取 Generated motion embedding（使用 timestep 进行噪声感知编码）
        gen_latent = self.model.get_motion_embedding(
            gen_motion_feats, 
            gen_len, 
            repr_type=self.repr_type,
            timestep=timestep
        )
        
        # 重新启用 Critic LoRA
        enable_lora(self.model.critic_lora_modules)
        
        # 计算 cosine similarity 作为 reward
        reward = F.cosine_similarity(
            gt_latent.squeeze(), 
            gen_latent.squeeze(), 
            dim=-1
        )
        
        return reward
    
    def get_ai_detection_reward(
        self,
        motion_feats: torch.Tensor,
        m_len: list,
        timestep: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        获取 AI Detection reward（Stage 3 AI Detection Head 输出）
        
        AI Detection 模型输出 logits [B, 2]，其中：
        - class 0: 真实 motion
        - class 1: AI 生成的 motion
        
        我们希望生成的 motion 看起来像真实的，所以 reward = P(class 0) = 1 - P(class 1)
        
        Args:
            motion_feats: motion 特征 [B, T, D]
            m_len: motion 长度列表
            timestep: 当前扩散步的 timestep（可选，用于噪声感知编码）
            
        Returns:
            reward: P(real) 分数 [B]，越高表示越像真实 motion
        """
        if self.model.ai_detection_head is None:
            raise RuntimeError("AI Detection head not initialized. Set lambda_ai_detection > 0 to enable.")
        
        # 使用 AI Detection 的独立 encoder
        latent, _ = self.model.encode_motion_ai_detection(
            motion_feats, 
            m_len, 
            repr_type=self.repr_type,
            timestep=timestep
        )
        
        # 通过 AI Detection Head 获取 logits
        latent_squeezed = latent.squeeze(0)  # [B, latent_dim]
        logits = self.model.ai_detection_head(latent_squeezed)  # [B, 2]
        
        # 计算 P(real) = softmax(logits)[:, 0]
        probs = F.softmax(logits, dim=-1)
        reward = probs[:, 0]  # P(real)，越高越好
        
        return reward
    
    def get_reward_t2m(
        self,
        raw_texts: list,
        motion_feats: torch.Tensor,
        m_len: list,
        t_len: list = None,
        sent_emb: torch.Tensor = None,
        timestep: torch.Tensor = None,
        return_m: bool = False,
        return_details: bool = False,
        gt_motion_feats: torch.Tensor = None,  # GT motion for m2m reward
        gt_len: list = None,                   # GT motion lengths
        **kwargs
    ) -> torch.Tensor:
        """
        计算组合 reward = lambda_critic * critic + lambda_retrieval * retrieval + lambda_m2m * m2m + lambda_ai_detection * ai_det
        
        Args:
            raw_texts: 文本列表
            motion_feats: Generated motion 特征 [B, T, D]
            m_len: motion 长度列表
            return_m: 是否返回 motion latent
            return_details: 是否返回详细的 reward 分解
            gt_motion_feats: GT motion 特征 [B, T, D] (用于 m2m reward)
            gt_len: GT motion 长度列表 (用于 m2m reward)
            其他参数: 保持接口兼容
            
        Returns:
            reward: 组合 reward [B]
            如果 return_details=True，返回 (reward, critic_reward, retrieval_reward, m2m_reward, ai_detection_reward)
        """
        with torch.enable_grad():
            reward = torch.zeros(motion_feats.shape[0], device=motion_feats.device)
            critic_reward = None
            retrieval_reward = None
            m2m_reward = None
            ai_detection_reward = None
            latent = None
            
            # Critic reward
            if self.lambda_critic > 0:
                critic_reward = self.get_critic_reward(motion_feats, m_len, timestep=timestep)
                reward = reward + self.lambda_critic * critic_reward
                
                # 获取 latent 用于 return_m
                if return_m:
                    latent, _ = self.model.encode_motion(
                        motion_feats, m_len, repr_type=self.repr_type, timestep=timestep
                    )
            
            # Retrieval reward
            if self.lambda_retrieval > 0:
                retrieval_reward = self.get_retrieval_reward(raw_texts, motion_feats, m_len, timestep=timestep)
                reward = reward + self.lambda_retrieval * retrieval_reward
            
            # M2M reward
            if self.lambda_m2m > 0 and gt_motion_feats is not None:
                gt_len_use = gt_len if gt_len is not None else m_len
                m2m_reward = self.get_m2m_reward(gt_motion_feats, motion_feats, gt_len_use, m_len, timestep=timestep)
                reward = reward + self.lambda_m2m * m2m_reward
            
            # AI Detection reward
            if self.lambda_ai_detection > 0:
                ai_detection_reward = self.get_ai_detection_reward(motion_feats, m_len, timestep=timestep)
                reward = reward + self.lambda_ai_detection * ai_detection_reward
        
        if return_details:
            return reward, critic_reward, retrieval_reward, m2m_reward, ai_detection_reward
        if return_m:
            return reward, latent
        return reward
    
    def forward(self, *args, **kwargs):
        return self.get_reward_t2m(*args, **kwargs)


def find_critic_checkpoints(checkpoint_dir, model_size='small'):
    """
    自动查找 Critic 相关的 checkpoint 文件
    
    Args:
        checkpoint_dir: checkpoint 目录
        model_size: 模型大小
        
    Returns:
        dict: {'backbone': path, 'critic_lora': path, 'critic_head': path}
    """
    result = {'backbone': None, 'critic_lora': None, 'critic_head': None}
    
    # 查找 backbone (优先 stage2, 其次 stage1)
    backbone_patterns = [
        '*_stage2_*_spm_backbone_*.pth',
        '*_stage2_*_retrieval_backbone_*.pth',
        '*_stage1_*_spm_backbone_*.pth',
        '*_stage1_*_retrieval_backbone_*.pth',
    ]
    for pattern in backbone_patterns:
        files = glob.glob(os.path.join(checkpoint_dir, pattern))
        if files:
            result['backbone'] = max(files, key=os.path.getmtime)
            break
    
    # 查找 critic_lora
    lora_files = glob.glob(os.path.join(checkpoint_dir, '*_critic_lora_*.pth'))
    if lora_files:
        result['critic_lora'] = max(lora_files, key=os.path.getmtime)
    
    # 查找 critic_head
    head_files = glob.glob(os.path.join(checkpoint_dir, '*_critic_head_*.pth'))
    if head_files:
        result['critic_head'] = max(head_files, key=os.path.getmtime)
    
    # 查找 ai_detection_lora
    ai_lora_files = glob.glob(os.path.join(checkpoint_dir, '*_ai_detection_lora_*.pth'))
    if ai_lora_files:
        result['ai_detection_lora'] = max(ai_lora_files, key=os.path.getmtime)
    
    # 查找 ai_detection_head
    ai_head_files = glob.glob(os.path.join(checkpoint_dir, '*_ai_detection_head_*.pth'))
    if ai_head_files:
        result['ai_detection_head'] = max(ai_head_files, key=os.path.getmtime)
    
    return result


def fuck(model):
    """统计模型参数"""
    para_cnt, para_sum = {}, {}
    for name, para in model.named_parameters():
        try:
            para_cnt[name.split('.')[0]] += para.sum().item()
            para_sum[name.split('.')[0]] += para.numel()
        except:
            para_cnt[name.split('.')[0]] = para.sum().item()
            para_sum[name.split('.')[0]] = para.numel()
    print(para_cnt, '\n', para_sum)
    return para_cnt, para_sum


def main():
    cfg = parse_args()
    
    ft_config = get_ft_config(
        ft_type=cfg.ft_type, m=cfg.ft_m, prob=cfg.ft_prob, t=cfg.ft_t, k=cfg.ft_k,
        skip=cfg.ft_skip, reverse=cfg.ft_reverse, custom=None, 
        lambda_reward=cfg.ft_lambda_reward, dy=cfg.ft_dy,
        curriculum=getattr(cfg, 'curriculum', False),
        sweep_ratio=getattr(cfg, 'sweep_ratio', 0.03)
    )
    
    # ==================== Critic Checkpoint 配置 ====================
    # 支持命令行指定或自动查找
    critic_backbone_ckpt = getattr(cfg, 'critic_backbone_ckpt', '')
    critic_lora_ckpt = getattr(cfg, 'critic_lora_ckpt', '')
    critic_head_ckpt = getattr(cfg, 'critic_head_ckpt', '')
    ai_detection_lora_ckpt = getattr(cfg, 'ai_detection_lora_ckpt', '')
    ai_detection_head_ckpt = getattr(cfg, 'ai_detection_head_ckpt', '')
    
    # 如果没有指定，自动查找
    if not critic_backbone_ckpt or not critic_lora_ckpt or not critic_head_ckpt:
        model_size = getattr(cfg, 'reward_model_size', 'small')
        
        # 优先从 motionreward 查找，其次从 spm_lora_models/retrieval_lora_models 查找
        checkpoint_dirs = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        '..', 'checkpoints', 'motionreward'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        '..', 'checkpoints', 'spm_lora_models', f'{model_size}_three_repr'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        '..', 'checkpoints', 'retrieval_lora_models', f'{model_size}_three_repr'),
        ]
        
        for ckpt_dir in checkpoint_dirs:
            if os.path.exists(ckpt_dir):
                found = find_critic_checkpoints(ckpt_dir, model_size)
                if not critic_backbone_ckpt and found['backbone']:
                    critic_backbone_ckpt = found['backbone']
                if not critic_lora_ckpt and found['critic_lora']:
                    critic_lora_ckpt = found['critic_lora']
                if not critic_head_ckpt and found['critic_head']:
                    critic_head_ckpt = found['critic_head']
                # AI Detection checkpoints
                if not ai_detection_lora_ckpt and found.get('ai_detection_lora'):
                    ai_detection_lora_ckpt = found['ai_detection_lora']
                if not ai_detection_head_ckpt and found.get('ai_detection_head'):
                    ai_detection_head_ckpt = found['ai_detection_head']
    
    ft_config['critic_backbone_ckpt'] = critic_backbone_ckpt
    ft_config['critic_lora_ckpt'] = critic_lora_ckpt
    ft_config['critic_head_ckpt'] = critic_head_ckpt
    ft_config['ai_detection_lora_ckpt'] = ai_detection_lora_ckpt
    ft_config['ai_detection_head_ckpt'] = ai_detection_head_ckpt
    ft_config['reward_type'] = 'critic'
    
    # 设置 reward model size
    ft_config['reward_model_size'] = getattr(cfg, 'reward_model_size', 'small')
    ft_config['reward_lora_rank'] = getattr(cfg, 'reward_lora_rank', 128)
    ft_config['reward_lora_alpha'] = getattr(cfg, 'reward_lora_alpha', 256)
    
    # 设置 reward 权重
    ft_config['lambda_critic'] = getattr(cfg, 'lambda_critic', 1.0)
    ft_config['lambda_retrieval'] = getattr(cfg, 'lambda_retrieval', 0.0)
    ft_config['lambda_m2m'] = getattr(cfg, 'lambda_m2m', 0.0)
    ft_config['lambda_ai_detection'] = getattr(cfg, 'lambda_ai_detection', 0.0)
    
    # 更新实验名称 — 反映所有实际传入的关键参数
    # Reward 权重部分
    reward_parts = []
    if ft_config['lambda_critic'] > 0:
        reward_parts.append(f'C{ft_config["lambda_critic"]:g}')
    if ft_config['lambda_retrieval'] > 0:
        reward_parts.append(f'Ret{ft_config["lambda_retrieval"]:g}')
    if ft_config['lambda_m2m'] > 0:
        reward_parts.append(f'M2M{ft_config["lambda_m2m"]:g}')
    if ft_config['lambda_ai_detection'] > 0:
        reward_parts.append(f'AI{ft_config["lambda_ai_detection"]:g}')
    reward_str = '_'.join(reward_parts) if reward_parts else 'NoReward'
    
    ft_config['name'] += f'_{reward_str}_lr{cfg.ft_lr:.0e}_E{cfg.ft_epochs}'
    ft_config['name'] += f'_maxT{getattr(cfg, "reward_max_t", 500)}'
    if getattr(cfg, 'reward_t_switch', 0) != 0:
        ft_config['name'] += f'_sw{cfg.reward_t_switch}'
    if cfg.ft_dy != 2:
        ft_config['name'] += f'_dy{cfg.ft_dy}'
    
    cfg.ft_config = ft_config

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(cfg.SEED_VALUE)

    cfg.output_dir = f"./checkpoints/rft_mld/{ft_config['name']}"
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/checkpoints", exist_ok=True)

    if cfg.vis == "tb":
        writer = SummaryWriter(cfg.output_dir)
    elif cfg.vis == "swanlab":
        cfg_dict = {
            'output_dir': cfg.output_dir,
            'ft_type': cfg.ft_type,
            'ft_m': cfg.ft_m,
            'ft_k': cfg.ft_k,
            'ft_lr': cfg.ft_lr,
            'ft_lambda_reward': cfg.ft_lambda_reward,
            'ft_epochs': cfg.ft_epochs,
            'reward_type': 'critic',
            'lambda_critic': ft_config.get('lambda_critic', 0.0),
            'lambda_retrieval': ft_config.get('lambda_retrieval', 0.0),
            'lambda_m2m': ft_config.get('lambda_m2m', 0.0),
            'lambda_ai_detection': ft_config.get('lambda_ai_detection', 0.0),
            'reward_max_t': getattr(cfg, 'reward_max_t', 500),
            'reward_t_switch': getattr(cfg, 'reward_t_switch', 0),
            'curriculum': getattr(cfg, 'curriculum', False),
            'sweep_ratio': getattr(cfg, 'sweep_ratio', 0.03),
            'reward_model_size': ft_config.get('reward_model_size', 'small'),
            'critic_backbone_ckpt': critic_backbone_ckpt,
            'critic_lora_ckpt': critic_lora_ckpt,
            'critic_head_ckpt': critic_head_ckpt,
            'ai_detection_lora_ckpt': ai_detection_lora_ckpt,
            'ai_detection_head_ckpt': ai_detection_head_ckpt,
            'TRAIN': {
                'BATCH_SIZE': cfg.TRAIN.BATCH_SIZE,
                'learning_rate': cfg.TRAIN.learning_rate,
                'max_ft_epochs': cfg.TRAIN.max_ft_epochs,
            },
            'VAL': {
                'BATCH_SIZE': cfg.VAL.BATCH_SIZE,
                'SPLIT': cfg.VAL.SPLIT,
            },
            'DATASET': {
                'NAME': cfg.DATASET.NAME,
            }
        }
        writer = swanlab.init(
            project="MotionLCM-Critic",
            experiment_name=os.path.normpath(cfg.output_dir).replace(os.path.sep, "-"),
            suffix=None, config=cfg_dict, logdir=cfg.output_dir
        )
    else:
        raise ValueError(f"Invalid vis method: {cfg.vis}")

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(osp.join(cfg.output_dir, 'output.log'))
    handlers = [file_handler, stream_handler]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=handlers
    )
    logger = logging.getLogger(__name__)
    
    OmegaConf.save(cfg, osp.join(cfg.output_dir, 'config.yaml'))

    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    dataset = get_dataset(cfg)
    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()
    test_dataloader = dataset.test_dataloader()

    model = MLD(cfg, dataset)

    assert cfg.TRAIN.PRETRAINED, "cfg.TRAIN.PRETRAINED must not be None."
    logger.info(f"Loading pre-trained model: {cfg.TRAIN.PRETRAINED}")
    state_dict = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]
    logger.info(model.load_state_dict(state_dict, strict=False))

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.vae.eval()
    model.text_encoder.eval()
    model.to(device)

    cfg.TRAIN.learning_rate = float(cfg.ft_lr)
    logger.info("learning_rate: {}".format(cfg.TRAIN.learning_rate))
    optimizer = torch.optim.AdamW(
        model.denoiser.parameters(),
        lr=cfg.TRAIN.learning_rate,
        betas=(cfg.TRAIN.adam_beta1, cfg.TRAIN.adam_beta2),
        weight_decay=cfg.TRAIN.adam_weight_decay,
        eps=cfg.TRAIN.adam_epsilon
    )
    if cfg.TRAIN.max_ft_steps == -1:
        cfg.TRAIN.max_ft_steps = cfg.TRAIN.max_ft_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        cfg.TRAIN.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.TRAIN.lr_warmup_steps,
        num_training_steps=cfg.TRAIN.max_ft_steps
    )

    # Train!
    logger.info("***** Running training with Critic Reward *****")
    logging.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logging.info(f"  Num Epochs = {cfg.TRAIN.max_ft_epochs}")
    logging.info(f"  Instantaneous batch size per device = {cfg.TRAIN.BATCH_SIZE}")
    logging.info(f"  Total optimization steps = {cfg.TRAIN.max_ft_steps}")
    logging.info(f"  1 Epoch == {len(train_dataloader)} Step")

    global_step = 0

    @torch.no_grad()
    def validation(target_model: MLD, reward_model_for_val=None, critic_test_loader_for_val=None, ema: bool = False) -> tuple:
        target_model.denoiser.eval()
        val_loss_list = []
        
        vis_batch_idx = random.randint(0, len(val_dataloader) - 1)
        vis_joints = None
        vis_text = None
        
        # 收集生成样本的 reward scores
        all_combined_scores = []
        all_critic_scores = []
        all_retrieval_scores = []
        all_m2m_scores = []
        all_ai_detection_scores = []
        
        for batch_idx, val_batch in enumerate(tqdm(val_dataloader)):
            val_batch = move_batch_to_device(val_batch, device)
            val_loss_dict = target_model.allsplit_step(split='val', batch=val_batch)
            val_loss_list.append(val_loss_dict)
            
            # 生成 motion 并用 reward model 打分
            try:
                rs_set = target_model.t2m_eval(val_batch)
                joints_rst = rs_set['joints_rst']  # [B, T, J, 3]
                feats_rst_raw = rs_set['feats_rst_raw']  # [B, T, D] - 原始 263 dim features（未经 renorm）
                batch_size = joints_rst.shape[0]
                lengths = val_batch['length']
                if isinstance(lengths, torch.Tensor):
                    lengths = lengths.tolist()
                
                # 用 reward model 给生成的 motion 打分（返回详细分解）
                if reward_model_for_val is not None:
                    # 获取 GT motion 用于 M2M reward
                    gt_motion_feats = val_batch.get('motion', None)
                    gt_len = val_batch.get('length', None)
                    if isinstance(gt_len, torch.Tensor):
                        gt_len = gt_len.tolist()
                    
                    combined_score, critic_score, retrieval_score, m2m_score, ai_detection_score = reward_model_for_val.get_reward_t2m(
                        raw_texts=val_batch['text'],
                        motion_feats=feats_rst_raw,
                        m_len=lengths,
                        gt_motion_feats=gt_motion_feats,
                        gt_len=gt_len,
                        return_details=True
                    )
                    all_combined_scores.append(combined_score.cpu())
                    if critic_score is not None:
                        all_critic_scores.append(critic_score.cpu())
                    if retrieval_score is not None:
                        all_retrieval_scores.append(retrieval_score.cpu())
                    if m2m_score is not None:
                        all_m2m_scores.append(m2m_score.cpu())
                    if ai_detection_score is not None:
                        all_ai_detection_scores.append(ai_detection_score.cpu())
                
                if batch_idx == vis_batch_idx and vis_joints is None:
                    sample_idx = random.randint(0, batch_size - 1)
                    
                    text = val_batch['text'][sample_idx]
                    if isinstance(text, list):
                        text = text[0]
                    length = val_batch['length'][sample_idx]
                    if isinstance(length, torch.Tensor):
                        length = length.item()
                    
                    vis_joints = joints_rst[sample_idx, :length].cpu().numpy()
                    vis_text = text
                    logger.info(f"Generated motion for visualization: '{text[:50]}...' with shape {vis_joints.shape}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate motion for batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        if vis_joints is not None:
            vis_dir = os.path.join(cfg.output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            safe_text = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in vis_text[:30])
            safe_text = safe_text.replace(' ', '_')
            video_path = os.path.join(vis_dir, f"step{global_step}_{safe_text}.gif")
            
            try:
                plot_3d_motion(
                    save_path=video_path,
                    joints=vis_joints,
                    title=f"Step {global_step}: {vis_text[:50]}",
                    fps=20
                )
                logger.info(f"Saved visualization video: {video_path}")
                
                if cfg.vis == "swanlab":
                    writer.log({"visualization/motion_video": swanlab.Video(video_path)}, step=global_step)
                elif cfg.vis == "tb":
                    writer.add_text("visualization/video_path", video_path, global_step=global_step)
                    
            except Exception as e:
                logger.warning(f"Failed to save visualization: {e}")
                import traceback
                traceback.print_exc()
        
        # 计算生成样本的平均 reward scores
        avg_combined_score = 0.0
        avg_critic_score = 0.0
        avg_retrieval_score = 0.0
        avg_m2m_score = 0.0
        avg_ai_detection_score = 0.0
        
        if len(all_combined_scores) > 0:
            all_combined_scores = torch.cat(all_combined_scores, dim=0)
            avg_combined_score = all_combined_scores.mean().item()
        if len(all_critic_scores) > 0:
            all_critic_scores = torch.cat(all_critic_scores, dim=0)
            avg_critic_score = all_critic_scores.mean().item()
        if len(all_retrieval_scores) > 0:
            all_retrieval_scores = torch.cat(all_retrieval_scores, dim=0)
            avg_retrieval_score = all_retrieval_scores.mean().item()
        if len(all_m2m_scores) > 0:
            all_m2m_scores = torch.cat(all_m2m_scores, dim=0)
            avg_m2m_score = all_m2m_scores.mean().item()
        if len(all_ai_detection_scores) > 0:
            all_ai_detection_scores = torch.cat(all_ai_detection_scores, dim=0)
            avg_ai_detection_score = all_ai_detection_scores.mean().item()
        
        logger.info(f"Validation Generated Samples - Combined: {avg_combined_score:.4f}, Critic: {avg_critic_score:.4f}, Retrieval: {avg_retrieval_score:.4f}, M2M: {avg_m2m_score:.4f}, AI_Det: {avg_ai_detection_score:.4f}")
        
        metrics = target_model.allsplit_epoch_end()
        metrics[f"Val/loss"] = sum([d['loss'] for d in val_loss_list]).item() / len(val_dataloader)
        metrics[f"Val/diff_loss"] = sum([d['diff_loss'] for d in val_loss_list]).item() / len(val_dataloader)
        metrics[f"Val/reward_combined"] = avg_combined_score      # 组合 reward
        metrics[f"Val/reward_critic"] = avg_critic_score          # Critic reward
        metrics[f"Val/reward_retrieval"] = avg_retrieval_score    # Retrieval reward
        metrics[f"Val/reward_m2m"] = avg_m2m_score                # M2M reward
        metrics[f"Val/reward_ai_detection"] = avg_ai_detection_score  # AI Detection reward
        max_val_rp1 = metrics['Metrics/R_precision_top_1']
        min_val_fid = metrics['Metrics/FID']
        print_table(f'Validation@Step-{global_step}', metrics)
        for mk, mv in metrics.items():
            mk = mk + '_EMA' if ema else mk
            if cfg.vis == "tb":
                writer.add_scalar(mk, mv, global_step=global_step)
            elif cfg.vis == "swanlab":
                writer.log({mk: mv}, step=global_step)
        target_model.denoiser.train()
        return max_val_rp1, min_val_fid, metrics

    max_rp1, min_fid = 0.80862, 0.40636

    progress_bar = tqdm(range(0, cfg.TRAIN.max_ft_steps), desc="Steps")
    
    # ==================== 加载 Critic Reward Model ====================
    t5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'deps', 'sentence-t5-large')
    
    # 获取 reward 权重配置
    lambda_critic = ft_config.get('lambda_critic', 1.0)
    lambda_retrieval = ft_config.get('lambda_retrieval', 0.0)
    lambda_m2m = ft_config.get('lambda_m2m', 0.0)
    lambda_ai_detection = ft_config.get('lambda_ai_detection', 0.0)
    
    logger.info("="*60)
    logger.info("Loading Reward Model...")
    logger.info("="*60)
    logger.info(f"  Backbone: {critic_backbone_ckpt}")
    logger.info(f"  Critic LoRA: {critic_lora_ckpt}")
    logger.info(f"  Critic Head: {critic_head_ckpt}")
    if lambda_ai_detection > 0:
        logger.info(f"  AI Detection LoRA: {ai_detection_lora_ckpt}")
        logger.info(f"  AI Detection Head: {ai_detection_head_ckpt}")
    logger.info(f"  Model Size: {ft_config['reward_model_size']}")
    logger.info(f"  Lambda Critic: {lambda_critic}")
    logger.info(f"  Lambda Retrieval: {lambda_retrieval}")
    logger.info(f"  Lambda M2M: {lambda_m2m}")
    logger.info(f"  Lambda AI Detection: {lambda_ai_detection}")
    
    reward_model = CriticRewardAdapter(
        backbone_ckpt=critic_backbone_ckpt,
        critic_lora_ckpt=critic_lora_ckpt,
        critic_head_ckpt=critic_head_ckpt,
        ai_detection_lora_ckpt=ai_detection_lora_ckpt,
        ai_detection_head_ckpt=ai_detection_head_ckpt,
        t5_path=t5_path,
        repr_type='263',
        model_size=ft_config['reward_model_size'],
        lora_rank=ft_config['reward_lora_rank'],
        lora_alpha=ft_config['reward_lora_alpha'],
        device='cuda',
        lambda_critic=lambda_critic,
        lambda_retrieval=lambda_retrieval,
        lambda_m2m=lambda_m2m,
        lambda_ai_detection=lambda_ai_detection
    ).cuda()
    
    logger.info("Reward Model loaded successfully!")
    logger.info("="*60)
    
    # ==================== 加载 Critic 测试数据集 ====================
    PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 使用 .pth 文件格式的 Critic 数据 (参考 train_motion_reward_tiny.sh)
    critic_data_263 = os.path.join(PROJ_DIR, 'datasets', 'critic', 'critic_val_263.pth')
    critic_data_22x3 = os.path.join(PROJ_DIR, 'datasets', 'critic', 'critic_val_22x3.pth')
    critic_data_135 = os.path.join(PROJ_DIR, 'datasets', 'critic', 'critic_val_135.pth')
    
    # 加载归一化参数，与 MotionReward 训练一致 (Z-normalization)
    humanml3d_root = os.path.join(PROJ_DIR, 'datasets', 'humanml3d')
    mean_263, std_263, mean_22x3, std_22x3, mean_135, std_135 = load_normalization_stats(humanml3d_root)
    critic_mean_std_dict = {}
    if os.path.exists(critic_data_263):
        critic_mean_std_dict['263'] = (mean_263, std_263)
    if os.path.exists(critic_data_22x3):
        critic_mean_std_dict['22x3'] = (mean_22x3, std_22x3)
    if os.path.exists(critic_data_135):
        critic_mean_std_dict['135'] = (mean_135, std_135)
    logger.info(f"Critic Z-normalization enabled for repr types: {list(critic_mean_std_dict.keys())}")
    
    # 创建 Critic 测试数据集（传入 mean_std_dict 进行 Z-归一化，与训练一致）
    critic_test_dataset = CriticPairDataset(
        data_path_263=critic_data_263 if os.path.exists(critic_data_263) else None,
        data_path_22x3=critic_data_22x3 if os.path.exists(critic_data_22x3) else None,
        data_path_135=critic_data_135 if os.path.exists(critic_data_135) else None,
        max_motion_length=196,
        split='eval',
        mean_std_dict=critic_mean_std_dict
    )
    
    if len(critic_test_dataset) > 0:
        critic_test_sampler = CriticReprTypeBatchSampler(
            critic_test_dataset,
            batch_size=32,
            shuffle=False,
            drop_last=False
        )
        critic_test_loader = torch.utils.data.DataLoader(
            critic_test_dataset,
            batch_sampler=critic_test_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=critic_collate_fn
        )
        logger.info(f"Critic test dataset loaded: {len(critic_test_dataset)} pairs")
    else:
        critic_test_loader = None
        logger.warning("No Critic test data found, skipping Critic evaluation")
    
    # ==================== 评估 Critic Reward Model ====================
    def evaluate_critic_on_testset(reward_model, critic_loader):
        """在 Critic 测试集上评估模型（pairwise ranking 准确率）"""
        if critic_loader is None:
            raise RuntimeError("Critic test loader is None, cannot evaluate!")
        
        logger.info("="*60)
        logger.info("Evaluating Critic Reward Model on test set...")
        logger.info("="*60)
        
        # 使用 eval_critic 函数进行评估
        result = eval_critic(critic_loader, reward_model.model, device=device, rank=0)
        
        logger.info(f"Critic Test Set Evaluation:")
        logger.info(f"  Overall Accuracy: {result['acc']:.4f}")
        logger.info(f"  Overall Loss: {result['loss']:.4f}")
        for repr_type in result.get('repr_types', []):
            acc_key = f'acc_{repr_type}'
            loss_key = f'loss_{repr_type}'
            if acc_key in result:
                logger.info(f"  [{repr_type}] Accuracy: {result[acc_key]:.4f}, Loss: {result.get(loss_key, 0):.4f}")
        logger.info("="*60)
        
        return result
    
    # 在 finetune 之前评估 critic reward model
    if cfg.get('eval_reward_model', True) and critic_test_loader is not None:
        critic_metrics = evaluate_critic_on_testset(reward_model, critic_test_loader)
        logger.info(f"Critic Reward Model Evaluation Complete: {critic_metrics}")
    elif critic_test_loader is None:
        logger.warning("Skipping Critic evaluation: critic_test_loader is None")
    
    # ==================== 评估 Retrieval Reward Model ====================
    if lambda_retrieval > 0:
        logger.info("="*60)
        logger.info("Evaluating Retrieval Reward Model (Stage 1)...")
        logger.info("="*60)
        
        # 使用与 eval_three_stage.py (A 阶段) 完全一致的数据加载方式
        try:
            from motionreward.models.lora_modules import disable_lora
            
            # 构建数据路径（与 A 阶段一致，使用原始 .npy + test.txt split）
            motion_dir_263 = os.path.join(humanml3d_root, 'new_joint_vecs')
            motion_dir_22x3 = os.path.join(humanml3d_root, 'new_joints')
            motion_dir_135 = os.path.join(humanml3d_root, 'joints_6d')
            text_dir = os.path.join(humanml3d_root, 'texts')
            test_split = os.path.join(humanml3d_root, 'test.txt')
            
            retrieval_repr_types = ['263', '22x3', '135']
            retrieval_test_loaders = {}
            
            if os.path.exists(motion_dir_263) and os.path.exists(test_split):
                test_ds_263 = Text2MotionDataset263(mean_263, std_263, test_split, motion_dir_263, text_dir)
                retrieval_test_loaders['263'] = torch.utils.data.DataLoader(
                    test_ds_263, batch_size=32, shuffle=False, num_workers=4,
                    pin_memory=True, collate_fn=retrieval_collate_fn, drop_last=False)
                logger.info(f"Loaded retrieval test data [263]: {len(test_ds_263)} samples")
            
            if os.path.exists(motion_dir_22x3) and os.path.exists(test_split):
                test_ds_22x3 = JointLevelText2MotionDataset(mean_22x3, std_22x3, test_split, motion_dir_22x3, text_dir)
                retrieval_test_loaders['22x3'] = torch.utils.data.DataLoader(
                    test_ds_22x3, batch_size=32, shuffle=False, num_workers=4,
                    pin_memory=True, collate_fn=retrieval_collate_fn, drop_last=False)
                logger.info(f"Loaded retrieval test data [22x3]: {len(test_ds_22x3)} samples")
            
            if os.path.exists(motion_dir_135) and os.path.exists(test_split):
                test_ds_135 = Text2MotionDataset135(mean_135, std_135, test_split, motion_dir_135, text_dir)
                retrieval_test_loaders['135'] = torch.utils.data.DataLoader(
                    test_ds_135, batch_size=32, shuffle=False, num_workers=4,
                    pin_memory=True, collate_fn=retrieval_collate_fn, drop_last=False)
                logger.info(f"Loaded retrieval test data [135]: {len(test_ds_135)} samples")
            
            # 禁用所有 LoRA 进行 Retrieval 评估（只使用 Stage 1 backbone）
            disable_lora(reward_model.model.critic_lora_modules)
            if hasattr(reward_model.model, 'ai_detection_lora_modules') and reward_model.model.ai_detection_lora_modules:
                disable_lora(reward_model.model.ai_detection_lora_modules)
            
            # 逐表征评估（与 eval_three_stage.py 一致）
            for repr_type, test_loader in retrieval_test_loaders.items():
                logger.info(f"Evaluating retrieval [{repr_type}]...")
                
                text_list, text_latents, motion_latents = [], [], []
                
                with torch.no_grad():
                    for batch in tqdm(test_loader, desc=f"Retrieval eval [{repr_type}]"):
                        motion = batch['motion'].to(device)
                        texts = batch['text']
                        lengths = batch['length']
                        
                        if isinstance(lengths, torch.Tensor):
                            lengths = lengths.tolist()
                        
                        t_latent = reward_model.model.get_text_embedding(texts)
                        m_latent = reward_model.model.get_motion_embedding(motion, lengths, repr_type=repr_type)
                        
                        text_list.extend(texts)
                        text_latents.extend(t_latent.cpu().numpy())
                        motion_latents.extend(m_latent.cpu().numpy())
                
                test_result = [text_list, text_latents, motion_latents]
                
                # 与 eval_three_stage.py 一致：使用固定种子 shuffle
                random.seed(42)
                shuffle_index = list(range(len(test_result[2])))
                random.shuffle(shuffle_index)
                test_result[0] = [test_result[0][i] for i in shuffle_index]
                test_result[1] = [test_result[1][i] for i in shuffle_index]
                test_result[2] = [test_result[2][i] for i in shuffle_index]
                
                bs32_metrics = calculate_retrieval_metrics_small_batches(test_result, epoch=0, fptr=None)
                full_metrics = calculate_retrieval_metrics(test_result, epoch=0, fptr=None)
                
                logger.info(f"  [{repr_type}] BS32 - R@1: {bs32_metrics['R1']:.2f}% | R@5: {bs32_metrics['R5']:.2f}% | R@10: {bs32_metrics['R10']:.2f}%")
                logger.info(f"  [{repr_type}] Full - R@1: {full_metrics['R1']:.2f}% | R@5: {full_metrics['R5']:.2f}% | R@10: {full_metrics['R10']:.2f}%")
            
            # 重新启用所有 LoRA
            enable_lora(reward_model.model.critic_lora_modules)
            if hasattr(reward_model.model, 'ai_detection_lora_modules') and reward_model.model.ai_detection_lora_modules:
                enable_lora(reward_model.model.ai_detection_lora_modules)
            
            logger.info("="*60)
            
        except Exception as e:
            logger.warning(f"Failed to evaluate retrieval reward model: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== 评估 Critic Reward Model 结束 ====================
    
    model.reward_model = reward_model
    model.reward_max_t = getattr(cfg, 'reward_max_t', 500)  # MotionReward 训练时的 maxT，用于 clamp timestep
    model.reward_t_switch = getattr(cfg, 'reward_t_switch', 50)  # Reward 策略分界点: i < switch 用 x_0 预测, i >= switch 用 x_t 直接
    model.mem_dict = {'Before Forward Finetune': [], 'Before Diffusion Reverse': [], 'Before T5': [], 'After T5': [],}
    model.mem_dict.update({f'After Reverse Timestep {i}': [] for i in range(50)})
    model.mem_dict.update({
        'After Reverse': [], 'Before VAE Decode': [],
        'Afrer VAE Decode/Before Reward func': [], 'After Reward func': [],
        'After Forward Finetune': [], 'Before Backward': [], 'After Backward': [],
    })
    model.ft_config = ft_config
    model.lambda_reward = ft_config['lambda_reward']
    model.reward_record = [[] for _ in range(50)]
    model.trn_reward = []
    
    logger.info(f'FineTune Config : {ft_config} ')
    logger.info(f'reward_t_switch : {model.reward_t_switch} (i < {model.reward_t_switch}: R(x_0,0), i >= {model.reward_t_switch}: R(x_t,t))')
    if ft_config.get('curriculum', False):
        logger.info(f'Motion Reward Timestep Scheduling: ENABLED (sweep_ratio={ft_config.get("sweep_ratio", 0.03)}, 剩余时间固定最后k步)')
    global_step = 0
    epochs = cfg.ft_epochs
    max_training_steps = epochs * len(train_dataloader)
    model.training_progress = 0.0  # 初始化训练进度
    logger.info(f"Training for {epochs} epochs, max_training_steps={max_training_steps}")
    rcd_reward = []
    
    validation_steps = getattr(cfg, 'validation_steps', 100)
    no_save = getattr(cfg, 'no_save', False)
    fid_save_threshold = getattr(cfg, 'fid_save_threshold', 0.15)
    logger.info(f"Validation every {validation_steps} steps, no_save={no_save}, fid_save_threshold={fid_save_threshold}")
    
    # ==================== Step 0 初始验证 ====================
    logger.info("="*60)
    logger.info("Running initial validation at step 0...")
    logger.info("="*60)
    cur_rp1, cur_fid, metrics = validation(model, reward_model_for_val=reward_model, critic_test_loader_for_val=critic_test_loader)
    if not no_save and cur_fid < fid_save_threshold:
        save_path = os.path.join(cfg.output_dir, 'checkpoints', 
                                f"step0-R1-{round(cur_rp1, 3)}-FID-{round(cur_fid, 3)}.ckpt")
        ckpt = dict(
            state_dict=model.state_dict(), 
            ft_config=model.ft_config, metrics=metrics, 
            reward_record=[model.reward_record, model.trn_reward]
        )
        model.on_save_checkpoint(ckpt)
        torch.save(ckpt, save_path)
        logger.info(f"Step 0: FID={round(cur_fid, 3)} < {fid_save_threshold}, saved to {save_path}")
    elif not no_save:
        logger.info(f"Step 0: FID={round(cur_fid, 3)} >= {fid_save_threshold}, skip saving")
    logger.info(f"Step 0: Initial validation complete | R@1={round(cur_rp1, 3)}, FID={round(cur_fid, 3)}")
    
    device = torch.device("cuda:0")
    for epoch in range(epochs):
        para_cnt, para_sum = fuck(model)
        logger.info(f"Epoch {epoch}: Para Sum: {para_sum} Para Cnt: {para_cnt}\n\n")
        for step, batch in enumerate(train_dataloader):
            batch = move_batch_to_device(batch, device)
            # 更新训练进度 (用于 Motion Reward timestep scheduling)
            model.training_progress = global_step / max(max_training_steps, 1)
            torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
            peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
            model.mem_dict['Before Forward Finetune'].append(peak_memory)
            
            if ft_config['type'] == 'NIPS':
                loss_dict = model.allsplit_step('finetune_nips', batch, optimizer, lr_scheduler)
            else:
                loss_dict = model.allsplit_step('finetune', batch)
            
            if ft_config['type'] != 'NIPS':
                peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 ** 2)
                model.mem_dict['After Forward Finetune'].append(peak_memory)
                torch.cuda.reset_peak_memory_stats(device)
                peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                model.mem_dict['Before Backward'].append(peak_memory)
                loss = loss_dict['loss']
                loss.backward()
                device = torch.device("cuda:0")
                torch.cuda.reset_peak_memory_stats(device)
                peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                model.mem_dict['After Backward'].append(peak_memory)
                torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), cfg.TRAIN.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            diff_loss = loss_dict['diff_loss']
            reward = loss_dict['reward']
            rcd_reward.append(reward.item())
            loss = loss_dict['loss']
            progress_bar.update(1)
            global_step += 1
            logs = {
                'Epoch': epoch,
                "loss": loss.item(),
                "diff_loss": 0,
                "lr": lr_scheduler.get_last_lr()[0],
                "reward": reward.item()
            }
            # 记录 Motion Reward 窗口信息
            if ft_config.get('curriculum', False):
                eg = model.ft_config['enable_grad']
                logs['cur_window_start'] = eg[0] if eg else -1
                logs['cur_window_end'] = eg[-1] if eg else -1
                logs['training_progress'] = model.training_progress
            progress_bar.set_postfix(**logs)
            for k, v in logs.items():
                if cfg.vis == "tb":
                    writer.add_scalar(f"ft/{k}", v, global_step=global_step)
                elif cfg.vis == "swanlab":
                    writer.log({f"ft/{k}": v}, step=global_step)
            
            if global_step % validation_steps == 0:
                cur_rp1, cur_fid, metrics = validation(model, reward_model_for_val=reward_model, critic_test_loader_for_val=critic_test_loader)
                if not no_save and cur_fid < fid_save_threshold:
                    save_path = os.path.join(cfg.output_dir, 'checkpoints', 
                                            f"step{global_step}-R1-{round(cur_rp1, 3)}-FID-{round(cur_fid, 3)}.ckpt")
                    ckpt = dict(
                        state_dict=model.state_dict(), 
                        ft_config=model.ft_config, metrics=metrics, 
                        reward_record=[model.reward_record, model.trn_reward]
                    )
                    model.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)
                    logger.info(f"Step {global_step}: FID={round(cur_fid, 3)} < {fid_save_threshold}, saved to {save_path}")
                elif not no_save:
                    logger.info(f"Step {global_step}: FID={round(cur_fid, 3)} >= {fid_save_threshold}, skip saving")
                main_metrics = {k: round(v, 4) for k, v in metrics.items() if 'gt' not in k}
                logger.info(f"Step {global_step}: R@1={round(cur_rp1, 3)}, FID={round(cur_fid, 3)}")
                logger.info(f"Step {global_step}: Metrics={main_metrics}\n")

        avg_mem_dict = {k: 0 for k in model.mem_dict.keys()}
        
        cur_rp1, cur_fid, metrics = validation(model, reward_model_for_val=reward_model, critic_test_loader_for_val=critic_test_loader)
        if not no_save and cur_fid < fid_save_threshold:
            save_path = os.path.join(cfg.output_dir, 'checkpoints', f"E{epoch}-R1-{round(cur_rp1, 3)}-FID-{round(cur_fid, 3)}.ckpt")
            ckpt = dict(
                state_dict=model.state_dict(), mem_use=[model.mem_dict, avg_mem_dict],
                ft_config=model.ft_config, metrics=metrics, 
                reward_record=[model.reward_record, model.trn_reward]
            )
            model.on_save_checkpoint(ckpt)
            torch.save(ckpt, save_path)
            logger.info(f"Epoch: {epoch}, FID={round(cur_fid, 3)} < {fid_save_threshold}, saved to {save_path}")
        elif not no_save:
            logger.info(f"Epoch: {epoch}, FID={round(cur_fid, 3)} >= {fid_save_threshold}, skip saving")
        main_metrics = {k: round(v, 4) for k, v in metrics.items() if 'gt' not in k}
        logger.info(f"Epoch: {epoch}, R@1:{round(cur_rp1, 3)}, FID:{round(cur_fid, 3)} Metrics: {metrics}")
        logger.info(f"Epoch: {epoch}, Main Metrics: {main_metrics}\n\n")


if __name__ == "__main__":
    main()
