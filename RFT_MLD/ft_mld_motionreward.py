#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
    使用 MotionReward 微调 MLD 的脚本

    核心思路:
    1. 加载预训练的 MLD 模型
    2. 加载 MotionReward 作为 reward model (完整复用 train_retrieval_lora_new.py 的逻辑)
    3. MLD 生成 motion → MotionReward 计算相似度 → 作为 reward 信号反向传播
    
    与原 ft_mld.py 的区别:
    - 更清晰的代码结构
    - 完整的注释和文档
    - 更灵活的配置系统
    - 更详细的日志输出
===============================================================================
"""

import os
import sys
import glob
import logging
import datetime
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from omegaconf import OmegaConf

# SwanLab 支持
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

# 导入 MLD 相关模块
from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.utils.utils import print_table, set_seed, move_batch_to_device

# 导入 diffusers 的优化器和调度器
from diffusers.optimization import get_scheduler

# 添加 motionreward 到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入 MotionReward 相关模块
from motionreward.models import MultiReprRetrievalWithLoRA
from motionreward.utils.config_utils import get_model_config
from motionreward.evaluation.retrieval_metrics import (
    calculate_retrieval_metrics,
    calculate_retrieval_metrics_small_batches
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===============================================================================
#                           MotionReward 加载器
# ===============================================================================

class MotionRewardLoader:
    """
    MotionReward 模型加载器
    
    完整复用 train_retrieval_lora_new.py 的加载逻辑
    """
    
    @staticmethod
    def load_model(
        model_size: str = 'small',
        stage1_ckpt: Optional[str] = None,
        critic_lora_ckpt: Optional[str] = None,
        t5_path: str = '../deps/sentence-t5-large',
        repr_type: str = '263',
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.1,
        device: str = 'cuda',
        verbose: bool = True
    ) -> nn.Module:
        """
        加载 MotionReward 模型（复用 train_retrieval_lora_new.py 的逻辑）
        
        Args:
            model_size: 模型规模 ('tiny', 'small', 'base', 'large', ...)
            stage1_ckpt: Stage 1 backbone checkpoint 路径
            critic_lora_ckpt: (可选) Stage 2 Critic LoRA checkpoint 路径
            t5_path: Sentence-T5 模型路径
            repr_type: motion 表征类型 ('263', '22x3', '135')
            lora_rank: LoRA 秩
            lora_alpha: LoRA 缩放因子
            lora_dropout: LoRA dropout
            device: 设备
            verbose: 是否打印详细信息
            
        Returns:
            加载好的 MotionReward 模型
        """
        if verbose:
            print("="*80)
            print("  MotionReward Model Loader")
            print("="*80)
            print(f"  Model Size       : {model_size}")
            print(f"  Repr Type        : {repr_type}")
            print(f"  Stage1 Ckpt      : {stage1_ckpt}")
            print(f"  Critic LoRA Ckpt : {critic_lora_ckpt or 'None'}")
            print(f"  T5 Path          : {t5_path}")
            print(f"  Device           : {device}")
            print("="*80)
        
        # 1. 通过 model_size 获取模型配置（复用 config_utils.py）
        model_cfg = get_model_config(model_size)
        
        if verbose:
            print(f"\n[Model Config] {model_size}")
            print(f"  latent_dim          : {model_cfg['latent_dim']}")
            print(f"  unified_dim         : {model_cfg['unified_dim']}")
            print(f"  encoder_num_layers  : {model_cfg['encoder_num_layers']}")
            print(f"  encoder_num_heads   : {model_cfg['encoder_num_heads']}")
            print(f"  encoder_ff_size     : {model_cfg['encoder_ff_size']}")
            print(f"  text_num_layers     : {model_cfg['text_num_layers']}")
            print(f"  text_num_heads      : {model_cfg['text_num_heads']}")
            print(f"  text_ff_size        : {model_cfg['text_ff_size']}")
            print(f"  proj_hidden_dim     : {model_cfg['proj_hidden_dim']}")
            print(f"  proj_num_layers     : {model_cfg['proj_num_layers']}")
        
        # 2. 创建 MultiReprRetrievalWithLoRA 模型（复用 lora_retrieval.py）
        model = MultiReprRetrievalWithLoRA(
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
        
        # 3. 加载 Stage 1 backbone（复用 train_retrieval_lora_new.py 的加载逻辑）
        if stage1_ckpt and os.path.exists(stage1_ckpt):
            if verbose:
                print(f"\n[Loading Stage 1 Backbone]")
                print(f"  Path: {stage1_ckpt}")
            
            checkpoint = torch.load(stage1_ckpt, map_location='cpu', weights_only=False)
            
            if 'state_dict' in checkpoint:
                missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
                if verbose and (missing or unexpected):
                    print(f"  Missing keys: {len(missing)}")
                    print(f"  Unexpected keys: {len(unexpected)}")
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            if verbose:
                print(f"  ✓ Stage 1 backbone loaded successfully")
        else:
            if verbose:
                print(f"\n[Warning] No Stage 1 checkpoint provided, using random weights")
        
        # 4. (可选) 加载 Critic LoRA（复用 train_retrieval_lora_new.py 的逻辑）
        if critic_lora_ckpt and os.path.exists(critic_lora_ckpt):
            if verbose:
                print(f"\n[Loading Critic LoRA]")
                print(f"  Path: {critic_lora_ckpt}")
            
            from motionreward.models.lora_modules import load_lora_state_dict
            
            # Inject Critic LoRA
            model.inject_critic_lora()
            
            # 加载 LoRA 权重
            critic_ckpt = torch.load(critic_lora_ckpt, map_location='cpu', weights_only=False)
            
            if 'lora_state_dict' in critic_ckpt:
                load_lora_state_dict(model.critic_lora_modules, critic_ckpt['lora_state_dict'])
            
            if 'head_state_dict' in critic_ckpt:
                model.critic_head.load_state_dict(critic_ckpt['head_state_dict'])
            
            if verbose:
                print(f"  ✓ Critic LoRA loaded successfully")
        
        # 5. 移动到设备并设为 eval 模式
        model.to(device)
        model.eval()
        
        # 6. 冻结所有参数（作为 reward model 不需要训练）
        for param in model.parameters():
            param.requires_grad = False
        
        if verbose:
            print(f"\n[Model Setup Complete]")
            print(f"  Device: {device}")
            print(f"  Mode: eval")
            print(f"  All parameters frozen: True")
            print("="*80)
        
        return model


class RewardModelWrapper(nn.Module):
    """
    Reward Model 包装器
    
    提供与原 ft_mld.py 兼容的 get_reward_t2m 接口
    """
    
    def __init__(
        self,
        model: MultiReprRetrievalWithLoRA,
        repr_type: str = '263',
        device: str = 'cuda'
    ):
        super().__init__()
        self.model = model
        self.repr_type = repr_type
        self.device = device
    
    def get_reward_t2m(
        self,
        raw_texts: List[str],
        motion_feats: torch.Tensor,
        m_len: List[int],
        t_len: Optional[List[int]] = None,
        sent_emb: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        return_m: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        计算 text-motion 相似度作为 reward
        
        Args:
            raw_texts: 文本列表
            motion_feats: motion 特征 [B, T, D]
            m_len: motion 长度列表
            t_len: (unused) 文本长度
            sent_emb: (unused) 句子嵌入
            timestep: (unused) 时间步
            return_m: 是否返回 motion latent
            
        Returns:
            reward: cosine similarity [B]
        """
        with torch.enable_grad():
            # 获取 text embedding
            t_latent = self.model.get_text_embedding(raw_texts)
            
            # 获取 motion embedding
            m_latent = self.model.get_motion_embedding(
                motion_feats,
                m_len,
                repr_type=self.repr_type
            )
            
            # 计算 cosine similarity 作为 reward
            reward = F.cosine_similarity(
                t_latent.squeeze(),
                m_latent.squeeze(),
                dim=-1
            )
        
        if return_m:
            return reward, m_latent
        return reward
    
    def forward(self, *args, **kwargs):
        """Forward 直接调用 get_reward_t2m"""
        return self.get_reward_t2m(*args, **kwargs)


# ===============================================================================
#                           Reward Model 评估器
# ===============================================================================

class RewardModelEvaluator:
    """Reward Model 评估器"""
    
    @staticmethod
    @torch.no_grad()
    def evaluate(
        reward_model: RewardModelWrapper,
        dataloader: DataLoader,
        device: str = 'cuda',
        logger: Optional[logging.Logger] = None
    ) -> Dict:
        """
        评估 Reward Model 的检索性能
        
        Args:
            reward_model: Reward model 包装器
            dataloader: 数据加载器
            device: 设备
            logger: 日志记录器
            
        Returns:
            评估指标字典
        """
        if logger:
            logger.info("="*80)
            logger.info("  Evaluating Reward Model Retrieval Performance")
            logger.info("="*80)
        
        reward_model.eval()
        text_list, text_latents, motion_latents = [], [], []
        
        for batch in tqdm(dataloader, desc="Evaluating Reward Model"):
            batch = move_batch_to_device(batch, device)
            texts = batch['text']
            motions = batch['motion']  # [B, T, D]
            lengths = batch['length']
            
            # 处理文本（可能是列表）
            processed_texts = []
            for text in texts:
                if isinstance(text, list):
                    processed_texts.append(text[0])
                else:
                    processed_texts.append(text)
            
            # 处理长度
            processed_lengths = []
            for length in lengths:
                if isinstance(length, torch.Tensor):
                    processed_lengths.append(length.item())
                else:
                    processed_lengths.append(int(length))
            
            # 获取 embeddings
            t_latent = reward_model.model.get_text_embedding(processed_texts)
            m_latent = reward_model.model.get_motion_embedding(
                motions,
                processed_lengths,
                repr_type=reward_model.repr_type
            )
            
            text_list.extend(processed_texts)
            text_latents.extend(t_latent.cpu().numpy())
            motion_latents.extend(m_latent.cpu().numpy())
        
        # 计算检索指标
        test_result = [text_list, text_latents, motion_latents]
        
        if logger:
            logger.info("\n[Retrieval Metrics - Small Batches (BS=32)]")
        bs32_metrics = calculate_retrieval_metrics_small_batches(test_result, epoch=0, fptr=None)
        
        if logger:
            logger.info("\n[Retrieval Metrics - Full Dataset]")
        full_metrics = calculate_retrieval_metrics(test_result, epoch=0, fptr=None)
        
        if logger:
            logger.info("="*80)
            logger.info(f"  BS32 - R@1: {bs32_metrics['R1']:.2f}% | R@5: {bs32_metrics['R5']:.2f}% | R@10: {bs32_metrics['R10']:.2f}%")
            logger.info(f"  Full - R@1: {full_metrics['R1']:.2f}% | R@5: {full_metrics['R5']:.2f}% | R@10: {full_metrics['R10']:.2f}%")
            logger.info("="*80)
        
        return {
            'bs32': bs32_metrics,
            'full': full_metrics
        }


# ===============================================================================
#                           微调配置
# ===============================================================================

class FinetuneConfig:
    """
    微调配置
    
    控制不同 loss 的权重:
    - lambda_diff: Diffusion loss 的权重 (MSE loss)
    - lambda_reward: Reward loss 的权重 (MotionReward 相似度)
    """
    
    def __init__(
        self,
        lambda_diff: float = 1.0,
        lambda_reward: float = 1.0,
        ft_type: str = 'default',
        name: str = 'ft_mld_motionreward'
    ):
        self.lambda_diff = lambda_diff
        self.lambda_reward = lambda_reward
        self.type = ft_type
        self.name = name
    
    def to_dict(self) -> Dict:
        return {
            'lambda_diff': self.lambda_diff,
            'lambda_reward': self.lambda_reward,
            'type': self.type,
            'name': self.name,
            'enable_grad': list(range(50)),  # 默认所有 timestep 都启用梯度
        }


# ===============================================================================
#                           主训练流程
# ===============================================================================

def main():
    """主训练流程"""
    
    # ===== 1. 解析配置 =====
    cfg = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.SEED_VALUE)
    
    # 微调配置
    ft_config = FinetuneConfig(
        lambda_diff=getattr(cfg, 'ft_lambda_diff', None) or 1.0,
        lambda_reward=getattr(cfg, 'ft_lambda_reward', None) or 1.0,
        ft_type=getattr(cfg, 'ft_type', None) or 'None',
        name=getattr(cfg, 'ft_name', None) or 'ft_mld_motionreward'
    )
    
    # 输出目录
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.output_dir = f"./checkpoints/ft_mld_motionreward/{ft_config.name}_{time_str}"
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/checkpoints", exist_ok=True)
    
    # ===== 2. 设置日志 =====
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 文件日志
    file_handler = logging.FileHandler(os.path.join(cfg.output_dir, 'train.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    
    # 控制台日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)
    
    # TensorBoard
    writer = SummaryWriter(cfg.output_dir)
    
    # SwanLab 初始化
    swanlab_run = None
    if SWANLAB_AVAILABLE:
        try:
            swanlab_run = swanlab.init(
                project="MLD-MotionReward-Finetune",
                experiment_name=f"{ft_config.name}_{time_str}",
                config={
                    "ft_type": ft_config.type,
                    "lambda_diff": ft_config.lambda_diff,
                    "lambda_reward": ft_config.lambda_reward,
                    "learning_rate": getattr(cfg, 'ft_lr', 1e-5),
                    "reward_model_size": getattr(cfg, 'reward_model_size', 'small'),
                    "debug": getattr(cfg, 'debug', False),
                }
            )
            logger.info("  ✓ SwanLab initialized")
        except Exception as e:
            logger.warning(f"  SwanLab initialization failed: {e}")
            swanlab_run = None
    
    # 保存配置
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, 'config.yaml'))
    
    logger.info("="*80)
    logger.info("  Fine-tuning MLD with MotionReward")
    logger.info("="*80)
    logger.info(f"  Output Dir       : {cfg.output_dir}")
    logger.info(f"  Lambda Diff      : {ft_config.lambda_diff}")
    logger.info(f"  Lambda Reward    : {ft_config.lambda_reward}")
    logger.info(f"  FT Type          : {ft_config.type}")
    logger.info(f"  Device           : {device}")
    logger.info(f"  Debug Mode       : {getattr(cfg, 'debug', False)}")
    logger.info("="*80)
    
    # ===== 3. 加载数据集 =====
    logger.info("\n[Loading Dataset]")
    dataset = get_dataset(cfg)
    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()
    
    # Debug 模式：只使用少量数据
    if getattr(cfg, 'debug', False):
        from torch.utils.data import Subset
        debug_train_size = min(64, len(train_dataloader.dataset))
        debug_val_size = min(32, len(val_dataloader.dataset))
        train_subset = Subset(train_dataloader.dataset, range(debug_train_size))
        val_subset = Subset(val_dataloader.dataset, range(debug_val_size))
        
        # 保存原始的 collate_fn（用于处理不同长度的 motion）
        original_collate_fn = train_dataloader.collate_fn
        
        train_dataloader = torch.utils.data.DataLoader(
            train_subset, batch_size=train_dataloader.batch_size, 
            shuffle=True, num_workers=0, drop_last=True,
            collate_fn=original_collate_fn
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_subset, batch_size=val_dataloader.batch_size,
            shuffle=False, num_workers=0,
            collate_fn=original_collate_fn
        )
        
        # Debug 模式下调整 diversity_times，避免样本不足导致断言失败
        cfg.TEST.DIVERSITY_TIMES = min(cfg.TEST.DIVERSITY_TIMES, debug_val_size)
        
        logger.info(f"  [DEBUG MODE] Using subset of data")
        logger.info(f"  Train samples: {len(train_subset)} (original: {len(dataset.train_dataloader().dataset)})")
        logger.info(f"  Val samples  : {len(val_subset)} (original: {len(dataset.val_dataloader().dataset)})")
        logger.info(f"  DIVERSITY_TIMES adjusted to: {cfg.TEST.DIVERSITY_TIMES}")
    else:
        logger.info(f"  Train samples: {len(train_dataloader.dataset)}")
        logger.info(f"  Val samples  : {len(val_dataloader.dataset)}")
    
    # ===== 4. 加载 MLD 模型 =====
    logger.info("\n[Loading MLD Model]")
    model = MLD(cfg, dataset)
    
    if cfg.TRAIN.PRETRAINED:
        logger.info(f"  Loading pre-trained MLD from: {cfg.TRAIN.PRETRAINED}")
        state_dict = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]
        load_result = model.load_state_dict(state_dict, strict=False)
        logger.info(f"  Load result: {load_result}")
    else:
        logger.info("  [Warning] No pre-trained MLD checkpoint provided")
    
    # 冻结 VAE 和 Text Encoder
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.vae.eval()
    model.text_encoder.eval()
    model.to(device)
    
    logger.info("  ✓ MLD model loaded and frozen (VAE, Text Encoder)")
    
    # ===== 5. 加载 MotionReward 模型 =====
    logger.info("\n[Loading MotionReward Model]")
    
    # 确定 checkpoint 路径
    reward_model_size = getattr(cfg, 'reward_model_size', None) or 'small'
    stage1_ckpt = getattr(cfg, 'reward_stage1_ckpt', None)
    
    # 如果命令行传入空字符串，视为 None
    if stage1_ckpt == '':
        stage1_ckpt = None
    
    if not stage1_ckpt:
        # 自动查找最新的 checkpoint
        checkpoint_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'checkpoints', 'motionreward'
        )
        stage1_files = glob.glob(os.path.join(checkpoint_dir, '*_stage1_*_retrieval_backbone_*.pth'))
        if stage1_files:
            stage1_ckpt = max(stage1_files, key=os.path.getmtime)
            logger.info(f"  [Auto-selected] {stage1_ckpt}")
        else:
            raise FileNotFoundError(f"No Stage 1 checkpoint found in {checkpoint_dir}")
    
    # T5 路径
    t5_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'deps', 'sentence-t5-large'
    )
    
    # 使用 MotionRewardLoader 加载模型
    reward_model_base = MotionRewardLoader.load_model(
        model_size=reward_model_size,
        stage1_ckpt=stage1_ckpt,
        critic_lora_ckpt=None,  # 可选：如果需要 Critic LoRA
        t5_path=t5_path,
        repr_type='263',
        device=device,
        verbose=True
    )
    
    # 包装为 RewardModelWrapper
    reward_model = RewardModelWrapper(
        model=reward_model_base,
        repr_type='263',
        device=device
    )
    
    logger.info("  ✓ MotionReward model loaded successfully")
    
    # ===== 6. 评估 Reward Model（训练前）=====
    eval_reward_before_ft = getattr(cfg, 'eval_reward_before_ft', True)
    # 处理字符串形式的布尔值
    if isinstance(eval_reward_before_ft, str):
        eval_reward_before_ft = eval_reward_before_ft.lower() in ('true', '1', 'yes')
    
    if eval_reward_before_ft:
        logger.info("\n" + "="*80)
        logger.info("  [Pre-Training Evaluation] Evaluating Reward Model Performance")
        logger.info("="*80)
        reward_metrics = RewardModelEvaluator.evaluate(
            reward_model=reward_model,
            dataloader=val_dataloader,
            device=device,
            logger=logger
        )
        
        # 记录到 TensorBoard
        writer.add_scalar("RewardModel/BS32_R1", reward_metrics['bs32']['R1'], global_step=0)
        writer.add_scalar("RewardModel/BS32_R5", reward_metrics['bs32']['R5'], global_step=0)
        writer.add_scalar("RewardModel/BS32_R10", reward_metrics['bs32']['R10'], global_step=0)
        writer.add_scalar("RewardModel/Full_R1", reward_metrics['full']['R1'], global_step=0)
        writer.add_scalar("RewardModel/Full_R5", reward_metrics['full']['R5'], global_step=0)
        writer.add_scalar("RewardModel/Full_R10", reward_metrics['full']['R10'], global_step=0)
        
        logger.info("\n  [Pre-Training Evaluation Complete]")
    else:
        logger.info("\n[Skipping Reward Model Evaluation (eval_reward_before_ft=False)]")
    
    # 将 reward model 挂载到 MLD
    model.reward_model = reward_model
    model.ft_config = ft_config.to_dict()
    model.lambda_diff = ft_config.lambda_diff
    model.lambda_reward = ft_config.lambda_reward
    model.trn_reward = []  # 用于记录训练过程中的 reward
    model.reward_record = [[] for _ in range(50)]  # 用于记录每个 timestep 的 reward
    
    # ===== 7. 设置优化器和调度器 =====
    logger.info("\n[Setting up Optimizer and Scheduler]")
    
    cfg.TRAIN.learning_rate = float(getattr(cfg, 'ft_lr', 1e-5))
    logger.info(f"  Learning rate: {cfg.TRAIN.learning_rate}")
    
    optimizer = torch.optim.AdamW(
        model.denoiser.parameters(),
        lr=cfg.TRAIN.learning_rate,
        betas=(cfg.TRAIN.adam_beta1, cfg.TRAIN.adam_beta2),
        weight_decay=cfg.TRAIN.adam_weight_decay,
        eps=cfg.TRAIN.adam_epsilon
    )
    
    max_ft_epochs = getattr(cfg.TRAIN, 'max_ft_epochs', 10)
    max_ft_steps = max_ft_epochs * len(train_dataloader)
    
    lr_scheduler = get_scheduler(
        cfg.TRAIN.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.TRAIN.lr_warmup_steps,
        num_training_steps=max_ft_steps
    )
    
    logger.info(f"  Max epochs       : {max_ft_epochs}")
    logger.info(f"  Max steps        : {max_ft_steps}")
    logger.info(f"  Warmup steps     : {cfg.TRAIN.lr_warmup_steps}")
    
    # 验证频率（每 100 步验证一次）
    validation_steps = getattr(cfg.TRAIN, 'validation_steps', 100)
    if validation_steps <= 0:
        validation_steps = 100
    logger.info(f"  Validation steps : {validation_steps}")
    
    # ===== 8. 验证函数 =====
    def compute_val_reward(target_model: MLD, batch: dict) -> float:
        """计算验证集上的 reward（使用生成的 motion）"""
        with torch.no_grad():
            # 生成 motion
            feats_ref = batch["motion"]
            mask = batch['mask']
            text = batch["text"]
            
            # 编码文本
            # 对于 CFG，需要同时编码条件文本和无条件文本（空字符串）
            if target_model.do_classifier_free_guidance:
                texts_for_encoding = text + [""] * len(text)
            else:
                texts_for_encoding = text
            text_emb = target_model.text_encoder(texts_for_encoding)
            
            # 生成 latent
            latent_dim = target_model.latent_dim
            x_T = torch.randn((len(text), *latent_dim), device=text_emb.device)
            
            # Diffusion reverse
            target_model.scheduler.set_timesteps(target_model.cfg.model.scheduler.num_inference_steps)
            latents = x_T * target_model.scheduler.init_noise_sigma
            
            for t in target_model.scheduler.timesteps:
                t = t.to(latents.device)  # 确保 timestep 在正确的设备上
                latent_model_input = torch.cat([latents] * 2) if target_model.do_classifier_free_guidance else latents
                latent_model_input = target_model.scheduler.scale_model_input(latent_model_input, t)
                
                noise_pred, _ = target_model.denoiser(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=text_emb
                )
                
                if target_model.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + target_model.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                latents = target_model.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Decode to motion
            x_0 = latents / target_model.vae_scale_factor
            recons_motion = target_model.vae.decode(x_0, mask)
            
            # 计算 reward
            if target_model.reward_model is not None:
                reward = target_model.reward_model.get_reward_t2m(
                    raw_texts=batch["text"],
                    motion_feats=recons_motion,
                    m_len=batch["length"],
                    t_len=None,
                    sent_emb=None,
                    timestep=torch.tensor(0, dtype=torch.long).to(x_0.device)
                )
                return reward.mean().item()
            return 0.0
    
    @torch.no_grad()
    def validation(target_model: MLD, global_step: int) -> Tuple[float, float, Dict]:
        """验证函数"""
        target_model.denoiser.eval()
        val_loss_list = []
        val_rewards = []
        
        for val_batch in tqdm(val_dataloader, desc="Validation"):
            val_batch = move_batch_to_device(val_batch, device)
            val_loss_dict = target_model.allsplit_step(split='val', batch=val_batch)
            val_loss_list.append(val_loss_dict)
            
            # 计算验证集上的 reward（使用生成的 motion）
            val_reward = compute_val_reward(target_model, val_batch)
            val_rewards.append(val_reward)
        
        metrics = target_model.allsplit_epoch_end()
        metrics["Val/loss"] = sum([d['loss'] for d in val_loss_list]).item() / len(val_dataloader)
        metrics["Val/diff_loss"] = sum([d['diff_loss'] for d in val_loss_list]).item() / len(val_dataloader)
        metrics["Val/reward"] = sum(val_rewards) / len(val_rewards) if val_rewards else 0.0
        
        max_val_rp1 = metrics['Metrics/R_precision_top_1']
        min_val_fid = metrics['Metrics/FID']
        
        print_table(f'Validation@Step-{global_step}', metrics)
        
        for mk, mv in metrics.items():
            writer.add_scalar(mk, mv, global_step=global_step)
        
        # SwanLab 记录
        if swanlab_run is not None:
            swanlab.log(metrics, step=global_step)
        
        target_model.denoiser.train()
        return max_val_rp1, min_val_fid, metrics
    
    # ===== 9. 训练循环 =====
    logger.info("\n" + "="*80)
    logger.info("  Starting Fine-tuning")
    logger.info("="*80)
    
    global_step = 0
    max_rp1, min_fid = 0.0, float('inf')
    
    progress_bar = tqdm(range(max_ft_steps), desc="Training")
    
    for epoch in range(max_ft_epochs):
        logger.info(f"\n[Epoch {epoch+1}/{max_ft_epochs}]")
        
        for step, batch in enumerate(train_dataloader):
            batch = move_batch_to_device(batch, device)
            
            # Forward + Backward
            if ft_config.type == 'NIPS':
                loss_dict = model.allsplit_step('finetune_nips', batch, optimizer, lr_scheduler)
            else:
                loss_dict = model.allsplit_step('finetune', batch)
                
                loss = loss_dict['loss']
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), cfg.TRAIN.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # 记录日志
            diff_loss = loss_dict['diff_loss']
            reward = loss_dict['reward']
            loss = loss_dict['loss']
            
            progress_bar.update(1)
            global_step += 1
            
            logs = {
                'Epoch': epoch,
                'loss': loss.item(),
                'diff_loss': diff_loss.item(),
                'reward': reward.item(),
                'lr': lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)
            
            for k, v in logs.items():
                writer.add_scalar(f"train/{k}", v, global_step=global_step)
            
            # SwanLab 记录训练指标
            if swanlab_run is not None:
                swanlab.log({f"train/{k}": v for k, v in logs.items()}, step=global_step)
            
            # 每 validation_steps 步验证一次
            if global_step % validation_steps == 0:
                cur_rp1, cur_fid, metrics = validation(model, global_step)
                
                # 保存 checkpoint
                save_path = os.path.join(
                    cfg.output_dir, 'checkpoints',
                    f"step{global_step}_R1-{cur_rp1:.4f}_FID-{cur_fid:.4f}.ckpt"
                )
                
                ckpt = {
                    'state_dict': model.state_dict(),
                    'ft_config': model.ft_config,
                    'metrics': metrics,
                    'epoch': epoch,
                    'global_step': global_step
                }
                model.on_save_checkpoint(ckpt)
                torch.save(ckpt, save_path)
                
                logger.info(f"  ✓ Checkpoint saved: {save_path}")
                logger.info(f"    R@1: {cur_rp1:.4f} | FID: {cur_fid:.4f}")
                
                # 更新最佳指标
                if cur_rp1 > max_rp1:
                    max_rp1 = cur_rp1
                if cur_fid < min_fid:
                    min_fid = cur_fid
    
    # ===== 10. 训练结束 =====
    logger.info("\n" + "="*80)
    logger.info("  Fine-tuning Complete")
    logger.info("="*80)
    logger.info(f"  Best R@1 : {max_rp1:.4f}")
    logger.info(f"  Best FID : {min_fid:.4f}")
    logger.info(f"  Output   : {cfg.output_dir}")
    logger.info("="*80)
    
    writer.close()
    
    # 关闭 SwanLab
    if swanlab_run is not None:
        swanlab.finish()


if __name__ == "__main__":
    main()
