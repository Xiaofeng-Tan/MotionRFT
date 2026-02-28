"""
eval_hy.py

HY-Motion 模型评估脚本 - 使用 SPM 模型计算评估指标

功能：
1. 评估 GT 数据集的指标（使用数据集的 mean/std 归一化）
2. 评估生成的 motion 指标（使用整个数据集的所有 prompt）
3. 指标：R-Precision (Top-1/2/3)、FID、Matching Score、Diversity

用法:
    # ==================== HY-Motion-1.0-Lite ====================

    # 评估 Lite pretrained (GT + Generated)
    CUDA_VISIBLE_DEVICES=1 python eval_hy.py \
        --spm_path ./t2m/evaluator.pth \
        --t5_path ../deps/sentence-t5-large \
        --data_root ../datasets/humanml3d \
        --motion_dir ../datasets/humanml3d/joints_6d \
        --model_path ../pretrain/hymotion/HY-Motion-1.0-Lite \
        --eval_mode both \
        --repeat_times 5 \
        --split test

    # 评估 Lite LoRA 微调后的模型
    CUDA_VISIBLE_DEVICES=1 python eval_hy.py \
        --spm_path ./t2m/evaluator.pth \
        --t5_path ../deps/sentence-t5-large \
        --data_root ../datasets/humanml3d \
        --motion_dir ../datasets/humanml3d/joints_6d \
        --model_path ../pretrain/hymotion/HY-Motion-1.0-Lite \
        --lora_path /path/to/lora_checkpoint.pth \
        --eval_mode both \
        --repeat_times 5 \
        --split test

    # ==================== HY-Motion-1.0 (Full) ====================

    # 评估 Full pretrained (GT + Generated)
    CUDA_VISIBLE_DEVICES=1 python eval_hy.py \
        --spm_path ./t2m/evaluator.pth \
        --t5_path ../deps/sentence-t5-large \
        --data_root ../datasets/humanml3d \
        --motion_dir ../datasets/humanml3d/joints_6d \
        --model_path ../pretrain/hymotion/HY-Motion-1.0 \
        --eval_mode generated \
        --repeat_times 1 \
        --split test

    # 评估 Full LoRA 微调后的模型
    CUDA_VISIBLE_DEVICES=1 python eval_hy.py \
        --spm_path ./t2m/evaluator.pth \
        --t5_path ../deps/sentence-t5-large \
        --data_root ../datasets/humanml3d \
        --motion_dir ../datasets/humanml3d/joints_6d \
        --model_path ../pretrain/hymotion/HY-Motion-1.0 \
        --lora_path /path/to/lora_checkpoint.pth \
        --eval_mode both \
        --repeat_times 5 \
        --split test

    # ==================== 仅评估 GT ====================

    # 仅评估 GT 数据集 (不需要 model_path)
    python eval_hy.py \
        --spm_path ./t2m/evaluator.pth \
        --t5_path ../deps/sentence-t5-large \
        --data_root ../datasets/humanml3d \
        --motion_dir ../datasets/humanml3d/joints_6d \
        --eval_mode gt \
        --repeat_times 5 \
        --split test
"""

import os

# 通过环境变量控制 GPU，需在 import torch 之前设置
# 用法: CUDA_VISIBLE_DEVICES=1 python eval_hy.py ...
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import json
import argparse
import random
import datetime
from os.path import join as pjoin
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy import linalg

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 切换工作目录到 RFT_HY，以便 WoodenMesh 能找到相对路径的资源文件
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_script_dir)

# 导入 SPM 模型
from ReAlignModule.models.spm import SPM, process_T5_outputs, load_SPM

# 导入 HY-Motion 相关模块
from hymotion.pipeline.motion_diffusion import MotionFlowMatching, length_to_mask, randn_tensor
from hymotion.utils.loaders import load_object


#################################################################################
#                              数据集定义                                        #
#################################################################################

class HumanML3DEvalDataset(Dataset):
    """HumanML3D 评估数据集 (135维 Hunyuan 格式) - 使用 mean/std 归一化"""
    
    def __init__(
        self,
        data_root: str,
        split: str = 'test',
        max_motion_length: int = 196,
        min_motion_length: int = 40,
        motion_dir: str = None,
    ):
        self.data_root = data_root
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        
        # 动作和文本目录
        if motion_dir is not None:
            self.motion_dir = motion_dir
        else:
            self.motion_dir = pjoin(data_root, 'joints_hunyuan')
            if not os.path.exists(self.motion_dir):
                self.motion_dir = pjoin(data_root, 'new_joint_vecs')
                print(f'Using motion dir: {self.motion_dir}')
        self.text_dir = pjoin(data_root, 'texts')
        
        # 加载 mean/std - 优先从 motion_dir 加载
        if os.path.exists(pjoin(self.motion_dir, 'Mean.npy')):
            mean_path = pjoin(self.motion_dir, 'Mean.npy')
            std_path = pjoin(self.motion_dir, 'Std.npy')
            print(f'Loading Mean/Std from motion_dir: {self.motion_dir}')
        elif os.path.exists(pjoin(data_root, 'Mean_135.npy')):
            mean_path = pjoin(data_root, 'Mean_135.npy')
            std_path = pjoin(data_root, 'Std_135.npy')
            print(f'Loading Mean_135/Std_135 from data_root: {data_root}')
        else:
            mean_path = pjoin(data_root, 'Mean.npy')
            std_path = pjoin(data_root, 'Std.npy')
            print(f'Loading Mean/Std from data_root: {data_root}')
        
        if os.path.exists(mean_path):
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
            print(f'Loaded Mean/Std with shape: {self.mean.shape}')
        else:
            self.mean = np.zeros(135)
            self.std = np.ones(135)
            print(f'Warning: Mean/Std not found, using default values')
        
        # 加载 split 文件
        split_file = pjoin(data_root, f'{split}.txt')
        with open(split_file, 'r') as f:
            self.name_list = [line.strip() for line in f.readlines()]
        
        # 过滤有效样本
        self.data_list = []
        for name in tqdm(self.name_list, desc='Loading dataset'):
            motion_path = pjoin(self.motion_dir, f'{name}.npy')
            text_path = pjoin(self.text_dir, f'{name}.txt')
            
            if not os.path.exists(motion_path) or not os.path.exists(text_path):
                continue
                
            motion = np.load(motion_path)
            if motion.shape[0] < self.min_motion_length:
                continue
                
            with open(text_path, 'r') as f:
                texts = []
                for line in f.readlines():
                    text = line.strip().split('#')[0]
                    if text:
                        texts.append(text)
                
            if len(texts) == 0:
                continue
                
            self.data_list.append({
                'name': name,
                'motion_path': motion_path,
                'texts': texts,
                'length': min(motion.shape[0], self.max_motion_length)
            })
        
        print(f'Loaded {len(self.data_list)} samples from {split} set')
        
        # 更新 nfeats
        sample_motion = np.load(self.data_list[0]['motion_path'])
        self.nfeats = sample_motion.shape[-1]
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        motion = np.load(data['motion_path'])
        length = data['length']
        
        if motion.shape[0] > self.max_motion_length:
            motion = motion[:self.max_motion_length]
        
        # 使用数据集的 mean/std 归一化
        motion = (motion - self.mean) / self.std
        text = data['texts'][0]  # 使用第一个文本描述
        
        return {
            'name': data['name'],
            'motion': torch.from_numpy(motion).float(),
            'text': text,
            'length': length,
            'all_texts': data['texts'],  # 保留所有文本用于可能的多次评估
        }


def eval_collate_fn(batch):
    """数据批次整理函数"""
    names = [item['name'] for item in batch]
    texts = [item['text'] for item in batch]
    lengths = [item['length'] for item in batch]
    all_texts = [item['all_texts'] for item in batch]
    
    max_len = max(lengths)
    motions = []
    for item in batch:
        motion = item['motion']
        if motion.shape[0] < max_len:
            padding = torch.zeros(max_len - motion.shape[0], motion.shape[1])
            motion = torch.cat([motion, padding], dim=0)
        motions.append(motion)
    
    motions = torch.stack(motions, dim=0)
    
    return {
        'name': names,
        'motion': motions,
        'text': texts,
        'length': lengths,
        'all_texts': all_texts,
    }


#################################################################################
#                              LoRA 层定义                                       #
#################################################################################

class LoRALayer(nn.Module):
    """LoRA 层 - 与训练代码保持一致"""
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 与训练代码一致: lora_A (rank, in_features), lora_B (out_features, rank)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(x.shape[:-1] + (self.lora_B.shape[0],), device=x.device, dtype=x.dtype)
        # 与训练代码一致: F.linear(F.linear(x, lora_A), lora_B)
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


def inject_lora_layers(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: List[str] = None,
) -> Dict[str, LoRALayer]:
    """注入 LoRA 层到模型 - 与训练代码保持一致"""
    if target_modules is None:
        # 与训练代码一致
        target_modules = ['qkv', 'proj', 'fc1', 'fc2', 'to_q', 'to_k', 'to_v', 'to_out']
    
    lora_layers = {}
    
    def _apply_lora_recursive(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                # 与训练代码一致：使用 name.lower()
                should_apply = any(target in name.lower() for target in target_modules)
                if should_apply:
                    lora_layer = LoRALayer(
                        child.in_features,
                        child.out_features,
                        rank=rank,
                        alpha=alpha
                    ).to(child.weight.device).to(child.weight.dtype)
                    
                    lora_layers[full_name] = lora_layer
                    
                    original_forward = child.forward
                    
                    def make_lora_forward(orig_forward, lora):
                        def lora_forward(x):
                            return orig_forward(x) + lora(x)
                        return lora_forward
                    
                    child.forward = make_lora_forward(original_forward, lora_layer)
            else:
                _apply_lora_recursive(child, full_name)
    
    _apply_lora_recursive(model)
    print(f'Injected LoRA layers into {len(lora_layers)} modules')
    return lora_layers


#################################################################################
#                              评价指标计算                                      #
#################################################################################

def euclidean_distance_matrix(matrix1, matrix2):
    """计算欧氏距离矩阵"""
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    dists = np.sqrt(np.maximum(d1 + d2 + d3, 0))
    return dists


def calculate_top_k(mat, top_k):
    """计算 Top-K 准确率"""
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k=3, sum_all=False):
    """计算 R-Precision"""
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_activation_statistics(activations):
    """计算激活统计量 (用于 FID)"""
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算 Frechet Distance (FID)"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_diversity(activation, diversity_times=300):
    """计算 Diversity"""
    num_samples = activation.shape[0]
    if num_samples < diversity_times * 2:
        diversity_times = num_samples // 2
    
    if diversity_times <= 0:
        return 0.0
    
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation, multimodality_times=20):
    """计算 Multimodality"""
    num_samples = activation.shape[0]
    if num_samples < multimodality_times * 2:
        multimodality_times = num_samples // 2
    
    if multimodality_times <= 0:
        return 0.0
    
    first_indices = np.random.choice(num_samples, multimodality_times, replace=False)
    second_indices = np.random.choice(num_samples, multimodality_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def get_metric_statistics(values, replication_times):
    """
    计算指标的均值和 95% 置信区间 (参考 MotionLCM test.py)
    
    Args:
        values: np.ndarray, shape (replication_times,) 或 (replication_times, ...)
        replication_times: 重复次数
    
    Returns:
        mean: 均值
        conf_interval: 95% 置信区间
    """
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


#################################################################################
#                              采样器                                           #
#################################################################################

class FlowMatchingSampler:
    """Flow Matching 采样器（用于生成 motion）"""
    
    def __init__(
        self,
        model: MotionFlowMatching,
        num_steps: int = 50,
        cfg_scale: float = 5.0,
    ):
        self.model = model
        self.num_steps = num_steps
        self.cfg_scale = cfg_scale
    
    @torch.no_grad()
    def sample(
        self,
        texts: List[str],
        lengths: List[int],
        hidden_state_dict: Optional[Dict] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """采样生成 motion"""
        device = next(self.model.parameters()).device
        bs = len(texts)
        max_length = max(lengths)
        
        # 编码文本
        if hidden_state_dict is None:
            hidden_state_dict = self.model.encode_text({"text": texts})
        
        vtxt_input = hidden_state_dict["text_vec_raw"]
        ctxt_input = hidden_state_dict["text_ctxt_raw"]
        ctxt_length = hidden_state_dict["text_ctxt_raw_length"]
        
        # 处理 shape
        if len(vtxt_input.shape) == 2:
            vtxt_input = vtxt_input.unsqueeze(0).repeat(bs, 1, 1)
            ctxt_input = ctxt_input.unsqueeze(0).repeat(bs, 1, 1)
            ctxt_length = ctxt_length.repeat(bs)
        
        ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
        x_length = torch.LongTensor(lengths).to(device)
        x_mask_temporal = length_to_mask(x_length, self.model.train_frames)
        
        # CFG 设置
        do_cfg = self.cfg_scale > 1.0 and not self.model.uncondition_mode
        if do_cfg:
            silent_text_feat = self.model.null_vtxt_feat.expand(*vtxt_input.shape)
            vtxt_input = torch.cat([silent_text_feat, vtxt_input], dim=0)
            
            if self.model.enable_ctxt_null_feat:
                silent_ctxt_input = self.model.null_ctxt_input.expand(*ctxt_input.shape)
            else:
                silent_ctxt_input = ctxt_input
            ctxt_input = torch.cat([silent_ctxt_input, ctxt_input], dim=0)
            
            ctxt_mask_temporal = torch.cat([ctxt_mask_temporal] * 2, dim=0)
            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)
        
        # 定义速度场预测函数
        def predict_velocity(x: torch.Tensor, t_value: float) -> torch.Tensor:
            x_input = torch.cat([x] * 2, dim=0) if do_cfg else x
            t_tensor = torch.tensor([t_value], device=device, dtype=x.dtype)
            
            x_pred = self.model.motion_transformer(
                x=x_input,
                ctxt_input=ctxt_input,
                vtxt_input=vtxt_input,
                timesteps=t_tensor.expand(x_input.shape[0]),
                x_mask_temporal=x_mask_temporal,
                ctxt_mask_temporal=ctxt_mask_temporal,
            )
            if do_cfg:
                x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
                x_pred = x_pred_basic + self.cfg_scale * (x_pred_text - x_pred_basic)
            return x_pred
        
        # 初始化噪声
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None
        
        y0 = randn_tensor(
            (bs, self.model.train_frames, self.model._network_module_args["input_dim"]),
            generator=generator,
            device=device,
            dtype=vtxt_input.dtype
        )
        
        dt = 1.0 / self.num_steps
        x_t = y0
        
        # 完整采样
        for step in range(self.num_steps):
            t_current = step / self.num_steps
            v = predict_velocity(x_t, t_current)
            x_t = x_t + dt * v
        
        sampled = x_t
        
        # 截断到目标长度
        sampled = sampled[:, :max_length, :]
        
        # 反归一化
        std_zero = self.model.std < 1e-3
        std = torch.where(std_zero, torch.zeros_like(self.model.std), self.model.std)
        latent_denorm = sampled * std + self.model.mean
        
        return sampled, {
            'latent_denorm': latent_denorm,
            'lengths': lengths,
        }


#################################################################################
#                              评估器                                           #
#################################################################################

class HYMotionEvaluator:
    """HY-Motion 评估器"""
    
    def __init__(
        self,
        spm_model: SPM,
        eval_loader: DataLoader,
        device: torch.device,
        nfeats: int = 135,
        hymotion_model: Optional[MotionFlowMatching] = None,
        num_inference_steps: int = 50,
        cfg_scale: float = 5.0,
    ):
        self.spm_model = spm_model
        self.eval_loader = eval_loader
        self.device = device
        self.nfeats = nfeats
        self.hymotion_model = hymotion_model
        
        # 创建采样器（如果有 HY-Motion 模型）
        if hymotion_model is not None:
            self.sampler = FlowMatchingSampler(
                model=hymotion_model,
                num_steps=num_inference_steps,
                cfg_scale=cfg_scale,
            )
        else:
            self.sampler = None
    
    @torch.no_grad()
    def evaluate_gt(
        self,
        num_samples: Optional[int] = None,
        repeat_times: int = 1,
    ) -> Dict:
        """
        评估 GT 数据集的指标
        
        Args:
            num_samples: 评估样本数（None 表示使用全部）
            repeat_times: 重复评估次数
        
        Returns:
            metrics: 评估指标字典
        """
        self.spm_model.eval()
        
        all_results = {
            'fid': [], 'div': [], 
            'top1': [], 'top2': [], 'top3': [],
            'matching': [],
        }
        
        for rep in range(repeat_times):
            if repeat_times > 1:
                print(f'\n  Repeat {rep + 1}/{repeat_times}')
            
            gt_motion_latents = []
            text_latents = []
            
            R_precision_sum = np.array([0., 0., 0.])
            matching_score_sum = 0.0
            nb_sample = 0
            sample_count = 0
            
            for batch in tqdm(self.eval_loader, desc=f'Evaluating GT (Rep {rep + 1})'):
                if num_samples is not None and sample_count >= num_samples:
                    break
                
                texts = batch['text']
                gt_motions = batch['motion'].to(self.device)
                lengths = batch['length']
                bs = len(texts)
                
                # 限制样本数
                if num_samples is not None:
                    remaining = num_samples - sample_count
                    if bs > remaining:
                        texts = texts[:remaining]
                        gt_motions = gt_motions[:remaining]
                        lengths = lengths[:remaining]
                        bs = remaining
                
                # 编码文本
                t_len, token_emb, cls_token = process_T5_outputs(texts, self.spm_model.clip)
                token_emb = token_emb.to(self.device).float()
                
                # 编码 GT 动作（已归一化）
                gt_m_latent = self.spm_model.encode_motion(gt_motions, lengths)[0].squeeze()
                
                # 编码文本 latent
                t_latent = self.spm_model.encode_text(token_emb, t_len)[0].squeeze()
                
                # 归一化
                gt_m_latent_norm = F.normalize(gt_m_latent, dim=-1)
                t_latent_norm = F.normalize(t_latent, dim=-1)
                
                # 收集用于 FID 和 Diversity 计算
                gt_motion_latents.append(gt_m_latent_norm.cpu().numpy())
                text_latents.append(t_latent_norm.cpu().numpy())
                
                # 计算 R-Precision（GT motion vs text）
                temp_R = calculate_R_precision(
                    t_latent_norm.cpu().numpy(),
                    gt_m_latent_norm.cpu().numpy(),
                    top_k=3, sum_all=True
                )
                R_precision_sum += temp_R
                
                # 计算 Matching Score
                temp_match = euclidean_distance_matrix(
                    t_latent_norm.cpu().numpy(),
                    gt_m_latent_norm.cpu().numpy()
                ).trace()
                matching_score_sum += temp_match
                
                nb_sample += bs
                sample_count += bs
            
            # 汇总结果
            gt_motion_latents = np.concatenate(gt_motion_latents, axis=0)
            text_latents_np = np.concatenate(text_latents, axis=0)
            
            # 计算 FID（GT vs GT，应该接近 0）
            gt_mu, gt_cov = calculate_activation_statistics(gt_motion_latents)
            fid = 0.0  # GT vs GT 的 FID 为 0
            
            # 计算 Diversity
            diversity = calculate_diversity(gt_motion_latents, min(300, nb_sample // 2))
            
            # 计算 R-Precision
            R_precision = R_precision_sum / nb_sample
            
            # 计算 Matching Score
            matching_score = matching_score_sum / nb_sample
            
            # 记录结果
            all_results['fid'].append(fid)
            all_results['div'].append(diversity)
            all_results['top1'].append(R_precision[0])
            all_results['top2'].append(R_precision[1])
            all_results['top3'].append(R_precision[2])
            all_results['matching'].append(matching_score)
            
            if repeat_times > 1:
                print(f'  Rep {rep + 1}: FID={fid:.4f}, Div={diversity:.4f}, '
                      f'Top1={R_precision[0]:.4f}, Top2={R_precision[1]:.4f}, Top3={R_precision[2]:.4f}, '
                      f'Matching={matching_score:.4f}')
        
        # 计算平均值和置信区间 (参考 MotionLCM: 20次重复 + 95% CI)
        metrics = {}
        for key, values in all_results.items():
            values = np.array(values)
            mean, conf_interval = get_metric_statistics(values, repeat_times)
            metrics[key] = float(mean)
            metrics[f'{key}_conf'] = float(conf_interval)
            metrics[f'{key}_all'] = values.tolist()
        
        return metrics
    
    @torch.no_grad()
    def evaluate_generated(
        self,
        num_samples: Optional[int] = None,
        repeat_times: int = 1,
    ) -> Dict:
        """
        评估生成的 motion 指标
        
        Args:
            num_samples: 评估样本数（None 表示使用全部）
            repeat_times: 重复评估次数
        
        Returns:
            metrics: 评估指标字典
        """
        if self.hymotion_model is None or self.sampler is None:
            raise ValueError("HY-Motion model is required for evaluating generated motion")
        
        self.spm_model.eval()
        self.hymotion_model.eval()
        
        # 获取数据集的 mean/std
        base_dataset = self.eval_loader.dataset
        if hasattr(base_dataset, 'dataset'):
            base_dataset = base_dataset.dataset
        mean = torch.from_numpy(base_dataset.mean[:self.nfeats]).float().to(self.device)
        std = torch.from_numpy(base_dataset.std[:self.nfeats]).float().to(self.device)
        
        all_results = {
            'fid': [], 'div': [], 
            'top1': [], 'top2': [], 'top3': [],
            'matching': [],
        }
        
        for rep in range(repeat_times):
            if repeat_times > 1:
                print(f'\n  Repeat {rep + 1}/{repeat_times}')
            
            gt_motion_latents = []
            pred_motion_latents = []
            text_latents = []
            
            R_precision_sum = np.array([0., 0., 0.])
            matching_score_sum = 0.0
            nb_sample = 0
            sample_count = 0
            
            for batch in tqdm(self.eval_loader, desc=f'Evaluating Generated (Rep {rep + 1})'):
                if num_samples is not None and sample_count >= num_samples:
                    break
                
                texts = batch['text']
                gt_motions = batch['motion'].to(self.device)
                lengths = batch['length']
                bs = len(texts)
                
                # 限制样本数
                if num_samples is not None:
                    remaining = num_samples - sample_count
                    if bs > remaining:
                        texts = texts[:remaining]
                        gt_motions = gt_motions[:remaining]
                        lengths = lengths[:remaining]
                        bs = remaining
                
                # 编码文本
                t_len, token_emb, cls_token = process_T5_outputs(texts, self.spm_model.clip)
                token_emb = token_emb.to(self.device).float()
                
                # 编码 GT 动作（已归一化）
                gt_m_latent = self.spm_model.encode_motion(gt_motions, lengths)[0].squeeze()
                
                # 编码文本 latent
                t_latent = self.spm_model.encode_text(token_emb, t_len)[0].squeeze()
                
                # 生成 motion
                try:
                    seed = random.randint(0, 99999)
                    sampled, output_dict = self.sampler.sample(
                        texts=texts,
                        lengths=lengths,
                        seed=seed,
                    )
                    
                    latent_denorm = output_dict['latent_denorm']
                    pred_motion_feats = latent_denorm[:, :, :self.nfeats].to(self.device)
                    
                    # 标准化生成的 motion（与 GT 保持一致）
                    max_len = max(lengths)
                    pred_motions_norm = []
                    for i in range(bs):
                        m = pred_motion_feats[i, :lengths[i], :]
                        m_norm = (m - mean) / std
                        if m_norm.shape[0] < max_len:
                            padding = torch.zeros(max_len - m_norm.shape[0], self.nfeats, device=self.device)
                            m_norm = torch.cat([m_norm, padding], dim=0)
                        pred_motions_norm.append(m_norm)
                    pred_motions_norm = torch.stack(pred_motions_norm, dim=0)
                    
                    # 编码生成的动作
                    pred_m_latent = self.spm_model.encode_motion(pred_motions_norm, lengths)[0].squeeze()
                    
                except Exception as e:
                    print(f'Generation failed: {e}')
                    import traceback
                    traceback.print_exc()
                    pred_m_latent = gt_m_latent
                
                # 归一化
                gt_m_latent_norm = F.normalize(gt_m_latent, dim=-1)
                pred_m_latent_norm = F.normalize(pred_m_latent, dim=-1)
                t_latent_norm = F.normalize(t_latent, dim=-1)
                
                # 收集用于 FID 和 Diversity 计算
                gt_motion_latents.append(gt_m_latent_norm.cpu().numpy())
                pred_motion_latents.append(pred_m_latent_norm.cpu().numpy())
                text_latents.append(t_latent_norm.cpu().numpy())
                
                # 计算 R-Precision（生成 motion vs text）
                temp_R = calculate_R_precision(
                    t_latent_norm.cpu().numpy(),
                    pred_m_latent_norm.cpu().numpy(),
                    top_k=3, sum_all=True
                )
                R_precision_sum += temp_R
                
                # 计算 Matching Score
                temp_match = euclidean_distance_matrix(
                    t_latent_norm.cpu().numpy(),
                    pred_m_latent_norm.cpu().numpy()
                ).trace()
                matching_score_sum += temp_match
                
                nb_sample += bs
                sample_count += bs
            
            # 汇总结果
            gt_motion_latents = np.concatenate(gt_motion_latents, axis=0)
            pred_motion_latents = np.concatenate(pred_motion_latents, axis=0)
            text_latents_np = np.concatenate(text_latents, axis=0)
            
            # 计算 FID（生成 vs GT）
            gt_mu, gt_cov = calculate_activation_statistics(gt_motion_latents)
            pred_mu, pred_cov = calculate_activation_statistics(pred_motion_latents)
            fid = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)
            
            # 计算 Diversity（生成 motion 的多样性）
            diversity = calculate_diversity(pred_motion_latents, min(300, nb_sample // 2))
            
            # 计算 R-Precision
            R_precision = R_precision_sum / nb_sample
            
            # 计算 Matching Score
            matching_score = matching_score_sum / nb_sample
            
            # 记录结果
            all_results['fid'].append(fid)
            all_results['div'].append(diversity)
            all_results['top1'].append(R_precision[0])
            all_results['top2'].append(R_precision[1])
            all_results['top3'].append(R_precision[2])
            all_results['matching'].append(matching_score)
            
            if repeat_times > 1:
                print(f'  Rep {rep + 1}: FID={fid:.4f}, Div={diversity:.4f}, '
                      f'Top1={R_precision[0]:.4f}, Top2={R_precision[1]:.4f}, Top3={R_precision[2]:.4f}, '
                      f'Matching={matching_score:.4f}')
        
        # 计算平均值和置信区间 (参考 MotionLCM: 20次重复 + 95% CI)
        metrics = {}
        for key, values in all_results.items():
            values = np.array(values)
            mean_val, conf_interval = get_metric_statistics(values, repeat_times)
            metrics[key] = float(mean_val)
            metrics[f'{key}_conf'] = float(conf_interval)
            metrics[f'{key}_all'] = values.tolist()
        
        return metrics


#################################################################################
#                              主函数                                           #
#################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description='HY-Motion Evaluation Script')
    
    # 模型路径
    parser.add_argument('--spm_path', type=str, required=True,
                        help='Path to SPM model checkpoint')
    parser.add_argument('--t5_path', type=str, required=True,
                        help='Path to Sentence-T5 model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to HY-Motion model (required for generated mode)')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Path to LoRA checkpoint (optional)')
    
    # 数据路径
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to HumanML3D dataset root')
    parser.add_argument('--motion_dir', type=str, default=None,
                        help='Path to motion directory (default: data_root/joints_hunyuan)')
    
    # 评估配置
    parser.add_argument('--eval_mode', type=str, default='gt', choices=['gt', 'generated', 'both'],
                        help='Evaluation mode: gt, generated, or both')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate (train/val/test)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None for all)')
    parser.add_argument('--repeat_times', type=int, default=20,
                        help='Number of times to repeat evaluation (default: 20, for mean±CI)')
    
    # 生成配置
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of inference steps for generation')
    parser.add_argument('--cfg_scale', type=float, default=5.0,
                        help='CFG scale for generation')
    
    # LoRA 配置
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=16.0,
                        help='LoRA alpha')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results (default: model_path/eval_results)')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载 SPM 模型
    print(f'\nLoading SPM model from: {args.spm_path}')
    # 1. 先创建 SPM 实例
    spm_model = SPM(
        t5_path=args.t5_path,
        nfeats=135,
        temp=0.1,
        thr=0.9
    )
    # 2. 加载权重
    load_SPM(args.spm_path, spm_model)
    # 3. 移动到设备
    spm_model = spm_model.to(device)
    
    # 4. 设置 spm_mode（根据 checkpoint 路径推断）
    spm_path_lower = args.spm_path.lower()
    if 'm1t1' in spm_path_lower or 'sam1t1' in spm_path_lower:
        spm_model.spm_mode = 'M1T1'
    elif 'm1t0' in spm_path_lower or 'sam1t0' in spm_path_lower:
        spm_model.spm_mode = 'M1T0'
    elif 'm0t1' in spm_path_lower or 'sam0t1' in spm_path_lower:
        spm_model.spm_mode = 'M0T1'
    else:
        spm_model.spm_mode = 'M0T0'
    print(f'SPM mode: {spm_model.spm_mode}')
    
    spm_model.eval()
    print('SPM model loaded successfully')
    
    # 加载数据集
    print(f'\nLoading dataset from: {args.data_root}')
    eval_dataset = HumanML3DEvalDataset(
        data_root=args.data_root,
        split=args.split,
        motion_dir=args.motion_dir,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=eval_collate_fn,
    )
    
    # 加载 HY-Motion 模型（如果需要评估生成的 motion）
    hymotion_model = None
    lora_layers = None
    
    if args.eval_mode in ['generated', 'both'] and args.model_path is not None:
        print(f'\nLoading HY-Motion model from: {args.model_path}')
        
        import yaml
        cfg_path = pjoin(args.model_path, 'config.yml')
        ckpt_path = pjoin(args.model_path, 'latest.ckpt')
        
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
        
        hymotion_model = MotionFlowMatching(
            network_module=config['network_module'],
            network_module_args=config['network_module_args'],
            **config['train_pipeline_args']
        )
        hymotion_model.load_in_demo(ckpt_path, build_text_encoder=True)
        hymotion_model = hymotion_model.to(device)
        hymotion_model.eval()
        print('HY-Motion model loaded successfully')
        
        # 加载 LoRA（如果有）
        if args.lora_path is not None and os.path.exists(args.lora_path):
            print(f'\nLoading LoRA checkpoint from: {args.lora_path}')
            
            # 注入 LoRA 层
            lora_layers = inject_lora_layers(
                hymotion_model.motion_transformer,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
            )
            
            # 加载 LoRA 权重
            lora_checkpoint = torch.load(args.lora_path, map_location=device)
            lora_state_dict = lora_checkpoint.get('lora_state_dict', lora_checkpoint)
            
            loaded_count = 0
            missing_keys = []
            for name, layer in lora_layers.items():
                if name in lora_state_dict:
                    layer.lora_A.data = lora_state_dict[name]['lora_A'].to(device)
                    layer.lora_B.data = lora_state_dict[name]['lora_B'].to(device)
                    loaded_count += 1
                else:
                    missing_keys.append(name)
            
            # 打印缺失的 key（如果有）
            if missing_keys:
                print(f'  Warning: {len(missing_keys)} LoRA layers not found in checkpoint:')
                for k in missing_keys[:5]:
                    print(f'    - {k}')
                if len(missing_keys) > 5:
                    print(f'    ... and {len(missing_keys) - 5} more')
            
            if loaded_count == 0:
                print(f'\n  Checkpoint keys: {list(lora_state_dict.keys())[:10]}')
                print(f'  Expected keys: {list(lora_layers.keys())[:10]}')
                raise RuntimeError(
                    f'LoRA path specified but no LoRA layers were loaded! '
                    f'Key mismatch between checkpoint and model.'
                )
            
            print(f'LoRA checkpoint loaded successfully: {loaded_count}/{len(lora_layers)} layers')
            
            # 验证 LoRA 权重不是全零
            print('\nLoRA weight verification (first 3 layers):')
            for name, layer in list(lora_layers.items())[:3]:
                a_sum = layer.lora_A.abs().sum().item()
                b_sum = layer.lora_B.abs().sum().item()
                print(f'  {name}: lora_A sum={a_sum:.6f}, lora_B sum={b_sum:.6f}')
    
    # 创建评估器
    evaluator = HYMotionEvaluator(
        spm_model=spm_model,
        eval_loader=eval_loader,
        device=device,
        nfeats=eval_dataset.nfeats,
        hymotion_model=hymotion_model,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
    )
    
    # 执行评估
    print(f'\n{"="*60}')
    print(f'Starting evaluation on {args.split} set')
    print(f'Evaluation mode: {args.eval_mode}')
    print(f'Number of samples: {args.num_samples if args.num_samples else "all"}')
    print(f'Repeat times: {args.repeat_times}')
    print(f'{"="*60}')
    
    results = {}
    
    # 评估 GT
    if args.eval_mode in ['gt', 'both']:
        print('\n--- Evaluating GT Dataset ---')
        gt_metrics = evaluator.evaluate_gt(
            num_samples=args.num_samples,
            repeat_times=args.repeat_times,
        )
        results['gt'] = gt_metrics
        
        print(f'\nGT Dataset Results (repeat={args.repeat_times}):')
        print(f'  R-Precision Top-1: {gt_metrics["top1"]:.4f} ± {gt_metrics["top1_conf"]:.4f}')
        print(f'  R-Precision Top-2: {gt_metrics["top2"]:.4f} ± {gt_metrics["top2_conf"]:.4f}')
        print(f'  R-Precision Top-3: {gt_metrics["top3"]:.4f} ± {gt_metrics["top3_conf"]:.4f}')
        print(f'  FID:               {gt_metrics["fid"]:.4f} ± {gt_metrics["fid_conf"]:.4f}')
        print(f'  Matching Score:    {gt_metrics["matching"]:.4f} ± {gt_metrics["matching_conf"]:.4f}')
        print(f'  Diversity:         {gt_metrics["div"]:.4f} ± {gt_metrics["div_conf"]:.4f}')
    
    # 评估生成的 motion
    if args.eval_mode in ['generated', 'both']:
        if hymotion_model is None:
            print('\nWarning: HY-Motion model not loaded, skipping generated evaluation')
        else:
            print('\n--- Evaluating Generated Motion ---')
            gen_metrics = evaluator.evaluate_generated(
                num_samples=args.num_samples,
                repeat_times=args.repeat_times,
            )
            results['generated'] = gen_metrics
            
            print(f'\nGenerated Motion Results (repeat={args.repeat_times}):')
            print(f'  R-Precision Top-1: {gen_metrics["top1"]:.4f} ± {gen_metrics["top1_conf"]:.4f}')
            print(f'  R-Precision Top-2: {gen_metrics["top2"]:.4f} ± {gen_metrics["top2_conf"]:.4f}')
            print(f'  R-Precision Top-3: {gen_metrics["top3"]:.4f} ± {gen_metrics["top3_conf"]:.4f}')
            print(f'  FID:               {gen_metrics["fid"]:.4f} ± {gen_metrics["fid_conf"]:.4f}')
            print(f'  Matching Score:    {gen_metrics["matching"]:.4f} ± {gen_metrics["matching_conf"]:.4f}')
            print(f'  Diversity:         {gen_metrics["div"]:.4f} ± {gen_metrics["div_conf"]:.4f}')
    
    print(f'\n{"="*60}')
    print('Evaluation completed!')
    print(f'{"="*60}')
    
    # 保存结果到日志文件
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.model_path is not None:
        output_dir = os.path.join(args.model_path, 'eval_results')
    else:
        output_dir = None
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        lora_suffix = '_lora' if args.lora_path else '_pretrained'
        
        # ==================== 1. 保存 metrics.json (参考 MotionLCM) ====================
        # 包含每次重复的值、均值、置信区间
        metrics_json = {
            'configuration': {
                'model_path': args.model_path,
                'lora_path': args.lora_path,
                'spm_path': args.spm_path,
                'data_root': args.data_root,
                'split': args.split,
                'eval_mode': args.eval_mode,
                'num_samples': args.num_samples,
                'repeat_times': args.repeat_times,
                'num_inference_steps': args.num_inference_steps,
                'cfg_scale': args.cfg_scale,
                'lora_rank': args.lora_rank,
                'lora_alpha': args.lora_alpha,
                'seed': args.seed,
                'timestamp': timestamp,
            }
        }
        
        metric_names = ['top1', 'top2', 'top3', 'fid', 'matching', 'div']
        
        for mode_key in ['gt', 'generated']:
            if mode_key not in results:
                continue
            m = results[mode_key]
            mode_metrics = {}
            for name in metric_names:
                mode_metrics[name] = {
                    'mean': m[name],
                    'conf_interval': m[f'{name}_conf'],
                    'all_values': m.get(f'{name}_all', []),
                }
            metrics_json[mode_key] = mode_metrics
        
        json_filename = f'eval_{args.split}_{args.eval_mode}{lora_suffix}_rep{args.repeat_times}_{timestamp}.json'
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f'\nMetrics JSON saved to: {json_path}')
        
        # 同时保存一份固定名称的 metrics.json（方便读取最新结果）
        latest_json_path = os.path.join(output_dir, 'metrics.json')
        with open(latest_json_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f'Latest metrics saved to: {latest_json_path}')
        
        # ==================== 2. 保存可读的 txt 日志 ====================
        log_filename = f'eval_{args.split}_{args.eval_mode}{lora_suffix}_rep{args.repeat_times}_{timestamp}.txt'
        log_path = os.path.join(output_dir, log_filename)
        
        def fmt_metric(m, key):
            """格式化为 mean ± conf_interval"""
            return f'{m[key]:.4f} ± {m[f"{key}_conf"]:.4f}'
        
        with open(log_path, 'w') as f:
            f.write('='*60 + '\n')
            f.write('HY-Motion Evaluation Results\n')
            f.write('='*60 + '\n\n')
            
            f.write('Configuration:\n')
            f.write(f'  Model Path: {args.model_path}\n')
            f.write(f'  LoRA Path: {args.lora_path}\n')
            f.write(f'  SPM Path: {args.spm_path}\n')
            f.write(f'  Data Root: {args.data_root}\n')
            f.write(f'  Split: {args.split}\n')
            f.write(f'  Eval Mode: {args.eval_mode}\n')
            f.write(f'  Num Samples: {args.num_samples if args.num_samples else "all"}\n')
            f.write(f'  Repeat Times: {args.repeat_times}\n')
            f.write(f'  Num Inference Steps: {args.num_inference_steps}\n')
            f.write(f'  CFG Scale: {args.cfg_scale}\n')
            f.write(f'  LoRA Rank: {args.lora_rank}\n')
            f.write(f'  LoRA Alpha: {args.lora_alpha}\n')
            f.write(f'  Seed: {args.seed}\n')
            f.write('\n')
            
            if 'gt' in results:
                gt = results['gt']
                f.write(f'GT Dataset Results (repeat={args.repeat_times}):\n')
                f.write(f'  R-Precision Top-1: {fmt_metric(gt, "top1")}\n')
                f.write(f'  R-Precision Top-2: {fmt_metric(gt, "top2")}\n')
                f.write(f'  R-Precision Top-3: {fmt_metric(gt, "top3")}\n')
                f.write(f'  FID:               {fmt_metric(gt, "fid")}\n')
                f.write(f'  Matching Score:    {fmt_metric(gt, "matching")}\n')
                f.write(f'  Diversity:         {fmt_metric(gt, "div")}\n')
                f.write('\n')
            
            if 'generated' in results:
                gen = results['generated']
                f.write(f'Generated Motion Results (repeat={args.repeat_times}):\n')
                f.write(f'  R-Precision Top-1: {fmt_metric(gen, "top1")}\n')
                f.write(f'  R-Precision Top-2: {fmt_metric(gen, "top2")}\n')
                f.write(f'  R-Precision Top-3: {fmt_metric(gen, "top3")}\n')
                f.write(f'  FID:               {fmt_metric(gen, "fid")}\n')
                f.write(f'  Matching Score:    {fmt_metric(gen, "matching")}\n')
                f.write(f'  Diversity:         {fmt_metric(gen, "div")}\n')
                f.write('\n')
            
            f.write('='*60 + '\n')
        
        print(f'Log saved to: {log_path}')
        
        # ==================== 3. 保存简洁 summary ====================
        summary_filename = f'eval_summary_{args.split}_{args.eval_mode}{lora_suffix}.txt'
        summary_path = os.path.join(output_dir, summary_filename)
        with open(summary_path, 'w') as f:
            f.write(f'Evaluation Summary ({timestamp}, repeat={args.repeat_times})\n')
            f.write(f'Model: {args.model_path}\n')
            f.write(f'LoRA: {args.lora_path}\n')
            f.write(f'Split: {args.split}, Mode: {args.eval_mode}\n\n')
            
            if 'gt' in results:
                gt = results['gt']
                f.write(f'GT:  Top1={fmt_metric(gt, "top1")}, Top2={fmt_metric(gt, "top2")}, Top3={fmt_metric(gt, "top3")}, ')
                f.write(f'FID={fmt_metric(gt, "fid")}, Match={fmt_metric(gt, "matching")}, Div={fmt_metric(gt, "div")}\n')
            
            if 'generated' in results:
                gen = results['generated']
                f.write(f'Gen: Top1={fmt_metric(gen, "top1")}, Top2={fmt_metric(gen, "top2")}, Top3={fmt_metric(gen, "top3")}, ')
                f.write(f'FID={fmt_metric(gen, "fid")}, Match={fmt_metric(gen, "matching")}, Div={fmt_metric(gen, "div")}\n')
        
        print(f'Summary saved to: {summary_path}')
    
    return results


if __name__ == '__main__':
    main()
