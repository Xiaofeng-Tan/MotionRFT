"""
motionrft_hy.py

使用 RL (Reward Learning) 微调 HY-Motion 模型 - V2 版本
核心改进：
- 随机采样 k ∈ [1, T]
- 阶段1 (无梯度): t=0 → t=k/T，多步欧拉采样
- 阶段2 (有梯度): t=k/T → t=1，单步跳跃采样
- 使用 LoRA 进行参数高效微调
- 最大化 SPM reward (text-motion 和 motion-motion 相似度的加权组合)
  - text-motion: 生成动作与文本描述的相似度
  - motion-motion: 生成动作与GT动作的相似度
  - 可通过 --reward_tm_weight 和 --reward_mm_weight 调整权重

支持模型: HY-Motion-1.0-Lite, HY-Motion-1.0 (Full)
  通过 --model_path 指定即可，代码自动读取 config.yml 适配不同规模

用法:
    # ==================== HY-Motion-1.0-Lite ====================

    # Lite + SPM reward
    CUDA_VISIBLE_DEVICES=0 python motionrft_hy.py \
        --model_path ../pretrain/hymotion/HY-Motion-1.0-Lite \
        --spm_path ./t2m/evaluator.pth \
        --t5_path ../deps/sentence-t5-large \
        --data_root ../datasets/humanml3d \
        --motion_dir ../datasets/humanml3d/joints_6d \
        --device_ids 0 --batch_size 4 --num_epochs 100 --eval_every 50 \
        --lora_rank 8 --min_k 1 --num_inference_steps 50 \
        --use_swanlab --swanlab_project HY-Motion-RL --swanlab_experiment lite_spm \
        --train_split train \
        --output_dir ../pretrain/hymotion/HY-Motion-1.0-Lite/rl_finetune_v2_train_outputs

    # Lite + MotionReward only
    CUDA_VISIBLE_DEVICES=0 python motionrft_hy.py \
        --model_path ../pretrain/hymotion/HY-Motion-1.0-Lite \
        --spm_path ./t2m/evaluator.pth \
        --t5_path ../deps/sentence-t5-large \
        --data_root ../datasets/humanml3d \
        --motion_dir ../datasets/humanml3d/joints_6d \
        --device_ids 0 --batch_size 4 --num_epochs 100 --eval_every 50 \
        --lora_rank 8 --min_k 1 --num_inference_steps 50 \
        --use_swanlab --swanlab_project HY-Motion-RL --swanlab_experiment lite_mr_only \
        --train_split train \
        --output_dir ../pretrain/hymotion/HY-Motion-1.0-Lite/rl_finetune_v2_mr_only \
        --use_motion_reward \
        --motion_reward_path ../checkpoints/motionreward \
        --motion_reward_repr_type 135 --reward_mode mr_only

    # ==================== HY-Motion-1.0 (Full) ====================

    # Full + MotionReward only
    CUDA_VISIBLE_DEVICES=0 python motionrft_hy.py \
        --model_path ../pretrain/hymotion/HY-Motion-1.0 \
        --spm_path ./t2m/evaluator.pth \
        --t5_path ../deps/sentence-t5-large \
        --data_root ../datasets/humanml3d \
        --motion_dir ../datasets/humanml3d/joints_6d \
        --device_ids 0 --batch_size 4 --num_epochs 100 --eval_every 500 \
        --lora_rank 8 --min_k 1 --num_inference_steps 50 \
        --use_swanlab --swanlab_project HY-Motion-RL --swanlab_experiment full_mr_only \
        --train_split train \
        --output_dir ../pretrain/hymotion/HY-Motion-1.0/rl_finetune_v2_mr_only \
        --use_motion_reward \
        --motion_reward_path ../checkpoints/motionreward \
        --motion_reward_repr_type 135 --reward_mode mr_only

"""

import os
import sys
import random
import argparse
import json
import math
from os.path import join as pjoin
from typing import List, Dict, Optional, Tuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from scipy import linalg

# SwanLab 日志记录
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: SwanLab not installed. Run 'pip install swanlab' to enable experiment tracking.")

# 添加路径 (使用当前目录)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 添加 motionreward 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入 SPM 模型
from ReAlignModule.models.spm import SPM, process_T5_outputs, load_SPM

# 导入 Motion Reward 模型
from motionreward.models import MultiReprRetrievalWithLoRA
from motionreward.utils.config_utils import get_model_config

# 导入 HY-Motion 相关模块
from hymotion.pipeline.motion_diffusion import MotionFlowMatching, length_to_mask, randn_tensor
from hymotion.utils.loaders import load_object
from hymotion.utils.visualize_mesh_web import save_visualization_data, generate_static_html_content


#################################################################################
#                              Motion Reward 模型加载                            #
#################################################################################

def load_motion_reward_model(ckpt_path: str, t5_path: str, device: torch.device):
    """
    加载 Motion Reward 模型（Stage 1 检索模型）
    
    Follow train_retrieval_lora_new.py 的加载方式，从 state_dict 推断所有关键参数。
    默认启用 use_unified_dim=True。
    
    关键维度关系：
    - encoder_dim = unified_dim if use_unified_dim else latent_dim（用于 motion encoder）
    - latent_dim 始终用于 text encoder
    - 当 use_unified_dim=True 时，encoder_dim 和 latent_dim 可以不同
    
    Args:
        ckpt_path: 检查点路径（目录或文件）
        t5_path: Sentence-T5 模型路径
        device: 设备
    
    Returns:
        model: Motion Reward 模型
    """
    import glob
    
    # 查找 backbone checkpoint
    if os.path.isdir(ckpt_path):
        # 目录：查找 stage1_best_retrieval_backbone 文件
        backbone_files = glob.glob(os.path.join(ckpt_path, '*stage1_best_retrieval_backbone*.pth'))
        if backbone_files:
            backbone_path = backbone_files[0]
        else:
            # 尝试其他命名模式
            backbone_files = glob.glob(os.path.join(ckpt_path, '*backbone*.pth'))
            if backbone_files:
                backbone_path = backbone_files[0]
            else:
                raise FileNotFoundError(f"No backbone found in {ckpt_path}")
    else:
        backbone_path = ckpt_path
    
    print(f'Loading Motion Reward backbone from: {backbone_path}')
    
    # 加载 checkpoint 获取模型配置
    checkpoint = torch.load(backbone_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    
    # ========== 从 state_dict 推断关键参数 ==========
    model_cfg = {}
    
    # 1. 推断 encoder_dim（motion encoder 维度）: 从 global_motion_token
    #    encoder_dim = unified_dim if use_unified_dim else latent_dim
    if 'global_motion_token' in state_dict:
        encoder_dim = state_dict['global_motion_token'].shape[1]
        print(f'Inferred encoder_dim={encoder_dim} from global_motion_token')
    elif 'encoder.norm.weight' in state_dict:
        encoder_dim = state_dict['encoder.norm.weight'].shape[0]
        print(f'Inferred encoder_dim={encoder_dim} from encoder.norm.weight')
    else:
        encoder_dim = 256
        print(f'Warning: Cannot infer encoder_dim, using default {encoder_dim}')
    
    # 2. 推断 latent_dim（text encoder 维度）: 从 global_text_token
    if 'global_text_token' in state_dict:
        latent_dim = state_dict['global_text_token'].shape[1]
        print(f'Inferred latent_dim={latent_dim} from global_text_token')
    elif 'text_encoder.norm.weight' in state_dict:
        latent_dim = state_dict['text_encoder.norm.weight'].shape[0]
        print(f'Inferred latent_dim={latent_dim} from text_encoder.norm.weight')
    else:
        latent_dim = 128
        print(f'Warning: Cannot infer latent_dim, using default {latent_dim}')
    
    model_cfg['latent_dim'] = latent_dim
    
    # 3. 判断 use_unified_dim：如果 encoder_dim != latent_dim，则使用 unified_dim
    #    默认启用 use_unified_dim=True
    if encoder_dim != latent_dim:
        model_cfg['use_unified_dim'] = True
        model_cfg['unified_dim'] = encoder_dim
        print(f'Detected use_unified_dim=True (encoder_dim={encoder_dim} != latent_dim={latent_dim})')
    else:
        # 即使相等，也默认启用 use_unified_dim
        model_cfg['use_unified_dim'] = True
        model_cfg['unified_dim'] = encoder_dim
        print(f'Using use_unified_dim=True with unified_dim={encoder_dim}')
    
    # 4. 推断 proj_hidden_dim: 从 proj_xxx.proj.0.weight 的 shape[0]
    for proj_key in ['proj_263.proj.0.weight', 'proj_22x3.proj.0.weight', 'proj_135.proj.0.weight']:
        if proj_key in state_dict:
            model_cfg['proj_hidden_dim'] = state_dict[proj_key].shape[0]
            print(f'Inferred proj_hidden_dim={model_cfg["proj_hidden_dim"]} from {proj_key}')
            break
    
    # 5. 推断 encoder_num_layers: 从 encoder.input_blocks/output_blocks 的最大索引
    max_input_idx = -1
    max_output_idx = -1
    for key in state_dict.keys():
        if key.startswith('encoder.input_blocks.'):
            parts = key.split('.')
            if len(parts) > 2:
                try:
                    idx = int(parts[2])
                    max_input_idx = max(max_input_idx, idx)
                except ValueError:
                    pass
        elif key.startswith('encoder.output_blocks.'):
            parts = key.split('.')
            if len(parts) > 2:
                try:
                    idx = int(parts[2])
                    max_output_idx = max(max_output_idx, idx)
                except ValueError:
                    pass
    
    if max_input_idx >= 0:
        model_cfg['encoder_num_layers'] = (max_input_idx + 1) + 1 + (max_output_idx + 1)
        print(f'Inferred encoder_num_layers={model_cfg["encoder_num_layers"]} (input={max_input_idx+1}, middle=1, output={max_output_idx+1})')
    
    # 6. 推断 encoder_ff_size: 从 encoder.input_blocks.0.linear1.weight
    if 'encoder.input_blocks.0.linear1.weight' in state_dict:
        model_cfg['encoder_ff_size'] = state_dict['encoder.input_blocks.0.linear1.weight'].shape[0]
        print(f'Inferred encoder_ff_size={model_cfg["encoder_ff_size"]}')
    elif 'encoder.middle_block.linear1.weight' in state_dict:
        model_cfg['encoder_ff_size'] = state_dict['encoder.middle_block.linear1.weight'].shape[0]
        print(f'Inferred encoder_ff_size={model_cfg["encoder_ff_size"]} from middle_block')
    
    # 7. 推断 text_num_layers: 从 text_encoder.input_blocks 的最大索引
    max_text_input_idx = -1
    max_text_output_idx = -1
    for key in state_dict.keys():
        if key.startswith('text_encoder.input_blocks.'):
            parts = key.split('.')
            if len(parts) > 2:
                try:
                    idx = int(parts[2])
                    max_text_input_idx = max(max_text_input_idx, idx)
                except ValueError:
                    pass
        elif key.startswith('text_encoder.output_blocks.'):
            parts = key.split('.')
            if len(parts) > 2:
                try:
                    idx = int(parts[2])
                    max_text_output_idx = max(max_text_output_idx, idx)
                except ValueError:
                    pass
    
    if max_text_input_idx >= 0:
        model_cfg['text_num_layers'] = (max_text_input_idx + 1) + 1 + (max_text_output_idx + 1)
        print(f'Inferred text_num_layers={model_cfg["text_num_layers"]}')
    
    # 8. 推断 text_ff_size: 从 text_encoder.input_blocks.0.linear1.weight
    if 'text_encoder.input_blocks.0.linear1.weight' in state_dict:
        model_cfg['text_ff_size'] = state_dict['text_encoder.input_blocks.0.linear1.weight'].shape[0]
        print(f'Inferred text_ff_size={model_cfg["text_ff_size"]}')
    
    # 9. 推断 proj_num_layers: 从 proj_xxx.proj 的最大层索引
    max_proj_layer = 0
    for key in state_dict.keys():
        if key.startswith('proj_') and '.proj.' in key:
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'proj' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        max_proj_layer = max(max_proj_layer, layer_idx)
                    except ValueError:
                        pass
    if max_proj_layer > 0:
        model_cfg['proj_num_layers'] = (max_proj_layer // 3) + 1
        print(f'Inferred proj_num_layers={model_cfg["proj_num_layers"]}')
    
    # 从 meta 文件获取其他配置（作为补充）
    meta_path = backbone_path.replace('_retrieval_backbone_', '_meta_')
    if os.path.exists(meta_path):
        try:
            meta = torch.load(meta_path, map_location='cpu', weights_only=False)
            saved_cfg = meta.get('model_config', {})
            # 只使用 state_dict 无法推断的参数
            for key in ['temp', 'thr', 'encoder_num_heads', 'text_num_heads']:
                if key in saved_cfg and key not in model_cfg:
                    model_cfg[key] = saved_cfg[key]
            print(f'Loaded additional config from meta: temp, thr, num_heads')
        except Exception as e:
            print(f'Warning: Failed to load meta file: {e}')
    
    # 设置默认值
    default_cfg = {
        'temp': 0.1,
        'thr': 0.9,
        'latent_dim': 128,
        'unified_dim': 256,
        'encoder_num_layers': 9,
        'encoder_num_heads': 4,
        'encoder_ff_size': 512,
        'text_num_layers': 9,
        'text_num_heads': 4,
        'text_ff_size': 512,
        'proj_hidden_dim': 256,
        'proj_num_layers': 3,
        'use_unified_dim': True,
    }
    
    # 合并配置（model_cfg 优先）
    final_cfg = {**default_cfg, **model_cfg}
    
    print(f'Final model config:')
    print(f'  latent_dim={final_cfg["latent_dim"]}, unified_dim={final_cfg["unified_dim"]}, use_unified_dim={final_cfg["use_unified_dim"]}')
    print(f'  encoder_num_layers={final_cfg["encoder_num_layers"]}, encoder_ff_size={final_cfg["encoder_ff_size"]}')
    print(f'  text_num_layers={final_cfg["text_num_layers"]}, text_ff_size={final_cfg["text_ff_size"]}')
    print(f'  proj_hidden_dim={final_cfg["proj_hidden_dim"]}, proj_num_layers={final_cfg["proj_num_layers"]}')
    
    # 创建模型
    model = MultiReprRetrievalWithLoRA(
        t5_path=t5_path,
        temp=final_cfg['temp'],
        thr=final_cfg['thr'],
        latent_dim=final_cfg['latent_dim'],
        unified_dim=final_cfg['unified_dim'],
        encoder_num_layers=final_cfg['encoder_num_layers'],
        encoder_num_heads=final_cfg['encoder_num_heads'],
        encoder_ff_size=final_cfg['encoder_ff_size'],
        text_num_layers=final_cfg['text_num_layers'],
        text_num_heads=final_cfg['text_num_heads'],
        text_ff_size=final_cfg['text_ff_size'],
        proj_hidden_dim=final_cfg['proj_hidden_dim'],
        proj_num_layers=final_cfg['proj_num_layers'],
        use_unified_dim=final_cfg['use_unified_dim'],
    )
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        # 过滤掉 clip 相关的 key（T5 模型单独加载）
        missing_keys = [k for k in missing_keys if not k.startswith('clip.')]
        if missing_keys:
            print(f'Warning: Missing keys (non-clip): {missing_keys[:5]}...')
    if unexpected_keys:
        print(f'Warning: Unexpected keys: {unexpected_keys[:5]}...')
    
    model.to(device)
    model.eval()
    
    # 冻结参数
    for param in model.parameters():
        param.requires_grad = False
    
    print(f'Motion Reward model loaded, params: {sum(p.numel() for p in model.parameters()):,}')
    
    return model


#################################################################################
#                              LoRA 实现                                         #
#################################################################################

class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) 层"""
    
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.enabled = True  # 控制是否启用 LoRA
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA 低秩分解矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # 冻结原始层
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # 原始输出
        original_output = self.original_layer(x)
        
        # 如果禁用 LoRA，直接返回原始输出
        if not self.enabled:
            return original_output
        
        # LoRA 增量
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return original_output + lora_output


def apply_lora_to_model(model: nn.Module, rank: int = 8, alpha: float = 16.0, 
                        target_modules: List[str] = None) -> Dict[str, LoRALayer]:
    """
    对模型应用 LoRA
    
    Args:
        model: 要修改的模型
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        target_modules: 目标模块名称列表 (默认: qkv, proj, mlp)
    
    Returns:
        lora_layers: LoRA 层字典
    """
    if target_modules is None:
        target_modules = ['qkv', 'proj', 'fc1', 'fc2', 'to_q', 'to_k', 'to_v', 'to_out']
    
    lora_layers = {}
    
    def _apply_lora_recursive(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                # 检查是否是目标模块
                should_apply = any(target in name.lower() for target in target_modules)
                if should_apply:
                    lora_layer = LoRALayer(child, rank=rank, alpha=alpha)
                    setattr(module, name, lora_layer)
                    lora_layers[full_name] = lora_layer
                    print(f'Applied LoRA to: {full_name}')
            else:
                _apply_lora_recursive(child, full_name)
    
    _apply_lora_recursive(model)
    return lora_layers


def get_lora_parameters(lora_layers: Dict[str, LoRALayer]) -> List[nn.Parameter]:
    """获取所有 LoRA 参数"""
    params = []
    for layer in lora_layers.values():
        params.extend([layer.lora_A, layer.lora_B])
    return params


#################################################################################
#                              数据集定义                                        #
#################################################################################

class HumanML3DTrainDataset(Dataset):
    """HumanML3D 训练集数据加载器 (135维 Hunyuan 格式) - 用于训练，不标准化"""
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
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
        self.text_dir = pjoin(data_root, 'texts')
        
        # 加载 mean/std - 优先从 motion_dir 加载，使用 135 维版本
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
        for name in tqdm(self.name_list, desc=f'Loading {split} dataset'):
            motion_path = pjoin(self.motion_dir, f'{name}.npy')
            text_path = pjoin(self.text_dir, f'{name}.txt')
            
            if not os.path.exists(motion_path) or not os.path.exists(text_path):
                continue
                
            motion = np.load(motion_path)
            if motion.shape[0] < self.min_motion_length:
                continue
                
            # 读取文本描述
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
        
        # 加载动作
        motion = np.load(data['motion_path'])
        length = data['length']
        
        # 截断
        if motion.shape[0] > self.max_motion_length:
            motion = motion[:self.max_motion_length]
        
        # 随机选择一个文本描述
        text = random.choice(data['texts'])
        
        return {
            'name': data['name'],
            'motion': torch.from_numpy(motion).float(),
            'text': text,
            'length': length
        }


class HumanML3DEvalDataset(Dataset):
    """HumanML3D 测试集数据加载器 (135维 Hunyuan 格式) - 用于评估"""
    
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
        
        # 加载 mean/std - 优先从 motion_dir 加载，使用 135 维版本
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
        
        motion = (motion - self.mean) / self.std
        text = data['texts'][0]
        
        return {
            'name': data['name'],
            'motion': torch.from_numpy(motion).float(),
            'text': text,
            'length': length
        }


def train_collate_fn(batch):
    """数据批次整理函数 - 用于训练"""
    names = [item['name'] for item in batch]
    texts = [item['text'] for item in batch]
    lengths = [item['length'] for item in batch]
    
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
        'length': lengths
    }


def eval_collate_fn(batch):
    """数据批次整理函数 - 用于评估"""
    names = [item['name'] for item in batch]
    texts = [item['text'] for item in batch]
    lengths = [item['length'] for item in batch]
    
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
        'length': lengths
    }


#################################################################################
#                              可微分采样器 V2                                    #
#################################################################################

class DifferentiableFlowMatchingSamplerV2:
    """
    可微分的 Flow Matching 采样器 V2
    
    采样策略:
    1. 随机采样 k ∈ [1, T]
    2. 阶段1 (无梯度): t=0 → t=k/T，多步欧拉采样
    3. 阶段2 (有梯度): t=k/T → t=1，单步跳跃采样
    
    这样设计的优势：
    - 显存效率：只在最后一步保留梯度
    - 训练稳定性：随机 k 提供多样的梯度信号
    - 与原始模型兼容：LoRA 可以随时禁用
    """
    
    def __init__(
        self,
        model: MotionFlowMatching,
        num_steps: int = 50,
        cfg_scale: float = 5.0,
        min_k: int = 1,  # k 的最小值
    ):
        """
        Args:
            model: HY-Motion 模型
            num_steps: 总采样步数 T
            cfg_scale: CFG 缩放因子
            min_k: k 的最小值 (默认 1)
        """
        self.model = model
        self.num_steps = num_steps
        self.cfg_scale = cfg_scale
        self.min_k = min_k
    
    def sample(
        self,
        texts: List[str],
        lengths: List[int],
        hidden_state_dict: Optional[Dict] = None,
        seed: Optional[int] = None,
        training: bool = True,  # 训练模式 vs 评估模式
    ) -> Tuple[torch.Tensor, Dict]:
        """
        两阶段可微分采样
        
        Args:
            texts: 文本描述列表
            lengths: 动作长度列表
            hidden_state_dict: 预编码的文本特征 (可选)
            seed: 随机种子 (可选)
            training: 是否为训练模式 (True: 随机k+单步跳跃, False: 完整采样)
        
        Returns:
            sampled: 生成的动作 (B, L, D)
            output_dict: 包含 latent_denorm, k 等信息
        """
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
            """预测速度场 v(x, t)"""
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
        
        if not training:
            # ============ 评估模式: 完整的无梯度采样 ============
            with torch.no_grad():
                for step in range(self.num_steps):
                    t_current = step / self.num_steps
                    v = predict_velocity(x_t, t_current)
                    x_t = x_t + dt * v
            sampled = x_t
            k = self.num_steps
        else:
            # ============ 训练模式: 随机 k + 单步跳跃 ============
            # 随机采样 k ∈ [min_k, num_steps]
            k = torch.randint(self.min_k, self.num_steps + 1, (1,)).item()
            t_k = k / self.num_steps
            
            # ============ 阶段1: 无梯度多步采样 (t=0 → t=k/T) ============
            with torch.no_grad():
                for step in range(k):
                    t_current = step / self.num_steps
                    v = predict_velocity(x_t, t_current)
                    x_t = x_t + dt * v
            
            # ============ 阶段2: 有梯度单步采样 (t=k/T → t=1) ============
            # 断开梯度，重新开启追踪
            x_t = x_t.detach().requires_grad_(True)
            
            # 剩余时间
            remaining_dt = 1.0 - t_k
            
            # 在 t_k 处预测速度，单步跳跃到 t=1
            v_final = predict_velocity(x_t, t_k)
            sampled = x_t + remaining_dt * v_final
        
        # 截断到目标长度
        sampled = sampled[:, :max_length, :]
        
        # 反归一化
        std_zero = self.model.std < 1e-3
        std = torch.where(std_zero, torch.zeros_like(self.model.std), self.model.std)
        latent_denorm = sampled * std + self.model.mean
        
        return sampled, {
            'latent_denorm': latent_denorm,
            'lengths': lengths,
            'k': k,
            't_k': k / self.num_steps if training else 1.0,
        }


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


#################################################################################
#                              训练器 V2                                         #
#################################################################################

class RLTrainerV2:
    """RL 微调训练器 V2 - 使用随机 k + 单步跳跃采样"""
    
    def __init__(
        self,
        hymotion_model: MotionFlowMatching,
        spm_model: SPM,
        lora_layers: Dict[str, LoRALayer],
        train_loader: DataLoader,
        eval_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        min_k: int = 1,  # k 的最小值
        grad_clip: float = 1.0,
        output_dir: str = './rl_outputs_v2',
        nfeats: int = 135,
        num_eval_samples: int = 100,  # 评估样本数
        use_swanlab: bool = False,  # 是否使用 SwanLab
        reward_tm_weight: float = 0.5,  # text-motion 相似度权重
        reward_mm_weight: float = 0.5,  # motion-motion 相似度权重
        # ===== 新增参数 =====
        motion_reward_model: Optional['MultiReprRetrievalWithLoRA'] = None,
        motion_reward_weight: float = 0.5,
        spm_weight: float = 0.5,
        motion_reward_repr_type: str = '135',
        reward_mode: str = 'combined',  # 新增：reward 模式
    ):
        self.hymotion_model = hymotion_model
        self.spm_model = spm_model
        self.lora_layers = lora_layers
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.cfg_scale = cfg_scale
        self.num_inference_steps = num_inference_steps
        self.min_k = min_k
        self.grad_clip = grad_clip
        self.output_dir = output_dir
        self.nfeats = nfeats
        self.num_eval_samples = num_eval_samples  # 保存评估样本数
        self.use_swanlab = use_swanlab and SWANLAB_AVAILABLE
        self.reward_tm_weight = reward_tm_weight  # text-motion 权重
        self.reward_mm_weight = reward_mm_weight  # motion-motion 权重
        
        # ===== 新增：Motion Reward 模型 =====
        self.motion_reward_model = motion_reward_model
        self.motion_reward_weight = motion_reward_weight
        self.spm_weight = spm_weight
        self.motion_reward_repr_type = motion_reward_repr_type
        self.use_motion_reward = motion_reward_model is not None
        self.reward_mode = reward_mode  # 新增：reward 模式
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(pjoin(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(pjoin(output_dir, 'html_outputs'), exist_ok=True)
        
        # 只优化 LoRA 参数
        self.lora_params = get_lora_parameters(lora_layers)
        self.optimizer = torch.optim.AdamW(self.lora_params, lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        # 训练用采样器 (随机 k + 单步跳跃)
        self.train_sampler = DifferentiableFlowMatchingSamplerV2(
            model=hymotion_model,
            num_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            min_k=min_k,
        )
        
        # 评估用采样器 (完整 50 步)
        self.eval_sampler = DifferentiableFlowMatchingSamplerV2(
            model=hymotion_model,
            num_steps=50,
            cfg_scale=cfg_scale,
            min_k=1,
        )
        
        # 训练统计
        self.global_step = 0
        self.best_reward = -float('inf')
        self.best_r_precision = 0.0
        self.training_log = []
        
        # 预定义用于可视化的固定样本 (取 eval_dataset 前 k 个)
        self.num_vis_samples = 10
        self.vis_samples = self._prepare_vis_samples()
    
    def compute_reward(
        self,
        texts: List[str],
        gt_motions: torch.Tensor,
        pred_motions: torch.Tensor,
        lengths: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 reward (text-motion 和 motion-motion 相似度的加权组合)
        
        Args:
            texts: 文本描述列表
            gt_motions: GT动作 (B, L, D)，未标准化的原始数据
            pred_motions: 生成的动作 (B, L, D)，未标准化的原始数据
            lengths: 动作长度列表
        
        Returns:
            reward: 加权组合的相似度分数 (标量)
            reward_tm: text-motion 相似度 (标量)
            reward_mm: motion-motion 相似度 (标量)
        
        注意：gt_motions 和 pred_motions 都是未标准化的原始数据，
        需要先标准化后再送入 SPM 编码，与验证时保持一致。
        """
        # 获取 mean/std 用于标准化
        base_dataset = self.train_loader.dataset
        if hasattr(base_dataset, 'dataset'):
            base_dataset = base_dataset.dataset
        mean = torch.from_numpy(base_dataset.mean[:self.nfeats]).float().to(self.device)
        std = torch.from_numpy(base_dataset.std[:self.nfeats]).float().to(self.device)
        
        # 提取 SPM 需要的特征维度 (135维)
        gt_feats = gt_motions[:, :, :self.nfeats].to(self.device)
        pred_feats = pred_motions[:, :, :self.nfeats].to(self.device)
        
        # 标准化（与验证时保持一致）
        gt_feats_norm = (gt_feats - mean) / std
        pred_feats_norm = (pred_feats - mean) / std
        
        # 编码文本
        t_len, token_emb, cls_token = process_T5_outputs(texts, self.spm_model.clip)
        token_emb = token_emb.to(self.device).float()
        t_latent, _ = self.spm_model.encode_text(token_emb, t_len)
        
        # 编码 GT motion
        gt_m_latent, _ = self.spm_model.encode_motion(gt_feats_norm, lengths)
        
        # 编码生成的 motion
        pred_m_latent, _ = self.spm_model.encode_motion(pred_feats_norm, lengths)
        
        # 计算 text-motion 相似度 (生成动作与文本的相似度)
        reward_tm = torch.nn.functional.cosine_similarity(
            t_latent.squeeze(), pred_m_latent.squeeze(), dim=-1
        ).mean()
        
        # 计算 motion-motion 相似度 (生成动作与GT动作的相似度)
        reward_mm = torch.nn.functional.cosine_similarity(
            gt_m_latent.squeeze(), pred_m_latent.squeeze(), dim=-1
        ).mean()
        
        # 加权组合
        reward = self.reward_tm_weight * reward_tm + self.reward_mm_weight * reward_mm
        
        return reward, reward_tm, reward_mm
    
    def compute_motion_reward(
        self,
        texts: List[str],
        gt_motions: torch.Tensor,
        pred_motions: torch.Tensor,
        lengths: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用 Motion Reward 模型计算 reward
        
        Args:
            texts: 文本描述列表
            gt_motions: GT动作 (B, L, D)，未标准化的原始数据
            pred_motions: 生成的动作 (B, L, D)，未标准化的原始数据
            lengths: 动作长度列表
        
        Returns:
            reward: 加权组合的相似度分数 (标量)
            reward_tm: text-motion 相似度 (标量)
            reward_mm: motion-motion 相似度 (标量)
        """
        if self.motion_reward_model is None:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        
        # 获取 mean/std 用于标准化
        base_dataset = self.train_loader.dataset
        if hasattr(base_dataset, 'dataset'):
            base_dataset = base_dataset.dataset
        mean = torch.from_numpy(base_dataset.mean[:self.nfeats]).float().to(self.device)
        std = torch.from_numpy(base_dataset.std[:self.nfeats]).float().to(self.device)
        
        # 提取特征维度
        gt_feats = gt_motions[:, :, :self.nfeats].to(self.device)
        pred_feats = pred_motions[:, :, :self.nfeats].to(self.device)
        
        # 标准化
        gt_feats_norm = (gt_feats - mean) / std
        pred_feats_norm = (pred_feats - mean) / std
        
        # 获取文本嵌入
        t_latent = self.motion_reward_model.get_text_embedding(texts)
        
        # 获取 GT motion 嵌入
        gt_m_latent = self.motion_reward_model.get_motion_embedding(
            gt_feats_norm, lengths, repr_type=self.motion_reward_repr_type
        )
        
        # 获取生成 motion 嵌入
        pred_m_latent = self.motion_reward_model.get_motion_embedding(
            pred_feats_norm, lengths, repr_type=self.motion_reward_repr_type
        )
        
        # 计算 text-motion 相似度
        reward_tm = torch.nn.functional.cosine_similarity(
            t_latent, pred_m_latent, dim=-1
        ).mean()
        
        # 计算 motion-motion 相似度
        reward_mm = torch.nn.functional.cosine_similarity(
            gt_m_latent, pred_m_latent, dim=-1
        ).mean()
        
        # 加权组合
        reward = self.reward_tm_weight * reward_tm + self.reward_mm_weight * reward_mm
        
        return reward, reward_tm, reward_mm
    
    def compute_combined_reward(
        self,
        texts: List[str],
        gt_motions: torch.Tensor,
        pred_motions: torch.Tensor,
        lengths: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合 reward（SPM + Motion Reward）
        
        根据 reward_mode 决定用于优化的 combined_reward:
        - 'spm_only': 仅用 SPM reward 优化
        - 'mr_only': 仅用 Motion Reward 优化
        - 'combined': 加权组合
        
        注意：无论 reward_mode 如何，都会计算所有可用的 reward 用于日志记录
        
        Returns:
            reward_dict: 包含所有 reward 指标的字典
        """
        # 始终计算 SPM reward（用于日志记录）
        spm_reward, spm_tm, spm_mm = self.compute_reward(texts, gt_motions, pred_motions, lengths)
        
        reward_dict = {
            'spm_reward': spm_reward,
            'spm_tm': spm_tm,
            'spm_mm': spm_mm,
        }
        
        # 始终计算 Motion Reward（如果模型可用，用于日志记录）
        if self.use_motion_reward:
            mr_reward, mr_tm, mr_mm = self.compute_motion_reward(texts, gt_motions, pred_motions, lengths)
            reward_dict['mr_reward'] = mr_reward
            reward_dict['mr_tm'] = mr_tm
            reward_dict['mr_mm'] = mr_mm
        
        # 根据 reward_mode 决定用于优化的 combined_reward
        if self.reward_mode == 'mr_only':
            # 仅用 Motion Reward 优化
            if self.use_motion_reward:
                reward_dict['combined_reward'] = mr_reward
            else:
                print('Warning: reward_mode=mr_only but Motion Reward model not available, falling back to SPM')
                reward_dict['combined_reward'] = spm_reward
        elif self.reward_mode == 'spm_only':
            # 仅用 SPM 优化
            reward_dict['combined_reward'] = spm_reward
        else:  # 'combined'
            # 加权组合
            if self.use_motion_reward:
                combined_reward = self.spm_weight * spm_reward + self.motion_reward_weight * mr_reward
                reward_dict['combined_reward'] = combined_reward
            else:
                reward_dict['combined_reward'] = spm_reward
        
        return reward_dict
    
    def _prepare_vis_samples(self) -> Dict:
        """预先准备固定的可视化样本 (取 eval_dataset 前 k 个)"""
        vis_texts = []
        vis_lengths = []
        vis_names = []
        
        # 获取底层数据集
        base_dataset = self.eval_loader.dataset
        if hasattr(base_dataset, 'dataset'):
            base_dataset = base_dataset.dataset
        
        for i in range(min(self.num_vis_samples, len(base_dataset))):
            sample = base_dataset[i]
            vis_texts.append(sample['text'])
            vis_lengths.append(sample['length'])
            vis_names.append(sample.get('name', f'{i:06d}'))
        
        # 为每个样本分配固定种子
        vis_seeds = [42 + i for i in range(len(vis_texts))]
        
        print(f'Prepared {len(vis_texts)} fixed samples for visualization')
        
        return {
            'texts': vis_texts,
            'lengths': vis_lengths,
            'names': vis_names,
            'seeds': vis_seeds,
        }
    
    def train_step(self, batch: Dict) -> Dict:
        """单步训练"""
        self.hymotion_model.train()
        self.spm_model.eval()
        if self.motion_reward_model is not None:
            self.motion_reward_model.eval()
        
        texts = batch['text']
        lengths = batch['length']
        gt_motions = batch['motion'].to(self.device)  # GT动作
        
        # 可微分采样生成动作
        self.optimizer.zero_grad()
        
        sampled, output_dict = self.train_sampler.sample(
            texts=texts,
            lengths=lengths,
            seed=random.randint(0, 99999),
            training=True,  # 训练模式
        )
        
        # 获取反归一化的动作
        latent_denorm = output_dict['latent_denorm']
        k = output_dict['k']
        t_k = output_dict['t_k']
        
        # 计算组合 reward
        reward_dict = self.compute_combined_reward(texts, gt_motions, latent_denorm, lengths)
        
        # 最大化组合 reward (即最小化 -reward)
        loss = -reward_dict['combined_reward']
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.lora_params, self.grad_clip)
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        # 返回所有指标
        result = {
            'loss': loss.item(),
            'reward': reward_dict['combined_reward'].item(),
            'spm_reward': reward_dict['spm_reward'].item(),
            'spm_tm': reward_dict['spm_tm'].item(),
            'spm_mm': reward_dict['spm_mm'].item(),
            'lr': self.scheduler.get_last_lr()[0],
            'k': k,
            't_k': t_k,
        }
        
        if self.use_motion_reward:
            result['mr_reward'] = reward_dict['mr_reward'].item()
            result['mr_tm'] = reward_dict['mr_tm'].item()
            result['mr_mm'] = reward_dict['mr_mm'].item()
        
        return result
    
    @torch.no_grad()
    def evaluate(
        self,
        num_samples: int = 100,
        save_html: bool = False,
        repeat_times: int = 1,
    ) -> Dict:
        """
        评估模型
        
        Args:
            num_samples: 评估样本数
            save_html: 是否保存 HTML 可视化
            repeat_times: 重复评估次数
        
        Returns:
            results: 评估指标
        """
        self.hymotion_model.eval()
        self.spm_model.eval()
        if self.motion_reward_model is not None:
            self.motion_reward_model.eval()
        
        # 如果需要保存 HTML，先用固定样本生成可视化
        if save_html:
            print(f'  Generating visualizations for {len(self.vis_samples["texts"])} fixed samples...')
            self._generate_fixed_visualizations()
        
        all_results = {
            'fid': [], 'div': [], 
            'top1': [], 'top2': [], 'top3': [],
            'matching': [],
            'reward_mean': [], 'reward_std': [],
        }
        
        # Motion Reward 检索评估结果
        if self.use_motion_reward:
            all_results['mr_top1'] = []
            all_results['mr_top2'] = []
            all_results['mr_top3'] = []
            all_results['mr_matching'] = []
        
        for rep in range(repeat_times):
            if repeat_times > 1:
                print(f'\n  Repeat {rep + 1}/{repeat_times}')
            
            gt_motion_latents = []
            pred_motion_latents = []
            text_latents = []
            
            # Motion Reward 检索评估
            mr_text_latents = []
            mr_pred_motion_latents = []
            
            R_precision_sum = np.array([0., 0., 0.])
            matching_score_sum = 0.0
            mr_R_precision_sum = np.array([0., 0., 0.])
            mr_matching_score_sum = 0.0
            nb_sample = 0
            sample_count = 0
            
            rep_rewards = []
            
            # 计算实际需要的 batch 数，让 tqdm 显示正确的进度
            if num_samples is not None:
                bs_est = self.eval_loader.batch_size if hasattr(self.eval_loader, 'batch_size') else 4
                total_batches = (num_samples + bs_est - 1) // bs_est
            else:
                total_batches = len(self.eval_loader)
            
            for batch in tqdm(self.eval_loader, desc=f'Evaluating (Rep {rep + 1})', total=total_batches):
                if num_samples is not None and sample_count >= num_samples:
                    break
                
                texts = batch['text']
                gt_motions = batch['motion'].to(self.device)
                lengths = batch['length']
                names = batch.get('name', [f'{sample_count + i:06d}' for i in range(len(texts))])
                bs = len(texts)
                
                # 限制样本数
                if num_samples is not None:
                    remaining = num_samples - sample_count
                    if bs > remaining:
                        texts = texts[:remaining]
                        gt_motions = gt_motions[:remaining]
                        lengths = lengths[:remaining]
                        names = names[:remaining]
                        bs = remaining
                
                # 编码文本
                t_len, token_emb, cls_token = process_T5_outputs(texts, self.spm_model.clip)
                token_emb = token_emb.to(self.device).float()
                
                # 编码 GT 动作
                gt_m_latent = self.spm_model.encode_motion(gt_motions, lengths)[0].squeeze()
                
                # 编码文本 latent
                t_latent = self.spm_model.encode_text(token_emb, t_len)[0].squeeze()
                
                # 使用评估采样器生成动作
                seeds = [random.randint(0, 99999) for _ in range(bs)]
                
                try:
                    sampled, output_dict = self.eval_sampler.sample(
                        texts=texts,
                        lengths=lengths,
                        seed=seeds[0],
                        training=False,  # 评估模式
                    )
                    
                    latent_denorm = output_dict['latent_denorm']
                    pred_motion_feats = latent_denorm[:, :, :self.nfeats].to(self.device)
                    
                    # 获取 mean/std 用于标准化
                    base_dataset = self.eval_loader.dataset
                    if hasattr(base_dataset, 'dataset'):
                        base_dataset = base_dataset.dataset
                    mean = torch.from_numpy(base_dataset.mean[:self.nfeats]).float().to(self.device)
                    std = torch.from_numpy(base_dataset.std[:self.nfeats]).float().to(self.device)
                    
                    # 计算每个样本的 reward (motion-motion 相似度)
                    # 注意：gt_motions 已经标准化（来自 EvalDataset）
                    # pred_motion_feats 是未标准化的，需要标准化
                    for i in range(bs):
                        sample_pred_motion = pred_motion_feats[i:i+1, :lengths[i], :]
                        sample_gt_motion = gt_motions[i:i+1, :lengths[i], :self.nfeats]
                        
                        # 标准化生成的 motion（与 GT 保持一致）
                        sample_pred_motion_norm = (sample_pred_motion - mean) / std
                        
                        # 编码 GT motion（已标准化）
                        gt_m_latent_i, _ = self.spm_model.encode_motion(sample_gt_motion, [lengths[i]])
                        # 编码生成的 motion（标准化后）
                        pred_m_latent_i, _ = self.spm_model.encode_motion(sample_pred_motion_norm, [lengths[i]])
                        
                        # 计算 cosine similarity
                        reward = torch.nn.functional.cosine_similarity(
                            gt_m_latent_i.squeeze(), pred_m_latent_i.squeeze(), dim=-1
                        ).mean()
                        rep_rewards.append(reward.item())
                    
                    # 填充并标准化
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
                    
                    # ===== Motion Reward 检索评估 =====
                    if self.use_motion_reward:
                        # 获取 Motion Reward 文本嵌入
                        mr_t_latent = self.motion_reward_model.get_text_embedding(texts)
                        # 获取 Motion Reward 生成 motion 嵌入
                        mr_pred_m_latent = self.motion_reward_model.get_motion_embedding(
                            pred_motions_norm, lengths, repr_type=self.motion_reward_repr_type
                        )
                        mr_text_latents.append(F.normalize(mr_t_latent, dim=-1).cpu().numpy())
                        mr_pred_motion_latents.append(F.normalize(mr_pred_m_latent, dim=-1).cpu().numpy())
                    
                except Exception as e:
                    print(f'Generation failed: {e}')
                    import traceback
                    traceback.print_exc()
                    pred_m_latent = gt_m_latent
                    for i in range(bs):
                        rep_rewards.append(0.0)
                
                # 归一化
                gt_m_latent_norm = F.normalize(gt_m_latent, dim=-1)
                pred_m_latent_norm = F.normalize(pred_m_latent, dim=-1)
                t_latent_norm = F.normalize(t_latent, dim=-1)
                
                # 收集用于 FID 和 Diversity 计算
                gt_motion_latents.append(gt_m_latent_norm.cpu().numpy())
                pred_motion_latents.append(pred_m_latent_norm.cpu().numpy())
                text_latents.append(t_latent_norm.cpu().numpy())
                
                # 计算 R-Precision 和 Matching Score
                temp_R = calculate_R_precision(
                    t_latent_norm.cpu().numpy(),
                    pred_m_latent_norm.cpu().numpy(),
                    top_k=3, sum_all=True
                )
                R_precision_sum += temp_R
                
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
            
            # 计算 FID
            gt_mu, gt_cov = calculate_activation_statistics(gt_motion_latents)
            pred_mu, pred_cov = calculate_activation_statistics(pred_motion_latents)
            fid = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)
            
            # 计算 Diversity
            diversity = calculate_diversity(pred_motion_latents, min(300, nb_sample // 2))
            
            # 计算 R-Precision
            R_precision = R_precision_sum / nb_sample
            
            # 计算 Matching Score
            matching_score = matching_score_sum / nb_sample
            
            # 计算 Reward 统计
            reward_mean = np.mean(rep_rewards)
            reward_std = np.std(rep_rewards)
            
            # 记录结果
            all_results['fid'].append(fid)
            all_results['div'].append(diversity)
            all_results['top1'].append(R_precision[0])
            all_results['top2'].append(R_precision[1])
            all_results['top3'].append(R_precision[2])
            all_results['matching'].append(matching_score)
            all_results['reward_mean'].append(reward_mean)
            all_results['reward_std'].append(reward_std)
            
            # ===== Motion Reward 检索评估 =====
            if self.use_motion_reward and len(mr_text_latents) > 0:
                mr_text_latents_np = np.concatenate(mr_text_latents, axis=0)
                mr_pred_motion_latents_np = np.concatenate(mr_pred_motion_latents, axis=0)
                
                # 计算 Motion Reward R-Precision
                mr_R_precision = calculate_R_precision(
                    mr_text_latents_np,
                    mr_pred_motion_latents_np,
                    top_k=3, sum_all=True
                )
                mr_R_precision = mr_R_precision / len(mr_text_latents_np)
                
                # 计算 Motion Reward Matching Score
                mr_matching = euclidean_distance_matrix(
                    mr_text_latents_np,
                    mr_pred_motion_latents_np
                ).trace() / len(mr_text_latents_np)
                
                all_results['mr_top1'].append(mr_R_precision[0])
                all_results['mr_top2'].append(mr_R_precision[1])
                all_results['mr_top3'].append(mr_R_precision[2])
                all_results['mr_matching'].append(mr_matching)
            
            if repeat_times > 1:
                print(f'  Rep {rep + 1}: FID={fid:.4f}, Div={diversity:.4f}, '
                      f'Top1={R_precision[0]:.4f}, Top2={R_precision[1]:.4f}, Top3={R_precision[2]:.4f}, '
                      f'Matching={matching_score:.4f}')
                print(f'  Reward: Mean={reward_mean:.4f}, Std={reward_std:.4f}')
        
        # 计算平均值
        metrics = {}
        for key, values in all_results.items():
            values = np.array(values)
            mean = np.mean(values)
            metrics[key] = mean
            if repeat_times > 1:
                conf = np.std(values) * 1.96 / np.sqrt(repeat_times)
                metrics[f'{key}_conf'] = conf
        
        return metrics
    
    def _generate_fixed_visualizations(self):
        """使用固定样本生成可视化 HTML"""
        vis = self.vis_samples
        
        for i in range(len(vis['texts'])):
            text = vis['texts'][i]
            length = vis['lengths'][i]
            name = vis['names'][i]
            seed = vis['seeds'][i]
            
            try:
                # 使用固定种子生成动作
                sampled, output_dict = self.eval_sampler.sample(
                    texts=[text],
                    lengths=[length],
                    seed=seed,
                    training=False,
                )
                
                # 注意：decode_motion_from_latent 内部会做反归一化
                # 所以这里传入 sampled（归一化空间），而不是 latent_denorm（已反归一化）
                # 避免双重反归一化导致可视化异常
                output_dict_decode = self.hymotion_model.decode_motion_from_latent(
                    sampled, should_apply_smooothing=True
                )
                
                model_output = {
                    'rot6d': output_dict_decode['rot6d'][:1, :length],
                    'transl': output_dict_decode['transl'][:1, :length],
                }
                
                output_filename = f'step{self.global_step:06d}_{name}'
                html_output_dir = os.path.abspath(pjoin(self.output_dir, 'html_outputs'))
                
                save_data, base_filename = save_visualization_data(
                    output=model_output,
                    text=text,
                    rewritten_text=text,
                    timestamp=output_filename,
                    output_dir=html_output_dir,
                    output_filename=output_filename,
                )
                
                html_content = generate_static_html_content(
                    folder_name=html_output_dir,
                    file_name=base_filename,
                    hide_captions=False,
                )
                
                html_path = pjoin(html_output_dir, f'{output_filename}.html')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                    
            except Exception as e:
                print(f'Failed to generate visualization for sample {i} ({name}): {e}')
    
    def _save_html_batch(
        self,
        sampled: torch.Tensor,  # 归一化空间的 latent，decode_motion_from_latent 内部会反归一化
        texts: List[str],
        lengths: List[int],
        names: List[str],
    ):
        """保存 HTML 可视化
        
        Args:
            sampled: 归一化空间的 latent，不是 latent_denorm
                     decode_motion_from_latent 内部会做反归一化
        """
        try:
            for i in range(len(texts)):
                # 解码动作（内部会反归一化）
                output_dict = self.hymotion_model.decode_motion_from_latent(
                    sampled[i:i+1], should_apply_smooothing=True
                )
                
                model_output = {
                    'rot6d': output_dict['rot6d'][:1, :lengths[i]],
                    'transl': output_dict['transl'][:1, :lengths[i]],
                }
                
                output_filename = f'step{self.global_step:06d}_{names[i]}'
                html_output_dir = os.path.abspath(pjoin(self.output_dir, 'html_outputs'))
                
                save_data, base_filename = save_visualization_data(
                    output=model_output,
                    text=texts[i],
                    rewritten_text=texts[i],
                    timestamp=output_filename,
                    output_dir=html_output_dir,
                    output_filename=output_filename,
                )
                
                html_content = generate_static_html_content(
                    folder_name=html_output_dir,
                    file_name=base_filename,
                    hide_captions=False,
                )
                
                html_path = pjoin(html_output_dir, f'{output_filename}.html')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                    
        except Exception as e:
            print(f'Failed to save HTML: {e}')
    
    def enable_lora(self):
        """启用 LoRA"""
        for layer in self.lora_layers.values():
            layer.enabled = True
    
    def disable_lora(self):
        """禁用 LoRA"""
        for layer in self.lora_layers.values():
            layer.enabled = False
    
    def save_checkpoint(self, filename: str = None):
        """保存检查点"""
        if filename is None:
            filename = f'lora_step{self.global_step:06d}.pth'
        
        checkpoint = {
            'global_step': self.global_step,
            'lora_state_dict': {
                name: {
                    'lora_A': layer.lora_A.data,
                    'lora_B': layer.lora_B.data,
                }
                for name, layer in self.lora_layers.items()
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_reward': self.best_reward,
            'best_r_precision': self.best_r_precision,
            'training_log': self.training_log,
        }
        
        save_path = pjoin(self.output_dir, 'checkpoints', filename)
        torch.save(checkpoint, save_path)
        print(f'Saved checkpoint to {save_path}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.global_step = checkpoint['global_step']
        self.best_reward = checkpoint.get('best_reward', -float('inf'))
        self.best_r_precision = checkpoint.get('best_r_precision', 0.0)
        self.training_log = checkpoint.get('training_log', [])
        
        for name, layer in self.lora_layers.items():
            if name in checkpoint['lora_state_dict']:
                layer.lora_A.data = checkpoint['lora_state_dict'][name]['lora_A']
                layer.lora_B.data = checkpoint['lora_state_dict'][name]['lora_B']
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f'Loaded checkpoint from {checkpoint_path}')
    
    def train(self, num_epochs: int, eval_every: int = 50):
        """
        训练循环
        
        Args:
            num_epochs: 训练轮数
            eval_every: 每多少步评估一次
        """
        print(f'\n{"="*70}')
        print(f'Starting RL Training V2 (Random k + Single-Step Jump)')
        print(f'{"="*70}')
        print(f'Total epochs: {num_epochs}')
        print(f'Eval every: {eval_every} steps')
        print(f'LoRA parameters: {sum(p.numel() for p in self.lora_params)}')
        print(f'Sampling: {self.num_inference_steps} total steps')
        print(f'  - Random k ∈ [{self.min_k}, {self.num_inference_steps}]')
        print(f'  - Phase 1 (no grad): t=0 → t=k/T, multi-step Euler')
        print(f'  - Phase 2 (with grad): t=k/T → t=1, single-step jump')
        print(f'Reward Config:')
        print(f'  - Reward Mode: {self.reward_mode}')
        print(f'  - SPM weight: {self.spm_weight}')
        print(f'  - Motion Reward: {"Enabled" if self.use_motion_reward else "Disabled"}')
        if self.use_motion_reward:
            print(f'  - Motion Reward weight: {self.motion_reward_weight}')
            print(f'  - Motion Reward repr_type: {self.motion_reward_repr_type}')
        print(f'SwanLab logging: {self.use_swanlab}')
        print(f'{"="*70}\n')
        
        # 初始评估
        print(f'\n>>> Initial evaluation at step 0...')
        eval_metrics = self.evaluate(num_samples=self.num_eval_samples, save_html=True, repeat_times=1)
        print(f"  FID={eval_metrics['fid']:.4f}, Div={eval_metrics['div']:.4f}, "
              f"Top1={eval_metrics['top1']:.4f}, Top2={eval_metrics['top2']:.4f}, Top3={eval_metrics['top3']:.4f}, "
              f"Reward={eval_metrics['reward_mean']:.4f}")
        
        self.training_log.append({
            'step': 0,
            'epoch': 0,
            'metrics': eval_metrics,
        })
        
        # SwanLab 记录初始评估
        if self.use_swanlab:
            init_log_dict = {
                'eval/fid': eval_metrics['fid'],
                'eval/diversity': eval_metrics['div'],
                'eval/top1': eval_metrics['top1'],
                'eval/top2': eval_metrics['top2'],
                'eval/top3': eval_metrics['top3'],
                'eval/matching': eval_metrics['matching'],
                'eval/reward_mean': eval_metrics['reward_mean'],
                'eval/reward_std': eval_metrics['reward_std'],
            }
            if self.use_motion_reward:
                init_log_dict['eval/mr_top1'] = eval_metrics.get('mr_top1', 0.0)
                init_log_dict['eval/mr_top2'] = eval_metrics.get('mr_top2', 0.0)
                init_log_dict['eval/mr_top3'] = eval_metrics.get('mr_top3', 0.0)
                init_log_dict['eval/mr_matching'] = eval_metrics.get('mr_matching', 0.0)
            swanlab.log(init_log_dict, step=0)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_reward = 0.0
            epoch_spm_reward = 0.0
            epoch_spm_tm = 0.0
            epoch_spm_mm = 0.0
            epoch_mr_reward = 0.0
            epoch_mr_tm = 0.0
            epoch_mr_mm = 0.0
            epoch_k_sum = 0
            num_batches = 0
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch in pbar:
                batch['motion'] = batch['motion'].to(self.device)
                
                metrics = self.train_step(batch)
                
                epoch_loss += metrics['loss']
                epoch_reward += metrics['reward']
                epoch_spm_reward += metrics['spm_reward']
                epoch_spm_tm += metrics['spm_tm']
                epoch_spm_mm += metrics['spm_mm']
                if self.use_motion_reward:
                    epoch_mr_reward += metrics.get('mr_reward', 0.0)
                    epoch_mr_tm += metrics.get('mr_tm', 0.0)
                    epoch_mr_mm += metrics.get('mr_mm', 0.0)
                epoch_k_sum += metrics['k']
                num_batches += 1
                
                # 更新进度条显示
                postfix = {
                    'loss': f"{metrics['loss']:.4f}",
                    'reward': f"{metrics['reward']:.4f}",
                    'spm': f"{metrics['spm_reward']:.4f}",
                    'k': metrics['k'],
                }
                if self.use_motion_reward:
                    postfix['mr'] = f"{metrics.get('mr_reward', 0.0):.4f}"
                pbar.set_postfix(postfix)
                
                # SwanLab 记录训练指标
                if self.use_swanlab:
                    log_dict = {
                        'train/loss': metrics['loss'],
                        'train/reward': metrics['reward'],
                        'train/spm_reward': metrics['spm_reward'],
                        'train/spm_tm': metrics['spm_tm'],
                        'train/spm_mm': metrics['spm_mm'],
                        'train/k': metrics['k'],
                        'train/t_k': metrics['t_k'],
                        'train/lr': metrics['lr'],
                    }
                    if self.use_motion_reward:
                        log_dict['train/mr_reward'] = metrics.get('mr_reward', 0.0)
                        log_dict['train/mr_tm'] = metrics.get('mr_tm', 0.0)
                        log_dict['train/mr_mm'] = metrics.get('mr_mm', 0.0)
                    swanlab.log(log_dict, step=self.global_step)
                
                # 评估
                if self.global_step % eval_every == 0:
                    print(f'\n>>> Evaluating at step {self.global_step}...')
                    eval_metrics = self.evaluate(num_samples=self.num_eval_samples, save_html=True, repeat_times=1)
                    
                    print(f"  FID={eval_metrics['fid']:.4f}, Div={eval_metrics['div']:.4f}, "
                          f"Top1={eval_metrics['top1']:.4f}, Top2={eval_metrics['top2']:.4f}, Top3={eval_metrics['top3']:.4f}, "
                          f"Reward={eval_metrics['reward_mean']:.4f}")
                    
                    self.training_log.append({
                        'step': self.global_step,
                        'epoch': epoch + 1,
                        'metrics': eval_metrics,
                    })
                    
                    # SwanLab 记录评估指标
                    if self.use_swanlab:
                        eval_log_dict = {
                            'eval/fid': eval_metrics['fid'],
                            'eval/diversity': eval_metrics['div'],
                            'eval/top1': eval_metrics['top1'],
                            'eval/top2': eval_metrics['top2'],
                            'eval/top3': eval_metrics['top3'],
                            'eval/matching': eval_metrics['matching'],
                            'eval/reward_mean': eval_metrics['reward_mean'],
                            'eval/reward_std': eval_metrics['reward_std'],
                            'eval/best_reward': self.best_reward,
                            'eval/best_r_precision': self.best_r_precision,
                        }
                        if self.use_motion_reward:
                            eval_log_dict['eval/mr_top1'] = eval_metrics.get('mr_top1', 0.0)
                            eval_log_dict['eval/mr_top2'] = eval_metrics.get('mr_top2', 0.0)
                            eval_log_dict['eval/mr_top3'] = eval_metrics.get('mr_top3', 0.0)
                            eval_log_dict['eval/mr_matching'] = eval_metrics.get('mr_matching', 0.0)
                        swanlab.log(eval_log_dict, step=self.global_step)
                    
                    # 保存最佳模型
                    if eval_metrics['reward_mean'] > self.best_reward:
                        self.best_reward = eval_metrics['reward_mean']
                        self.save_checkpoint('best_reward.pth')
                    
                    if eval_metrics['top3'] > self.best_r_precision:
                        self.best_r_precision = eval_metrics['top3']
                        self.save_checkpoint('best_r_precision.pth')
                    
                    self.save_checkpoint()
            
            # 打印 epoch 统计
            avg_loss = epoch_loss / num_batches
            avg_reward = epoch_reward / num_batches
            avg_spm_reward = epoch_spm_reward / num_batches
            avg_spm_tm = epoch_spm_tm / num_batches
            avg_spm_mm = epoch_spm_mm / num_batches
            avg_k = epoch_k_sum / num_batches
            
            print_str = f'\nEpoch {epoch + 1} Summary: Loss={avg_loss:.4f}, Reward={avg_reward:.4f}, SPM={avg_spm_reward:.4f} (TM={avg_spm_tm:.4f}, MM={avg_spm_mm:.4f})'
            if self.use_motion_reward:
                avg_mr_reward = epoch_mr_reward / num_batches
                avg_mr_tm = epoch_mr_tm / num_batches
                avg_mr_mm = epoch_mr_mm / num_batches
                print_str += f', MR={avg_mr_reward:.4f} (TM={avg_mr_tm:.4f}, MM={avg_mr_mm:.4f})'
            print_str += f', Avg k={avg_k:.1f}'
            print(print_str)
            
            # SwanLab 记录 epoch 统计
            if self.use_swanlab:
                epoch_log_dict = {
                    'epoch/avg_loss': avg_loss,
                    'epoch/avg_reward': avg_reward,
                    'epoch/avg_spm_reward': avg_spm_reward,
                    'epoch/avg_spm_tm': avg_spm_tm,
                    'epoch/avg_spm_mm': avg_spm_mm,
                    'epoch/avg_k': avg_k,
                }
                if self.use_motion_reward:
                    epoch_log_dict['epoch/avg_mr_reward'] = epoch_mr_reward / num_batches
                    epoch_log_dict['epoch/avg_mr_tm'] = epoch_mr_tm / num_batches
                    epoch_log_dict['epoch/avg_mr_mm'] = epoch_mr_mm / num_batches
                swanlab.log(epoch_log_dict, step=self.global_step)
        
        # 保存最终模型
        self.save_checkpoint('final_lora.pth')
        
        # 保存训练日志
        log_path = pjoin(self.output_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        # 关闭 SwanLab
        if self.use_swanlab:
            swanlab.finish()
        
        print(f'\nTraining completed!')
        print(f'Best Reward: {self.best_reward:.4f}')
        print(f'Best R@3: {self.best_r_precision:.4f}')


#################################################################################
#                        Pre-training Validation                                #
#################################################################################

def validate_reward_model(
    hymotion_model: MotionFlowMatching,
    spm_model: SPM,
    eval_loader: DataLoader,
    device: torch.device,
    nfeats: int = 135,
    num_samples: int = 50,
    cfg_scale: float = 5.0,
):
    """
    在训练前验证 reward model 的效果
    
    计算三个指标：
    1. GT-GT Similarity (上界，理论上接近 1.0)
    2. GT-Generated Similarity (当前生成质量)
    3. Random-Random Similarity (下界，随机配对)
    
    Args:
        hymotion_model: HY-Motion 模型
        spm_model: SPM reward 模型
        eval_loader: 评估数据加载器
        device: 设备
        nfeats: 特征维度
        num_samples: 验证样本数
        cfg_scale: CFG 缩放因子
    
    Returns:
        results: 验证结果字典
    """
    print('\n' + '='*70)
    print('Pre-training Validation: Checking Reward Model')
    print('='*70)
    
    hymotion_model.eval()
    spm_model.eval()
    
    # 创建评估用采样器
    eval_sampler = DifferentiableFlowMatchingSamplerV2(
        model=hymotion_model,
        num_steps=50,
        cfg_scale=cfg_scale,
        min_k=1,
    )
    
    gt_gt_sims = []           # GT 与自身的相似度
    gt_gen_sims = []          # GT 与生成的相似度
    random_sims = []          # 随机配对的相似度
    
    all_gt_latents = []       # 收集所有 GT latent 用于随机配对
    
    sample_count = 0
    
    # 获取 mean/std 用于标准化
    base_dataset = eval_loader.dataset
    if hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset
    mean = torch.from_numpy(base_dataset.mean[:nfeats]).float().to(device)
    std = torch.from_numpy(base_dataset.std[:nfeats]).float().to(device)
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Validating Reward Model'):
            if sample_count >= num_samples:
                break
            
            texts = batch['text']
            gt_motions = batch['motion'].to(device)  # 已标准化
            lengths = batch['length']
            bs = len(texts)
            
            # 限制样本数
            remaining = num_samples - sample_count
            if bs > remaining:
                texts = texts[:remaining]
                gt_motions = gt_motions[:remaining]
                lengths = lengths[:remaining]
                bs = remaining
            
            # 1. 编码 GT motion (已标准化的)
            gt_feats = gt_motions[:, :, :nfeats]
            gt_latents, _ = spm_model.encode_motion(gt_feats, lengths)
            # gt_latents 形状是 (1, B, D)，需要变成 (B, D)
            gt_latents = gt_latents.squeeze(0)
            
            # 收集 GT latents
            all_gt_latents.append(gt_latents.cpu())
            
            # 2. 计算 GT-GT 相似度（自身与自身）
            gt_gt_sim = F.cosine_similarity(gt_latents, gt_latents, dim=-1)
            gt_gt_sims.extend(gt_gt_sim.cpu().tolist())
            
            # 3. 生成 motion
            try:
                sampled, output_dict = eval_sampler.sample(
                    texts=texts,
                    lengths=lengths,
                    seed=42,
                    training=False,
                )
                latent_denorm = output_dict['latent_denorm']
                pred_feats_raw = latent_denorm[:, :, :nfeats].to(device)
                
                # 标准化生成的 motion (与 GT 保持一致)
                pred_feats_norm = (pred_feats_raw - mean) / std
                
                # 编码生成的 motion
                pred_latents, _ = spm_model.encode_motion(pred_feats_norm, lengths)
                # pred_latents 形状是 (1, B, D)，需要变成 (B, D)
                pred_latents = pred_latents.squeeze(0)
                
                # 4. 计算 GT-Generated 相似度
                gt_gen_sim = F.cosine_similarity(gt_latents, pred_latents, dim=-1)
                gt_gen_sims.extend(gt_gen_sim.cpu().tolist())
                
            except Exception as e:
                print(f'Generation failed: {e}')
                import traceback
                traceback.print_exc()
                # 如果生成失败，用 0 填充
                gt_gen_sims.extend([0.0] * bs)
            
            sample_count += bs
    
    # 5. 计算随机配对相似度
    all_gt_latents = torch.cat(all_gt_latents, dim=0)  # (N, D)
    n = all_gt_latents.shape[0]
    
    # 随机打乱并配对
    perm = torch.randperm(n)
    shuffled_latents = all_gt_latents[perm]
    random_sim = F.cosine_similarity(all_gt_latents, shuffled_latents, dim=-1)
    random_sims = random_sim.tolist()
    
    # 打印结果
    print('\n' + '-'*50)
    print('Validation Results:')
    print('-'*50)
    print(f'  GT-GT Similarity (Upper Bound):')
    print(f'    Mean: {np.mean(gt_gt_sims):.4f}, Std: {np.std(gt_gt_sims):.4f}')
    print(f'    Min: {np.min(gt_gt_sims):.4f}, Max: {np.max(gt_gt_sims):.4f}')
    
    print(f'\n  GT-Generated Similarity (Current Quality):')
    print(f'    Mean: {np.mean(gt_gen_sims):.4f}, Std: {np.std(gt_gen_sims):.4f}')
    print(f'    Min: {np.min(gt_gen_sims):.4f}, Max: {np.max(gt_gen_sims):.4f}')
    
    print(f'\n  Random-Random Similarity (Lower Bound):')
    print(f'    Mean: {np.mean(random_sims):.4f}, Std: {np.std(random_sims):.4f}')
    print(f'    Min: {np.min(random_sims):.4f}, Max: {np.max(random_sims):.4f}')
    
    print('-'*50)
    
    # 计算 reward 的有效范围
    reward_range = np.mean(gt_gt_sims) - np.mean(random_sims)
    if reward_range > 0:
        current_position = (np.mean(gt_gen_sims) - np.mean(random_sims)) / reward_range * 100
    else:
        current_position = 0.0
    
    print(f'\n  Reward Range Analysis:')
    print(f'    Effective Range: {reward_range:.4f}')
    print(f'    Current Position: {current_position:.1f}% of optimal')
    print(f'    Room for Improvement: {100 - current_position:.1f}%')
    print('='*70 + '\n')
    
    return {
        'gt_gt_sim_mean': np.mean(gt_gt_sims),
        'gt_gt_sim_std': np.std(gt_gt_sims),
        'gt_gen_sim_mean': np.mean(gt_gen_sims),
        'gt_gen_sim_std': np.std(gt_gen_sims),
        'random_sim_mean': np.mean(random_sims),
        'random_sim_std': np.std(random_sims),
        'reward_range': reward_range,
        'current_position_pct': current_position,
    }


#################################################################################
#                              主函数                                            #
#################################################################################

def main():
    parser = argparse.ArgumentParser(description='RL Fine-tune HY-Motion V2 (Random k + Single-Step Jump)')
    
    # 模型路径
    parser.add_argument('--model_path', type=str, required=True,
                        help='HY-Motion 模型路径')
    parser.add_argument('--spm_path', type=str, required=True,
                        help='SPM 检索模型检查点路径')
    parser.add_argument('--t5_path', type=str, 
                        default='deps/sentence-t5-large',
                        help='Sentence-T5 模型路径')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, 
                        default='../datasets/humanml3d',
                        help='HumanML3D 数据集根目录')
    parser.add_argument('--motion_dir', type=str, 
                        default='../datasets/humanml3d/joints_6d',
                        help='joints_6d 目录路径')
    parser.add_argument('--train_split', type=str, default='train',
                        help='训练集 split (train/test/val)')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--eval_every', type=int, default=50,
                        help='每多少步评估一次')
    
    # LoRA 参数
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA 秩')
    parser.add_argument('--lora_alpha', type=float, default=16.0,
                        help='LoRA 缩放因子')
    
    # 采样参数
    parser.add_argument('--cfg_scale', type=float, default=5.0)
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='总推理步数 T')
    parser.add_argument('--min_k', type=int, default=1,
                        help='k 的最小值 (随机 k ∈ [min_k, T])')
    
    # Reward 权重参数
    parser.add_argument('--reward_tm_weight', type=float, default=0.5,
                        help='text-motion 相似度权重')
    parser.add_argument('--reward_mm_weight', type=float, default=0.5,
                        help='motion-motion 相似度权重')
    
    # Motion Reward 参数（新增）
    parser.add_argument('--motion_reward_path', type=str, default=None,
                        help='Motion Reward 模型检查点路径（Stage 1 检索模型）')
    parser.add_argument('--use_motion_reward', action='store_true',
                        help='是否使用 Motion Reward 模型')
    parser.add_argument('--motion_reward_weight', type=float, default=0.5,
                        help='Motion Reward 在总 reward 中的权重')
    parser.add_argument('--spm_weight', type=float, default=0.5,
                        help='SPM 在总 reward 中的权重')
    parser.add_argument('--motion_reward_repr_type', type=str, default='135',
                        choices=['263', '22x3', '135'],
                        help='Motion Reward 使用的表征类型')
    parser.add_argument('--reward_mode', type=str, default='combined',
                        choices=['spm_only', 'mr_only', 'combined'],
                        help='训练时使用的 reward 模式: spm_only=仅用SPM优化, mr_only=仅用MotionReward优化, combined=加权组合')
    
    # 设备参数
    parser.add_argument('--device_ids', type=str, default='0',
                        help='GPU 设备 ID')
    parser.add_argument('--seed', type=int, default=42)
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    
    # 评估参数
    parser.add_argument('--eval_only', action='store_true',
                        help='只进行评估，不训练')
    parser.add_argument('--validate_only', action='store_true',
                        help='只进行 reward model 验证，不训练 (在训练前检查 SPM 是否正常)')
    parser.add_argument('--num_eval_samples', type=int, default=100,
                        help='评估样本数 (训练中和 eval_only 模式都使用)')
    parser.add_argument('--num_validate_samples', type=int, default=50,
                        help='验证样本数 (validate_only 模式使用)')
    parser.add_argument('--save_html', action='store_true',
                        help='保存 HTML 可视化')
    
    # SwanLab 参数
    parser.add_argument('--use_swanlab', action='store_true',
                        help='使用 SwanLab 记录训练日志')
    parser.add_argument('--swanlab_project', type=str, default='HY-Motion-RL',
                        help='SwanLab 项目名称')
    parser.add_argument('--swanlab_experiment', type=str, default=None,
                        help='SwanLab 实验名称')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 解析设备
    device_ids = [int(x.strip()) for x in args.device_ids.split(',')]
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    
    print(f'Using device: {device}')
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = pjoin(args.model_path, 'rl_finetune_v2_outputs')
    
    # 初始化 SwanLab
    if args.use_swanlab:
        if SWANLAB_AVAILABLE:
            experiment_name = args.swanlab_experiment or f'rl_v2_lora{args.lora_rank}_lr{args.lr}'
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=experiment_name,
                config={
                    'model_path': args.model_path,
                    'spm_path': args.spm_path,
                    'data_root': args.data_root,
                    'motion_dir': args.motion_dir,
                    'batch_size': args.batch_size,
                    'num_epochs': args.num_epochs,
                    'lr': args.lr,
                    'grad_clip': args.grad_clip,
                    'eval_every': args.eval_every,
                    'lora_rank': args.lora_rank,
                    'lora_alpha': args.lora_alpha,
                    'cfg_scale': args.cfg_scale,
                    'num_inference_steps': args.num_inference_steps,
                    'min_k': args.min_k,
                    'seed': args.seed,
                    'num_eval_samples': args.num_eval_samples,
                    'reward_type': 'weighted text-motion + motion-motion',
                    'reward_tm_weight': args.reward_tm_weight,
                    'reward_mm_weight': args.reward_mm_weight,
                    # Motion Reward 相关参数
                    'use_motion_reward': args.use_motion_reward,
                    'motion_reward_path': args.motion_reward_path,
                    'motion_reward_weight': args.motion_reward_weight,
                    'spm_weight': args.spm_weight,
                    'motion_reward_repr_type': args.motion_reward_repr_type,
                    'reward_mode': args.reward_mode,
                }
            )
            print(f'SwanLab initialized: project={args.swanlab_project}, experiment={experiment_name}')
        else:
            print('Warning: SwanLab not available, logging disabled')
            args.use_swanlab = False
    
    # 加载数据集
    print('\nLoading datasets...')
    train_dataset = HumanML3DTrainDataset(
        data_root=args.data_root,
        split=args.train_split,
        max_motion_length=196,
        min_motion_length=40,
        motion_dir=args.motion_dir
    )
    
    eval_dataset = HumanML3DEvalDataset(
        data_root=args.data_root,
        split='test',
        max_motion_length=196,
        min_motion_length=40,
        motion_dir=args.motion_dir
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=train_collate_fn,
        drop_last=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=eval_collate_fn,
        drop_last=False
    )
    
    print(f'Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}')
    
    # 加载 HY-Motion 模型
    print('\nLoading HY-Motion model...')
    
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
    hymotion_model.to(device)
    
    # 应用 LoRA
    print(f'\nApplying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...')
    lora_layers = apply_lora_to_model(
        hymotion_model.motion_transformer,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    print(f'Applied LoRA to {len(lora_layers)} layers')
    
    hymotion_model.motion_transformer.to(device)
    
    # 加载 SPM 模型
    print('\nLoading SPM model...')
    nfeats = 135
    spm_model = SPM(
        t5_path=args.t5_path,
        nfeats=nfeats,
        temp=0.1,
        thr=0.9
    )
    load_SPM(args.spm_path, spm_model)
    
    # 设置 spm_mode
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
    
    spm_model.to(device)
    spm_model.eval()
    
    # 冻结 SPM
    for param in spm_model.parameters():
        param.requires_grad = False
    
    # ===== 新增：加载 Motion Reward 模型 =====
    motion_reward_model = None
    if args.use_motion_reward and args.motion_reward_path:
        print('\nLoading Motion Reward model...')
        motion_reward_model = load_motion_reward_model(
            ckpt_path=args.motion_reward_path,
            t5_path=args.t5_path,
            device=device
        )
    elif args.use_motion_reward and not args.motion_reward_path:
        print('Warning: --use_motion_reward is set but --motion_reward_path is not provided. Skipping Motion Reward model.')
        args.use_motion_reward = False
    
    # 创建训练器
    trainer = RLTrainerV2(
        hymotion_model=hymotion_model,
        spm_model=spm_model,
        lora_layers=lora_layers,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        lr=args.lr,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        min_k=args.min_k,
        grad_clip=args.grad_clip,
        output_dir=args.output_dir,
        nfeats=nfeats,
        num_eval_samples=args.num_eval_samples,
        use_swanlab=args.use_swanlab,
        reward_tm_weight=args.reward_tm_weight,
        reward_mm_weight=args.reward_mm_weight,
        # ===== 新增参数 =====
        motion_reward_model=motion_reward_model,
        motion_reward_weight=args.motion_reward_weight,
        spm_weight=args.spm_weight,
        motion_reward_repr_type=args.motion_reward_repr_type,
        reward_mode=args.reward_mode,
    )
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 评估模式
    if args.eval_only:
        print('\n=== Evaluation Only Mode ===')
        metrics = trainer.evaluate(
            num_samples=args.num_eval_samples,
            save_html=args.save_html,
            repeat_times=3
        )
        print('\nEvaluation Results:')
        for key, value in metrics.items():
            print(f'  {key}: {value:.4f}' if isinstance(value, float) else f'  {key}: {value}')
        
        # 关闭 SwanLab
        if args.use_swanlab and SWANLAB_AVAILABLE:
            swanlab.finish()
        return
    
    # === 训练前验证 Reward Model ===
    print('\n>>> Pre-training validation: Checking Reward Model...')
    validation_results = validate_reward_model(
        hymotion_model=hymotion_model,
        spm_model=spm_model,
        eval_loader=eval_loader,
        device=device,
        nfeats=nfeats,
        num_samples=args.num_validate_samples,
        cfg_scale=args.cfg_scale,
    )
    
    # 如果只是验证模式，直接返回
    if args.validate_only:
        print('\n=== Validation Only Mode - Exiting before training ===')
        print('Validation results saved. You can now review them and decide whether to proceed with training.')
        
        # 关闭 SwanLab
        if args.use_swanlab and SWANLAB_AVAILABLE:
            swanlab.finish()
        return
    
    # 开始训练
    trainer.train(
        num_epochs=args.num_epochs,
        eval_every=args.eval_every,
    )


if __name__ == '__main__':
    main()
