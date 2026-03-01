"""
AI 检测训练数据集

包含:
- AIDetectionDataset: AI 检测数据集（真实 vs AI 生成）
- AIDetectionReprTypeBatchSampler: 按表征类型分组的 BatchSampler
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class AIDetectionDataset(Dataset):
    """AI 检测数据集 - 用于训练判断 motion 是否为 AI 生成
    
    加载真实数据（label=0）和 AI 生成数据（label=1）
    支持多个 AI 生成来源（如 MLD、ACMDM 等）
    支持单独的验证集/测试集目录
    
    支持两种加载方式：
    1. 目录模式：ai_generated_dirs 为目录列表，逐个加载 .npy 文件
    2. 打包模式：ai_packed_file 为打包后的 .pth 文件，一次性加载（推荐，速度快）
    """
    
    # 已知的无效文件列表（通过 check_motion_data.py 检查得到）
    INVALID_FILES = {
        '22x3': {'M000990', '000990', 'M005836', '005836', '002503'},
        '263': set()
    }
    
    def __init__(self, real_data_dir, ai_generated_dirs=None, repr_types=['263', '22x3'],
                 max_motion_length=196, val_ratio=0.0, is_val=False, seed=42,
                 split='train', ai_packed_file=None, max_samples=None):
        """
        Args:
            real_data_dir: 真实数据目录（HumanML3D）
            ai_generated_dirs: AI 生成数据目录列表（目录模式）
            repr_types: 表征类型列表
            max_motion_length: 最大 motion 长度
            val_ratio: 验证集比例（仅当 split='train' 且未指定单独验证集时使用）
            is_val: 是否为验证集（仅当 val_ratio > 0 时使用）
            seed: 随机种子
            split: 数据集划分 ('train', 'val', 'test')
            ai_packed_file: 打包的 AI 数据文件路径（打包模式，优先使用）
            max_samples: 最大样本数（用于 debug 模式）
        """
        self.max_motion_length = max_motion_length
        self.repr_types = repr_types
        self.samples = []
        self.split = split
        self.use_packed = False
        self.max_samples = max_samples
        
        # 打包数据缓存（用于打包模式）
        self.packed_data = None
        self.packed_indices = []  # [(repr_type, idx), ...]
        
        # 加载真实数据 mean/std
        self.mean_263 = np.load(os.path.join(real_data_dir, 'Mean.npy'))
        self.std_263 = np.load(os.path.join(real_data_dir, 'Std.npy'))
        mean_22x3_path = os.path.join(real_data_dir, 'Mean_22x3.npy')
        if os.path.exists(mean_22x3_path):
            self.mean_22x3 = np.load(mean_22x3_path)
            self.std_22x3 = np.load(os.path.join(real_data_dir, 'Std_22x3.npy'))
        else:
            self.mean_22x3 = np.zeros((22, 3))
            self.std_22x3 = np.ones((22, 3))
        
        # 加载真实数据（label=0）
        if '263' in repr_types:
            self._load_real_data(real_data_dir, '263', split)
        if '22x3' in repr_types:
            self._load_real_data(real_data_dir, '22x3', split)
        
        # 加载 AI 生成数据（label=1）
        # 优先使用打包文件
        if ai_packed_file and os.path.exists(ai_packed_file):
            self._load_ai_packed(ai_packed_file, repr_types)
        elif ai_generated_dirs:
            # 目录模式
            for ai_dir in ai_generated_dirs:
                if ai_dir and os.path.exists(ai_dir):
                    if '263' in repr_types:
                        self._load_ai_data(ai_dir, '263')
                    if '22x3' in repr_types:
                        self._load_ai_data(ai_dir, '22x3')
        
        # 划分训练/验证集（仅当 split='train' 且 val_ratio > 0 时使用）
        if split == 'train' and val_ratio > 0:
            random.seed(seed)
            if self.use_packed:
                # 打包模式：shuffle indices
                combined = list(range(len(self.samples))) + list(range(len(self.packed_indices)))
                random.shuffle(combined)
                split_idx = int(len(combined) * (1 - val_ratio))
                if is_val:
                    selected = combined[split_idx:]
                else:
                    selected = combined[:split_idx]
                # 分离 samples 和 packed_indices
                new_samples = []
                new_packed_indices = []
                for idx in selected:
                    if idx < len(self.samples):
                        new_samples.append(self.samples[idx])
                    else:
                        new_packed_indices.append(self.packed_indices[idx - len(self.samples)])
                self.samples = new_samples
                self.packed_indices = new_packed_indices
            else:
                random.shuffle(self.samples)
                split_idx = int(len(self.samples) * (1 - val_ratio))
                if is_val:
                    self.samples = self.samples[split_idx:]
                else:
                    self.samples = self.samples[:split_idx]
        
        # Debug 模式：限制样本数
        if max_samples is not None:
            total_before = len(self.samples) + len(self.packed_indices)
            if total_before > max_samples:
                random.seed(seed)
                # 合并后随机采样
                if self.use_packed and len(self.packed_indices) > 0:
                    # 打包模式：按比例分配
                    samples_ratio = len(self.samples) / total_before if total_before > 0 else 0.5
                    max_samples_from_samples = int(max_samples * samples_ratio)
                    max_samples_from_packed = max_samples - max_samples_from_samples
                    
                    if len(self.samples) > max_samples_from_samples:
                        random.shuffle(self.samples)
                        self.samples = self.samples[:max_samples_from_samples]
                    if len(self.packed_indices) > max_samples_from_packed:
                        random.shuffle(self.packed_indices)
                        self.packed_indices = self.packed_indices[:max_samples_from_packed]
                else:
                    # 非打包模式
                    if len(self.samples) > max_samples:
                        random.shuffle(self.samples)
                        self.samples = self.samples[:max_samples]
                print(f"[DEBUG] AIDetectionDataset ({split}): limited from {total_before} to {len(self.samples) + len(self.packed_indices)} samples")
        
        # 统计
        real_count = sum(1 for s in self.samples if s['label'] == 0)
        ai_count_samples = sum(1 for s in self.samples if s['label'] == 1)
        ai_count_packed = len(self.packed_indices)
        total = len(self.samples) + ai_count_packed
        print(f"AIDetectionDataset ({split}): real={real_count}, AI={ai_count_samples + ai_count_packed} "
              f"(samples={ai_count_samples}, packed={ai_count_packed}), total={total}")
    
    def _load_real_data(self, data_dir, repr_type, split='train'):
        """加载真实数据"""
        if repr_type == '263':
            motion_dir = os.path.join(data_dir, 'new_joint_vecs')
        else:
            motion_dir = os.path.join(data_dir, 'new_joints')
        
        if not os.path.exists(motion_dir):
            print(f"Warning: {motion_dir} not found")
            return
        
        # 根据 split 选择对应的文件
        split_file_map = {
            'train': 'train.txt',
            'val': 'val.txt',
            'test': 'test.txt'
        }
        split_file = os.path.join(data_dir, split_file_map.get(split, 'train.txt'))
        
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                names = [line.strip() for line in f.readlines()]
        else:
            # fallback: 使用目录中的所有文件
            names = [f[:-4] for f in os.listdir(motion_dir) if f.endswith('.npy')]
        
        # 过滤无效文件
        invalid_set = self.INVALID_FILES.get(repr_type, set())
        skipped = 0
        
        for name in names:
            if name in invalid_set:
                skipped += 1
                continue
            motion_path = os.path.join(motion_dir, f'{name}.npy')
            if os.path.exists(motion_path) and os.path.getsize(motion_path) > 0:
                self.samples.append({
                    'path': motion_path,
                    'repr_type': repr_type,
                    'label': 0,  # 真实数据
                    'source': 'real'
                })
        
        if skipped > 0:
            print(f"Skipped {skipped} invalid real {repr_type} files")
    
    def _load_ai_data(self, ai_dir, repr_type):
        """加载 AI 生成数据（目录模式）"""
        if repr_type == '263':
            motion_dir = os.path.join(ai_dir, 'new_joint_vecs')
        else:
            motion_dir = os.path.join(ai_dir, 'new_joints')
        
        if not os.path.exists(motion_dir):
            print(f"Warning: {motion_dir} not found")
            return
        
        source_name = os.path.basename(ai_dir)
        
        for f in os.listdir(motion_dir):
            if f.endswith('.npy'):
                motion_path = os.path.join(motion_dir, f)
                if os.path.getsize(motion_path) > 0:
                    self.samples.append({
                        'path': motion_path,
                        'repr_type': repr_type,
                        'label': 1,  # AI 生成
                        'source': source_name
                    })
    
    def _load_ai_packed(self, packed_file, repr_types):
        """加载打包的 AI 数据（打包模式，速度快）"""
        print(f"Loading packed AI data from {packed_file}...")
        self.packed_data = torch.load(packed_file, map_location='cpu', weights_only=False)
        self.use_packed = True
        
        for repr_type in repr_types:
            if repr_type in self.packed_data:
                n_samples = self.packed_data[repr_type]['motions'].shape[0]
                for i in range(n_samples):
                    self.packed_indices.append((repr_type, i))
                print(f"  Loaded {n_samples} packed {repr_type} AI samples")
    
    def _normalize(self, motion, repr_type):
        """标准化 motion"""
        if repr_type == '263':
            return (motion - self.mean_263) / self.std_263
        else:
            return (motion - self.mean_22x3) / self.std_22x3
    
    def _pad_motion(self, motion, repr_type):
        """Padding motion 到固定长度"""
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion)
        
        frames = motion.shape[0]
        
        if frames < self.max_motion_length:
            if repr_type == '263':
                pad = torch.zeros(self.max_motion_length - frames, motion.shape[1])
            else:  # 22x3
                if len(motion.shape) == 3:
                    pad = torch.zeros(self.max_motion_length - frames, motion.shape[1], motion.shape[2])
                else:
                    pad = torch.zeros(self.max_motion_length - frames, motion.shape[1])
            motion = torch.cat([motion, pad], dim=0)
        elif frames > self.max_motion_length:
            motion = motion[:self.max_motion_length]
        
        return motion
    
    def __getitem__(self, index):
        # 判断是从 samples 还是 packed_indices 获取
        n_samples = len(self.samples)
        
        if index < n_samples:
            # 从 samples 获取（目录模式或真实数据）
            item = self.samples[index]
            repr_type = item['repr_type']
            label = item['label']
            
            # 加载 motion
            try:
                motion = np.load(item['path'])
            except Exception as e:
                # 文件损坏，返回零填充
                if repr_type == '263':
                    motion = np.zeros((self.max_motion_length, 263))
                else:
                    motion = np.zeros((self.max_motion_length, 22, 3))
                return {
                    'motion': torch.from_numpy(motion).float(),
                    'label': label,
                    'repr_type': repr_type,
                    'length': self.max_motion_length
                }
            
            # 处理维度
            if repr_type == '263':
                # 确保是 2D: [T, 263]
                if len(motion.shape) != 2 or motion.shape[1] != 263:
                    # 无效形状，返回零填充
                    motion = np.zeros((self.max_motion_length, 263))
            else:  # 22x3
                # 确保是 3D: [T, 22, 3] 或 2D: [T, 66]
                if len(motion.shape) == 2 and motion.shape[1] == 66:
                    motion = motion.reshape(motion.shape[0], 22, 3)
                elif len(motion.shape) != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
                    # 无效形状，返回零填充
                    motion = np.zeros((self.max_motion_length, 22, 3))
            
            # 标准化
            motion = self._normalize(motion, repr_type)
            
            # Padding
            motion = self._pad_motion(motion, repr_type)
            
            return {
                'motion': motion.float(),
                'label': label,
                'repr_type': repr_type,
                'length': min(motion.shape[0], self.max_motion_length)
            }
        else:
            # 从 packed_indices 获取（打包模式的 AI 数据）
            packed_idx = index - n_samples
            repr_type, data_idx = self.packed_indices[packed_idx]
            
            # 直接从打包数据获取
            motion = self.packed_data[repr_type]['motions'][data_idx]  # 已经是 tensor
            length = self.packed_data[repr_type]['lengths'][data_idx].item()
            
            # 标准化（打包数据未标准化）
            if repr_type == '263':
                mean = torch.from_numpy(self.mean_263).float()
                std = torch.from_numpy(self.std_263).float()
            else:
                mean = torch.from_numpy(self.mean_22x3).float()
                std = torch.from_numpy(self.std_22x3).float()
            motion = (motion - mean) / std
            
            return {
                'motion': motion.float(),
                'label': 1,  # AI 生成
                'repr_type': repr_type,
                'length': length
            }
    
    def __len__(self):
        total = len(self.samples) + len(self.packed_indices)
        if self.max_samples is not None:
            return min(total, self.max_samples)
        return total


def ai_detection_collate_fn(batch):
    """AI 检测数据集的 collate 函数"""
    repr_type = batch[0]['repr_type']
    motions = torch.stack([item['motion'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    lengths = [item['length'] for item in batch]
    
    return {
        'motion': motions,
        'label': labels,
        'repr_type': repr_type,
        'length': lengths
    }


class AIDetectionReprTypeBatchSampler(torch.utils.data.Sampler):
    """AI 检测数据集按表征类型分组的 BatchSampler
    
    支持目录模式和打包模式混合
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, 
                 rank=0, world_size=1, seed=1234):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        
        self.indices_263 = []
        self.indices_22x3 = []
        
        # 从 samples 获取索引
        for i, sample in enumerate(dataset.samples):
            if sample['repr_type'] == '263':
                self.indices_263.append(i)
            elif sample['repr_type'] == '22x3':
                self.indices_22x3.append(i)
        
        # 从 packed_indices 获取索引（偏移 len(samples)）
        n_samples = len(dataset.samples)
        for i, (repr_type, _) in enumerate(dataset.packed_indices):
            global_idx = n_samples + i
            if repr_type == '263':
                self.indices_263.append(global_idx)
            elif repr_type == '22x3':
                self.indices_22x3.append(global_idx)
        
        if rank == 0:
            print(f"AIDetectionReprTypeBatchSampler: 263={len(self.indices_263)}, 22x3={len(self.indices_22x3)}")
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices_263 = self.indices_263.copy()
        indices_22x3 = self.indices_22x3.copy()
        
        if self.shuffle:
            perm_263 = torch.randperm(len(indices_263), generator=g).tolist()
            perm_22x3 = torch.randperm(len(indices_22x3), generator=g).tolist()
            indices_263 = [indices_263[i] for i in perm_263]
            indices_22x3 = [indices_22x3[i] for i in perm_22x3]
        
        batches = []
        
        for i in range(0, len(indices_263), self.batch_size):
            batch = indices_263[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        for i in range(0, len(indices_22x3), self.batch_size):
            batch = indices_22x3[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        if self.shuffle:
            perm_batches = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm_batches]
        
        batches_per_rank = len(batches) // self.world_size
        start_idx = self.rank * batches_per_rank
        end_idx = start_idx + batches_per_rank if self.rank < self.world_size - 1 else len(batches)
        
        for batch in batches[start_idx:end_idx]:
            yield batch
    
    def __len__(self):
        n_263 = len(self.indices_263) // self.batch_size if self.drop_last else (len(self.indices_263) + self.batch_size - 1) // self.batch_size
        n_22x3 = len(self.indices_22x3) // self.batch_size if self.drop_last else (len(self.indices_22x3) + self.batch_size - 1) // self.batch_size
        total = n_263 + n_22x3
        return max(1, total // self.world_size)
