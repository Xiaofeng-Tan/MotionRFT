"""
Critic 训练数据集

包含:
- CriticPairDataset: Critic 配对数据集（pairwise ranking）
- CriticReprTypeBatchSampler: 按表征类型分组的 BatchSampler

支持的数据格式:
1. 单个 .pth 文件（旧格式）
2. 目录格式（新格式）：包含 train/ 和 eval/ 子目录，每个子目录下有多个 .pth 文件

支持的表征类型:
- 263: HumanML3D 263维表征
- 22x3: 关节级 22x3 表征
- 66: positions 66维表征
- 135: rot6d 135维表征

重要说明:
- Critic 训练数据 (critic_train_*.pth) 存储的是原始特征（未归一化）
- Stage 1 Retrieval backbone 训练时使用的是 Z-归一化数据: (x - Mean) / Std
- 为保持一致，CriticPairDataset 在 __getitem__ 中会对 motion 做 Z-归一化
- 这样 Critic Head 训练时看到的输入分布与 backbone 训练时一致
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class CriticPairDataset(Dataset):
    """Critic 配对数据集 - 用于 pairwise ranking 训练
    
    每个样本包含 (motion_better, motion_worse) 配对
    
    支持多种数据格式：
    1. 单个 .pth 文件
    2. 目录格式：包含 train/ 和 eval/ 子目录
    
    重要: 数据文件中存储的是原始特征（未归一化），通过 mean_std_dict 参数传入
    各表征类型的 Mean/Std，在 __getitem__ 中做 Z-归一化，与 Stage 1 backbone 保持一致。
    
    Args:
        data_path_263: 263维数据路径（文件或目录）
        data_path_22x3: 22x3维数据路径（文件或目录）
        data_path_66: 66维数据路径（目录格式）
        data_path_135: 135维数据路径（目录格式）
        max_motion_length: 最大动作长度
        max_samples: 最大样本数（用于 debug）
        split: 数据集划分（'train' 或 'eval'），用于目录格式
        mean_std_dict: 各表征类型的归一化参数字典，格式:
            {'263': (mean_263, std_263), '22x3': (mean_22x3, std_22x3), '135': (mean_135, std_135)}
            如果为 None，则不做归一化（向后兼容）
    """
    def __init__(self, data_path_263=None, data_path_22x3=None, 
                 data_path_66=None, data_path_135=None,
                 max_motion_length=196, max_samples=None, split='train',
                 mean_std_dict=None):
        self.pairs = []
        self.max_motion_length = max_motion_length
        self.max_samples = max_samples
        self.split = split
        
        # 归一化参数: {repr_type: (mean_tensor, std_tensor)}
        self.mean_std = {}
        if mean_std_dict is not None:
            for repr_type, (mean, std) in mean_std_dict.items():
                if isinstance(mean, np.ndarray):
                    mean = torch.from_numpy(mean).float()
                if isinstance(std, np.ndarray):
                    std = torch.from_numpy(std).float()
                self.mean_std[repr_type] = (mean, std)
            print(f"[CriticPairDataset] Z-normalization enabled for: {list(self.mean_std.keys())}")
        else:
            print(f"[CriticPairDataset] Warning: No mean_std_dict provided, data will NOT be normalized. "
                  f"This may cause mismatch with Stage 1 backbone training.")
        
        # 加载各种表征类型的数据
        if data_path_263 is not None and os.path.exists(data_path_263):
            self._load_data(data_path_263, repr_type='263')
        
        if data_path_22x3 is not None and os.path.exists(data_path_22x3):
            self._load_data(data_path_22x3, repr_type='22x3')
        
        if data_path_66 is not None and os.path.exists(data_path_66):
            # 66 维 = 22x3，映射为模型认识的 repr_type
            self._load_data(data_path_66, repr_type='22x3')
        
        if data_path_135 is not None and os.path.exists(data_path_135):
            self._load_data(data_path_135, repr_type='135')
        
        # Debug 模式：限制样本数
        if max_samples is not None and len(self.pairs) > max_samples:
            self.pairs = self.pairs[:max_samples]
        
        print(f"Total critic pairs: {len(self.pairs)}")
    
    def _load_data(self, data_path, repr_type):
        """加载数据
        
        支持两种格式：
        1. 单个 .pth 文件
        2. 目录格式：包含 train/ 和 eval/ 子目录
        """
        try:
            if os.path.isdir(data_path):
                # 检查是否是新的目录格式（包含 train/ 和 eval/ 子目录）
                train_dir = os.path.join(data_path, 'train')
                eval_dir = os.path.join(data_path, 'eval')
                
                if os.path.exists(train_dir) or os.path.exists(eval_dir):
                    # 新的目录格式
                    self._load_from_new_format_dir(data_path, repr_type)
                else:
                    # 旧的 chunk 文件格式
                    self._load_from_chunk_dir(data_path, repr_type)
            else:
                # 单个 .pth 文件
                self._load_from_file(data_path, repr_type)
        except Exception as e:
            print(f"Error loading {repr_type} data from {data_path}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        loaded_count = len([p for p in self.pairs if p['repr_type'] == repr_type])
        print(f"Loaded {repr_type} critic pairs: {loaded_count}")
    
    def _load_from_new_format_dir(self, data_path, repr_type):
        """从新格式目录加载数据
        
        新格式目录结构：
        data_path/
            train/
                xxx-train.pth
                ...
            eval/
                xxx-eval.pth
                ...
        """
        # 根据 split 选择子目录
        if self.split == 'train':
            sub_dir = os.path.join(data_path, 'train')
            pattern = '*-train.pth'
        else:
            sub_dir = os.path.join(data_path, 'eval')
            pattern = '*-eval.pth'
        
        if not os.path.exists(sub_dir):
            print(f"Warning: {sub_dir} does not exist")
            return
        
        pth_files = sorted(glob.glob(os.path.join(sub_dir, pattern)))
        if not pth_files:
            # 尝试通用模式
            pth_files = sorted(glob.glob(os.path.join(sub_dir, '*.pth')))
        
        if not pth_files:
            print(f"Warning: No .pth files found in {sub_dir}")
            return
        
        print(f"Loading {len(pth_files)} files from {sub_dir}")
        
        for pth_file in pth_files:
            self._load_from_file(pth_file, repr_type)
    
    def _load_from_chunk_dir(self, data_path, repr_type):
        """从旧的 chunk 目录格式加载数据"""
        chunk_files = sorted(glob.glob(os.path.join(data_path, "chunk_*.pth")))
        if not chunk_files:
            print(f"Warning: No chunk files found in {data_path}")
            return
        
        for chunk_file in chunk_files:
            chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
            motion_better = chunk_data['motion_better']
            motion_worse = chunk_data['motion_worse']
            N = motion_better.shape[0]
            for i in range(N):
                self.pairs.append({
                    'motion_better': motion_better[i],
                    'motion_worse': motion_worse[i],
                    'repr_type': repr_type
                })
    
    def _load_from_file(self, data_path, repr_type):
        """从单个 .pth 文件加载数据"""
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        
        if isinstance(data, list):
            for item in data:
                motion_better = item['motion_better']
                motion_worse = item['motion_worse']
                
                # 去除 batch 维度: (1, T, D) -> (T, D) 或 (1, T, J, 3) -> (T, J, 3)
                if motion_better.shape[0] == 1 and len(motion_better.shape) >= 3:
                    motion_better = motion_better.squeeze(0)
                if motion_worse.shape[0] == 1 and len(motion_worse.shape) >= 3:
                    motion_worse = motion_worse.squeeze(0)
                
                self.pairs.append({
                    'motion_better': motion_better,
                    'motion_worse': motion_worse,
                    'repr_type': repr_type
                })
        elif isinstance(data, dict) and 'motion_better' in data:
            motion_better = data['motion_better']
            motion_worse = data['motion_worse']
            N = motion_better.shape[0]
            for i in range(N):
                mb = motion_better[i]
                mw = motion_worse[i]
                
                # 去除 batch 维度
                if mb.shape[0] == 1 and len(mb.shape) >= 3:
                    mb = mb.squeeze(0)
                if mw.shape[0] == 1 and len(mw.shape) >= 3:
                    mw = mw.squeeze(0)
                
                self.pairs.append({
                    'motion_better': mb,
                    'motion_worse': mw,
                    'repr_type': repr_type
                })
    
    def _pad_motion(self, motion, repr_type):
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion)
        
        frames = motion.shape[0]
        
        if frames < self.max_motion_length:
            # 根据 motion 的实际 shape 进行 padding
            if len(motion.shape) == 2:
                # (T, D) 格式：263, 66, 135 等
                pad = torch.zeros(self.max_motion_length - frames, motion.shape[1])
            elif len(motion.shape) == 3:
                # (T, J, 3) 格式：22x3
                pad = torch.zeros(self.max_motion_length - frames, motion.shape[1], motion.shape[2])
            else:
                raise ValueError(f"Unexpected motion shape: {motion.shape}")
            motion = torch.cat([motion, pad], dim=0)
        elif frames > self.max_motion_length:
            motion = motion[:self.max_motion_length]
        
        return motion
    
    def _normalize_motion(self, motion, repr_type):
        """对 motion 做 Z-归一化: (x - mean) / std
        
        与 Stage 1 Retrieval 训练时 Text2MotionDataset263 等数据集保持一致。
        """
        if repr_type not in self.mean_std:
            return motion
        
        mean, std = self.mean_std[repr_type]
        # 确保 mean/std 与 motion 在同一设备上且类型匹配
        mean = mean.to(motion.device)
        std = std.to(motion.device)
        return (motion - mean) / std
    
    def __getitem__(self, index):
        item = self.pairs[index]
        repr_type = item['repr_type']
        
        # 先归一化，再 padding（与 Stage 1 Retrieval 训练一致：先归一化，collate 时再 pad）
        # 这样 padding 区域保持为 0，不会被归一化成 -mean/std
        motion_better = item['motion_better']
        motion_worse = item['motion_worse']
        if isinstance(motion_better, np.ndarray):
            motion_better = torch.from_numpy(motion_better)
        if isinstance(motion_worse, np.ndarray):
            motion_worse = torch.from_numpy(motion_worse)
        
        # Z-归一化（在 padding 之前），与 Stage 1 backbone 训练时保持一致
        motion_better = self._normalize_motion(motion_better.float(), repr_type)
        motion_worse = self._normalize_motion(motion_worse.float(), repr_type)
        
        # Padding（归一化之后）
        motion_better = self._pad_motion(motion_better, repr_type)
        motion_worse = self._pad_motion(motion_worse, repr_type)
        
        return {
            'motion_better': motion_better,
            'motion_worse': motion_worse,
            'repr_type': repr_type
        }
    
    def __len__(self):
        return len(self.pairs)


def critic_collate_fn(batch):
    """Critic 数据集的 collate 函数"""
    repr_type = batch[0]['repr_type']
    motion_better = torch.stack([item['motion_better'] for item in batch])
    motion_worse = torch.stack([item['motion_worse'] for item in batch])
    
    return {
        'motion_better': motion_better,
        'motion_worse': motion_worse,
        'repr_type': repr_type
    }


class CriticReprTypeBatchSampler(torch.utils.data.Sampler):
    """Critic 数据集按表征类型分组的 BatchSampler
    
    支持的表征类型：263, 22x3, 66, 135
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
        
        # 按表征类型分组索引
        self.indices_by_repr = {}
        for i, pair in enumerate(dataset.pairs):
            repr_type = pair['repr_type']
            if repr_type not in self.indices_by_repr:
                self.indices_by_repr[repr_type] = []
            self.indices_by_repr[repr_type].append(i)
        
        if rank == 0:
            for repr_type, indices in self.indices_by_repr.items():
                print(f"CriticReprTypeBatchSampler: {repr_type}={len(indices)}")
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        batches = []
        
        # 对每种表征类型生成 batches
        for repr_type, indices in self.indices_by_repr.items():
            indices_copy = indices.copy()
            
            if self.shuffle:
                perm = torch.randperm(len(indices_copy), generator=g).tolist()
                indices_copy = [indices_copy[i] for i in perm]
            
            for i in range(0, len(indices_copy), self.batch_size):
                batch = indices_copy[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        # 打乱所有 batches
        if self.shuffle:
            perm_batches = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm_batches]
        
        # 分配给当前 rank
        batches_per_rank = len(batches) // self.world_size
        start_idx = self.rank * batches_per_rank
        end_idx = start_idx + batches_per_rank if self.rank < self.world_size - 1 else len(batches)
        
        for batch in batches[start_idx:end_idx]:
            yield batch
    
    def __len__(self):
        total = 0
        for indices in self.indices_by_repr.values():
            n = len(indices) // self.batch_size if self.drop_last else (len(indices) + self.batch_size - 1) // self.batch_size
            total += n
        return max(1, total // self.world_size)
