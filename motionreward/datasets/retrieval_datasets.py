"""
Retrieval 训练数据集

包含:
- Text2MotionDataset263: 263维表征数据集
- JointLevelText2MotionDataset: 22x3 表征数据集
- PackedText2MotionDataset: 打包数据集（快速加载）
- PairedReprDataset: 配对表征数据集（跨表征对齐）
- PackedPairedReprDataset: 打包配对数据集
- ReprTypeBatchSampler: 按表征类型分组的 BatchSampler
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Text2MotionDataset263(Dataset):
    """263维表征数据集
    
    包含数据增强随机性（有助于模型泛化）：
    - 随机选择文本描述
    - 随机选择 'single' 或 'double' 模式
    - 随机位置裁剪 motion
    """
    def __init__(self, mean, std, split_file, motion_dir, text_dir,
                 unit_length=4, max_motion_length=200, min_motion_len=40, max_samples=None):
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.min_motion_len = min_motion_len
        self.mean = mean
        self.std = std
        
        data_dict = {}
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        if max_samples is not None:
            id_list = id_list[:max_samples]
        
        new_name_list = []
        length_list = []
        sub_motion_counter = 0  # 用于生成确定性的 key
        
        for name in tqdm(id_list, desc='Loading 263 data'):
            try:
                motion = np.load(os.path.join(motion_dir, name + '.npy'))
                if len(motion) < min_motion_len or len(motion) >= max_motion_length:
                    continue
                
                text_data = []
                flag = False
                with open(os.path.join(text_dir, name + '.txt'), 'r') as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20):int(to_tag*20)]
                                if len(n_motion) < min_motion_len or len(n_motion) >= max_motion_length:
                                    continue
                                # 使用确定性的 key：sub_{counter}_{name}
                                new_name = f'sub_{sub_motion_counter}_{name}'
                                sub_motion_counter += 1
                                data_dict[new_name] = {
                                    'motion': n_motion,
                                    'length': len(n_motion),
                                    'text': [text_dict]
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                pass
                
                if flag:
                    data_dict[name] = {
                        'motion': motion,
                        'length': len(motion),
                        'text': text_data
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                pass
        
        self.data_dict = data_dict
        self.name_list = new_name_list
        print(f"Total 263 motions: {len(self.data_dict)}")
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        
        # 随机选择一个文本描述（数据增强）
        text_data = random.choice(text_list)
        caption = text_data['caption']
        
        # 随机选择 'single' 或 'double' 模式（数据增强）
        coin2 = 'single' if self.unit_length >= 10 else np.random.choice(['single', 'single', 'double'])
        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
        
        # 确保 m_length > 0
        m_length = max(m_length, self.unit_length)
        
        # 随机位置裁剪（数据增强）
        if len(motion) > m_length:
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]
        else:
            m_length = len(motion)
        
        motion = (motion - self.mean) / self.std
        
        return {
            'motion': torch.tensor(motion).float(),
            'text': caption,
            'length': m_length,
            'repr_type': '263'
        }


class Text2MotionDataset135(Dataset):
    """135维表征数据集 (rot6d 表征，与 263 使用相同的 text 和 split 文件)
    
    注意：
    - 135 数据可能存在部分缺失，会自动跳过缺失的 motion 文件
    - 包含数据增强随机性（有助于模型泛化）
    """
    def __init__(self, mean, std, split_file, motion_dir, text_dir,
                 unit_length=4, max_motion_length=200, min_motion_len=40, max_samples=None):
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.min_motion_len = min_motion_len
        self.mean = mean
        self.std = std
        
        data_dict = {}
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        if max_samples is not None:
            id_list = id_list[:max_samples]
        
        new_name_list = []
        length_list = []
        skipped_count = 0
        sub_motion_counter = 0  # 用于生成确定性的 key
        
        for name in tqdm(id_list, desc='Loading 135 data'):
            motion_path = os.path.join(motion_dir, name + '.npy')
            text_path = os.path.join(text_dir, name + '.txt')
            
            # 检查 motion 文件是否存在，不存在则跳过
            if not os.path.exists(motion_path):
                skipped_count += 1
                continue
            
            # 检查 text 文件是否存在
            if not os.path.exists(text_path):
                skipped_count += 1
                continue
            
            try:
                motion = np.load(motion_path)
                if len(motion) < min_motion_len or len(motion) >= max_motion_length:
                    continue
                
                text_data = []
                flag = False
                with open(text_path, 'r') as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20):int(to_tag*20)]
                                if len(n_motion) < min_motion_len or len(n_motion) >= max_motion_length:
                                    continue
                                # 使用确定性的 key：sub_{counter}_{name}
                                new_name = f'sub_{sub_motion_counter}_{name}'
                                sub_motion_counter += 1
                                data_dict[new_name] = {
                                    'motion': n_motion,
                                    'length': len(n_motion),
                                    'text': [text_dict]
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                pass
                
                if flag:
                    data_dict[name] = {
                        'motion': motion,
                        'length': len(motion),
                        'text': text_data
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                pass
        
        self.data_dict = data_dict
        self.name_list = new_name_list
        print(f"Total 135 motions: {len(self.data_dict)} (skipped {skipped_count} missing files)")
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        
        # 随机选择一个文本描述（数据增强）
        text_data = random.choice(text_list)
        caption = text_data['caption']
        
        # 随机选择 'single' 或 'double' 模式（数据增强）
        coin2 = 'single' if self.unit_length >= 10 else np.random.choice(['single', 'single', 'double'])
        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
        
        # 确保 m_length > 0
        m_length = max(m_length, self.unit_length)
        
        # 随机位置裁剪（数据增强）
        if len(motion) > m_length:
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]
        else:
            m_length = len(motion)
        
        motion = (motion - self.mean) / self.std
        
        return {
            'motion': torch.tensor(motion).float(),
            'text': caption,
            'length': m_length,
            'repr_type': '135'
        }


class JointLevelText2MotionDataset(Dataset):
    """22x3 表征数据集
    
    包含数据增强随机性（有助于模型泛化）
    """
    def __init__(self, mean, std, split_file, motion_dir, text_dir,
                 unit_length=4, max_motion_length=200, min_motion_len=40, max_samples=None):
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.min_motion_len = min_motion_len
        self.mean = mean
        self.std = std
        
        data_dict = {}
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        if max_samples is not None:
            id_list = id_list[:max_samples]
        
        new_name_list = []
        length_list = []
        sub_motion_counter = 0  # 用于生成确定性的 key
        
        for name in tqdm(id_list, desc='Loading 22x3 data'):
            try:
                motion = np.load(os.path.join(motion_dir, name + '.npy'))
                if len(motion.shape) == 2:
                    motion = motion.reshape(motion.shape[0], -1, 3)
                
                if len(motion) < min_motion_len or len(motion) >= max_motion_length:
                    continue
                
                text_data = []
                flag = False
                with open(os.path.join(text_dir, name + '.txt'), 'r') as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20):int(to_tag*20)]
                                if len(n_motion) < min_motion_len or len(n_motion) >= max_motion_length:
                                    continue
                                # 使用确定性的 key：sub_{counter}_{name}
                                new_name = f'sub_{sub_motion_counter}_{name}'
                                sub_motion_counter += 1
                                data_dict[new_name] = {
                                    'motion': n_motion,
                                    'length': len(n_motion),
                                    'text': [text_dict]
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                pass
                
                if flag:
                    data_dict[name] = {
                        'motion': motion,
                        'length': len(motion),
                        'text': text_data
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                pass
        
        self.data_dict = data_dict
        self.name_list = new_name_list
        print(f"Total 22x3 motions: {len(self.data_dict)}")
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        
        # 随机选择一个文本描述（数据增强）
        text_data = random.choice(text_list)
        caption = text_data['caption']
        
        # 随机选择 'single' 或 'double' 模式（数据增强）
        coin2 = 'single' if self.unit_length >= 10 else np.random.choice(['single', 'single', 'double'])
        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
        
        # 确保 m_length > 0
        m_length = max(m_length, self.unit_length)
        
        # 随机位置裁剪（数据增强）
        if len(motion) > m_length:
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]
        else:
            m_length = len(motion)
        
        motion = (motion - self.mean) / self.std
        
        return {
            'motion': torch.tensor(motion).float(),
            'text': caption,
            'length': m_length,
            'repr_type': '22x3'
        }


class PackedText2MotionDataset(Dataset):
    """打包数据集 - 从预打包的 .pth 文件加载，速度快
    
    支持 263 和 22x3 两种表征，数据已预先标准化
    """
    def __init__(self, packed_file, repr_type='263', unit_length=4):
        """
        Args:
            packed_file: 打包文件路径 (.pth)
            repr_type: 表征类型 ('263' 或 '22x3')
            unit_length: 长度对齐单位
        """
        self.repr_type = repr_type
        self.unit_length = unit_length
        
        print(f"Loading packed data from {packed_file}...")
        packed = torch.load(packed_file, map_location='cpu', weights_only=False)
        
        if repr_type not in packed:
            raise ValueError(f"repr_type {repr_type} not found in packed file")
        
        self.data = packed[repr_type]
        self.motions = self.data['motions']  # List[np.ndarray]，已标准化
        self.lengths = self.data['lengths']  # List[int]
        self.texts = self.data['texts']      # List[List[str]]
        self.names = self.data['names']      # List[str]
        
        print(f"Loaded {len(self.motions)} {repr_type} samples from packed file")
    
    def __len__(self):
        return len(self.motions)
    
    def __getitem__(self, item):
        motion = self.motions[item]  # 已标准化的 np.ndarray
        m_length = self.lengths[item]
        text_list = self.texts[item]
        
        # 随机选择一个 caption
        caption = random.choice(text_list)
        
        # 长度对齐
        coin2 = 'single' if self.unit_length >= 10 else np.random.choice(['single', 'single', 'double'])
        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
        
        # 确保 m_length > 0
        m_length = max(m_length, self.unit_length)
        
        # 随机裁剪
        if len(motion) > m_length:
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]
        else:
            m_length = len(motion)
        
        return {
            'motion': torch.from_numpy(motion).float(),
            'text': caption,
            'length': m_length,
            'repr_type': self.repr_type
        }


class PairedReprDataset(Dataset):
    """配对表征数据集 - 用于跨表征对齐训练
    
    同一个 motion 的 263 和 22x3 表征配对，用于对齐训练
    """
    def __init__(self, mean_263, std_263, mean_22x3, std_22x3, 
                 split_file, motion_dir_263, motion_dir_22x3, text_dir,
                 unit_length=4, max_motion_length=200, min_motion_len=40):
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.min_motion_len = min_motion_len
        self.mean_263 = mean_263
        self.std_263 = std_263
        self.mean_22x3 = mean_22x3
        self.std_22x3 = std_22x3
        
        data_dict = {}
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        new_name_list = []
        
        for name in tqdm(id_list, desc='Loading paired data'):
            try:
                # 加载两种表征
                motion_263 = np.load(os.path.join(motion_dir_263, name + '.npy'))
                motion_22x3 = np.load(os.path.join(motion_dir_22x3, name + '.npy'))
                
                if len(motion_22x3.shape) == 2:
                    motion_22x3 = motion_22x3.reshape(motion_22x3.shape[0], -1, 3)
                
                # 长度必须一致
                if len(motion_263) != len(motion_22x3):
                    continue
                
                if len(motion_263) < min_motion_len or len(motion_263) >= max_motion_length:
                    continue
                
                # 加载文本
                text_data = []
                with open(os.path.join(text_dir, name + '.txt'), 'r') as f:
                    for line in f.readlines():
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        
                        if f_tag == 0.0 and to_tag == 0.0:
                            text_data.append(caption)
                
                if text_data:
                    data_dict[name] = {
                        'motion_263': motion_263,
                        'motion_22x3': motion_22x3,
                        'length': len(motion_263),
                        'text': text_data
                    }
                    new_name_list.append(name)
            except Exception as e:
                pass
        
        self.data_dict = data_dict
        self.name_list = new_name_list
        print(f"Total paired motions: {len(self.data_dict)}")
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        motion_263 = data['motion_263']
        motion_22x3 = data['motion_22x3']
        m_length = data['length']
        text_list = data['text']
        
        caption = random.choice(text_list)
        
        # 长度裁剪
        coin2 = 'single' if self.unit_length >= 10 else np.random.choice(['single', 'single', 'double'])
        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
        
        idx = random.randint(0, len(motion_263) - m_length)
        motion_263 = motion_263[idx:idx + m_length]
        motion_22x3 = motion_22x3[idx:idx + m_length]
        
        # 标准化
        motion_263 = (motion_263 - self.mean_263) / self.std_263
        motion_22x3 = (motion_22x3 - self.mean_22x3) / self.std_22x3
        
        return {
            'motion_263': torch.tensor(motion_263).float(),
            'motion_22x3': torch.tensor(motion_22x3).float(),
            'text': caption,
            'length': m_length
        }


class PackedPairedReprDataset(Dataset):
    """打包的配对表征数据集 - 用于跨表征对齐训练
    
    同一个 motion 的 263 和 22x3 表征配对，从打包文件加载
    """
    def __init__(self, packed_file, unit_length=4):
        """
        Args:
            packed_file: 打包文件路径 (.pth)
            unit_length: 长度对齐单位
        """
        self.unit_length = unit_length
        
        print(f"Loading packed paired data from {packed_file}...")
        packed = torch.load(packed_file, map_location='cpu', weights_only=False)
        
        if '263' not in packed or '22x3' not in packed:
            raise ValueError("Packed file must contain both '263' and '22x3' data")
        
        # 构建 name -> index 映射
        data_263 = packed['263']
        data_22x3 = packed['22x3']
        
        name_to_idx_263 = {name: i for i, name in enumerate(data_263['names'])}
        name_to_idx_22x3 = {name: i for i, name in enumerate(data_22x3['names'])}
        
        # 找出两者都有的 name
        common_names = set(data_263['names']) & set(data_22x3['names'])
        
        self.pairs = []
        for name in common_names:
            idx_263 = name_to_idx_263[name]
            idx_22x3 = name_to_idx_22x3[name]
            
            # 检查长度一致
            if data_263['lengths'][idx_263] == data_22x3['lengths'][idx_22x3]:
                self.pairs.append({
                    'motion_263': data_263['motions'][idx_263],
                    'motion_22x3': data_22x3['motions'][idx_22x3],
                    'length': data_263['lengths'][idx_263],
                    'texts': data_263['texts'][idx_263],
                    'name': name
                })
        
        print(f"Loaded {len(self.pairs)} paired samples from packed file")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, item):
        pair = self.pairs[item]
        motion_263 = pair['motion_263']
        motion_22x3 = pair['motion_22x3']
        m_length = pair['length']
        text_list = pair['texts']
        
        caption = random.choice(text_list)
        
        # 长度对齐
        coin2 = 'single' if self.unit_length >= 10 else np.random.choice(['single', 'single', 'double'])
        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
        
        m_length = max(m_length, self.unit_length)
        
        # 随机裁剪
        if len(motion_263) > m_length:
            idx = random.randint(0, len(motion_263) - m_length)
            motion_263 = motion_263[idx:idx + m_length]
            motion_22x3 = motion_22x3[idx:idx + m_length]
        else:
            m_length = len(motion_263)
        
        return {
            'motion_263': torch.from_numpy(motion_263).float(),
            'motion_22x3': torch.from_numpy(motion_22x3).float(),
            'text': caption,
            'length': m_length
        }


def retrieval_collate_fn(batch):
    """Retrieval collate 函数 - 支持多表征"""
    motion_list = [item['motion'] for item in batch]
    texts = [item['text'] for item in batch]
    lengths = [item['length'] for item in batch]
    repr_types = [item['repr_type'] for item in batch]
    
    max_len = max(m.shape[0] for m in motion_list)
    
    padded_motions = []
    for motion in motion_list:
        if motion.shape[0] < max_len:
            if len(motion.shape) == 2:
                padding = torch.zeros(max_len - motion.shape[0], motion.shape[1])
            else:
                padding = torch.zeros(max_len - motion.shape[0], motion.shape[1], motion.shape[2])
            padded_motion = torch.cat([motion, padding], dim=0)
        else:
            padded_motion = motion
        padded_motions.append(padded_motion)
    
    motions = torch.stack(padded_motions)
    
    return {
        'motion': motions,
        'text': texts,
        'length': lengths,
        'repr_type': repr_types[0]  # 假设同一 batch 内表征类型相同
    }


def paired_collate_fn(batch):
    """配对数据集的 collate 函数"""
    motion_263_list = [item['motion_263'] for item in batch]
    motion_22x3_list = [item['motion_22x3'] for item in batch]
    texts = [item['text'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    max_len = max(m.shape[0] for m in motion_263_list)
    
    padded_263 = []
    padded_22x3 = []
    for m263, m22x3 in zip(motion_263_list, motion_22x3_list):
        if m263.shape[0] < max_len:
            pad_263 = torch.zeros(max_len - m263.shape[0], m263.shape[1])
            pad_22x3 = torch.zeros(max_len - m22x3.shape[0], m22x3.shape[1], m22x3.shape[2])
            m263 = torch.cat([m263, pad_263], dim=0)
            m22x3 = torch.cat([m22x3, pad_22x3], dim=0)
        padded_263.append(m263)
        padded_22x3.append(m22x3)
    
    return {
        'motion_263': torch.stack(padded_263),
        'motion_22x3': torch.stack(padded_22x3),
        'text': texts,
        'length': lengths
    }


class ReprTypeBatchSampler(torch.utils.data.Sampler):
    """按表征类型分组的 BatchSampler
    
    确保同一 batch 内的样本具有相同的表征类型
    支持 263、22x3、135 三种表征类型
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
        self.indices_135 = []
        
        for i in range(len(dataset)):
            item = dataset[i]
            if item['repr_type'] == '263':
                self.indices_263.append(i)
            elif item['repr_type'] == '22x3':
                self.indices_22x3.append(i)
            elif item['repr_type'] == '135':
                self.indices_135.append(i)
        
        if rank == 0:
            print(f"ReprTypeBatchSampler: 263={len(self.indices_263)}, 22x3={len(self.indices_22x3)}, 135={len(self.indices_135)}")
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices_263 = self.indices_263.copy()
        indices_22x3 = self.indices_22x3.copy()
        indices_135 = self.indices_135.copy()
        
        if self.shuffle:
            perm_263 = torch.randperm(len(indices_263), generator=g).tolist()
            perm_22x3 = torch.randperm(len(indices_22x3), generator=g).tolist()
            perm_135 = torch.randperm(len(indices_135), generator=g).tolist()
            indices_263 = [indices_263[i] for i in perm_263]
            indices_22x3 = [indices_22x3[i] for i in perm_22x3]
            indices_135 = [indices_135[i] for i in perm_135]
        
        batches = []
        
        for i in range(0, len(indices_263), self.batch_size):
            batch = indices_263[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        for i in range(0, len(indices_22x3), self.batch_size):
            batch = indices_22x3[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        for i in range(0, len(indices_135), self.batch_size):
            batch = indices_135[i:i + self.batch_size]
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
        n_135 = len(self.indices_135) // self.batch_size if self.drop_last else (len(self.indices_135) + self.batch_size - 1) // self.batch_size
        total = n_263 + n_22x3 + n_135
        return max(1, total // self.world_size)
