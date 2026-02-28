"""
数据加载工具函数
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler

from motionreward.datasets import (
    Text2MotionDataset263,
    Text2MotionDataset135,
    JointLevelText2MotionDataset,
    CriticPairDataset,
    AIDetectionDataset,
    retrieval_collate_fn,
    critic_collate_fn,
    ai_detection_collate_fn,
    ReprTypeBatchSampler,
    CriticReprTypeBatchSampler,
    AIDetectionReprTypeBatchSampler,
    PackedText2MotionDataset,
    PairedReprDataset,
    PackedPairedReprDataset,
    paired_collate_fn,
)


def load_normalization_stats(data_root):
    """加载归一化统计量
    
    注意：所有表征类型的 Mean/Std 文件必须存在，否则会报错。
    这是为了确保训练时使用正确的标准化参数，避免数据分布不一致的问题。
    """
    # 263 维表征
    mean_263 = np.load(os.path.join(data_root, 'Mean.npy'))
    std_263 = np.load(os.path.join(data_root, 'Std.npy'))
    
    # 22x3 表征 - 必须存在
    mean_22x3_path = os.path.join(data_root, 'Mean_22x3.npy')
    std_22x3_path = os.path.join(data_root, 'Std_22x3.npy')
    if not os.path.exists(mean_22x3_path):
        raise FileNotFoundError(
            f"Mean_22x3.npy not found at {mean_22x3_path}. "
            f"Please copy from ACMDM/utils/22x3_mean_std/t2m/22x3_mean.npy"
        )
    if not os.path.exists(std_22x3_path):
        raise FileNotFoundError(
            f"Std_22x3.npy not found at {std_22x3_path}. "
            f"Please copy from ACMDM/utils/22x3_mean_std/t2m/22x3_std.npy"
        )
    mean_22x3 = np.load(mean_22x3_path)
    std_22x3 = np.load(std_22x3_path)
    
    # 135 维表征
    mean_135_path = os.path.join(data_root, 'Mean_135.npy')
    std_135_path = os.path.join(data_root, 'Std_135.npy')
    if os.path.exists(mean_135_path):
        mean_135 = np.load(mean_135_path)
        std_135 = np.load(std_135_path)
    else:
        mean_135 = np.zeros(135)
        std_135 = np.ones(135)
    
    return mean_263, std_263, mean_22x3, std_22x3, mean_135, std_135


def _limit_dataset(dataset, max_samples):
    """限制数据集大小（用于 debug 模式）"""
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    indices = list(range(min(len(dataset), max_samples)))
    return Subset(dataset, indices)


def create_retrieval_dataloaders(cfg, retrieval_repr_list, rank, world_size, is_distributed):
    """创建 Retrieval 训练和测试数据加载器"""
    data_root = cfg.DATASET.HUMANML3D.ROOT
    motion_dir_263 = os.path.join(data_root, 'new_joint_vecs')
    motion_dir_22x3 = os.path.join(data_root, 'new_joints')
    motion_dir_135 = os.path.join(data_root, 'joints_6d')
    text_dir = os.path.join(data_root, 'texts')
    train_split = os.path.join(data_root, 'train.txt')
    test_split = os.path.join(data_root, 'test.txt')
    
    mean_263, std_263, mean_22x3, std_22x3, mean_135, std_135 = load_normalization_stats(data_root)
    
    debug_samples = getattr(cfg, 'debug_samples', None) if getattr(cfg, 'debug', False) else None
    
    train_datasets = []
    test_datasets = {}
    
    if '263' in retrieval_repr_list:
        train_ds_263 = Text2MotionDataset263(mean_263, std_263, train_split, motion_dir_263, text_dir, max_samples=debug_samples)
        test_ds_263 = Text2MotionDataset263(mean_263, std_263, test_split, motion_dir_263, text_dir, max_samples=debug_samples)
        train_datasets.append(train_ds_263)
        test_datasets['263'] = test_ds_263
    
    if '22x3' in retrieval_repr_list:
        train_ds_22x3 = JointLevelText2MotionDataset(mean_22x3, std_22x3, train_split, motion_dir_22x3, text_dir, max_samples=debug_samples)
        test_ds_22x3 = JointLevelText2MotionDataset(mean_22x3, std_22x3, test_split, motion_dir_22x3, text_dir, max_samples=debug_samples)
        train_datasets.append(train_ds_22x3)
        test_datasets['22x3'] = test_ds_22x3
    
    # 135 维数据集
    if '135' in retrieval_repr_list:
        flowmdm_datapath = getattr(cfg, 'flowmdm_datapath', None)
        if flowmdm_datapath and os.path.exists(flowmdm_datapath):
            motion_dir_135 = flowmdm_datapath
        
        train_ds_135 = Text2MotionDataset135(mean_135, std_135, train_split, motion_dir_135, text_dir, max_samples=debug_samples)
        test_ds_135 = Text2MotionDataset135(mean_135, std_135, test_split, motion_dir_135, text_dir, max_samples=debug_samples)
        train_datasets.append(train_ds_135)
        test_datasets['135'] = test_ds_135
    
    # 创建训练 DataLoader
    seed = getattr(cfg, 'SEED_VALUE', 1234)
    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
        batch_sampler = ReprTypeBatchSampler(
            train_dataset, 
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            rank=rank,
            world_size=world_size,
            drop_last=True,
            seed=seed
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.TRAIN.num_workers,
            collate_fn=retrieval_collate_fn,
            pin_memory=True
        )
    else:
        train_dataset = train_datasets[0]
        if is_distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                sampler=train_sampler,
                num_workers=cfg.TRAIN.num_workers,
                collate_fn=retrieval_collate_fn,
                pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                shuffle=True,
                num_workers=cfg.TRAIN.num_workers,
                collate_fn=retrieval_collate_fn,
                pin_memory=True
            )
    
    # 创建测试 DataLoader
    test_loaders = {}
    for repr_type, test_ds in test_datasets.items():
        test_loaders[repr_type] = DataLoader(
            test_ds,
            batch_size=min(32, cfg.TRAIN.BATCH_SIZE),
            shuffle=False,
            num_workers=cfg.TRAIN.num_workers,
            collate_fn=retrieval_collate_fn,
            pin_memory=True
        )
    
    return train_loader, test_loaders


def create_retrieval_test_loaders_only(cfg, retrieval_repr_list, rank):
    """仅创建 Retrieval 测试数据加载器
    
    注意：如果 use_retrieval_packed=True，会使用 packed 数据以确保与训练时一致
    """
    use_packed = getattr(cfg, 'use_retrieval_packed', False)
    packed_test_path = getattr(cfg, 'retrieval_packed_test', None)
    
    # 如果没有设置 retrieval_packed_test，使用默认路径
    if use_packed and not packed_test_path:
        proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        packed_test_path = os.path.join(proj_dir, 'datasets/retrieval_packed', 'retrieval_test.pth')
    
    # 使用 packed 数据
    if use_packed and packed_test_path and os.path.exists(packed_test_path):
        debug_samples = getattr(cfg, 'debug_samples', None) if getattr(cfg, 'debug', False) else None
        test_datasets = {}
        
        for repr_type in retrieval_repr_list:
            try:
                test_ds = PackedText2MotionDataset(packed_test_path, repr_type=repr_type)
                test_ds = _limit_dataset(test_ds, debug_samples)
                test_datasets[repr_type] = test_ds
            except ValueError as e:
                if rank == 0:
                    print(f"[Warning] Could not load packed data for {repr_type}: {e}")
        
        test_loaders = {}
        for repr_type, test_ds in test_datasets.items():
            test_loaders[repr_type] = DataLoader(
                test_ds,
                batch_size=min(32, cfg.TRAIN.BATCH_SIZE),
                shuffle=False,
                num_workers=cfg.TRAIN.num_workers,
                collate_fn=retrieval_collate_fn,
                pin_memory=True
            )
        
        return test_loaders
    
    # 使用原始数据（非 packed 模式）
    data_root = cfg.DATASET.HUMANML3D.ROOT
    motion_dir_263 = os.path.join(data_root, 'new_joint_vecs')
    motion_dir_22x3 = os.path.join(data_root, 'new_joints')
    motion_dir_135 = os.path.join(data_root, 'joints_6d')
    text_dir = os.path.join(data_root, 'texts')
    test_split = os.path.join(data_root, 'test.txt')
    
    mean_263, std_263, mean_22x3, std_22x3, mean_135, std_135 = load_normalization_stats(data_root)
    
    debug_samples = getattr(cfg, 'debug_samples', None) if getattr(cfg, 'debug', False) else None
    
    test_datasets = {}
    
    if '263' in retrieval_repr_list:
        test_ds_263 = Text2MotionDataset263(mean_263, std_263, test_split, motion_dir_263, text_dir, max_samples=debug_samples)
        test_datasets['263'] = test_ds_263
    
    if '22x3' in retrieval_repr_list:
        test_ds_22x3 = JointLevelText2MotionDataset(mean_22x3, std_22x3, test_split, motion_dir_22x3, text_dir, max_samples=debug_samples)
        test_datasets['22x3'] = test_ds_22x3
    
    # 135 维数据集
    if '135' in retrieval_repr_list:
        flowmdm_datapath = getattr(cfg, 'flowmdm_datapath', None)
        if flowmdm_datapath and os.path.exists(flowmdm_datapath):
            motion_dir_135 = flowmdm_datapath
        
        test_ds_135 = Text2MotionDataset135(mean_135, std_135, test_split, motion_dir_135, text_dir, max_samples=debug_samples)
        test_datasets['135'] = test_ds_135
    
    test_loaders = {}
    for repr_type, test_ds in test_datasets.items():
        test_loaders[repr_type] = DataLoader(
            test_ds,
            batch_size=min(32, cfg.TRAIN.BATCH_SIZE),
            shuffle=False,
            num_workers=cfg.TRAIN.num_workers,
            collate_fn=retrieval_collate_fn,
            pin_memory=True
        )
    
    return test_loaders


def create_critic_dataloaders(cfg, retrieval_repr_list, rank, world_size):
    """创建 Critic 训练和验证数据加载器
    
    数据源优先级:
    1. 从 MotionCritic 转换的单文件格式 (critic_converted_train_*.pth)
    2. 新目录格式 (critic_data_dir_*)
    3. 旧单文件格式 (critic_train_data_*)
    
    重要: Critic 训练数据存储的是原始特征（未归一化），但 Stage 1 backbone 是在
    Z-归一化数据上训练的。为保持一致，这里加载 Mean/Std 并传给 CriticPairDataset，
    使其在 __getitem__ 中做 Z-归一化。
    """
    debug_samples = getattr(cfg, 'debug_samples', None) if getattr(cfg, 'debug', False) else None
    
    # === 加载归一化参数，传给 CriticPairDataset 做 Z-归一化 ===
    data_root = cfg.DATASET.HUMANML3D.ROOT
    mean_263, std_263, mean_22x3, std_22x3, mean_135, std_135 = load_normalization_stats(data_root)
    
    mean_std_dict = {}
    if '263' in retrieval_repr_list:
        mean_std_dict['263'] = (mean_263, std_263)
    if '22x3' in retrieval_repr_list:
        mean_std_dict['22x3'] = (mean_22x3, std_22x3)
    if '135' in retrieval_repr_list:
        mean_std_dict['135'] = (mean_135, std_135)
    
    if rank == 0:
        print(f"[Critic Data] Z-normalization enabled for repr types: {list(mean_std_dict.keys())}")
    
    # === 优先检查从 MotionCritic 转换的单文件格式 ===
    converted_263 = getattr(cfg, 'critic_converted_train_263', None)
    converted_22x3 = getattr(cfg, 'critic_converted_train_22x3', None)
    converted_135 = getattr(cfg, 'critic_converted_train_135', None)
    
    has_263 = '263' in retrieval_repr_list
    has_22x3 = '22x3' in retrieval_repr_list
    has_135 = '135' in retrieval_repr_list
    
    c_263 = converted_263 if (has_263 and converted_263 and os.path.exists(converted_263)) else None
    c_22x3 = converted_22x3 if (has_22x3 and converted_22x3 and os.path.exists(converted_22x3)) else None
    c_135 = converted_135 if (has_135 and converted_135 and os.path.exists(converted_135)) else None
    
    use_converted = any([c_263, c_22x3, c_135])
    
    if use_converted:
        if rank == 0:
            print("[Critic Data] 使用从 MotionCritic 转换的单文件格式")
            if c_263: print(f"  263: {c_263}")
            if c_22x3: print(f"  22x3: {c_22x3}")
            if c_135: print(f"  135: {c_135}")
        
        critic_train_dataset = CriticPairDataset(
            data_path_263=c_263,
            data_path_22x3=c_22x3,
            data_path_135=c_135,
            max_motion_length=196,
            max_samples=debug_samples,
            split='train',
            mean_std_dict=mean_std_dict
        )
        
        # 加载从 MotionCritic 转换的验证集
        converted_val_263 = getattr(cfg, 'critic_converted_val_263', None)
        converted_val_22x3 = getattr(cfg, 'critic_converted_val_22x3', None)
        converted_val_135 = getattr(cfg, 'critic_converted_val_135', None)
        
        cv_263 = converted_val_263 if (has_263 and converted_val_263 and os.path.exists(converted_val_263)) else None
        cv_22x3 = converted_val_22x3 if (has_22x3 and converted_val_22x3 and os.path.exists(converted_val_22x3)) else None
        cv_135 = converted_val_135 if (has_135 and converted_val_135 and os.path.exists(converted_val_135)) else None
        
        has_val = any([cv_263, cv_22x3, cv_135])
        
        if has_val:
            if rank == 0:
                print("[Critic Data] 使用从 MotionCritic 转换的验证集")
                if cv_263: print(f"  val 263: {cv_263}")
                if cv_22x3: print(f"  val 22x3: {cv_22x3}")
                if cv_135: print(f"  val 135: {cv_135}")
            critic_val_dataset = CriticPairDataset(
                data_path_263=cv_263,
                data_path_22x3=cv_22x3,
                data_path_135=cv_135,
                max_motion_length=196,
                max_samples=debug_samples,
                split='eval',
                mean_std_dict=mean_std_dict
            )
        else:
            if rank == 0:
                print("[Critic Data] 验证集文件不存在，使用空验证集")
            critic_val_dataset = CriticPairDataset(
                max_motion_length=196,
                max_samples=0,
                split='eval',
                mean_std_dict=mean_std_dict
            )
    else:
        # === 检查新目录格式 ===
        use_new_format = getattr(cfg, 'use_new_critic_data', False)
        
        if use_new_format:
            data_dir_263 = getattr(cfg, 'critic_data_dir_263', None)
            data_dir_66 = getattr(cfg, 'critic_data_dir_66', None)
            data_dir_135 = getattr(cfg, 'critic_data_dir_135', None)
            
            has_66 = '22x3' in retrieval_repr_list or '66' in retrieval_repr_list
            
            critic_data_263 = data_dir_263 if (has_263 and data_dir_263 and os.path.exists(data_dir_263)) else None
            critic_data_66 = data_dir_66 if (has_66 and data_dir_66 and os.path.exists(data_dir_66)) else None
            critic_data_135 = data_dir_135 if (has_135 and data_dir_135 and os.path.exists(data_dir_135)) else None
            
            has_train_data = any([critic_data_263, critic_data_66, critic_data_135])
            
            if not has_train_data:
                use_new_format = False
        
        if use_new_format:
            critic_train_dataset = CriticPairDataset(
                data_path_263=critic_data_263,
                data_path_66=critic_data_66,
                data_path_135=critic_data_135,
                max_motion_length=196,
                max_samples=debug_samples,
                split='train',
                mean_std_dict=mean_std_dict
            )
            
            critic_val_dataset = CriticPairDataset(
                data_path_263=critic_data_263,
                data_path_66=critic_data_66,
                data_path_135=critic_data_135,
                max_motion_length=196,
                max_samples=debug_samples,
                split='eval',
                mean_std_dict=mean_std_dict
            )
        else:
            # === 旧单文件格式 ===
            critic_train_data_263 = cfg.critic_train_data_263 if '263' in retrieval_repr_list else None
            critic_train_data_22x3 = cfg.critic_train_data_22x3 if '22x3' in retrieval_repr_list else None
            critic_test_data_263 = cfg.critic_val_data_263 if '263' in retrieval_repr_list else None
            critic_test_data_22x3 = cfg.critic_val_data_22x3 if '22x3' in retrieval_repr_list else None
            
            has_train_data = (
                (critic_train_data_263 and os.path.exists(critic_train_data_263)) or
                (critic_train_data_22x3 and os.path.exists(critic_train_data_22x3))
            )
            
            if not has_train_data:
                return None, None
            
            critic_train_dataset = CriticPairDataset(
                data_path_263=critic_train_data_263,
                data_path_22x3=critic_train_data_22x3,
                max_motion_length=196,
                max_samples=debug_samples,
                mean_std_dict=mean_std_dict
            )
            
            critic_val_dataset = CriticPairDataset(
                data_path_263=critic_test_data_263,
                data_path_22x3=critic_test_data_22x3,
                max_motion_length=196,
                max_samples=debug_samples,
                mean_std_dict=mean_std_dict
            )
    
    if len(critic_train_dataset) == 0:
        return None, None
    
    seed = getattr(cfg, 'SEED_VALUE', 1234)
    critic_batch_sampler = CriticReprTypeBatchSampler(
        critic_train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        rank=rank,
        world_size=world_size,
        drop_last=True,
        seed=seed
    )
    critic_train_loader = DataLoader(
        critic_train_dataset,
        batch_sampler=critic_batch_sampler,
        num_workers=cfg.TRAIN.num_workers,
        collate_fn=critic_collate_fn,
        pin_memory=True
    )
    
    critic_val_loader = None
    if len(critic_val_dataset) > 0:
        # 验证集不使用 DDP 分割，每个rank加载完整验证集
        # 这样可以确保每个rank都能看到所有表征类型的数据
        critic_test_sampler = CriticReprTypeBatchSampler(
            critic_val_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            rank=0,  # 使用 rank=0，不分割
            world_size=1,  # 使用 world_size=1，不分割
            seed=seed
        )
        critic_val_loader = DataLoader(
            critic_val_dataset,
            batch_sampler=critic_test_sampler,
            num_workers=cfg.TRAIN.num_workers,
            collate_fn=critic_collate_fn,
            pin_memory=True
        )
    
    return critic_train_loader, critic_val_loader


def create_ai_detection_dataloaders(cfg, retrieval_repr_list, rank, world_size):
    """创建 AI Detection 训练、验证和测试数据加载器"""
    data_root = cfg.DATASET.HUMANML3D.ROOT
    
    supported_repr_types = ['263', '22x3']
    available_repr_types = [r for r in retrieval_repr_list if r in supported_repr_types]
    
    if not available_repr_types:
        return None, None, None
    
    if not os.path.exists(cfg.ai_packed_train):
        return None, None, None
    
    debug_samples = getattr(cfg, 'debug_samples', None) if getattr(cfg, 'debug', False) else None
    
    ai_train_dataset = AIDetectionDataset(
        real_data_dir=data_root,
        ai_generated_dirs=None,
        repr_types=available_repr_types,
        max_motion_length=196,
        split='train',
        ai_packed_file=cfg.ai_packed_train,
        max_samples=debug_samples
    )
    
    ai_val_dataset = None
    if os.path.exists(cfg.ai_packed_test):
        ai_val_dataset = AIDetectionDataset(
            real_data_dir=data_root,
            ai_generated_dirs=None,
            repr_types=available_repr_types,
            max_motion_length=196,
            split='test',
            ai_packed_file=cfg.ai_packed_test,
            max_samples=debug_samples
        )
    
    ai_test_dataset = ai_val_dataset
    
    if len(ai_train_dataset) == 0:
        return None, None, None
    
    seed = getattr(cfg, 'SEED_VALUE', 1234)
    ai_batch_sampler = AIDetectionReprTypeBatchSampler(
        ai_train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        rank=rank,
        world_size=world_size,
        drop_last=True,
        seed=seed
    )
    ai_train_loader = DataLoader(
        ai_train_dataset,
        batch_sampler=ai_batch_sampler,
        num_workers=cfg.TRAIN.num_workers,
        collate_fn=ai_detection_collate_fn,
        pin_memory=True
    )
    
    ai_val_loader = None
    if ai_val_dataset is not None and len(ai_val_dataset) > 0:
        ai_val_sampler = AIDetectionReprTypeBatchSampler(
            ai_val_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            rank=rank,
            world_size=world_size,
            seed=seed
        )
        ai_val_loader = DataLoader(
            ai_val_dataset,
            batch_sampler=ai_val_sampler,
            num_workers=cfg.TRAIN.num_workers,
            collate_fn=ai_detection_collate_fn,
            pin_memory=True
        )
    
    ai_test_loader = ai_val_loader
    
    return ai_train_loader, ai_val_loader, ai_test_loader


def create_paired_dataloaders(cfg, retrieval_repr_list, rank, world_size):
    """创建配对表征数据加载器（用于跨表征对齐）
    
    加载同一 motion 在不同表征下的配对数据，用于显式对齐训练。
    需要同时有 263 和 22x3 表征类型。
    
    Args:
        cfg: 配置对象
        retrieval_repr_list: 表征类型列表
        rank: 进程 rank
        world_size: 进程总数
        
    Returns:
        paired_train_loader: 配对训练数据加载器（可能为 None）
    """
    # 检查是否同时有 263 和 22x3
    has_263 = '263' in retrieval_repr_list
    has_22x3 = '22x3' in retrieval_repr_list
    
    if not (has_263 and has_22x3):
        if rank == 0:
            print("[Paired Data] Skipped - requires both 263 and 22x3 repr types")
        return None
    
    debug_samples = getattr(cfg, 'debug_samples', None) if getattr(cfg, 'debug', False) else None
    
    # 优先使用 packed 数据
    use_packed = getattr(cfg, 'use_retrieval_packed', False)
    packed_train_path = getattr(cfg, 'retrieval_packed_train', None)
    
    if use_packed and packed_train_path and os.path.exists(packed_train_path):
        if rank == 0:
            print(f"[Paired Data] Loading from packed file: {packed_train_path}")
        paired_dataset = PackedPairedReprDataset(packed_train_path)
        paired_dataset = _limit_dataset(paired_dataset, debug_samples)
    else:
        # 使用原始数据
        data_root = cfg.DATASET.HUMANML3D.ROOT
        motion_dir_263 = os.path.join(data_root, 'new_joint_vecs')
        motion_dir_22x3 = os.path.join(data_root, 'new_joints')
        text_dir = os.path.join(data_root, 'texts')
        train_split = os.path.join(data_root, 'train.txt')
        
        mean_263, std_263, mean_22x3, std_22x3, _, _ = load_normalization_stats(data_root)
        
        if rank == 0:
            print(f"[Paired Data] Loading from raw data: {data_root}")
        
        paired_dataset = PairedReprDataset(
            mean_263, std_263, mean_22x3, std_22x3,
            train_split, motion_dir_263, motion_dir_22x3, text_dir
        )
        paired_dataset = _limit_dataset(paired_dataset, debug_samples)
    
    if len(paired_dataset) == 0:
        if rank == 0:
            print("[Paired Data] No paired samples found")
        return None
    
    paired_train_loader = DataLoader(
        paired_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.num_workers,
        collate_fn=paired_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    if rank == 0:
        print(f"[Paired Data] Created paired loader: {len(paired_dataset)} samples, "
              f"{len(paired_train_loader)} batches")
    
    return paired_train_loader
