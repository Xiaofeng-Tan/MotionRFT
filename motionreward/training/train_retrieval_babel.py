"""
MotionReward - 135维 FlowMDM/BABEL 表征 Retrieval 训练脚本

训练支持 135 维 rot6d 表征 (FlowMDM/BABEL 格式) 的 Retrieval 模型

135维表征结构:
    - 1 维: root_y (根节点高度)
    - 2 维: vel_trajectory (轨迹速度 x, y)
    - 132 维: 22 joints × 6 (rot6d 旋转表示)

Usage (单卡):
    python -m motionreward.training.train_retrieval_135 --babel_datapath /path/to/babel-smplh-30fps-male
    
Usage (Multi-GPU DDP):
    torchrun --nproc_per_node=2 --master_port=29502 -m motionreward.training.train_retrieval_135 --babel_datapath /path/to/babel-smplh-30fps-male
"""

import os
import sys
import random
import argparse
from datetime import datetime
from collections import OrderedDict
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

try:
    import swanlab
except ImportError:
    swanlab = None
    print("Warning: swanlab not installed, logging disabled")

from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler

# 导入模块
from motionreward.datasets import (
    BABELText2MotionDataset135,
    PackedBABELDataset,
    FlowMDMOfficialDataset,
    FlowMDMRetrievalDataset,
    babel_collate_fn,
    retrieval_collate_fn,
)
from motionreward.models import MultiReprRetrieval
from motionreward.evaluation import eval_tmr
from motionreward.utils import setup_ddp, cleanup_ddp, is_main_process

# 评估专用 collate_fn（batch_size <= 32）
def eval_collate_fn(batch):
    """评估用 collate_fn，确保 batch_size <= 32"""
    return babel_collate_fn(batch)

# 项目路径
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 默认 T5 模型路径（尝试多个位置）
def get_default_t5_path():
    """获取默认的 T5 模型路径"""
    possible_paths = [
        os.path.join(PROJ_DIR, 'deps/sentence-t5-large'),
        'deps/sentence-t5-large',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return possible_paths[0]  # 返回第一个作为默认值


def parse_args():
    parser = argparse.ArgumentParser(description='Train Retrieval with 135-dim FlowMDM/BABEL representation')
    
    # 默认数据路径（本脚本专用，不影响其他文件）
    default_retrieval_datapath = os.path.join(PROJ_DIR, 'datasets', 'retrieval_dataset')
    
    # 数据路径
    parser.add_argument('--babel_datapath', type=str, default=None,
                        help='Path to BABEL data (babel-smplh-30fps-male directory)')
    parser.add_argument('--flowmdm_datapath', type=str, default=default_retrieval_datapath,
                        help='Path to FlowMDM retrieval dataset (contains {split}_motions.npz and {split}_texts.json)')
    parser.add_argument('--mean_path', type=str, default=None,
                        help='Path to 135-dim mean file (.pt)')
    parser.add_argument('--std_path', type=str, default=None,
                        help='Path to 135-dim std file (.pt)')
    parser.add_argument('--packed_data', type=str, default=None,
                        help='Path to packed BABEL data (.pth)')
    
    # 模型配置
    parser.add_argument('--t5_path', type=str, 
                        default=None,
                        help='Path to T5 model (local path). Auto-detected if not specified.')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent space dimension')
    parser.add_argument('--unified_dim', type=int, default=512,
                        help='Unified projection dimension')
    parser.add_argument('--encoder_num_layers', type=int, default=9,
                        help='Number of encoder layers')
    parser.add_argument('--encoder_num_heads', type=int, default=4,
                        help='Number of encoder attention heads')
    parser.add_argument('--encoder_ff_size', type=int, default=1024,
                        help='Encoder FFN dimension')
    parser.add_argument('--proj_hidden_dim', type=int, default=512,
                        help='Projection layer hidden dimension')
    parser.add_argument('--proj_num_layers', type=int, default=3,
                        help='Number of projection layers')
    
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_warmup_steps', type=int, default=1000,
                        help='Learning rate warmup steps (aligned with train_retrieval_lora_new.py)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Checkpoint save frequency (epochs)')
    parser.add_argument('--eval_freq', type=int, default=5,
                        help='Evaluation frequency (epochs)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader num_workers')
    
    # Retrieval 配置（与 train_retrieval_lora_new.py 对齐）
    parser.add_argument('--step_aware', type=str, default='M1T0',
                        choices=['M0T0', 'M1T0', 'M0T1', 'M1T1'],
                        help='Step-aware mode (M1T0 aligned with train_retrieval_lora_new.py)')
    parser.add_argument('--maxT', type=int, default=500,
                        help='Max timestep for noise (aligned with train_retrieval_lora_new.py)')
    parser.add_argument('--noise_thr', type=float, default=0.9,
                        help='Noise threshold (probability of NOT adding noise)')
    parser.add_argument('--cl_temp', type=float, default=0.1,
                        help='Contrastive learning temperature')
    parser.add_argument('--cl_thr', type=float, default=0.9,
                        help='Text similarity threshold')
    
    # 其他
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/retrieval_135',
                        help='Checkpoint save directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--use_swanlab', action='store_true',
                        help='Use SwanLab for logging')
    parser.add_argument('--use_unified_dim', action='store_true', default=True,
                        help='Use unified_dim intermediate layer (aligned with train_retrieval_lora_new.py)')
    parser.add_argument('--no_unified_dim', action='store_true', default=False,
                        help='Disable unified_dim intermediate layer')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 自动检测 T5 路径
    if args.t5_path is None:
        args.t5_path = get_default_t5_path()
    
    # 设置种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup DDP
    rank, world_size, local_rank, is_distributed = setup_ddp()
    
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
        if rank == 0:
            print(f"Using DDP with {world_size} GPUs")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using single GPU: {device}")
    
    # ==================== 创建模型 ====================
    if rank == 0:
        print("\n" + "=" * 60)
        print("[Step 1] Creating model")
        print(f"  T5 path: {args.t5_path}")
        print("=" * 60)
    
    model = MultiReprRetrieval(
        t5_path=args.t5_path,
        temp=args.cl_temp,
        thr=args.cl_thr,
        latent_dim=args.latent_dim,
        unified_dim=args.unified_dim,
        encoder_num_layers=args.encoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
        encoder_ff_size=args.encoder_ff_size,
        text_num_layers=args.encoder_num_layers,
        text_num_heads=args.encoder_num_heads,
        text_ff_size=args.encoder_ff_size,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_num_layers=args.proj_num_layers,
        proj_dropout=0.1,
        use_unified_dim=args.use_unified_dim and not args.no_unified_dim
    )
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    model = model.to(device)
    
    # 加载 checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
        start_epoch = ckpt.get('epoch', 0) + 1
    
    # ==================== 加载数据 ====================
    if rank == 0:
        print("\n" + "=" * 60)
        print("[Step 2] Loading data")
        print("=" * 60)
    
    # 优先使用 FlowMDM 数据格式（自动检测数据格式）
    if args.flowmdm_datapath and os.path.exists(args.flowmdm_datapath):
        if rank == 0:
            print(f"Loading FlowMDM data from {args.flowmdm_datapath}")
        
        train_dataset = None
        val_dataset = None
        datapath = args.flowmdm_datapath
        
        # 格式 1：官方 pkl 缓存（train/val 子目录）
        train_dir = os.path.join(datapath, 'train')
        val_dir = os.path.join(datapath, 'val')
        
        if os.path.isdir(train_dir) or os.path.isdir(val_dir):
            # 检测 pkl 文件
            pkl_pattern = '*_motion_data_*.pkl'
            train_pkl_files = list(Path(train_dir).glob(pkl_pattern)) if os.path.isdir(train_dir) else []
            val_pkl_files = list(Path(val_dir).glob(pkl_pattern)) if os.path.isdir(val_dir) else []
            
            if train_pkl_files or val_pkl_files:
                if rank == 0:
                    print("  Detected FlowMDM official pkl format")
                
                # 从文件名提取 suffix
                suffix = 'single_eval_30_200_False'  # 默认
                if train_pkl_files:
                    # 从 train_motion_data_xxx.pkl 提取 xxx
                    fname = train_pkl_files[0].name
                    suffix = fname.replace('train_motion_data_', '').replace('.pkl', '')
                elif val_pkl_files:
                    fname = val_pkl_files[0].name
                    suffix = fname.replace('val_motion_data_', '').replace('.pkl', '')
                
                if rank == 0:
                    print(f"  Using suffix: {suffix}")
                
                if train_pkl_files:
                    train_dataset = FlowMDMOfficialDataset(
                        datapath, split='train', suffix=suffix, random_crop=True
                    )
                if val_pkl_files:
                    val_dataset = FlowMDMOfficialDataset(
                        datapath, split='val', suffix=suffix, random_crop=False
                    )
        
        # 格式 2：npz + json 格式
        if train_dataset is None and val_dataset is None:
            train_motions = os.path.join(datapath, 'train_motions.npz')
            val_motions = os.path.join(datapath, 'val_motions.npz')
            
            if os.path.exists(train_motions) or os.path.exists(val_motions):
                if rank == 0:
                    print("  Detected FlowMDM npz+json format")
                
                if os.path.exists(train_motions):
                    train_dataset = FlowMDMRetrievalDataset(datapath, split='train')
                if os.path.exists(val_motions):
                    val_dataset = FlowMDMRetrievalDataset(datapath, split='val')
        
        # 如果只有 val，用 val 作为训练集
        if train_dataset is None and val_dataset is not None:
            if rank == 0:
                print("  No train split found, using val split for both training and evaluation")
            train_dataset = val_dataset
        
        if train_dataset is None:
            raise ValueError(f"No valid data found in {args.flowmdm_datapath}. "
                           f"Expected either pkl files in train/val subdirs or npz+json files.")
            
    # 使用打包数据或原始数据
    elif args.packed_data and os.path.exists(args.packed_data):
        if rank == 0:
            print(f"Loading packed data from {args.packed_data}")
        train_dataset = PackedBABELDataset(args.packed_data, repr_type='135', split='train')
        # 使用测试集作为验证集（与 train_retrieval_lora_new.py 对齐）
        val_dataset = None
        for split_name in ['test', 'val']:
            try:
                val_dataset = PackedBABELDataset(args.packed_data, repr_type='135', split=split_name)
                if rank == 0:
                    print(f"Using '{split_name}' split as validation set")
                break
            except (ValueError, KeyError):
                continue
        if val_dataset is None and rank == 0:
            print("No test/val split found in packed data, skipping TMR evaluation")
    elif args.babel_datapath and os.path.exists(args.babel_datapath):
        if rank == 0:
            print(f"Loading BABEL data from {args.babel_datapath}")
        train_dataset = BABELText2MotionDataset135(
            datapath=args.babel_datapath,
            split='train',
            mean_path=args.mean_path,
            std_path=args.std_path,
            precompute=True
        )
        val_dataset = BABELText2MotionDataset135(
            datapath=args.babel_datapath,
            split='val',
            mean_path=args.mean_path,
            std_path=args.std_path,
            precompute=True
        )
    else:
        raise ValueError("Must provide --flowmdm_datapath, --packed_data, or --babel_datapath")
    
    # DataLoader
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=babel_collate_fn,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=babel_collate_fn,
            pin_memory=True
        )
    
    # 创建 test_loader 用于 TMR 检索评估（batch_size 必须 <= 32）
    test_loader = None
    if val_dataset is not None:
        test_loader = DataLoader(
            val_dataset,
            batch_size=32,  # TMR 评估要求 batch_size <= 32
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=eval_collate_fn,
            pin_memory=True
        )
    
    # test_loaders 字典（与 train_retrieval_lora_new.py 对齐）
    test_loaders = {'135': test_loader} if test_loader is not None else {}
    
    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        if val_dataset:
            print(f"Test samples (for TMR eval): {len(val_dataset)}")
    
    # SwanLab
    writer = None
    if is_main_process(rank) and args.use_swanlab and swanlab is not None:
        writer = swanlab.init(
            project="MotionReward-Retrieval-135",
            experiment_name=f"Retrieval_BABEL_135_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # 包装 DDP
    model.train()
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if rank == 0:
            print(f"Model wrapped with DDP on {world_size} GPUs")
    
    # Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        variance_type="fixed_small",
        clip_sample=False
    )
    
    # Optimizer（与 train_retrieval_lora_new.py 对齐）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,  # 对齐 retrieval_multi_repr.yaml
        eps=1e-8
    )
    
    max_train_steps = args.epochs * len(train_loader)
    warmup_steps = args.lr_warmup_steps
    if is_distributed:
        warmup_steps = max(1, warmup_steps // world_size)
    
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps
    )
    
    # 训练参数
    step_aware = args.step_aware
    maxT = args.maxT
    thr = args.noise_thr
    
    # 概率分布（与 train_retrieval_lora_new.py 对齐）
    probs = torch.zeros(maxT, device=device)
    probs[:maxT//10] = 0.8 / maxT
    probs[maxT//10:] = 0.2 / (maxT - 101)
    
    if rank == 0:
        print(f'\n================StepAware:{step_aware}, maxT: {maxT}, NoiseThr: {thr} ================')
    
    actual_model = model.module if hasattr(model, 'module') else model
    
    # 最佳模型追踪（使用 R1 指标）
    best_r1 = 0.0
    best_epoch = -1
    best_ckpt_path = None
    
    # ==================== 训练循环 ====================
    if rank == 0:
        print("\n" + "=" * 60)
        print("[Step 3] Starting training")
        print("=" * 60 + "\n")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_file = os.path.join(args.checkpoint_dir, 'train.log')
    fptr = open(log_file, 'a') if rank == 0 else None
    
    global_step = 0
    
    for epoch in range(start_epoch, args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', disable=rank != 0)
        for batch in pbar:
            feats_ref = batch['motion'].float().to(device)
            text = batch['text']
            m_len = batch['length']
            repr_type = batch['repr_type']
            
            if isinstance(m_len, torch.Tensor):
                m_len = m_len.tolist() if m_len.numel() > 1 else [m_len.item()] * len(text)
            elif not isinstance(m_len, list):
                m_len = list(m_len)
            
            timestep = torch.multinomial(probs, num_samples=feats_ref.shape[0], replacement=True).long()
            
            if random.random() > thr:
                with torch.no_grad():
                    noise = torch.randn_like(feats_ref)
                    noised_z = scheduler.add_noise(original_samples=feats_ref.clone(), noise=noise, timesteps=timestep)
                    feats_ref = noised_z
            
            # Retrieval Loss
            loss = actual_model(
                text=text, 
                motion_feature=feats_ref, 
                m_len=m_len, 
                repr_type=repr_type, 
                timestep=timestep, 
                mode=step_aware
            )
            
            loss_item = loss.item()
            epoch_loss += loss_item
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            pbar.set_postfix({'loss': f'{loss_item:.4f}'})
            
            if writer and global_step % 100 == 0:
                writer.log({'train/loss': loss_item, 'train/lr': lr_scheduler.get_last_lr()[0]}, step=global_step)
            
            global_step += 1
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        if is_main_process(rank):
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f'Time = {time_str}, Epoch = {epoch}, Avg loss = {avg_epoch_loss:.4f}'
            print(log_msg)
            if fptr:
                print(log_msg, file=fptr)
                fptr.flush()
            
            if writer:
                writer.log({'train/epoch_loss': avg_epoch_loss}, step=epoch)
        
        # ==================== TMR 检索评估 ====================
        # 每个 epoch 结束后进行评估（与 train_retrieval_lora_new.py 对齐）
        model.eval()
        epoch_eval_results = {}
        
        for repr_type, test_loader in test_loaders.items():
            eval_results = eval_tmr(
                test_loader, model, epoch, 
                repr_type=repr_type, 
                mode=step_aware, 
                writer=writer, 
                device=device, 
                rank=rank
            )
            epoch_eval_results[repr_type] = eval_results
        
        # 计算平均 R1 并保存最佳模型（使用 BS32 R1）
        if is_main_process(rank) and epoch_eval_results:
            bs32_r1_sum = 0.0
            n_repr = 0
            for repr_type, results in epoch_eval_results.items():
                if results.get('bs32') and results['bs32'].get('t2m') and results['bs32'].get('m2t'):
                    t2m_r1 = results['bs32']['t2m'].get('R1', 0)
                    m2t_r1 = results['bs32']['m2t'].get('R1', 0)
                    bs32_r1_sum += (t2m_r1 + m2t_r1) / 2
                    n_repr += 1
            
            if n_repr > 0:
                avg_bs32_r1 = bs32_r1_sum / n_repr
                print(f"Epoch {epoch} | Avg BS32 R1: {avg_bs32_r1:.3f}%")
                
                if fptr:
                    print(f"Epoch {epoch} | Avg BS32 R1: {avg_bs32_r1:.3f}%", file=fptr)
                    fptr.flush()
                
                # 保存最佳模型（基于 BS32 R1 指标）
                if avg_bs32_r1 > best_r1:
                    best_r1 = avg_bs32_r1
                    best_epoch = epoch
                    
                    new_state = OrderedDict()
                    for k, v in actual_model.state_dict().items():
                        if 'clip' not in k:
                            new_state[k] = v
                    
                    best_path = os.path.join(args.checkpoint_dir, 'Retrieval_BABEL_135_best.pth')
                    torch.save({
                        'state_dict': new_state,
                        'epoch': epoch,
                        'r1': avg_bs32_r1,
                        'args': vars(args)
                    }, best_path)
                    best_ckpt_path = best_path
                    print(f"[Best] Model saved: {best_path} (BS32 R1={avg_bs32_r1:.3f}%)")
        
        model.train()
        
        # 定期保存
        if epoch % args.save_freq == 0 and is_main_process(rank):
            new_state = OrderedDict()
            for k, v in actual_model.state_dict().items():
                if 'clip' not in k:
                    new_state[k] = v
            
            ckpt_path = os.path.join(args.checkpoint_dir, f'Retrieval_BABEL_135_E{epoch}.pth')
            torch.save({
                'state_dict': new_state,
                'epoch': epoch,
                'args': vars(args)
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
        
        if is_distributed:
            dist.barrier()
    
    # 训练完成
    if rank == 0:
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best model: Epoch {best_epoch}, BS32 R1={best_r1:.3f}%")
        if best_ckpt_path:
            print(f"Best checkpoint: {best_ckpt_path}")
        print("=" * 60 + "\n")
        if fptr:
            fptr.close()
    
    cleanup_ddp()


if __name__ == "__main__":
    main()
