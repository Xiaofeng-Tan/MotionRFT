"""
Critic 训练器

包含 Critic 二阶段训练逻辑
"""

import os
import torch
from tqdm import tqdm
from collections import OrderedDict
import torch.distributed as dist
from diffusers.optimization import get_scheduler

from motionreward.models import pairwise_loss
from motionreward.evaluation import eval_critic


def train_critic_phase(
    cfg, model, critic_train_loader, critic_val_loader,
    device, rank, world_size, is_distributed, writer,
    best_retrieval_ckpt_path, retrieval_repr_list, retrieval_test_loaders=None, fptr=None
):
    """二阶段 Critic 训练
    
    冻结 Retrieval 编码器，只训练 Critic head
    
    Args:
        cfg: 配置
        model: Retrieval 模型
        critic_train_loader: Critic 训练数据加载器
        critic_val_loader: Critic 验证数据加载器
        device: 设备
        rank: 进程 rank
        world_size: 进程数
        is_distributed: 是否分布式训练
        writer: 日志 writer
        best_retrieval_ckpt_path: 最佳 Retrieval checkpoint 路径
        retrieval_repr_list: 表征类型列表
        retrieval_test_loaders: Retrieval 检索评估的 test_loaders
        fptr: 日志文件指针
    
    Returns:
        best_critic_ckpt_path: 最佳 Critic checkpoint 路径
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    from motionreward.evaluation import eval_tmr
    
    actual_model = model.module if hasattr(model, 'module') else model
    
    # 记录使用的 Retrieval 权重
    if rank == 0:
        print("\n" + "="*60)
        print("[Phase 2] Starting Critic Training")
        print(f"[Phase 2] Loading best Retrieval checkpoint from: {best_retrieval_ckpt_path}")
        print("="*60 + "\n")
        if fptr:
            print(f"[Phase 2] Best Retrieval checkpoint: {best_retrieval_ckpt_path}", file=fptr)
            fptr.flush()
    
    # 加载最佳 Retrieval 权重
    if best_retrieval_ckpt_path and os.path.exists(best_retrieval_ckpt_path):
        checkpoint = torch.load(best_retrieval_ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        missing_keys, unexpected_keys = actual_model.load_state_dict(state_dict, strict=False)
        
        if rank == 0:
            print(f"[Phase 2] ✓ Loaded Retrieval weights from: {best_retrieval_ckpt_path}")
            if missing_keys:
                non_clip_missing = [k for k in missing_keys if 'clip' not in k]
                if non_clip_missing:
                    print(f"[Phase 2] Missing keys (non-clip): {non_clip_missing[:5]}...")
    else:
        if rank == 0:
            print(f"[Phase 2] ✗ Warning: Retrieval checkpoint not found at {best_retrieval_ckpt_path}")
    
    # 冻结 Retrieval 所有参数
    frozen_params = []
    for name, param in actual_model.named_parameters():
        if 'critic_head' not in name:
            param.requires_grad = False
            frozen_params.append(name)
    
    if rank == 0:
        print(f"[Phase 2] Frozen {len(frozen_params)} parameter groups")
    
    # 初始化 Critic head
    actual_model.init_critic_head(hidden_dim=cfg.MODEL.latent_dim)
    actual_model.critic_head = actual_model.critic_head.to(device)
    
    if rank == 0:
        trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
        critic_params = sum(p.numel() for p in actual_model.critic_head.parameters())
        print(f"[Phase 2] Trainable params: {trainable_params:,}")
        print(f"[Phase 2] Critic head params: {critic_params:,}")
    
    # 如果是 DDP，需要重新包装模型
    if is_distributed:
        model = actual_model
        model = DDP(model, device_ids=[int(os.environ.get('LOCAL_RANK', 0))],
                   output_device=int(os.environ.get('LOCAL_RANK', 0)),
                   find_unused_parameters=True)
        actual_model = model.module
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        actual_model.critic_head.parameters(),
        lr=cfg.critic_lr,
        weight_decay=1e-4
    )
    
    # LR scheduler
    lr_scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=cfg.critic_epochs * len(critic_train_loader)
    )
    
    # 最佳模型追踪
    best_critic_acc = 0.0
    best_critic_epoch = -1
    best_critic_ckpt_path = None
    
    checkpoint_dir = cfg.Retrieval_CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    repr_str = '_'.join(retrieval_repr_list)
    
    for epoch in range(cfg.critic_epochs):
        if hasattr(critic_train_loader, 'batch_sampler') and hasattr(critic_train_loader.batch_sampler, 'set_epoch'):
            critic_train_loader.batch_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        
        for batch_data in tqdm(critic_train_loader, desc=f'Critic Epoch {epoch}', disable=rank != 0):
            batch_data = {k: v.float().to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
            
            critic = actual_model.forward_critic(batch_data)
            if critic is None:
                continue
            
            loss, _, acc = pairwise_loss(critic)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            n_batches += 1
        
        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            avg_acc = epoch_acc / n_batches
        else:
            avg_loss = 0.0
            avg_acc = 0.0
        
        # 验证
        model.eval()
        val_result = eval_critic(critic_val_loader, model, device=device, rank=rank)
        
        # Retrieval 检索评估
        if retrieval_test_loaders is not None:
            if rank == 0:
                print(f"\n[Critic Epoch {epoch}] === Retrieval Retrieval Evaluation ===")
            for repr_type, test_loader in retrieval_test_loaders.items():
                eval_tmr(test_loader, model, epoch, repr_type=repr_type,
                        mode='M1T0', writer=writer, device=device, rank=rank)
            if rank == 0:
                print(f"[Critic Epoch {epoch}] === Retrieval Retrieval Evaluation Done ===\n")
        
        if rank == 0:
            print(f"[Critic Epoch {epoch}] Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}")
            print(f"[Critic Epoch {epoch}] Val Loss: {val_result['loss']:.4f}, Val Acc: {val_result['acc']:.4f}")
            if 'acc_263' in val_result:
                print(f"  - 263 Acc: {val_result['acc_263']:.4f}")
            if 'acc_22x3' in val_result:
                print(f"  - 22x3 Acc: {val_result['acc_22x3']:.4f}")
            
            if fptr:
                print(f"[Critic Epoch {epoch}] Train: {avg_loss:.4f}/{avg_acc:.4f}, Val: {val_result['loss']:.4f}/{val_result['acc']:.4f}", file=fptr)
                fptr.flush()
            
            if writer is not None:
                writer.log({
                    "Critic/train_loss": avg_loss,
                    "Critic/train_acc": avg_acc,
                    "Critic/val_loss": val_result['loss'],
                    "Critic/val_acc": val_result['acc']
                }, step=epoch)
            
            # 保存最佳 Critic
            if val_result['acc'] > best_critic_acc:
                best_critic_acc = val_result['acc']
                best_critic_epoch = epoch
                
                model_config = _build_model_config(cfg, retrieval_repr_list)
                critic_state = {
                    'critic_head': actual_model.critic_head.state_dict(),
                    'epoch': epoch,
                    'acc': val_result['acc'],
                    'retrieval_ckpt_path': best_retrieval_ckpt_path,
                    'model_config': model_config
                }
                best_critic_ckpt_path = os.path.join(checkpoint_dir, f'Critic_{repr_str}_best.pth')
                torch.save(critic_state, best_critic_ckpt_path)
                print(f"[Phase 2] Best Critic saved: {best_critic_ckpt_path} (Acc={val_result['acc']:.4f})")
        
        if is_distributed:
            dist.barrier()
    
    if rank == 0:
        print("\n" + "="*60)
        print("[Phase 2] Critic Training Complete!")
        print(f"Best Critic: Epoch {best_critic_epoch}, Acc={best_critic_acc:.4f}")
        print(f"Best Critic checkpoint: {best_critic_ckpt_path}")
        print(f"Retrieval checkpoint used: {best_retrieval_ckpt_path}")
        print("="*60 + "\n")
        if fptr:
            print(f"[Phase 2] Best Critic: Epoch {best_critic_epoch}, Acc={best_critic_acc:.4f}", file=fptr)
            print(f"[Phase 2] Critic checkpoint: {best_critic_ckpt_path}", file=fptr)
            fptr.flush()
    
    return best_critic_ckpt_path


def _build_model_config(cfg, retrieval_repr_list):
    """构建模型配置字典"""
    return {
        'model_size': cfg.model_size,
        'use_unified_dim': cfg.use_unified_dim,
        'latent_dim': cfg.MODEL.latent_dim,
        'unified_dim': cfg.MODEL.unified_dim,
        'encoder_num_layers': cfg.MODEL.encoder_num_layers,
        'encoder_num_heads': cfg.MODEL.encoder_num_heads,
        'encoder_ff_size': cfg.MODEL.encoder_ff_size,
        'text_num_layers': cfg.MODEL.text_num_layers,
        'text_num_heads': cfg.MODEL.text_num_heads,
        'text_ff_size': cfg.MODEL.text_ff_size,
        'repr_types': retrieval_repr_list,
    }
