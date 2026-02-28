"""
AI 检测训练器

包含 AI 检测训练逻辑
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
import torch.distributed as dist
from diffusers.optimization import get_scheduler

from motionreward.evaluation import eval_ai_detection


def train_ai_detection_phase(
    cfg, model, ai_train_loader, ai_val_loader,
    device, rank, world_size, is_distributed, writer,
    best_retrieval_ckpt_path, retrieval_repr_list, retrieval_test_loaders=None,
    ai_test_loader=None, fptr=None
):
    """AI 检测训练阶段
    
    冻结 Retrieval 编码器，只训练 AI 检测分类头
    
    Args:
        cfg: 配置
        model: Retrieval 模型
        ai_train_loader: AI 检测训练数据加载器
        ai_val_loader: AI 检测验证数据加载器
        device: 设备
        rank: 进程 rank
        world_size: 进程数
        is_distributed: 是否分布式训练
        writer: 日志 writer
        best_retrieval_ckpt_path: 最佳 Retrieval checkpoint 路径
        retrieval_repr_list: 表征类型列表
        retrieval_test_loaders: Retrieval 检索评估的 test_loaders
        ai_test_loader: AI 检测测试数据加载器
        fptr: 日志文件指针
    
    Returns:
        best_ckpt_path: 最佳 checkpoint 路径
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    actual_model = model.module if hasattr(model, 'module') else model
    
    # 记录使用的 Retrieval 权重
    if rank == 0:
        print("\n" + "="*60)
        print("[AI Detection] Starting AI Detection Training")
        print(f"[AI Detection] Loading Retrieval checkpoint from: {best_retrieval_ckpt_path}")
        print("="*60 + "\n")
        if fptr:
            print(f"[AI Detection] Retrieval checkpoint: {best_retrieval_ckpt_path}", file=fptr)
            fptr.flush()
    
    # 加载 Retrieval 权重
    if best_retrieval_ckpt_path and os.path.exists(best_retrieval_ckpt_path):
        checkpoint = torch.load(best_retrieval_ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        missing_keys, unexpected_keys = actual_model.load_state_dict(state_dict, strict=False)
        
        if rank == 0:
            print(f"[AI Detection] ✓ Loaded Retrieval weights from: {best_retrieval_ckpt_path}")
            if missing_keys:
                non_clip_missing = [k for k in missing_keys if 'clip' not in k]
                if non_clip_missing:
                    print(f"[AI Detection] Missing keys (non-clip): {non_clip_missing[:5]}...")
    else:
        if rank == 0:
            print(f"[AI Detection] ✗ Warning: Retrieval checkpoint not found at {best_retrieval_ckpt_path}")
    
    # 冻结 Retrieval 所有参数
    frozen_params = []
    for name, param in actual_model.named_parameters():
        if 'ai_detection_head' not in name:
            param.requires_grad = False
            frozen_params.append(name)
    
    if rank == 0:
        print(f"[AI Detection] Frozen {len(frozen_params)} parameter groups")
    
    # 初始化 AI 检测头
    actual_model.init_ai_detection_head(hidden_dim=cfg.MODEL.latent_dim)
    actual_model.ai_detection_head = actual_model.ai_detection_head.to(device)
    
    if rank == 0:
        trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
        ai_head_params = sum(p.numel() for p in actual_model.ai_detection_head.parameters())
        print(f"[AI Detection] Trainable params: {trainable_params:,}")
        print(f"[AI Detection] AI detection head params: {ai_head_params:,}")
    
    # 如果是 DDP，需要重新包装模型
    if is_distributed:
        model = actual_model
        model = DDP(model, device_ids=[int(os.environ.get('LOCAL_RANK', 0))],
                   output_device=int(os.environ.get('LOCAL_RANK', 0)),
                   find_unused_parameters=True)
        actual_model = model.module
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        actual_model.ai_detection_head.parameters(),
        lr=cfg.ai_detection_lr,
        weight_decay=1e-4
    )
    
    # LR scheduler
    lr_scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=cfg.ai_detection_epochs * len(ai_train_loader)
    )
    
    # 最佳模型追踪
    best_acc = 0.0
    best_f1 = 0.0
    best_epoch = -1
    best_ckpt_path = None
    
    checkpoint_dir = cfg.Retrieval_CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    repr_str = '_'.join(retrieval_repr_list)
    
    for epoch in range(cfg.ai_detection_epochs):
        if hasattr(ai_train_loader, 'batch_sampler') and hasattr(ai_train_loader.batch_sampler, 'set_epoch'):
            ai_train_loader.batch_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0
        
        for batch_data in tqdm(ai_train_loader, desc=f'AI Detection Epoch {epoch}', disable=rank != 0):
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
            
            logits, labels = actual_model.forward_ai_detection(batch_data)
            if logits is None:
                continue
            
            loss = F.cross_entropy(logits, labels)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += len(labels)
            n_batches += 1
        
        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            avg_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        else:
            avg_loss = 0.0
            avg_acc = 0.0
        
        # 验证
        model.eval()
        val_result = eval_ai_detection(ai_val_loader, model, device=device, rank=rank)
        
        if rank == 0:
            print(f"[AI Detection Epoch {epoch}] Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}")
            print(f"[AI Detection Epoch {epoch}] Val Loss: {val_result['loss']:.4f}, Val Acc: {val_result['acc']:.4f}")
            print(f"  - Precision: {val_result['precision']:.4f}, Recall: {val_result['recall']:.4f}, F1: {val_result['f1']:.4f}")
            if 'acc_263' in val_result:
                print(f"  - 263 Acc: {val_result['acc_263']:.4f}")
            if 'acc_22x3' in val_result:
                print(f"  - 22x3 Acc: {val_result['acc_22x3']:.4f}")
            
            if fptr:
                print(f"[AI Detection Epoch {epoch}] Train: {avg_loss:.4f}/{avg_acc:.4f}, Val: {val_result['loss']:.4f}/{val_result['acc']:.4f}, F1: {val_result['f1']:.4f}", file=fptr)
                fptr.flush()
            
            if writer is not None:
                writer.log({
                    "AIDetection/train_loss": avg_loss,
                    "AIDetection/train_acc": avg_acc,
                    "AIDetection/val_loss": val_result['loss'],
                    "AIDetection/val_acc": val_result['acc'],
                    "AIDetection/val_precision": val_result['precision'],
                    "AIDetection/val_recall": val_result['recall'],
                    "AIDetection/val_f1": val_result['f1']
                }, step=epoch)
            
            # 保存最佳模型（基于 F1 分数）
            if val_result['f1'] > best_f1:
                best_f1 = val_result['f1']
                best_acc = val_result['acc']
                best_epoch = epoch
                
                model_config = _build_model_config(cfg, retrieval_repr_list)
                ai_state = {
                    'ai_detection_head': actual_model.ai_detection_head.state_dict(),
                    'epoch': epoch,
                    'acc': val_result['acc'],
                    'f1': val_result['f1'],
                    'retrieval_ckpt_path': best_retrieval_ckpt_path,
                    'model_config': model_config
                }
                best_ckpt_path = os.path.join(checkpoint_dir, f'AIDetection_{repr_str}_best.pth')
                torch.save(ai_state, best_ckpt_path)
                print(f"[AI Detection] Best model saved: {best_ckpt_path} (F1={val_result['f1']:.4f}, Acc={val_result['acc']:.4f})")
        
        if is_distributed:
            dist.barrier()
    
    # 测试集评估（如果有）
    if ai_test_loader is not None:
        model.eval()
        test_result = eval_ai_detection(ai_test_loader, model, device=device, rank=rank)
        
        if rank == 0:
            print("\n" + "-"*60)
            print("[AI Detection] Test Set Evaluation:")
            print(f"  Loss: {test_result['loss']:.4f}, Acc: {test_result['acc']:.4f}")
            print(f"  Precision: {test_result['precision']:.4f}, Recall: {test_result['recall']:.4f}, F1: {test_result['f1']:.4f}")
            if 'acc_263' in test_result:
                print(f"  263 Acc: {test_result['acc_263']:.4f}")
            if 'acc_22x3' in test_result:
                print(f"  22x3 Acc: {test_result['acc_22x3']:.4f}")
            print("-"*60)
            
            if fptr:
                print(f"[AI Detection] Test: Acc={test_result['acc']:.4f}, F1={test_result['f1']:.4f}", file=fptr)
                fptr.flush()
            
            if writer is not None:
                writer.log({
                    "AIDetection/test_loss": test_result['loss'],
                    "AIDetection/test_acc": test_result['acc'],
                    "AIDetection/test_precision": test_result['precision'],
                    "AIDetection/test_recall": test_result['recall'],
                    "AIDetection/test_f1": test_result['f1']
                })
    
    if rank == 0:
        print("\n" + "="*60)
        print("[AI Detection] Training Complete!")
        print(f"Best model: Epoch {best_epoch}, F1={best_f1:.4f}, Acc={best_acc:.4f}")
        print(f"Best checkpoint: {best_ckpt_path}")
        print(f"Retrieval checkpoint used: {best_retrieval_ckpt_path}")
        print("="*60 + "\n")
        if fptr:
            print(f"[AI Detection] Best: Epoch {best_epoch}, F1={best_f1:.4f}, Acc={best_acc:.4f}", file=fptr)
            print(f"[AI Detection] Checkpoint: {best_ckpt_path}", file=fptr)
            fptr.flush()
    
    return best_ckpt_path


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
