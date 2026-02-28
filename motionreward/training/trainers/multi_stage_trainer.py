"""
多阶段训练器

包含多阶段训练逻辑：Stage1 检索 -> Stage2 Critic -> Stage3 AI检测
"""

import os
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
import torch.distributed as dist
from diffusers.optimization import get_scheduler

from motionreward.models import pairwise_loss
from motionreward.evaluation import eval_tmr, eval_critic, eval_ai_detection


def train_multi_stage(
    cfg, model, train_loader, test_loaders,
    critic_train_loader, critic_val_loader,
    ai_train_loader, ai_val_loader, ai_test_loader,
    device, rank, world_size, is_distributed, writer,
    retrieval_repr_list, scheduler, probs, text_src, ds_name, fptr=None
):
    """多阶段训练：阶段1-检索 -> 阶段2-Critic -> 阶段3-AI检测
    
    阶段1: 训练完整 Retrieval 模型（检索任务）
    阶段2: 冻结 Retrieval，只训练 Critic head
    阶段3: 冻结 Retrieval，只训练 AI Detection head
    
    Returns:
        best_retrieval_ckpt_path: 最佳 Retrieval checkpoint 路径
        best_critic_ckpt_path: 最佳 Critic checkpoint 路径
        best_ai_ckpt_path: 最佳 AI Detection checkpoint 路径
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    actual_model = model.module if hasattr(model, 'module') else model
    
    stage1_epochs = cfg.stage1_epochs
    stage2_epochs = cfg.stage2_epochs
    stage3_epochs = cfg.stage3_epochs
    
    # 生成时间戳用于 checkpoint 文件名
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if rank == 0:
        print("\n" + "="*60)
        print("[Multi-Stage Training] Configuration:")
        print(f"  Stage 1 (Retrieval): {stage1_epochs} epochs")
        print(f"  Stage 2 (Critic): {stage2_epochs} epochs")
        print(f"  Stage 3 (AI Detection): {stage3_epochs} epochs")
        print(f"  Timestamp: {timestamp}")
        print("="*60 + "\n")
    
    best_retrieval_ckpt_path = None
    best_metrics = {'bs32': {'r1': 0.0, 'epoch': -1}}
    
    # ==================== Stage 1: Retrieval Training ====================
    if rank == 0:
        print("\n" + "="*60)
        print("[Stage 1] Starting Retrieval (Retrieval) Training")
        print(f"[Stage 1] Epochs: {stage1_epochs}")
        print("="*60 + "\n")
    
    # 确保所有参数可训练
    for param in actual_model.parameters():
        param.requires_grad = True
    
    # Stage 1 optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.TRAIN.learning_rate,
        betas=(cfg.TRAIN.adam_beta1, cfg.TRAIN.adam_beta2),
        weight_decay=cfg.TRAIN.adam_weight_decay,
        eps=cfg.TRAIN.adam_epsilon
    )
    
    warmup_steps = cfg.TRAIN.lr_warmup_steps
    max_train_steps = stage1_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        cfg.TRAIN.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps
    )
    
    step_aware = cfg.Retrieval.step_aware
    thr = cfg.Retrieval.NoiseThr
    
    checkpoint_dir = cfg.Retrieval_CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    repr_str = '_'.join(retrieval_repr_list)
    
    for epoch in range(stage1_epochs):
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        
        for i, batch in tqdm(enumerate(train_loader), desc=f'[Stage1] Epoch {epoch}', total=len(train_loader), disable=rank != 0):
            feats_ref, text, m_len = batch['motion'], batch['text'], batch['length']
            repr_type = batch['repr_type']
            
            feats_ref = feats_ref.float().to(device)
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
            
            loss = actual_model(text=text, motion_feature=feats_ref, m_len=m_len,
                               repr_type=repr_type, timestep=timestep, mode=step_aware)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
            del loss, feats_ref, timestep
            torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / len(train_loader)
        if rank == 0:
            print(f'[Stage1] Epoch {epoch}, Avg Loss: {avg_loss:.4f}')
            if writer is not None:
                writer.log({"Stage1/loss": avg_loss, "Stage1/lr": lr_scheduler.get_last_lr()[0]}, step=epoch)
        
        # Evaluation
        model.eval()
        epoch_eval_results = {}
        for repr_type, test_loader in test_loaders.items():
            eval_results = eval_tmr(test_loader, model, epoch, repr_type=repr_type,
                                   mode=step_aware, writer=writer, device=device, rank=rank)
            epoch_eval_results[repr_type] = eval_results
        
        if rank == 0:
            bs32_r1_sum = 0.0
            n_repr = 0
            for repr_type, results in epoch_eval_results.items():
                if results['bs32']['t2m'] and results['bs32']['m2t']:
                    bs32_r1_sum += (results['bs32']['t2m']['R1'] + results['bs32']['m2t']['R1']) / 2
                    n_repr += 1
            
            if n_repr > 0:
                avg_bs32_r1 = bs32_r1_sum / n_repr
                print(f"[Stage1] Epoch {epoch} | Avg BS32 R1: {avg_bs32_r1:.3f}%")
                
                if avg_bs32_r1 > best_metrics['bs32']['r1']:
                    best_metrics['bs32']['r1'] = avg_bs32_r1
                    best_metrics['bs32']['epoch'] = epoch
                    
                    # Save best model
                    new_state = OrderedDict()
                    for k, v in actual_model.state_dict().items():
                        if 'clip' not in k:
                            new_state[k] = v
                    
                    model_config = _build_model_config(cfg, retrieval_repr_list)
                    best_retrieval_ckpt_path = os.path.join(checkpoint_dir, f'{text_src}_MultiStage_{ds_name}_{repr_str}_stage1_best_{timestamp}.pth')
                    torch.save({'state_dict': new_state, 'epoch': epoch, 'r1': avg_bs32_r1, 'model_config': model_config}, best_retrieval_ckpt_path)
                    print(f"[Stage1] Best model saved: {best_retrieval_ckpt_path} (BS32 R1={avg_bs32_r1:.3f}%)")
        
        if is_distributed:
            dist.barrier()
    
    if rank == 0:
        print("\n" + "="*60)
        print(f"[Stage 1] Complete! Best BS32 R1: {best_metrics['bs32']['r1']:.3f}% at epoch {best_metrics['bs32']['epoch']}")
        print("="*60 + "\n")
    
    # ==================== Stage 2: Critic Training ====================
    best_critic_ckpt_path = None
    best_critic_acc = 0.0
    
    if critic_train_loader is not None and len(critic_train_loader) > 0:
        if rank == 0:
            print("\n" + "="*60)
            print("[Stage 2] Starting Critic Head Training")
            print(f"[Stage 2] Epochs: {stage2_epochs}")
            print(f"[Stage 2] Loading Retrieval from: {best_retrieval_ckpt_path}")
            print("="*60 + "\n")
        
        # Load best Retrieval weights
        if best_retrieval_ckpt_path and os.path.exists(best_retrieval_ckpt_path):
            checkpoint = torch.load(best_retrieval_ckpt_path, map_location=device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            actual_model.load_state_dict(state_dict, strict=False)
            if rank == 0:
                print(f"[Stage2] Loaded Retrieval weights from {best_retrieval_ckpt_path}")
        
        # Freeze Retrieval, only train critic head
        for name, param in actual_model.named_parameters():
            if 'critic_head' not in name:
                param.requires_grad = False
        
        # Init critic head
        actual_model.init_critic_head(hidden_dim=cfg.MODEL.latent_dim)
        actual_model.critic_head = actual_model.critic_head.to(device)
        
        if rank == 0:
            trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
            print(f"[Stage2] Trainable params (Critic head only): {trainable_params:,}")
        
        # Re-wrap DDP if needed
        if is_distributed:
            model = actual_model
            model = DDP(model, device_ids=[int(os.environ.get('LOCAL_RANK', 0))],
                       output_device=int(os.environ.get('LOCAL_RANK', 0)),
                       find_unused_parameters=True)
            actual_model = model.module
        
        # Critic optimizer
        critic_optimizer = torch.optim.AdamW(
            actual_model.critic_head.parameters(),
            lr=cfg.critic_lr,
            weight_decay=1e-4
        )
        
        critic_lr_scheduler = get_scheduler(
            'cosine',
            optimizer=critic_optimizer,
            num_warmup_steps=100,
            num_training_steps=stage2_epochs * len(critic_train_loader)
        )
        
        best_critic_epoch = -1
        
        for epoch in range(stage2_epochs):
            if hasattr(critic_train_loader, 'batch_sampler') and hasattr(critic_train_loader.batch_sampler, 'set_epoch'):
                critic_train_loader.batch_sampler.set_epoch(epoch)
            
            model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            batch_count = 0
            
            for batch in tqdm(critic_train_loader, desc=f'[Stage2] Epoch {epoch}', disable=rank != 0):
                batch = {k: v.float().to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                critic_output = actual_model.forward_critic(batch)
                
                if critic_output is not None:
                    loss, _, acc = pairwise_loss(critic_output)
                    
                    loss.backward()
                    critic_optimizer.step()
                    critic_lr_scheduler.step()
                    critic_optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
                    epoch_acc += acc.item()
                    batch_count += 1
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                avg_acc = epoch_acc / batch_count
                if rank == 0:
                    print(f'[Stage2] Epoch {epoch}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')
                    if writer is not None:
                        writer.log({"Stage2/loss": avg_loss, "Stage2/acc": avg_acc}, step=epoch)
            
            # Validation
            if critic_val_loader is not None:
                model.eval()
                val_result = eval_critic(critic_val_loader, model, device=device, rank=rank)
                
                if rank == 0:
                    print(f"[Stage2] Val - Loss: {val_result['loss']:.4f}, Acc: {val_result['acc']:.4f}")
                    
                    if val_result['acc'] > best_critic_acc:
                        best_critic_acc = val_result['acc']
                        best_critic_epoch = epoch
                        
                        new_state = OrderedDict()
                        for k, v in actual_model.state_dict().items():
                            if 'clip' not in k:
                                new_state[k] = v
                        
                        model_config = _build_model_config(cfg, retrieval_repr_list)
                        best_critic_ckpt_path = os.path.join(checkpoint_dir, f'{text_src}_MultiStage_{ds_name}_{repr_str}_stage2_best_{timestamp}.pth')
                        torch.save({
                            'state_dict': new_state,
                            'critic_head': actual_model.critic_head.state_dict(),
                            'epoch': epoch,
                            'critic_acc': val_result['acc'],
                            'model_config': model_config
                        }, best_critic_ckpt_path)
                        print(f"[Stage2] Best model saved: {best_critic_ckpt_path} (Acc={val_result['acc']:.4f})")
            
            if is_distributed:
                dist.barrier()
        
        if rank == 0:
            print("\n" + "="*60)
            print(f"[Stage 2] Complete! Best Acc: {best_critic_acc:.4f} at epoch {best_critic_epoch}")
            print("="*60 + "\n")
    else:
        if rank == 0:
            print("[Stage 2] Skipped - No critic data available")
        best_critic_ckpt_path = best_retrieval_ckpt_path
    
    # ==================== Stage 3: AI Detection Training ====================
    best_ai_ckpt_path = None
    best_ai_f1 = 0.0
    best_ai_acc = 0.0
    
    if ai_train_loader is not None and len(ai_train_loader) > 0:
        if rank == 0:
            print("\n" + "="*60)
            print("[Stage 3] Starting AI Detection Head Training")
            print(f"[Stage 3] Epochs: {stage3_epochs}")
            print(f"[Stage 3] Loading from: {best_critic_ckpt_path}")
            print("="*60 + "\n")
        
        # Load best model (with critic head)
        if best_critic_ckpt_path and os.path.exists(best_critic_ckpt_path):
            checkpoint = torch.load(best_critic_ckpt_path, map_location=device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            actual_model.load_state_dict(state_dict, strict=False)
            
            if 'critic_head' in checkpoint and hasattr(actual_model, 'critic_head') and actual_model.critic_head is not None:
                actual_model.critic_head.load_state_dict(checkpoint['critic_head'])
            
            if rank == 0:
                print(f"[Stage3] Loaded weights from {best_critic_ckpt_path}")
        
        # Freeze Retrieval and critic head, only train AI detection head
        for name, param in actual_model.named_parameters():
            if 'ai_detection_head' not in name:
                param.requires_grad = False
        
        # Init AI detection head
        actual_model.init_ai_detection_head(hidden_dim=cfg.MODEL.latent_dim)
        actual_model.ai_detection_head = actual_model.ai_detection_head.to(device)
        
        if rank == 0:
            trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
            print(f"[Stage3] Trainable params (AI Detection head only): {trainable_params:,}")
        
        # Re-wrap DDP if needed
        if is_distributed:
            model = actual_model
            model = DDP(model, device_ids=[int(os.environ.get('LOCAL_RANK', 0))],
                       output_device=int(os.environ.get('LOCAL_RANK', 0)),
                       find_unused_parameters=True)
            actual_model = model.module
        
        # AI detection optimizer
        ai_optimizer = torch.optim.AdamW(
            actual_model.ai_detection_head.parameters(),
            lr=cfg.ai_detection_lr,
            weight_decay=1e-4
        )
        
        ai_lr_scheduler = get_scheduler(
            'cosine',
            optimizer=ai_optimizer,
            num_warmup_steps=100,
            num_training_steps=stage3_epochs * len(ai_train_loader)
        )
        
        best_ai_epoch = -1
        
        for epoch in range(stage3_epochs):
            if hasattr(ai_train_loader, 'batch_sampler') and hasattr(ai_train_loader.batch_sampler, 'set_epoch'):
                ai_train_loader.batch_sampler.set_epoch(epoch)
            
            model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            batch_count = 0
            
            for batch in tqdm(ai_train_loader, desc=f'[Stage3] Epoch {epoch}', disable=rank != 0):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                ai_logits, ai_labels = actual_model.forward_ai_detection(batch)
                
                if ai_logits is not None:
                    loss = F.cross_entropy(ai_logits, ai_labels)
                    acc = (ai_logits.argmax(dim=-1) == ai_labels).float().mean()
                    
                    loss.backward()
                    ai_optimizer.step()
                    ai_lr_scheduler.step()
                    ai_optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
                    epoch_acc += acc.item()
                    batch_count += 1
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                avg_acc = epoch_acc / batch_count
                if rank == 0:
                    print(f'[Stage3] Epoch {epoch}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')
                    if writer is not None:
                        writer.log({"Stage3/loss": avg_loss, "Stage3/acc": avg_acc}, step=epoch)
            
            # Validation
            if ai_val_loader is not None:
                model.eval()
                val_result = eval_ai_detection(ai_val_loader, model, device=device, rank=rank)
                
                if rank == 0:
                    print(f"[Stage3] Val - Loss: {val_result['loss']:.4f}, Acc: {val_result['acc']:.4f}, F1: {val_result['f1']:.4f}")
                    
                    if val_result['f1'] > best_ai_f1:
                        best_ai_f1 = val_result['f1']
                        best_ai_acc = val_result['acc']
                        best_ai_epoch = epoch
                        
                        new_state = OrderedDict()
                        for k, v in actual_model.state_dict().items():
                            if 'clip' not in k:
                                new_state[k] = v
                        
                        model_config = _build_model_config(cfg, retrieval_repr_list)
                        best_ai_ckpt_path = os.path.join(checkpoint_dir, f'{text_src}_MultiStage_{ds_name}_{repr_str}_stage3_best_{timestamp}.pth')
                        save_dict = {
                            'state_dict': new_state,
                            'ai_detection_head': actual_model.ai_detection_head.state_dict(),
                            'epoch': epoch,
                            'ai_f1': val_result['f1'],
                            'ai_acc': val_result['acc'],
                            'model_config': model_config
                        }
                        if hasattr(actual_model, 'critic_head') and actual_model.critic_head is not None:
                            save_dict['critic_head'] = actual_model.critic_head.state_dict()
                        torch.save(save_dict, best_ai_ckpt_path)
                        print(f"[Stage3] Best model saved: {best_ai_ckpt_path} (F1={val_result['f1']:.4f})")
            
            if is_distributed:
                dist.barrier()
        
        # Test evaluation
        if ai_test_loader is not None:
            model.eval()
            test_result = eval_ai_detection(ai_test_loader, model, device=device, rank=rank)
            
            if rank == 0:
                print("\n" + "-"*60)
                print("[Stage3] AI Detection Test Set Evaluation:")
                print(f"  Loss: {test_result['loss']:.4f}, Acc: {test_result['acc']:.4f}")
                print(f"  Precision: {test_result['precision']:.4f}, Recall: {test_result['recall']:.4f}, F1: {test_result['f1']:.4f}")
                print("-"*60)
        
        if rank == 0:
            print("\n" + "="*60)
            print(f"[Stage 3] Complete! Best F1: {best_ai_f1:.4f}, Acc: {best_ai_acc:.4f} at epoch {best_ai_epoch}")
            print("="*60 + "\n")
    else:
        if rank == 0:
            print("[Stage 3] Skipped - No AI detection data available")
    
    # Final summary
    if rank == 0:
        print("\n" + "="*60)
        print("[Multi-Stage Training] Complete!")
        print(f"  Stage 1 (Retrieval): Best BS32 R1 = {best_metrics['bs32']['r1']:.3f}%")
        if critic_train_loader is not None:
            print(f"  Stage 2 (Critic): Best Acc = {best_critic_acc:.4f}")
        if ai_train_loader is not None:
            print(f"  Stage 3 (AI Detection): Best F1 = {best_ai_f1:.4f}")
        print("="*60 + "\n")
    
    return best_retrieval_ckpt_path, best_critic_ckpt_path, best_ai_ckpt_path


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
