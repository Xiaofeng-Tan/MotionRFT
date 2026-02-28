"""
Retrieval 训练器

包含 Retrieval 模型的训练逻辑
"""

import os
import random
import torch
from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict

from motionreward.models import pairwise_loss


def train_retrieval_epoch(
    model, train_loader, optimizer, lr_scheduler, scheduler, probs,
    epoch, cfg, device, rank, writer=None,
    paired_train_loader=None, paired_iter=None,
    critic_train_iter=None, ai_train_iter=None,
    joint_train=False, joint_ai_detection=False,
    use_cross_repr_align=False, lambda_cross_repr=0.1,
    fptr=None
):
    """训练一个 epoch
    
    Args:
        model: Retrieval 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        scheduler: DDPM scheduler
        probs: 时间步采样概率
        epoch: 当前 epoch
        cfg: 配置
        device: 设备
        rank: 进程 rank
        writer: 日志 writer
        paired_train_loader: 配对数据加载器（跨表征对齐）
        paired_iter: 配对数据迭代器
        critic_train_iter: Critic 数据迭代器
        ai_train_iter: AI 检测数据迭代器
        joint_train: 是否联合训练 Critic
        joint_ai_detection: 是否联合训练 AI 检测
        use_cross_repr_align: 是否使用跨表征对齐
        lambda_cross_repr: 跨表征对齐权重
        fptr: 日志文件指针
    
    Returns:
        avg_loss: 平均损失
        paired_iter: 更新后的配对数据迭代器
        critic_train_iter: 更新后的 Critic 数据迭代器
        ai_train_iter: 更新后的 AI 检测数据迭代器
        epoch_metrics: epoch 级别的指标字典
    """
    import torch.nn.functional as F
    
    actual_model = model.module if hasattr(model, 'module') else model
    model.train()
    
    step_aware = cfg.Retrieval.step_aware
    thr = cfg.Retrieval.NoiseThr
    
    epoch_loss = 0.0
    epoch_cross_loss = 0.0
    epoch_critic_loss = 0.0
    epoch_critic_acc = 0.0
    critic_batch_count = 0
    epoch_ai_detection_loss = 0.0
    epoch_ai_detection_acc = 0.0
    ai_detection_batch_count = 0
    total_loss = 0.0
    global_step = epoch * len(train_loader)
    
    for i, batch in tqdm(enumerate(train_loader), desc=f'Epoch {epoch}', total=len(train_loader), disable=rank != 0):
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
        
        # Retrieval Loss
        tfs_loss = actual_model(text=text, motion_feature=feats_ref, m_len=m_len, repr_type=repr_type, timestep=timestep, mode=step_aware)
        
        # 跨表征对齐 Loss
        cross_repr_loss = torch.tensor(0.0, device=device)
        cross_details = {}
        if use_cross_repr_align and paired_iter is not None and i % 2 == 0:
            try:
                paired_batch = next(paired_iter)
            except StopIteration:
                paired_iter = iter(paired_train_loader)
                paired_batch = next(paired_iter)
            
            motion_263 = paired_batch['motion_263'].float().to(device)
            motion_22x3 = paired_batch['motion_22x3'].float().to(device)
            paired_m_len = paired_batch['length']
            if isinstance(paired_m_len, torch.Tensor):
                paired_m_len = paired_m_len.tolist()
            
            paired_timestep = torch.multinomial(probs, num_samples=motion_263.shape[0], replacement=True).long()
            
            cross_repr_loss, cross_details = actual_model.forward_cross_repr(
                motion_263, motion_22x3, paired_m_len, timestep=paired_timestep
            )
            cross_repr_loss = cross_repr_loss * lambda_cross_repr
            epoch_cross_loss += cross_repr_loss.item()
        
        # 联合训练: Critic Loss
        critic_loss = torch.tensor(0.0, device=device)
        critic_acc_item = 0.0
        if joint_train and epoch >= cfg.critic_start_epoch and critic_train_iter is not None and i % 2 == 0:
            try:
                critic_batch = next(critic_train_iter)
            except StopIteration:
                from motionreward.datasets import CriticPairDataset, CriticReprTypeBatchSampler, critic_collate_fn
                # 需要重新创建迭代器
                critic_train_iter = None  # 标记需要外部重置
            
            if critic_train_iter is not None:
                critic_batch = {k: v.float().to(device) if isinstance(v, torch.Tensor) else v for k, v in critic_batch.items()}
                critic_output = actual_model.forward_critic(critic_batch)
                
                if critic_output is not None:
                    critic_loss_raw, _, critic_acc = pairwise_loss(critic_output)
                    critic_loss = critic_loss_raw * cfg.lambda_critic
                    critic_acc_item = critic_acc.item()
                    epoch_critic_loss += critic_loss.item()
                    epoch_critic_acc += critic_acc_item
                    critic_batch_count += 1
        
        # 联合训练: AI Detection Loss
        ai_detection_loss = torch.tensor(0.0, device=device)
        ai_detection_acc_item = 0.0
        if joint_ai_detection and epoch >= cfg.ai_detection_start_epoch and ai_train_iter is not None and i % 2 == 0:
            try:
                ai_batch = next(ai_train_iter)
            except StopIteration:
                ai_train_iter = None  # 标记需要外部重置
            
            if ai_train_iter is not None:
                ai_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in ai_batch.items()}
                ai_logits, ai_labels = actual_model.forward_ai_detection(ai_batch)
                
                if ai_logits is not None:
                    ai_detection_loss_raw = F.cross_entropy(ai_logits, ai_labels)
                    ai_detection_loss = ai_detection_loss_raw * cfg.lambda_ai_detection
                    ai_detection_acc_item = (ai_logits.argmax(dim=-1) == ai_labels).float().mean().item()
                    epoch_ai_detection_loss += ai_detection_loss.item()
                    epoch_ai_detection_acc += ai_detection_acc_item
                    ai_detection_batch_count += 1
        
        total_batch_loss = tfs_loss + cross_repr_loss + critic_loss + ai_detection_loss
        loss_item = total_batch_loss.item()
        total_loss += loss_item
        epoch_loss += loss_item
        
        total_batch_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        # 保存 loss 值用于日志
        critic_loss_item = critic_loss.item() if isinstance(critic_loss, torch.Tensor) else 0
        ai_detection_loss_item = ai_detection_loss.item() if isinstance(ai_detection_loss, torch.Tensor) else 0
        
        del tfs_loss, feats_ref, timestep, total_batch_loss
        if use_cross_repr_align:
            del cross_repr_loss
        if joint_train:
            del critic_loss
        if joint_ai_detection:
            del ai_detection_loss
        torch.cuda.empty_cache()
        
        if writer is not None and rank == 0:
            log_dict = {"Train/loss": loss_item, "Train/lr": lr_scheduler.get_last_lr()[0]}
            if use_cross_repr_align and i % 2 == 0:
                log_dict["Train/cross_repr_loss"] = cross_details.get('latent_align', 0)
            if joint_train and epoch >= cfg.critic_start_epoch and i % 2 == 0 and critic_loss_item > 0:
                log_dict["Train/critic_loss"] = critic_loss_item
                log_dict["Train/critic_acc"] = critic_acc_item
            if joint_ai_detection and epoch >= cfg.ai_detection_start_epoch and i % 2 == 0 and ai_detection_loss_item > 0:
                log_dict["Train/ai_detection_loss"] = ai_detection_loss_item
                log_dict["Train/ai_detection_acc"] = ai_detection_acc_item
            writer.log(log_dict, step=global_step + i)
    
    avg_epoch_loss = total_loss / len(train_loader)
    
    # 构建 epoch 指标
    epoch_metrics = {
        'loss': avg_epoch_loss,
        'cross_repr_loss': epoch_cross_loss / (len(train_loader) // 2 + 1) if use_cross_repr_align else 0,
    }
    
    if joint_train and critic_batch_count > 0:
        epoch_metrics['critic_loss'] = epoch_critic_loss / critic_batch_count
        epoch_metrics['critic_acc'] = epoch_critic_acc / critic_batch_count
    
    if joint_ai_detection and ai_detection_batch_count > 0:
        epoch_metrics['ai_detection_loss'] = epoch_ai_detection_loss / ai_detection_batch_count
        epoch_metrics['ai_detection_acc'] = epoch_ai_detection_acc / ai_detection_batch_count
    
    # 打印日志
    if rank == 0:
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f'Time = {time_str}, Epoch = {epoch}, Avg loss = {avg_epoch_loss:.4f}'
        if use_cross_repr_align:
            log_msg += f', Cross-repr loss = {epoch_metrics["cross_repr_loss"]:.4f}'
        if joint_train and 'critic_loss' in epoch_metrics:
            log_msg += f', Critic loss = {epoch_metrics["critic_loss"]:.4f}, Critic acc = {epoch_metrics["critic_acc"]:.4f}'
        if joint_ai_detection and 'ai_detection_loss' in epoch_metrics:
            log_msg += f', AI loss = {epoch_metrics["ai_detection_loss"]:.4f}, AI acc = {epoch_metrics["ai_detection_acc"]:.4f}'
        print(log_msg + '\n')
        if fptr:
            print(log_msg + '\n', file=fptr)
            fptr.flush()
    
    return avg_epoch_loss, paired_iter, critic_train_iter, ai_train_iter, epoch_metrics


def save_retrieval_checkpoint(model, cfg, epoch, metrics, retrieval_repr_list, checkpoint_dir, prefix='Retrieval'):
    """保存 Retrieval checkpoint
    
    Args:
        model: 模型
        cfg: 配置
        epoch: epoch 数
        metrics: 指标字典
        retrieval_repr_list: 表征类型列表
        checkpoint_dir: 保存目录
        prefix: 文件名前缀
    
    Returns:
        ckpt_path: 保存的 checkpoint 路径
    """
    actual_model = model.module if hasattr(model, 'module') else model
    
    new_state = OrderedDict()
    for k, v in actual_model.state_dict().items():
        if 'clip' not in k:
            new_state[k] = v
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    repr_str = '_'.join(retrieval_repr_list)
    
    model_config = {
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
    
    save_dict = {
        'state_dict': new_state,
        'epoch': epoch,
        'model_config': model_config,
        **metrics
    }
    
    # 保存任务头
    if hasattr(actual_model, 'critic_head') and actual_model.critic_head is not None:
        save_dict['critic_head'] = actual_model.critic_head.state_dict()
    if hasattr(actual_model, 'ai_detection_head') and actual_model.ai_detection_head is not None:
        save_dict['ai_detection_head'] = actual_model.ai_detection_head.state_dict()
    
    ckpt_path = os.path.join(checkpoint_dir, f'{prefix}_{repr_str}_E{epoch}.pth')
    torch.save(save_dict, ckpt_path)
    
    return ckpt_path
