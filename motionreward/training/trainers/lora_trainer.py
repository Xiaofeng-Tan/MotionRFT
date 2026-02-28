"""
LoRA 三阶段训练器

Stage 1: 训练 Retrieval（全参数），保存最佳检索模型
Stage 2: 冻结 Retrieval 主干，训练 Critic LoRA + Critic Head
Stage 3: 冻结 Retrieval 主干，训练 AI Detection LoRA + AI Detection Head
"""

import os
import random
from collections import OrderedDict, defaultdict
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from diffusers.optimization import get_scheduler

from motionreward.evaluation import eval_tmr, eval_critic, eval_ai_detection
from motionreward.models.heads import pairwise_loss
from motionreward.models.lora_modules import get_lora_state_dict, load_lora_state_dict, disable_lora, enable_lora
from motionreward.utils.ddp_utils import is_main_process
from motionreward.utils.config_utils import build_model_config_for_save

# 全局时间戳，在第一次调用时生成
_TRAINING_TIMESTAMP = None

def get_training_timestamp():
    """获取训练时间戳（单次训练会话内保持一致）"""
    global _TRAINING_TIMESTAMP
    if _TRAINING_TIMESTAMP is None:
        _TRAINING_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _TRAINING_TIMESTAMP


def save_model_modular(model, save_dir, prefix, epoch, metrics, model_config, rank=0):
    """分模块保存模型
    
    Args:
        model: 模型实例（可能是 DDP 包装的）
        save_dir: 保存目录
        prefix: 文件名前缀（如 '263_22x3_stage1_best'）
        epoch: 当前 epoch
        metrics: 指标字典
        model_config: 模型配置
        rank: 进程 rank
    
    Returns:
        saved_paths: 保存的文件路径字典
    """
    if rank != 0:
        return {}
    
    actual_model = model.module if hasattr(model, 'module') else model
    os.makedirs(save_dir, exist_ok=True)
    
    saved_paths = {}
    timestamp = get_training_timestamp()
    
    # 1. 保存 Retrieval 主干（motion encoder, text encoder, decoder, projections）
    # 注意：如果 LoRA 已注入，state_dict 的 key 会变成 xxx.original_linear.weight
    # 需要将其转换回原始格式 xxx.weight
    retrieval_backbone_state = OrderedDict()
    for k, v in actual_model.state_dict().items():
        # 排除 clip, lora, critic, ai_detection
        if 'clip' in k:
            continue
        if 'lora' in k.lower():
            continue
        if 'critic' in k.lower():
            continue
        if 'ai_detection' in k.lower():
            continue
        
        # 处理 LoRA 注入后的 key 名称变化
        # xxx.original_linear.weight -> xxx.weight
        # xxx.original_linear.bias -> xxx.bias
        new_key = k.replace('.original_linear.', '.').replace('.original_mha.', '.')
        retrieval_backbone_state[new_key] = v
    
    retrieval_path = os.path.join(save_dir, f'{prefix}_retrieval_backbone_{timestamp}.pth')
    torch.save({
        'state_dict': retrieval_backbone_state,
        'epoch': epoch,
        'metrics': metrics,
        'model_config': model_config,
    }, retrieval_path)
    saved_paths['retrieval_backbone'] = retrieval_path
    print(f"  [Save] Retrieval backbone: {retrieval_path} ({len(retrieval_backbone_state)} keys)")
    
    # 2. 保存 Critic LoRA（如果存在）
    if actual_model.critic_lora_modules is not None:
        critic_lora_state = get_lora_state_dict(actual_model.critic_lora_modules)
        critic_lora_path = os.path.join(save_dir, f'{prefix}_critic_lora_{timestamp}.pth')
        torch.save({
            'lora_state_dict': critic_lora_state,
            'lora_rank': actual_model.lora_rank,
            'lora_alpha': actual_model.lora_alpha,
            'epoch': epoch,
            'metrics': metrics,
        }, critic_lora_path)
        saved_paths['critic_lora'] = critic_lora_path
        print(f"  [Save] Critic LoRA: {critic_lora_path} ({len(critic_lora_state)} keys)")
    
    # 3. 保存 Critic Head（如果存在）
    if actual_model.critic_head is not None:
        critic_head_path = os.path.join(save_dir, f'{prefix}_critic_head_{timestamp}.pth')
        torch.save({
            'state_dict': actual_model.critic_head.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
        }, critic_head_path)
        saved_paths['critic_head'] = critic_head_path
        print(f"  [Save] Critic head: {critic_head_path}")
    
    # 4. 保存 AI Detection LoRA（如果存在）
    if hasattr(actual_model, 'ai_detection_lora_modules') and actual_model.ai_detection_lora_modules is not None:
        ai_lora_state = get_lora_state_dict(actual_model.ai_detection_lora_modules)
        ai_lora_path = os.path.join(save_dir, f'{prefix}_ai_detection_lora_{timestamp}.pth')
        torch.save({
            'lora_state_dict': ai_lora_state,
            'lora_rank': actual_model.lora_rank,
            'lora_alpha': actual_model.lora_alpha,
            'epoch': epoch,
            'metrics': metrics,
        }, ai_lora_path)
        saved_paths['ai_detection_lora'] = ai_lora_path
        print(f"  [Save] AI Detection LoRA: {ai_lora_path} ({len(ai_lora_state)} keys)")
    
    # 5. 保存 AI Detection Head（如果存在）
    if actual_model.ai_detection_head is not None:
        ai_head_path = os.path.join(save_dir, f'{prefix}_ai_detection_head_{timestamp}.pth')
        torch.save({
            'state_dict': actual_model.ai_detection_head.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
        }, ai_head_path)
        saved_paths['ai_detection_head'] = ai_head_path
        print(f"  [Save] AI Detection head: {ai_head_path}")
    
    # 6. 保存元信息文件
    meta_path = os.path.join(save_dir, f'{prefix}_meta_{timestamp}.pth')
    torch.save({
        'epoch': epoch,
        'metrics': metrics,
        'model_config': model_config,
        'saved_modules': list(saved_paths.keys()),
        'module_paths': saved_paths,
        'timestamp': timestamp,
    }, meta_path)
    saved_paths['meta'] = meta_path
    print(f"  [Save] Meta info: {meta_path}")
    
    return saved_paths


def load_model_modular(model, load_dir, prefix, device, load_modules=None, rank=0, timestamp=None):
    """分模块加载模型
    
    Args:
        model: 模型实例
        load_dir: 加载目录
        prefix: 文件名前缀
        device: 设备
        load_modules: 要加载的模块列表，None 表示加载所有可用模块
            可选: ['retrieval_backbone', 'critic_lora', 'critic_head', 'ai_detection_lora', 'ai_detection_head']
        rank: 进程 rank
        timestamp: 时间戳，None 表示使用当前训练时间戳
    
    Returns:
        loaded_modules: 成功加载的模块列表
    """
    actual_model = model.module if hasattr(model, 'module') else model
    loaded_modules = []
    ts = timestamp or get_training_timestamp()
    
    # 1. 加载 Retrieval 主干
    if load_modules is None or 'retrieval_backbone' in load_modules:
        retrieval_path = os.path.join(load_dir, f'{prefix}_retrieval_backbone_{ts}.pth')
        if os.path.exists(retrieval_path):
            checkpoint = torch.load(retrieval_path, map_location=device, weights_only=False)
            actual_model.load_state_dict(checkpoint['state_dict'], strict=False)
            loaded_modules.append('retrieval_backbone')
            if rank == 0:
                print(f"  [Load] Retrieval backbone from {retrieval_path}")
    
    # 2. 加载 Critic LoRA
    if load_modules is None or 'critic_lora' in load_modules:
        critic_lora_path = os.path.join(load_dir, f'{prefix}_critic_lora_{ts}.pth')
        if os.path.exists(critic_lora_path):
            checkpoint = torch.load(critic_lora_path, map_location=device, weights_only=False)
            if actual_model.critic_lora_modules is None:
                actual_model.inject_critic_lora()
            load_lora_state_dict(actual_model.critic_lora_modules, checkpoint['lora_state_dict'])
            loaded_modules.append('critic_lora')
            if rank == 0:
                print(f"  [Load] Critic LoRA from {critic_lora_path}")
    
    # 3. 加载 Critic Head
    if load_modules is None or 'critic_head' in load_modules:
        critic_head_path = os.path.join(load_dir, f'{prefix}_critic_head_{ts}.pth')
        if os.path.exists(critic_head_path):
            checkpoint = torch.load(critic_head_path, map_location=device, weights_only=False)
            if actual_model.critic_head is not None:
                actual_model.critic_head.load_state_dict(checkpoint['state_dict'])
                loaded_modules.append('critic_head')
                if rank == 0:
                    print(f"  [Load] Critic head from {critic_head_path}")
    
    # 4. 加载 AI Detection LoRA
    if load_modules is None or 'ai_detection_lora' in load_modules:
        ai_lora_path = os.path.join(load_dir, f'{prefix}_ai_detection_lora_{ts}.pth')
        if os.path.exists(ai_lora_path):
            checkpoint = torch.load(ai_lora_path, map_location=device, weights_only=False)
            if not hasattr(actual_model, 'ai_detection_lora_modules') or actual_model.ai_detection_lora_modules is None:
                actual_model.inject_ai_detection_lora()
            load_lora_state_dict(actual_model.ai_detection_lora_modules, checkpoint['lora_state_dict'])
            loaded_modules.append('ai_detection_lora')
            if rank == 0:
                print(f"  [Load] AI Detection LoRA from {ai_lora_path}")
    
    # 5. 加载 AI Detection Head
    if load_modules is None or 'ai_detection_head' in load_modules:
        ai_head_path = os.path.join(load_dir, f'{prefix}_ai_detection_head_{ts}.pth')
        if os.path.exists(ai_head_path):
            checkpoint = torch.load(ai_head_path, map_location=device, weights_only=False)
            if actual_model.ai_detection_head is not None:
                actual_model.ai_detection_head.load_state_dict(checkpoint['state_dict'])
                loaded_modules.append('ai_detection_head')
                if rank == 0:
                    print(f"  [Load] AI Detection head from {ai_head_path}")
    
    return loaded_modules


def train_stage1_retrieval(cfg, model, train_loader, test_loaders, device, rank, world_size, 
                     is_distributed, writer, retrieval_repr_list, scheduler, probs, fptr=None,
                     paired_train_loader=None):
    """Stage 1: 训练 Retrieval（全参数）
    
    Args:
        cfg: 配置对象
        model: 模型（可能是 DDP 包装的）
        train_loader: 训练数据加载器
        test_loaders: 测试数据加载器字典 {repr_type: loader}
        device: 设备
        rank: 进程 rank
        world_size: 进程总数
        is_distributed: 是否分布式训练
        writer: SwanLab writer
        retrieval_repr_list: 表征类型列表
        scheduler: 噪声调度器
        probs: 时间步采样概率
        fptr: 日志文件指针
        paired_train_loader: 配对数据加载器（跨表征对齐，可选）
        
    Returns:
        best_retrieval_ckpt_path: 最佳模型路径
        best_metrics: 最佳指标
    """
    actual_model = model.module if hasattr(model, 'module') else model
    timestamp = get_training_timestamp()
    
    if rank == 0:
        print("\n" + "="*60)
        print("[Stage 1] Retrieval Training (Full Parameters)")
        print(f"  Epochs: {cfg.stage1_epochs}")
        print(f"  Learning Rate: {cfg.stage1_lr}")
        print(f"  Timestamp: {timestamp}")
        use_cross_repr = getattr(cfg, 'use_cross_repr_align', False) and paired_train_loader is not None
        lambda_cross_repr = getattr(cfg, 'lambda_cross_repr', 0.1)
        if use_cross_repr:
            print(f"  Cross-Repr Alignment: ON (lambda={lambda_cross_repr})")
        else:
            print(f"  Cross-Repr Alignment: OFF")
        print("="*60 + "\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.stage1_lr,
        betas=(cfg.TRAIN.adam_beta1, cfg.TRAIN.adam_beta2),
        weight_decay=cfg.TRAIN.adam_weight_decay,
        eps=cfg.TRAIN.adam_epsilon
    )
    
    max_train_steps = cfg.stage1_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        cfg.TRAIN.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.TRAIN.lr_warmup_steps,
        num_training_steps=max_train_steps
    )
    
    step_aware = cfg.Retrieval.step_aware
    thr = cfg.Retrieval.NoiseThr
    maxT = cfg.Retrieval.maxT
    
    # 跨表征对齐配置
    use_cross_repr = getattr(cfg, 'use_cross_repr_align', False) and paired_train_loader is not None
    lambda_cross_repr = getattr(cfg, 'lambda_cross_repr', 0.1)
    paired_iter = iter(paired_train_loader) if use_cross_repr else None
    
    best_metrics = {'bs32': {'r1': 0.0, 'epoch': -1}}
    best_retrieval_ckpt_path = None
    
    checkpoint_dir = cfg.Retrieval_CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    repr_str = '_'.join(retrieval_repr_list)
    
    # 支持自定义前缀
    base_prefix = getattr(cfg, 'checkpoint_prefix', None) or repr_str
    
    for epoch in range(cfg.stage1_epochs):
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
        
        # 每个 epoch 重置配对迭代器
        if use_cross_repr:
            paired_iter = iter(paired_train_loader)
        
        model.train()
        epoch_loss = 0.0
        epoch_cross_loss = 0.0
        
        for i, batch in tqdm(enumerate(train_loader), desc=f'[Stage1] Epoch {epoch}', 
                             total=len(train_loader), disable=rank != 0):
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
            
            # 跨表征对齐损失（每隔 2 个 batch 计算一次，降低额外开销）
            cross_repr_loss_val = 0.0
            if use_cross_repr and i % 2 == 0:
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
                loss = loss + cross_repr_loss * lambda_cross_repr
                cross_repr_loss_val = cross_repr_loss.item()
                epoch_cross_loss += cross_repr_loss_val
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
            del loss, feats_ref, timestep
            torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / len(train_loader)
        if rank == 0:
            log_msg = f'[Stage1] Epoch {epoch}, Avg Loss: {avg_loss:.4f}'
            if use_cross_repr:
                avg_cross_loss = epoch_cross_loss / (len(train_loader) // 2 + 1)
                log_msg += f', Cross-Repr Loss: {avg_cross_loss:.4f}'
            print(log_msg)
            if writer is not None:
                log_dict = {"Stage1/loss": avg_loss, "Stage1/lr": lr_scheduler.get_last_lr()[0]}
                if use_cross_repr:
                    log_dict["Stage1/cross_repr_loss"] = avg_cross_loss
                writer.log(log_dict, step=epoch)
        
        # Evaluation
        model.eval()
        epoch_eval_results = {}
        for repr_type, test_loader in test_loaders.items():
            eval_results = eval_tmr(test_loader, model, epoch, repr_type=repr_type, 
                                   mode=step_aware, writer=writer, device=device, rank=rank)
            epoch_eval_results[repr_type] = eval_results
        
        if is_main_process(rank):
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
                    
                    # 分模块保存最佳模型
                    model_config = build_model_config_for_save(cfg, retrieval_repr_list)
                    metrics = {'r1': avg_bs32_r1, 'epoch': epoch}
                    
                    saved_paths = save_model_modular(
                        model, checkpoint_dir, f'{base_prefix}_stage1_best',
                        epoch, metrics, model_config, rank
                    )
                    best_retrieval_ckpt_path = saved_paths.get('retrieval_backbone')
                    print(f"[Stage1] Best model saved (BS32 R1={avg_bs32_r1:.3f}%)")
        
        if is_distributed:
            dist.barrier()
    
    if rank == 0:
        print("\n" + "="*60)
        print(f"[Stage 1] Complete! Best BS32 R1: {best_metrics['bs32']['r1']:.3f}% at epoch {best_metrics['bs32']['epoch']}")
        print("="*60 + "\n")
        if fptr:
            print(f"[Stage 1] Best BS32 R1: {best_metrics['bs32']['r1']:.3f}% at epoch {best_metrics['bs32']['epoch']}", file=fptr)
            fptr.flush()
    
    return best_retrieval_ckpt_path, best_metrics


def train_stage2_critic_lora(cfg, model, critic_train_loader, critic_val_loader, test_loaders,
                              device, rank, world_size, is_distributed, writer, 
                              retrieval_repr_list, best_retrieval_ckpt_path, fptr=None):
    """Stage 2: 训练 Critic LoRA + Critic Head（或全参训练）
    
    Args:
        cfg: 配置对象
        model: 模型
        critic_train_loader: Critic 训练数据加载器
        critic_val_loader: Critic 验证数据加载器
        test_loaders: 测试数据加载器字典
        device: 设备
        rank: 进程 rank
        world_size: 进程总数
        is_distributed: 是否分布式训练
        writer: SwanLab writer
        retrieval_repr_list: 表征类型列表
        best_retrieval_ckpt_path: Stage 1 最佳模型路径
        fptr: 日志文件指针
        
    Returns:
        best_critic_ckpt_path: 最佳 Critic 模型路径
        best_critic_acc: 最佳准确率
    """
    actual_model = model.module if hasattr(model, 'module') else model
    timestamp = get_training_timestamp()
    
    # 检查是否使用全参训练模式
    full_finetune = getattr(cfg, 'critic_full_finetune', False)
    
    if rank == 0:
        print("\n" + "="*60)
        if full_finetune:
            print("[Stage 2] Critic Full Finetune (全参训练)")
        else:
            print("[Stage 2] Critic LoRA Training")
        print(f"  Epochs: {cfg.stage2_epochs}")
        print(f"  Learning Rate: {cfg.stage2_lr}")
        if not full_finetune:
            print(f"  LoRA Rank: {cfg.lora_rank}, Alpha: {cfg.lora_alpha}")
        print(f"  Timestamp: {timestamp}")
        if best_retrieval_ckpt_path:
            print(f"  Loading Retrieval from: {best_retrieval_ckpt_path}")
        else:
            print(f"  Retrieval: Random initialization (no pretrained weights)")
        print("="*60 + "\n")
    
    # 加载 Retrieval 权重（如果有）
    if best_retrieval_ckpt_path and os.path.exists(best_retrieval_ckpt_path):
        checkpoint = torch.load(best_retrieval_ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        actual_model.load_state_dict(state_dict, strict=False)
        if rank == 0:
            print(f"[Stage2] Loaded Retrieval weights from {best_retrieval_ckpt_path}")
    elif rank == 0:
        print(f"[Stage2] Training from scratch (no Retrieval checkpoint)")
    
    if full_finetune:
        # 全参训练模式：训练整个 motion encoder + critic head
        # 不冻结任何参数，不注入 LoRA
        
        # 初始化 Critic Head（如果还没有）
        if actual_model.critic_head is None:
            actual_model.init_critic_head()
        
        # 所有参数都可训练
        for _, param in actual_model.named_parameters():
            param.requires_grad = True
        
        actual_model = actual_model.to(device)
        
        if rank == 0:
            trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in actual_model.parameters())
            print(f"[Stage2] Full finetune - Trainable params: {trainable_params:,} / {total_params:,}")
    else:
        # LoRA 训练模式：冻结 Retrieval 主干，只训练 LoRA + Critic Head
        
        # 冻结 Retrieval 主干
        for name, param in actual_model.named_parameters():
            if 'critic' not in name and 'lora' not in name:
                param.requires_grad = False
        
        # 注入 Critic LoRA
        critic_lora_params = actual_model.inject_critic_lora()
        # 确保整个模型（包括新注入的 LoRA 模块）在正确的设备上
        actual_model = actual_model.to(device)
        
        if rank == 0:
            trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
            print(f"[Stage2] Trainable params: {trainable_params:,}")
    
    # 重新包装 DDP
    if is_distributed:
        model = actual_model
        model = DDP(model, device_ids=[int(os.environ.get('LOCAL_RANK', 0))],
                   output_device=int(os.environ.get('LOCAL_RANK', 0)),
                   find_unused_parameters=True)
        actual_model = model.module
    
    # Optimizer - 优化可训练参数
    trainable_params_list = [p for p in actual_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params_list, lr=cfg.stage2_lr, weight_decay=1e-4)
    
    lr_scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=cfg.stage2_epochs * len(critic_train_loader)
    )
    
    best_critic_acc = 0.0
    best_critic_epoch = -1
    best_critic_ckpt_path = None
    
    checkpoint_dir = cfg.Retrieval_CHECKPOINT_DIR
    repr_str = '_'.join(retrieval_repr_list)
    
    # 支持自定义前缀
    base_prefix = getattr(cfg, 'checkpoint_prefix', None) or repr_str
    
    # 辅助损失权重（重建 + KL）
    lambda_aux = getattr(cfg, 'lambda_recons', 0.0)
    use_aux_loss = lambda_aux > 0
    if rank == 0 and use_aux_loss:
        print(f"[Stage2] Using auxiliary loss (recons + KL) with lambda={lambda_aux}")
    
    for epoch in range(cfg.stage2_epochs):
        if hasattr(critic_train_loader, 'batch_sampler') and hasattr(critic_train_loader.batch_sampler, 'set_epoch'):
            critic_train_loader.batch_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        epoch_critic_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_acc = 0.0
        batch_count = 0
        
        # 按表征类型统计训练指标
        repr_train_stats = defaultdict(lambda: {'loss': 0.0, 'acc': 0.0, 'count': 0})
        
        for batch in tqdm(critic_train_loader, desc=f'[Stage2] Epoch {epoch}', disable=rank != 0):
            repr_type = batch.get('repr_type', 'unknown')
            batch = {k: v.float().to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            if use_aux_loss:
                critic_output, aux_loss = actual_model.forward_critic(batch, return_aux_loss=True)
            else:
                critic_output = actual_model.forward_critic(batch, return_aux_loss=False)
                aux_loss = 0.0
            
            if critic_output is not None:
                critic_loss, _, acc = pairwise_loss(critic_output)
                
                # 总损失 = critic 损失 + 辅助损失（重建 + KL）
                if use_aux_loss:
                    loss = critic_loss + lambda_aux * aux_loss
                else:
                    loss = critic_loss
                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                epoch_critic_loss += critic_loss.item()
                if use_aux_loss:
                    epoch_aux_loss += aux_loss.item()
                epoch_acc += acc.item()
                batch_count += 1
                
                # 按表征类型统计
                repr_train_stats[repr_type]['loss'] += critic_loss.item()
                repr_train_stats[repr_type]['acc'] += acc.item()
                repr_train_stats[repr_type]['count'] += 1
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_critic_loss = epoch_critic_loss / batch_count
            avg_aux_loss = epoch_aux_loss / batch_count if use_aux_loss else 0.0
            avg_acc = epoch_acc / batch_count
            if rank == 0:
                if use_aux_loss:
                    print(f'[Stage2] Epoch {epoch}, Loss: {avg_loss:.4f} (critic: {avg_critic_loss:.4f}, aux: {avg_aux_loss:.4f}), Acc: {avg_acc:.4f}')
                else:
                    print(f'[Stage2] Epoch {epoch}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')
                
                # 打印每个表征类型的训练指标
                for repr_type, stats in repr_train_stats.items():
                    if stats['count'] > 0:
                        repr_loss = stats['loss'] / stats['count']
                        repr_acc = stats['acc'] / stats['count']
                        print(f'  [{repr_type}] Train Loss: {repr_loss:.4f}, Train Acc: {repr_acc:.4f}, Batches: {stats["count"]}')
                
                if writer is not None:
                    log_dict = {"Stage2/loss": avg_loss, "Stage2/critic_loss": avg_critic_loss, "Stage2/acc": avg_acc}
                    if use_aux_loss:
                        log_dict["Stage2/aux_loss"] = avg_aux_loss
                    # 记录每个表征类型的训练指标
                    for repr_type, stats in repr_train_stats.items():
                        if stats['count'] > 0:
                            log_dict[f"Stage2/train_loss_{repr_type}"] = stats['loss'] / stats['count']
                            log_dict[f"Stage2/train_acc_{repr_type}"] = stats['acc'] / stats['count']
                    writer.log(log_dict, step=epoch)
        
        # Validation (按 eval_freq 频率评估)
        stage2_eval_freq = getattr(cfg, 'stage2_eval_freq', 1)
        if critic_val_loader is not None and (epoch % stage2_eval_freq == 0 or epoch == cfg.stage2_epochs - 1):
            model.eval()
            val_result = eval_critic(critic_val_loader, model, device=device, rank=rank)
            
            # Retrieval Retrieval Evaluation（与 multi-repr 对齐）
            # 注意：评估 Retrieval 时需要禁用 LoRA，以获得纯 Retrieval 的检索性能
            skip_retrieval_eval = getattr(cfg, 'skip_retrieval_eval_in_stage2', False)
            if test_loaders is not None and not skip_retrieval_eval:
                if rank == 0:
                    print(f"\n[Stage2 Epoch {epoch}] === Retrieval Retrieval Evaluation (LoRA disabled) ===")
                # 禁用 Critic LoRA
                disable_lora(actual_model.critic_lora_modules)
                for repr_type, test_loader in test_loaders.items():
                    eval_tmr(test_loader, model, epoch, repr_type=repr_type,
                             mode='M1T0', writer=writer, device=device, rank=rank)
                # 重新启用 Critic LoRA
                enable_lora(actual_model.critic_lora_modules)
                if rank == 0:
                    print(f"[Stage2 Epoch {epoch}] === Retrieval Retrieval Evaluation Done ===\n")
            
            if is_main_process(rank):
                print(f"[Stage2] Val - Loss: {val_result['loss']:.4f}, Acc: {val_result['acc']:.4f}")
                
                # 打印每个表征类型的指标
                repr_types = val_result.get('repr_types', [])
                for repr_type in repr_types:
                    acc_key = f'acc_{repr_type}'
                    loss_key = f'loss_{repr_type}'
                    count_key = f'count_{repr_type}'
                    if acc_key in val_result:
                        count = val_result.get(count_key, 'N/A')
                        print(f"  [{repr_type}] Acc: {val_result[acc_key]:.4f}, "
                              f"Loss: {val_result.get(loss_key, 0):.4f}, Count: {count}")
                
                # 记录验证指标
                if writer is not None:
                    log_dict = {
                        "Stage2/val_loss": val_result['loss'],
                        "Stage2/val_acc": val_result['acc']
                    }
                    # 记录每个表征类型的指标
                    for repr_type in repr_types:
                        acc_key = f'acc_{repr_type}'
                        loss_key = f'loss_{repr_type}'
                        if acc_key in val_result:
                            log_dict[f"Stage2/val_acc_{repr_type}"] = val_result[acc_key]
                        if loss_key in val_result:
                            log_dict[f"Stage2/val_loss_{repr_type}"] = val_result[loss_key]
                    writer.log(log_dict, step=epoch)
                
                if val_result['acc'] > best_critic_acc:
                    best_critic_acc = val_result['acc']
                    best_critic_epoch = epoch
                    
                    if full_finetune:
                        # 保存整模（与 multi-repr 命名对齐）
                        model_config = build_model_config_for_save(cfg, retrieval_repr_list)
                        critic_state = {
                            'state_dict': actual_model.state_dict(),
                            'epoch': epoch,
                            'acc': val_result['acc'],
                            'model_config': model_config,
                            'full_finetune': True
                        }
                        repr_str = '_'.join(retrieval_repr_list)
                        best_critic_ckpt_path = os.path.join(checkpoint_dir, f'Critic_FullFT_{repr_str}_best.pth')
                        torch.save(critic_state, best_critic_ckpt_path)
                        print(f"[Stage2] Best FullFT model saved: {best_critic_ckpt_path} (Acc={val_result['acc']:.4f})")
                    else:
                        # 分模块保存（LoRA 路径保持不变）
                        model_config = build_model_config_for_save(cfg, retrieval_repr_list)
                        model_config['lora_rank'] = cfg.lora_rank
                        model_config['lora_alpha'] = cfg.lora_alpha
                        metrics = {'critic_acc': val_result['acc'], 'epoch': epoch}
                        for repr_type in repr_types:
                            acc_key = f'acc_{repr_type}'
                            if acc_key in val_result:
                                metrics[f'critic_acc_{repr_type}'] = val_result[acc_key]
                        saved_paths = save_model_modular(
                            model, checkpoint_dir, f'{base_prefix}_stage2_best',
                            epoch, metrics, model_config, rank
                        )
                        best_critic_ckpt_path = saved_paths.get('critic_lora')
                        print(f"[Stage2] Best model saved (Acc={val_result['acc']:.4f})")
        
        if is_distributed:
            dist.barrier()
    
    if rank == 0:
        print("\n" + "="*60)
        print(f"[Stage 2] Complete! Best Acc: {best_critic_acc:.4f} at epoch {best_critic_epoch}")
        print("="*60 + "\n")
        if fptr:
            print(f"[Stage 2] Best Acc: {best_critic_acc:.4f} at epoch {best_critic_epoch}", file=fptr)
            fptr.flush()
    
    return best_critic_ckpt_path, best_critic_acc


def train_stage3_ai_detection_lora(cfg, model_class, ai_train_loader, ai_val_loader, ai_test_loader,
                                    device, rank, world_size, is_distributed, writer,
                                    retrieval_repr_list, best_retrieval_ckpt_path, fptr=None):
    """Stage 3: 训练 AI Detection LoRA + AI Detection Head（或全参训练）
    
    注意：这里需要重新创建模型以避免 LoRA 注入污染
    
    Args:
        cfg: 配置对象
        model_class: 模型类（用于创建新实例）
        ai_train_loader: AI Detection 训练数据加载器
        ai_val_loader: AI Detection 验证数据加载器
        ai_test_loader: AI Detection 测试数据加载器
        device: 设备
        rank: 进程 rank
        world_size: 进程总数
        is_distributed: 是否分布式训练
        writer: SwanLab writer
        retrieval_repr_list: 表征类型列表
        best_retrieval_ckpt_path: Stage 1 最佳模型路径
        fptr: 日志文件指针
        
    Returns:
        best_ai_ckpt_path: 最佳 AI Detection 模型路径
        best_ai_f1: 最佳 F1 分数
    """
    # 检查是否使用全参训练模式
    full_finetune = getattr(cfg, 'ai_full_finetune', False)
    
    # 重新创建模型用于 AI Detection 训练
    model_cfg = cfg.MODEL
    ai_model = model_class(
        t5_path=cfg.t5_path,
        temp=cfg.Retrieval.CLTemp,
        thr=cfg.Retrieval.CLThr,
        latent_dim=model_cfg.latent_dim,
        unified_dim=model_cfg.unified_dim,
        encoder_num_layers=model_cfg.encoder_num_layers,
        encoder_num_heads=model_cfg.encoder_num_heads,
        encoder_ff_size=model_cfg.encoder_ff_size,
        text_num_layers=model_cfg.text_num_layers,
        text_num_heads=model_cfg.text_num_heads,
        text_ff_size=model_cfg.text_ff_size,
        proj_hidden_dim=model_cfg.proj_hidden_dim,
        proj_num_layers=model_cfg.proj_num_layers,
        use_unified_dim=cfg.use_unified_dim,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout
    ).to(device)
    
    if rank == 0:
        timestamp = get_training_timestamp()
        print("\n" + "="*60)
        if full_finetune:
            print("[Stage 3] AI Detection Full Finetune (全参训练)")
        else:
            print("[Stage 3] AI Detection LoRA Training")
        print(f"  Epochs: {cfg.stage3_epochs}")
        print(f"  Learning Rate: {cfg.stage3_lr}")
        if not full_finetune:
            print(f"  LoRA Rank: {cfg.lora_rank}, Alpha: {cfg.lora_alpha}")
        print(f"  Timestamp: {timestamp}")
        if best_retrieval_ckpt_path:
            print(f"  Loading Retrieval from: {best_retrieval_ckpt_path}")
        else:
            print(f"  Retrieval: Random initialization (no pretrained weights)")
        print("="*60 + "\n")
    
    # 加载 Retrieval 权重（如果有，不加载 Critic LoRA）
    if best_retrieval_ckpt_path and os.path.exists(best_retrieval_ckpt_path):
        checkpoint = torch.load(best_retrieval_ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        # 只加载 Retrieval 主干，不加载 Critic 相关
        retrieval_state = {k: v for k, v in state_dict.items() if 'critic' not in k.lower()}
        ai_model.load_state_dict(retrieval_state, strict=False)
        if rank == 0:
            print(f"[Stage3] Loaded Retrieval weights from {best_retrieval_ckpt_path}")
    elif rank == 0:
        print(f"[Stage3] Training from scratch (no Retrieval checkpoint)")
    
    actual_model = ai_model
    
    if full_finetune:
        # 全参训练模式：训练整个 motion encoder + ai detection head
        # 不冻结任何参数，不注入 LoRA
        
        # 初始化 AI Detection Head（如果还没有）
        if actual_model.ai_detection_head is None:
            actual_model.init_ai_detection_head()
        
        # 所有参数都可训练（除了 T5 等预训练模型）
        for name, param in actual_model.named_parameters():
            if 't5' in name.lower() or 'clip' in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        actual_model = actual_model.to(device)
        
        if rank == 0:
            trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in actual_model.parameters())
            print(f"[Stage3] Full finetune - Trainable params: {trainable_params:,} / {total_params:,}")
    else:
        # LoRA 训练模式：只训练 LoRA + AI Detection Head
        
        # 先冻结所有参数（包括 ai_detection_* 分支的非 LoRA 部分）
        for _, param in actual_model.named_parameters():
            param.requires_grad = False
        
        # 注入 AI Detection LoRA（会创建独立 encoder + projections，并替换为 LoRA 模块）
        ai_lora_params = actual_model.inject_ai_detection_lora()
        
        # 仅解冻 LoRA 参数 + AI Detection head
        for name, param in actual_model.named_parameters():
            if ('lora' in name.lower()) or ('ai_detection_head' in name):
                param.requires_grad = True
        
        # 确保整个模型（包括新注入的 LoRA 模块）在正确的设备上
        actual_model = actual_model.to(device)
        
        if rank == 0:
            trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
            print(f"[Stage3] Trainable params (LoRA+Head only): {trainable_params:,}")
    
    model = actual_model
    
    # 重新包装 DDP
    if is_distributed:
        model = DDP(model, device_ids=[int(os.environ.get('LOCAL_RANK', 0))],
                   output_device=int(os.environ.get('LOCAL_RANK', 0)),
                   find_unused_parameters=True)
        actual_model = model.module
    
    # Optimizer
    trainable_params_list = [p for p in actual_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params_list, lr=cfg.stage3_lr, weight_decay=1e-4)
    
    lr_scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=cfg.stage3_epochs * len(ai_train_loader)
    )
    
    best_ai_f1 = 0.0
    best_ai_acc = 0.0
    best_ai_epoch = -1
    best_ai_ckpt_path = None
    
    checkpoint_dir = cfg.Retrieval_CHECKPOINT_DIR
    repr_str = '_'.join(retrieval_repr_list)
    
    # 支持自定义前缀
    base_prefix = getattr(cfg, 'checkpoint_prefix', None) or repr_str
    
    # 辅助损失权重（重建 + KL）
    lambda_aux = getattr(cfg, 'lambda_recons', 0.0)
    use_aux_loss = lambda_aux > 0
    if rank == 0 and use_aux_loss:
        print(f"[Stage3] Using auxiliary loss (recons + KL) with lambda={lambda_aux}")
    
    for epoch in range(cfg.stage3_epochs):
        if hasattr(ai_train_loader, 'batch_sampler') and hasattr(ai_train_loader.batch_sampler, 'set_epoch'):
            ai_train_loader.batch_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_acc = 0.0
        batch_count = 0
        
        # 按表征类型统计训练指标
        repr_train_stats = defaultdict(lambda: {'loss': 0.0, 'acc': 0.0, 'correct': 0, 'total': 0, 'count': 0})
        
        for batch in tqdm(ai_train_loader, desc=f'[Stage3] Epoch {epoch}', disable=rank != 0):
            repr_type = batch.get('repr_type', 'unknown')
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            if use_aux_loss:
                ai_logits, ai_labels, aux_loss = actual_model.forward_ai_detection(batch, return_aux_loss=True)
            else:
                ai_logits, ai_labels = actual_model.forward_ai_detection(batch, return_aux_loss=False)
                aux_loss = 0.0
            
            if ai_logits is not None:
                cls_loss = F.cross_entropy(ai_logits, ai_labels)
                acc = (ai_logits.argmax(dim=-1) == ai_labels).float().mean()
                
                # 总损失 = 分类损失 + 辅助损失（重建 + KL）
                if use_aux_loss:
                    loss = cls_loss + lambda_aux * aux_loss
                else:
                    loss = cls_loss
                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                epoch_cls_loss += cls_loss.item()
                if use_aux_loss:
                    epoch_aux_loss += aux_loss.item()
                epoch_acc += acc.item()
                batch_count += 1
                
                # 按表征类型统计
                batch_correct = (ai_logits.argmax(dim=-1) == ai_labels).sum().item()
                batch_total = len(ai_labels)
                repr_train_stats[repr_type]['loss'] += cls_loss.item()
                repr_train_stats[repr_type]['correct'] += batch_correct
                repr_train_stats[repr_type]['total'] += batch_total
                repr_train_stats[repr_type]['count'] += 1
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            avg_cls_loss = epoch_cls_loss / batch_count
            avg_aux_loss = epoch_aux_loss / batch_count if use_aux_loss else 0.0
            avg_acc = epoch_acc / batch_count
            if rank == 0:
                if use_aux_loss:
                    print(f'[Stage3] Epoch {epoch}, Loss: {avg_loss:.4f} (cls: {avg_cls_loss:.4f}, aux: {avg_aux_loss:.4f}), Acc: {avg_acc:.4f}')
                else:
                    print(f'[Stage3] Epoch {epoch}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')
                
                # 打印每个表征类型的训练指标
                for repr_type, stats in repr_train_stats.items():
                    if stats['count'] > 0:
                        repr_loss = stats['loss'] / stats['count']
                        repr_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
                        print(f'  [{repr_type}] Train Loss: {repr_loss:.4f}, Train Acc: {repr_acc:.4f}, Samples: {stats["total"]}')
                
                if writer is not None:
                    log_dict = {"Stage3/loss": avg_loss, "Stage3/cls_loss": avg_cls_loss, "Stage3/acc": avg_acc}
                    if use_aux_loss:
                        log_dict["Stage3/aux_loss"] = avg_aux_loss
                    # 记录每个表征类型的训练指标
                    for repr_type, stats in repr_train_stats.items():
                        if stats['count'] > 0:
                            log_dict[f"Stage3/train_loss_{repr_type}"] = stats['loss'] / stats['count']
                            log_dict[f"Stage3/train_acc_{repr_type}"] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
                    writer.log(log_dict, step=epoch)
        
        # Validation
        if ai_val_loader is not None:
            model.eval()
            val_result = eval_ai_detection(ai_val_loader, model, device=device, rank=rank)
            
            if is_main_process(rank):
                print(f"[Stage3] Val - Loss: {val_result['loss']:.4f}, Acc: {val_result['acc']:.4f}, F1: {val_result['f1']:.4f}")
                
                # 打印每个表征类型的指标
                repr_types = val_result.get('repr_types', [])
                for repr_type in repr_types:
                    acc_key = f'acc_{repr_type}'
                    f1_key = f'f1_{repr_type}'
                    count_key = f'count_{repr_type}'
                    if acc_key in val_result:
                        count = val_result.get(count_key, 'N/A')
                        f1_val = val_result.get(f1_key, 0)
                        print(f"  [{repr_type}] Acc: {val_result[acc_key]:.4f}, F1: {f1_val:.4f}, Count: {count}")
                
                # 记录验证指标
                if writer is not None:
                    log_dict = {
                        "Stage3/val_loss": val_result['loss'],
                        "Stage3/val_acc": val_result['acc'],
                        "Stage3/val_f1": val_result['f1']
                    }
                    # 记录每个表征类型的指标
                    for repr_type in repr_types:
                        acc_key = f'acc_{repr_type}'
                        f1_key = f'f1_{repr_type}'
                        if acc_key in val_result:
                            log_dict[f"Stage3/val_acc_{repr_type}"] = val_result[acc_key]
                        if f1_key in val_result:
                            log_dict[f"Stage3/val_f1_{repr_type}"] = val_result[f1_key]
                    writer.log(log_dict, step=epoch)
                
                if val_result['f1'] > best_ai_f1:
                    best_ai_f1 = val_result['f1']
                    best_ai_acc = val_result['acc']
                    best_ai_epoch = epoch
                    
                    # 分模块保存最佳模型
                    model_config = build_model_config_for_save(cfg, retrieval_repr_list)
                    model_config['lora_rank'] = cfg.lora_rank
                    model_config['lora_alpha'] = cfg.lora_alpha
                    
                    # 保存所有表征的指标
                    metrics = {'ai_f1': val_result['f1'], 'ai_acc': val_result['acc'], 'epoch': epoch}
                    for repr_type in repr_types:
                        acc_key = f'acc_{repr_type}'
                        f1_key = f'f1_{repr_type}'
                        if acc_key in val_result:
                            metrics[f'ai_acc_{repr_type}'] = val_result[acc_key]
                        if f1_key in val_result:
                            metrics[f'ai_f1_{repr_type}'] = val_result[f1_key]
                    
                    saved_paths = save_model_modular(
                        model, checkpoint_dir, f'{base_prefix}_stage3_best',
                        epoch, metrics, model_config, rank
                    )
                    best_ai_ckpt_path = saved_paths.get('ai_detection_lora')
                    print(f"[Stage3] Best model saved (F1={val_result['f1']:.4f})")
        
        if is_distributed:
            dist.barrier()
    
    # Test evaluation
    if ai_test_loader is not None:
        model.eval()
        test_result = eval_ai_detection(ai_test_loader, model, device=device, rank=rank)
        
        if is_main_process(rank):
            print("\n" + "-"*60)
            print("[Stage3] AI Detection Test Set Evaluation:")
            print(f"  Loss: {test_result['loss']:.4f}, Acc: {test_result['acc']:.4f}")
            print(f"  Precision: {test_result['precision']:.4f}, Recall: {test_result['recall']:.4f}, F1: {test_result['f1']:.4f}")
            
            # 打印每个表征类型的测试指标
            repr_types = test_result.get('repr_types', [])
            if repr_types:
                print("  Per-representation results:")
                for repr_type in repr_types:
                    acc_key = f'acc_{repr_type}'
                    f1_key = f'f1_{repr_type}'
                    count_key = f'count_{repr_type}'
                    if acc_key in test_result:
                        count = test_result.get(count_key, 'N/A')
                        f1_val = test_result.get(f1_key, 0)
                        print(f"    [{repr_type}] Acc: {test_result[acc_key]:.4f}, F1: {f1_val:.4f}, Count: {count}")
            print("-"*60)
    
    if rank == 0:
        print("\n" + "="*60)
        print(f"[Stage 3] Complete! Best F1: {best_ai_f1:.4f}, Acc: {best_ai_acc:.4f} at epoch {best_ai_epoch}")
        print("="*60 + "\n")
        if fptr:
            print(f"[Stage 3] Best F1: {best_ai_f1:.4f}, Acc: {best_ai_acc:.4f} at epoch {best_ai_epoch}", file=fptr)
            fptr.flush()
    
    return best_ai_ckpt_path, best_ai_f1
