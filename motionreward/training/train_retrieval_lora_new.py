"""
MotionReward - Retrieval + LoRA 训练脚本（重构版）

新架构：Retrieval 检索为主任务（全参数），Critic 和 AI 检测使用 LoRA（极少参数）

核心思路：
1. Retrieval（检索任务）作为主任务，使用完整的 motion encoder 全参数训练
2. Critic（评分任务）和 AI Detection（检测任务）复用 motion encoder，但只训练 LoRA 适配器
3. 不同任务使用独立的 LoRA 参数，避免信息混合

三阶段训练：
    Stage 1: 训练 Retrieval（全参数），保存最佳检索模型
    Stage 2: 冻结 Retrieval 主干，训练 Critic LoRA + Critic Head
    Stage 3: 冻结 Retrieval 主干，训练 AI Detection LoRA + AI Detection Head

Usage:
    # 三阶段训练
    CUDA_VISIBLE_DEVICES=2 python -m motionreward.training.train_retrieval_lora_new \\
        --stage1_epochs 50 --stage2_epochs 30 --stage3_epochs 30 \\
        --lora_rank 16 --lora_alpha 32 --use_retrieval_packed

    # 多卡训练
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29502 \\
        -m motionreward.training.train_retrieval_lora_new \\
        --stage1_epochs 50 --stage2_epochs 30 --stage3_epochs 30 --use_retrieval_packed

完全对齐 train_retrieval_lora.py 的运行行为
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import DDPMScheduler

try:
    import swanlab
except ImportError:
    swanlab = None
    print("Warning: swanlab not installed, logging disabled")

# 导入模块化组件
from motionreward.models import MultiReprRetrievalWithLoRA
from .trainers import (
    train_stage1_retrieval,
    train_stage2_critic_lora,
    train_stage3_ai_detection_lora,
)
from motionreward.utils import (
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    parse_args_lora,
    create_retrieval_dataloaders,
    create_ai_detection_dataloaders,
    create_paired_dataloaders,
)
from .trainers.lora_trainer import get_training_timestamp
from motionreward.models.lora_modules import load_lora_state_dict
from torch.utils.data import DataLoader
from motionreward.datasets import CriticPairDataset, CriticReprTypeBatchSampler, critic_collate_fn
from motionreward.evaluation import eval_tmr, eval_critic, eval_ai_detection, calculate_retrieval_metrics_small_batches

# 日志文件
os.makedirs('./logs', exist_ok=True)
fptr = open('./logs/retrieval_lora.log', 'a')


def create_model(cfg, device, rank=0):
    """创建模型
    
    Args:
        cfg: 配置对象
        device: 设备
        rank: 进程 rank
    
    Returns:
        model: 模型实例
    """
    model_cfg = cfg.MODEL
    model = MultiReprRetrievalWithLoRA(
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
    )
    
    if rank == 0:
        print(f"Model size: {cfg.model_size}")
        print(f"  latent_dim={model_cfg.latent_dim}, unified_dim={model_cfg.unified_dim}")
        print(f"  LoRA rank={cfg.lora_rank}, alpha={cfg.lora_alpha}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {total_params:,}")
    
    return model.to(device)


def create_noise_scheduler(cfg, device):
    """创建噪声调度器和采样概率
    
    Args:
        cfg: 配置对象
        device: 设备
    
    Returns:
        tuple: (scheduler, probs)
    """
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        variance_type="fixed_small",
        clip_sample=False
    )
    
    maxT = cfg.Retrieval.maxT
    probs = torch.zeros(maxT, device=device)
    probs[:maxT//10] = 0.8 / maxT
    probs[maxT//10:] = 0.2 / (maxT - 101)
    
    return scheduler, probs


def init_swanlab(cfg, retrieval_repr_list, rank):
    """初始化 SwanLab 日志
    
    Args:
        cfg: 配置对象
        retrieval_repr_list: 表征类型列表
        rank: 进程 rank
    
    Returns:
        writer: SwanLab writer 或 None
    """
    if not is_main_process(rank):
        return None
    
    if swanlab is None:
        print("[SwanLab] Not installed, logging disabled")
        return None
    
    config_dict = dict(cfg)
    repr_str = '_'.join(retrieval_repr_list)
    
    # 支持自定义实验名称，默认使用 Retrieval_LoRA_{repr_str}
    exp_name = getattr(cfg, 'exp_name', None) or f"Retrieval_LoRA_{repr_str}"
    
    try:
        writer = swanlab.init(
            project="MotionReward-Retrieval-LoRA",
            experiment_name=exp_name,
            config=config_dict
        )
        print(f"[SwanLab] Initialized: project=MotionReward-Retrieval-LoRA, exp={exp_name}")
        return writer
    except Exception as e:
        print(f"[SwanLab] Failed to initialize: {e}")
        return None


def print_summary(rank, retrieval_metrics, best_retrieval_ckpt_path, 
                  best_critic_ckpt_path, best_critic_acc,
                  best_ai_ckpt_path, best_ai_f1, fptr, timestamp):
    """打印训练总结
    
    Args:
        rank: 进程 rank
        retrieval_metrics: Retrieval 指标
        best_retrieval_ckpt_path: Retrieval 最佳模型路径
        best_critic_ckpt_path: Critic 最佳模型路径
        best_critic_acc: Critic 最佳准确率
        best_ai_ckpt_path: AI Detection 最佳模型路径
        best_ai_f1: AI Detection 最佳 F1
        fptr: 日志文件指针
        timestamp: 训练时间戳
    """
    if rank != 0:
        return
    
    # 安全获取 Retrieval R1 指标（使用 bs32）
    retrieval_r1 = 0.0
    if isinstance(retrieval_metrics, dict):
        if 'bs32' in retrieval_metrics and isinstance(retrieval_metrics['bs32'], dict):
            retrieval_r1 = retrieval_metrics['bs32'].get('r1', 0.0)
        elif 'r1' in retrieval_metrics:
            retrieval_r1 = retrieval_metrics.get('r1', 0.0)
        elif 'R1' in retrieval_metrics:
            retrieval_r1 = retrieval_metrics.get('R1', 0.0)
    
    print("\n" + "="*60)
    print("[Training Complete]")
    print(f"  Timestamp: {timestamp}")
    print(f"  Stage 1 (Retrieval): Best BS32 R1 = {retrieval_r1:.3f}%")
    print(f"    Checkpoint: {best_retrieval_ckpt_path}")
    if best_critic_ckpt_path:
        print(f"  Stage 2 (Critic LoRA): Best Acc = {best_critic_acc:.4f}")
        print(f"    Checkpoint: {best_critic_ckpt_path}")
    if best_ai_ckpt_path:
        print(f"  Stage 3 (AI Detection LoRA): Best F1 = {best_ai_f1:.4f}")
        print(f"    Checkpoint: {best_ai_ckpt_path}")
    print("="*60 + "\n")
    
    print(f"\n[Summary] Timestamp: {timestamp}", file=fptr)
    print(f"  Stage 1: BS32 R1={retrieval_r1:.3f}%, ckpt={best_retrieval_ckpt_path}", file=fptr)
    if best_critic_ckpt_path:
        print(f"  Stage 2: Acc={best_critic_acc:.4f}, ckpt={best_critic_ckpt_path}", file=fptr)
    if best_ai_ckpt_path:
        print(f"  Stage 3: F1={best_ai_f1:.4f}, ckpt={best_ai_ckpt_path}", file=fptr)
    fptr.flush()


def load_lora_checkpoint(model, lora_ckpt_path, lora_type, device, rank=0, head_ckpt_path=None):
    """加载 LoRA checkpoint
    
    Args:
        model: 模型（可能是 DDP 包装的）
        lora_ckpt_path: LoRA checkpoint 路径
        lora_type: 'critic' 或 'ai_detection'
        device: 设备
        rank: 进程 rank
        head_ckpt_path: 独立的 head checkpoint 路径（可选）
    """
    actual_model = model.module if hasattr(model, 'module') else model
    
    if not os.path.exists(lora_ckpt_path):
        if rank == 0:
            print(f"[Warning] LoRA checkpoint not found: {lora_ckpt_path}")
        return False
    
    checkpoint = torch.load(lora_ckpt_path, map_location=device, weights_only=False)
    
    if lora_type == 'critic':
        # 注入 Critic LoRA
        if actual_model.critic_lora_modules is None:
            actual_model.inject_critic_lora()
        
        # 加载 LoRA 参数
        load_lora_state_dict(actual_model.critic_lora_modules, checkpoint['lora_state_dict'])
        
        # 加载 Critic Head
        head_loaded = False
        if head_ckpt_path and os.path.exists(head_ckpt_path):
            # 从独立的 head checkpoint 加载
            head_checkpoint = torch.load(head_ckpt_path, map_location=device, weights_only=False)
            if 'state_dict' in head_checkpoint:
                actual_model.critic_head.load_state_dict(head_checkpoint['state_dict'])
                head_loaded = True
                if rank == 0:
                    print(f"[Eval] Loaded Critic Head from: {head_ckpt_path}")
        elif 'head_state_dict' in checkpoint and actual_model.critic_head is not None:
            # 从 LoRA checkpoint 中加载（旧格式）
            actual_model.critic_head.load_state_dict(checkpoint['head_state_dict'])
            head_loaded = True
            if rank == 0:
                print(f"[Eval] Loaded Critic Head from LoRA checkpoint")
        
        if not head_loaded and rank == 0:
            print(f"[Warning] Critic Head not loaded - using random initialization")
        
        if rank == 0:
            print(f"[Eval] Loaded Critic LoRA from: {lora_ckpt_path}")
            
    elif lora_type == 'ai_detection':
        # 注入 AI Detection LoRA（创建独立的 encoder 副本）
        if not hasattr(actual_model, 'ai_detection_lora_modules') or actual_model.ai_detection_lora_modules is None:
            actual_model.inject_ai_detection_lora()
        
        # 加载 LoRA 参数
        load_lora_state_dict(actual_model.ai_detection_lora_modules, checkpoint['lora_state_dict'])
        
        # 加载 AI Detection Head
        head_loaded = False
        if head_ckpt_path and os.path.exists(head_ckpt_path):
            # 从独立的 head checkpoint 加载
            head_checkpoint = torch.load(head_ckpt_path, map_location=device, weights_only=False)
            if 'state_dict' in head_checkpoint:
                actual_model.ai_detection_head.load_state_dict(head_checkpoint['state_dict'])
                head_loaded = True
                if rank == 0:
                    print(f"[Eval] Loaded AI Detection Head from: {head_ckpt_path}")
        elif 'head_state_dict' in checkpoint and actual_model.ai_detection_head is not None:
            # 从 LoRA checkpoint 中加载（旧格式）
            actual_model.ai_detection_head.load_state_dict(checkpoint['head_state_dict'])
            head_loaded = True
            if rank == 0:
                print(f"[Eval] Loaded AI Detection Head from LoRA checkpoint")
        
        if not head_loaded and rank == 0:
            print(f"[Warning] AI Detection Head not loaded - using random initialization")
        
        if rank == 0:
            print(f"[Eval] Loaded AI Detection LoRA from: {lora_ckpt_path}")
    
    return True


def run_evaluation(cfg, model, test_loaders, device, rank, retrieval_repr_list, fptr):
    """运行评估模式 - 为每个任务创建独立的模型实例
    
    重要：Critic LoRA 会修改 encoder 结构（替换为 LoRALinear/LoRAMultiheadAttention），
    因此不能在同一个模型上切换 LoRA。必须为每个任务创建新的模型实例。
    
    评估流程：
    - Retrieval 检索：使用当前模型（纯 backbone）
    - Critic 任务：创建新模型，加载 backbone + Critic LoRA
    - AI Detection 任务：创建新模型，加载 backbone + AI Detection LoRA
    
    Args:
        cfg: 配置对象
        model: 模型（用于 Retrieval 评估）
        test_loaders: Retrieval 测试数据加载器
        device: 设备
        rank: 进程 rank
        retrieval_repr_list: 表征类型列表
        fptr: 日志文件指针
    """
    actual_model = model.module if hasattr(model, 'module') else model
    
    if rank == 0:
        print("\n" + "="*60)
        print("[Evaluation Mode]")
        print("="*60)
        print("\n  重要：每个任务使用独立的模型实例")
        print("  - Retrieval: 纯 backbone")
        print("  - Critic: 新模型 + Critic LoRA")
        print("  - AI Detection: 新模型 + AI Detection LoRA")
    
    results = {}
    
    # ==================== 1. 加载 Stage 1 主干权重（用于 Retrieval 评估）====================
    if rank == 0:
        print("\n[Step 1] Loading Retrieval backbone for retrieval evaluation...")
    
    if cfg.resume_stage1_ckpt and os.path.exists(cfg.resume_stage1_ckpt):
        checkpoint = torch.load(cfg.resume_stage1_ckpt, map_location=device, weights_only=False)
        actual_model.load_state_dict(checkpoint['state_dict'], strict=False)
        if rank == 0:
            print(f"  ✓ Retrieval backbone: {cfg.resume_stage1_ckpt}")
    else:
        if rank == 0:
            print("  ✗ [Warning] No Retrieval backbone checkpoint specified!")
    
    model.eval()
    
    # ==================== 2.1 评估 Retrieval 检索任务（纯 backbone，无 LoRA）====================
    if cfg.eval_retrieval or (not cfg.eval_retrieval and not cfg.eval_critic and not cfg.eval_ai_detection):
        if rank == 0:
            print("\n" + "-"*40)
            print("[Step 2.1] Retrieval Retrieval Task (No LoRA)")
            print("-"*40)
        
        retrieval_results = {}
        for repr_type, test_loader in test_loaders.items():
            if rank == 0:
                print(f"\n  Evaluating {repr_type}...")
            
            # 收集 latents
            text_list, text_latents, motion_latents = [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    texts = batch.get('text', [])
                    motion = batch['motion']
                    lengths = batch.get('length', batch.get('m_len', [motion.shape[1]] * motion.shape[0]))
                    if isinstance(lengths, torch.Tensor):
                        lengths = lengths.tolist()
                    
                    t_latent = actual_model.get_text_embedding(texts)
                    m_latent = actual_model.get_motion_embedding(motion, lengths, repr_type=repr_type)
                    
                    text_list.extend(texts)
                    text_latents.extend(t_latent.cpu().numpy())
                    motion_latents.extend(m_latent.cpu().numpy())
            
            test_result = [text_list, text_latents, motion_latents]
            bs32_metrics = calculate_retrieval_metrics_small_batches(test_result, epoch=0, fptr=fptr)
            
            retrieval_results[repr_type] = {'bs32': bs32_metrics}
            
            if rank == 0:
                print(f"  {repr_type} - BS32: R@1={bs32_metrics['R1']:.2f}%, R@2={bs32_metrics['R2']:.2f}%, R@3={bs32_metrics['R3']:.2f}%, R@5={bs32_metrics['R5']:.2f}%, R@10={bs32_metrics['R10']:.2f}%")
        
        results['retrieval'] = retrieval_results
    
    # ==================== 2.2 评估 Critic 任务（创建新模型 + Critic LoRA）====================
    if cfg.eval_critic or (not cfg.eval_retrieval and not cfg.eval_critic and not cfg.eval_ai_detection):
        if rank == 0:
            print("\n" + "-"*40)
            print("[Step 2.2] Critic Scoring Task (New Model + Critic LoRA)")
            print("-"*40)
        
        if cfg.resume_critic_lora_ckpt and os.path.exists(cfg.resume_critic_lora_ckpt):
            # 创建新的模型实例用于 Critic 评估
            if rank == 0:
                print("  Creating new model instance for Critic evaluation...")
            
            model_cfg = cfg.MODEL
            critic_model = MultiReprRetrievalWithLoRA(
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
            
            # 加载 Retrieval backbone
            if cfg.resume_stage1_ckpt and os.path.exists(cfg.resume_stage1_ckpt):
                checkpoint = torch.load(cfg.resume_stage1_ckpt, map_location=device, weights_only=False)
                critic_model.load_state_dict(checkpoint['state_dict'], strict=False)
                if rank == 0:
                    print(f"  ✓ Loaded Retrieval backbone")
            
            # 注入并加载 Critic LoRA
            critic_loaded = load_lora_checkpoint(critic_model, cfg.resume_critic_lora_ckpt, 'critic', device, rank, 
                                               head_ckpt_path=getattr(cfg, 'resume_critic_head_ckpt', None))
            
            if critic_loaded:
                if rank == 0:
                    print(f"  ✓ Critic LoRA loaded: {cfg.resume_critic_lora_ckpt}")
                
                critic_model.eval()
                
                # 创建 Critic 训练/验证数据加载器（评估时也跑训练集以输出各表征的 train loss/acc）
                from motionreward.utils import create_critic_dataloaders
                critic_train_loader, critic_val_loader = create_critic_dataloaders(cfg, retrieval_repr_list, rank, 1)
                
                train_result = None
                if critic_train_loader is not None:
                    train_result = eval_critic(critic_train_loader, critic_model, device=device, rank=rank)
                    results['critic_train'] = train_result
                
                val_result = None
                if critic_val_loader is not None:
                    val_result = eval_critic(critic_val_loader, critic_model, device=device, rank=rank)
                    results['critic'] = val_result  # 保持原有 key 兼容后续汇总
                
                if rank == 0:
                    print(f"\n  Critic Evaluation Results (Eval-only Mode):")
                    if train_result is not None:
                        print(f"    [Train] Overall Acc: {train_result['acc']:.4f}, Loss: {train_result['loss']:.4f}")
                        print(f"[Eval] Critic Train: Acc={train_result['acc']:.4f}, Loss={train_result['loss']:.4f}", file=fptr)
                        for repr_type in train_result.get('repr_types', []):
                            acc_key = f'acc_{repr_type}'
                            loss_key = f'loss_{repr_type}'
                            if acc_key in train_result:
                                print(f"      {repr_type}: Train Acc={train_result[acc_key]:.4f}, Train Loss={train_result.get(loss_key, 0):.4f}")
                                print(f"[Eval] Critic Train: [{repr_type}] Acc={train_result[acc_key]:.4f}, Loss={train_result.get(loss_key, 0):.4f}", file=fptr)
                        fptr.flush()
                    else:
                        print("    [Train] No critic train data available")

                    if val_result is not None:
                        print(f"    [Val]   Overall Acc: {val_result['acc']:.4f}, Loss: {val_result['loss']:.4f}")
                        print(f"[Eval] Critic Val: Acc={val_result['acc']:.4f}, Loss={val_result['loss']:.4f}", file=fptr)
                        for repr_type in val_result.get('repr_types', []):
                            acc_key = f'acc_{repr_type}'
                            loss_key = f'loss_{repr_type}'
                            if acc_key in val_result:
                                print(f"      {repr_type}: Val Acc={val_result[acc_key]:.4f}, Val Loss={val_result.get(loss_key, 0):.4f}")
                                print(f"[Eval] Critic Val: [{repr_type}] Acc={val_result[acc_key]:.4f}, Loss={val_result.get(loss_key, 0):.4f}", file=fptr)
                        fptr.flush()
                    else:
                        print("    [Val]   No critic validation data available")
                
            # 清理 Critic 模型
            del critic_model
            torch.cuda.empty_cache()
        else:
            if rank == 0:
                print("  [Skip] No Critic LoRA checkpoint provided")
    
    # ==================== 2.3 评估 AI Detection 任务（创建新模型 + AI Detection LoRA）====================
    if cfg.eval_ai_detection or (not cfg.eval_retrieval and not cfg.eval_critic and not cfg.eval_ai_detection):
        if rank == 0:
            print("\n" + "-"*40)
            print("[Step 2.3] AI Detection Task (New Model + AI Detection LoRA)")
            print("-"*40)
        
        if cfg.resume_ai_detection_lora_ckpt and os.path.exists(cfg.resume_ai_detection_lora_ckpt):
            # 创建新的模型实例用于 AI Detection 评估
            if rank == 0:
                print("  Creating new model instance for AI Detection evaluation...")
            
            model_cfg = cfg.MODEL
            ai_model = MultiReprRetrievalWithLoRA(
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
            
            # 加载 Retrieval backbone
            if cfg.resume_stage1_ckpt and os.path.exists(cfg.resume_stage1_ckpt):
                checkpoint = torch.load(cfg.resume_stage1_ckpt, map_location=device, weights_only=False)
                ai_model.load_state_dict(checkpoint['state_dict'], strict=False)
                if rank == 0:
                    print(f"  ✓ Loaded Retrieval backbone")
            
            # 注入并加载 AI Detection LoRA
            ai_detection_loaded = load_lora_checkpoint(ai_model, cfg.resume_ai_detection_lora_ckpt, 'ai_detection', device, rank,
                                                      head_ckpt_path=getattr(cfg, 'resume_ai_detection_head_ckpt', None))
            
            if ai_detection_loaded:
                if rank == 0:
                    print(f"  ✓ AI Detection LoRA loaded: {cfg.resume_ai_detection_lora_ckpt}")
                
                ai_model.eval()
                
                # 创建 AI Detection 测试数据加载器
                from motionreward.utils import create_ai_detection_dataloaders
                _, ai_val_loader, ai_test_loader = create_ai_detection_dataloaders(cfg, retrieval_repr_list, rank, 1)
                
                eval_loader = ai_test_loader if ai_test_loader is not None else ai_val_loader
                
                if eval_loader is not None:
                    result = eval_ai_detection(eval_loader, ai_model, device=device, rank=rank)
                    results['ai_detection'] = result
                    
                    if rank == 0:
                        print(f"\n  AI Detection Evaluation Results:")
                        print(f"    Accuracy: {result['acc']:.4f}")
                        print(f"    Precision: {result['precision']:.4f}")
                        print(f"    Recall: {result['recall']:.4f}")
                        print(f"    F1 Score: {result['f1']:.4f}")
                        print(f"    Confusion: TP={result['tp']}, FP={result['fp']}, FN={result['fn']}, TN={result['tn']}")
                        
                        for repr_type in result.get('repr_types', []):
                            f1_key = f'f1_{repr_type}'
                            acc_key = f'acc_{repr_type}'
                            if f1_key in result:
                                print(f"    {repr_type}: Acc={result.get(acc_key, 0):.4f}, F1={result[f1_key]:.4f}")
                        
                        print(f"\n[Eval] AI Detection: Acc={result['acc']:.4f}, F1={result['f1']:.4f}", file=fptr)
                else:
                    if rank == 0:
                        print("  [Warning] No AI detection test data available")
            
            # 清理 AI Detection 模型
            del ai_model
            torch.cuda.empty_cache()
        else:
            if rank == 0:
                print("  [Skip] No AI Detection LoRA checkpoint provided")
    
    # ==================== 3. 打印总结 ====================
    if rank == 0:
        print("\n" + "="*60)
        print("[Evaluation Summary]")
        print("="*60)
        
        if 'retrieval' in results:
            print("\n  Retrieval Retrieval (BS32):")
            for repr_type, metrics in results['retrieval'].items():
                bs32 = metrics['bs32']
                print(f"    {repr_type}: R@1={bs32['R1']:.2f}%, R@2={bs32['R2']:.2f}%, R@3={bs32['R3']:.2f}%, R@5={bs32['R5']:.2f}%, R@10={bs32['R10']:.2f}%")
        
        if 'critic' in results:
            print(f"\n  Critic: Acc={results['critic']['acc']:.4f}")
        
        if 'ai_detection' in results:
            print(f"\n  AI Detection: Acc={results['ai_detection']['acc']:.4f}, F1={results['ai_detection']['f1']:.4f}")
        
        print("\n" + "="*60)
        print("[Evaluation Complete]")
        print("="*60 + "\n")
    
    fptr.flush()
    return results


def main():
    # 解析配置
    cfg = parse_args_lora()
    
    # 设置种子
    from motionreward.utils import set_seed
    set_seed(cfg.SEED_VALUE)
    
    # 获取训练时间戳（整个训练会话使用同一个时间戳）
    timestamp = get_training_timestamp()
    
    # Setup DDP
    rank, world_size, local_rank, is_distributed = setup_ddp()
    
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
        if rank == 0:
            print(f"Using DDP with {world_size} GPUs")
            print(f"Training timestamp: {timestamp}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using single GPU: {device}")
        print(f"Training timestamp: {timestamp}")
    
    # 解析表征类型
    retrieval_repr_list = [r.strip() for r in cfg.retrieval_repr_types.split(',') if r.strip()]
    if rank == 0:
        print(f"Training with representations: {retrieval_repr_list}")
    
    # 创建模型
    model = create_model(cfg, device, rank)
    
    # 重新设置种子，确保数据加载的随机性一致
    # （模型初始化可能消耗了随机数）
    set_seed(cfg.SEED_VALUE)
    
    # 判断是否需要 Stage 1 训练数据
    # 如果跳过 Stage 1 或者从 checkpoint 恢复，只需要测试集用于验证
    need_retrieval_train_data = (
        cfg.stage1_epochs > 0 and 
        not cfg.skip_stage1 and 
        not (cfg.resume_stage1_ckpt and os.path.exists(cfg.resume_stage1_ckpt))
    )
    
    # 创建数据加载器
    if need_retrieval_train_data:
        # Stage 1 需要训练：加载完整的训练集和测试集
        train_loader, test_loaders = create_retrieval_dataloaders(
            cfg, retrieval_repr_list, rank, world_size, is_distributed
        )
    else:
        # 跳过 Stage 1：只加载测试集用于验证
        if rank == 0:
            print("[Retrieval] Skipping Stage 1 training data loading (only loading test set for validation)")
        train_loader = None
        from motionreward.utils import create_retrieval_test_loaders_only
        test_loaders = create_retrieval_test_loaders_only(cfg, retrieval_repr_list, rank)
    
    # 初始化日志
    writer = init_swanlab(cfg, retrieval_repr_list, rank)
    
    # DDP 包装
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # 创建噪声调度器
    scheduler, probs = create_noise_scheduler(cfg, device)
    
    # ==================== 评估模式 ====================
    if cfg.eval_only:
        run_evaluation(cfg, model, test_loaders, device, rank, retrieval_repr_list, fptr)
        cleanup_ddp()
        return
    
    # ==================== Stage 1: Retrieval Training ====================
    best_retrieval_ckpt_path = None
    retrieval_metrics = {'full': {'r1': 0.0}}
    
    # 检查是否从 Stage 1 checkpoint 恢复
    if cfg.resume_stage1_ckpt and os.path.exists(cfg.resume_stage1_ckpt):
        if rank == 0:
            print(f"\n[Resume] Loading Stage 1 checkpoint from: {cfg.resume_stage1_ckpt}")
        
        actual_model = model.module if hasattr(model, 'module') else model
        checkpoint = torch.load(cfg.resume_stage1_ckpt, map_location=device, weights_only=False)
        actual_model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        best_retrieval_ckpt_path = cfg.resume_stage1_ckpt
        if 'metrics' in checkpoint:
            retrieval_metrics = checkpoint.get('metrics', retrieval_metrics)
        
        if rank == 0:
            print(f"[Resume] Successfully loaded Retrieval backbone")
            if 'metrics' in checkpoint:
                print(f"[Resume] Loaded metrics: R1={retrieval_metrics.get('full', {}).get('r1', 'N/A')}")
    
    # 是否跳过 Stage 1
    if cfg.skip_stage1 or (cfg.resume_stage1_ckpt and os.path.exists(cfg.resume_stage1_ckpt)):
        if rank == 0:
            print("[Stage 1] Skipped - Using loaded checkpoint")
    else:
        # 创建配对数据加载器（跨表征对齐）
        paired_train_loader = None
        use_cross_repr = getattr(cfg, 'use_cross_repr_align', False)
        if use_cross_repr:
            paired_train_loader = create_paired_dataloaders(
                cfg, retrieval_repr_list, rank, world_size
            )
            if paired_train_loader is None and rank == 0:
                print("[Warning] Cross-repr alignment requested but paired data not available")
        
        best_retrieval_ckpt_path, retrieval_metrics = train_stage1_retrieval(
            cfg, model, train_loader, test_loaders, device, rank, world_size,
            is_distributed, writer, retrieval_repr_list, scheduler, probs, fptr,
            paired_train_loader=paired_train_loader
        )
    
    # ==================== Stage 2: Critic LoRA Training ====================
    best_critic_ckpt_path = None
    best_critic_acc = 0.0
    
    if cfg.skip_stage2:
        if rank == 0:
            print("[Stage 2] Skipped by --skip_stage2")
    else:
        # 使用 create_critic_dataloaders 函数，支持新格式目录数据
        from motionreward.utils import create_critic_dataloaders
        critic_train_loader, critic_val_loader = create_critic_dataloaders(
            cfg, retrieval_repr_list, rank, world_size
        )
        
        if critic_train_loader is not None:
            best_critic_ckpt_path, best_critic_acc = train_stage2_critic_lora(
                cfg, model, critic_train_loader, critic_val_loader, test_loaders,
                device, rank, world_size, is_distributed, writer,
                retrieval_repr_list, best_retrieval_ckpt_path, fptr
            )
        else:
            if rank == 0:
                print("[Stage 2] Skipped - No critic data available")
    
    # ==================== Stage 3: AI Detection LoRA Training ====================
    best_ai_ckpt_path = None
    best_ai_f1 = 0.0
    
    if cfg.skip_stage3:
        if rank == 0:
            print("[Stage 3] Skipped by --skip_stage3")
    else:
        ai_train_loader, ai_val_loader, ai_test_loader = create_ai_detection_dataloaders(
            cfg, retrieval_repr_list, rank, world_size
        )
        
        if ai_train_loader is not None:
            best_ai_ckpt_path, best_ai_f1 = train_stage3_ai_detection_lora(
                cfg, MultiReprRetrievalWithLoRA, ai_train_loader, ai_val_loader, ai_test_loader,
                device, rank, world_size, is_distributed, writer,
                retrieval_repr_list, best_retrieval_ckpt_path, fptr
            )
        else:
            if rank == 0:
                print("[Stage 3] Skipped - No AI detection data available")
    
    # 打印总结
    print_summary(
        rank, retrieval_metrics, best_retrieval_ckpt_path,
        best_critic_ckpt_path, best_critic_acc,
        best_ai_ckpt_path, best_ai_f1, fptr, timestamp
    )
    
    cleanup_ddp()


if __name__ == "__main__":
    main()
