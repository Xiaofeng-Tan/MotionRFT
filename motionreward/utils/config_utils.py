"""
配置工具函数
"""

import os
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf


# 项目路径
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CFG = os.path.join(os.path.dirname(MODULE_DIR), 'configs', 'retrieval_multi_repr.yaml')


def get_model_config(model_size):
    """获取模型配置
    
    关键对齐点（与原版 Retrieval 一致）：
    - 原版 Retrieval: latent_dim=256, num_layers=9, num_heads=4, ff_size=1024
    
    新增 unified_dim:
    - 当 use_unified_dim=True 时，投影层输出到 unified_dim，再经过编码器映射到 latent_dim
    - 当 use_unified_dim=False 时，投影层直接输出到 latent_dim（原版行为）
    
    Args:
        model_size: 模型规模名称
    
    Returns:
        dict: 模型配置字典
    """
    configs = {
        # 完全对齐原版 Retrieval 的配置
        'retrieval_original': {
            'latent_dim': 256,
            'unified_dim': 256,
            'encoder_num_layers': 9,
            'encoder_num_heads': 4,
            'encoder_ff_size': 1024,
            'text_num_layers': 9,
            'text_num_heads': 4,
            'text_ff_size': 1024,
            'proj_hidden_dim': 256,
            'proj_num_layers': 1,
        },
        'tiny': {
            'latent_dim': 128,
            'unified_dim': 256,
            'encoder_num_layers': 3,
            'encoder_num_heads': 4,
            'encoder_ff_size': 512,
            'text_num_layers': 3,
            'text_num_heads': 4,
            'text_ff_size': 512,
            'proj_hidden_dim': 256,
            'proj_num_layers': 2,
        },
        'small': {
            'latent_dim': 192,
            'unified_dim': 384,
            'encoder_num_layers': 5,
            'encoder_num_heads': 6,
            'encoder_ff_size': 768,
            'text_num_layers': 5,
            'text_num_heads': 6,
            'text_ff_size': 768,
            'proj_hidden_dim': 384,
            'proj_num_layers': 2,
        },
        'base': {
            'latent_dim': 256,
            'unified_dim': 512,
            'encoder_num_layers': 9,
            'encoder_num_heads': 4,
            'encoder_ff_size': 1024,
            'text_num_layers': 9,
            'text_num_heads': 4,
            'text_ff_size': 1024,
            'proj_hidden_dim': 512,
            'proj_num_layers': 3,
        },
        'large': {
            'latent_dim': 384,
            'unified_dim': 768,
            'encoder_num_layers': 11,
            'encoder_num_heads': 6,
            'encoder_ff_size': 1536,
            'text_num_layers': 11,
            'text_num_heads': 6,
            'text_ff_size': 1536,
            'proj_hidden_dim': 768,
            'proj_num_layers': 3,
        },
        'xlarge': {
            'latent_dim': 512,
            'unified_dim': 1024,
            'encoder_num_layers': 11,
            'encoder_num_heads': 8,
            'encoder_ff_size': 2048,
            'text_num_layers': 11,
            'text_num_heads': 8,
            'text_ff_size': 2048,
            'proj_hidden_dim': 1024,
            'proj_num_layers': 3,
        },
        'xxlarge': {
            'latent_dim': 384,
            'unified_dim': 768,
            'encoder_num_layers': 11,
            'encoder_num_heads': 12,
            'encoder_ff_size': 3072,
            'text_num_layers': 11,
            'text_num_heads': 6,
            'text_ff_size': 3072,
            'proj_hidden_dim': 768,
            'proj_num_layers': 3,
        },
        'giant': {
            'latent_dim': 448,
            'unified_dim': 896,
            'encoder_num_layers': 13,
            'encoder_num_heads': 14,
            'encoder_ff_size': 3584,
            'text_num_layers': 13,
            'text_num_heads': 7,
            'text_ff_size': 3584,
            'proj_hidden_dim': 896,
            'proj_num_layers': 3,
        },
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model_size: {model_size}. Choose from {list(configs.keys())}")
    
    return configs[model_size]


def parse_args():
    """解析命令行参数
    
    Returns:
        OmegaConf: 配置对象
    """
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG, help="Config file path")
    
    # 模型规模
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['retrieval_original', 'tiny', 'small', 'base', 'large', 'xlarge', 'xxlarge', 'giant'],
                        help='Model size preset')
    
    # 其他参数省略...（完整实现见原文件）
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = OmegaConf.load(args.cfg)
    
    # 应用模型规模配置
    model_cfg = get_model_config(args.model_size)
    cfg.MODEL = OmegaConf.create(model_cfg)
    cfg.model_size = args.model_size
    
    return cfg


def parse_args_lora():
    """解析 LoRA 版本训练的命令行参数
    
    Returns:
        OmegaConf: 配置对象
    """
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG, help="Config file path")
    
    # Debug 模式
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--debug_samples', type=int, default=200)
    
    # 模型规模
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['retrieval_original', 'tiny', 'small', 'base', 'large', 'xlarge', 'xxlarge', 'giant'])
    
    # LoRA 配置
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    
    # 三阶段训练配置
    parser.add_argument('--stage1_epochs', type=int, default=50)
    parser.add_argument('--stage2_epochs', type=int, default=30)
    parser.add_argument('--stage3_epochs', type=int, default=30)
    parser.add_argument('--stage2_eval_freq', type=int, default=1,
                        help='Stage 2 evaluation frequency (epochs), default=1 means every epoch')
    parser.add_argument('--skip_retrieval_eval_in_stage2', action='store_true', default=False,
                        help='Skip Retrieval evaluation during Stage 2 training')
    parser.add_argument('--stage1_lr', type=float, default=1e-4)
    parser.add_argument('--stage2_lr', type=float, default=1e-4)
    parser.add_argument('--stage3_lr', type=float, default=1e-4)
    parser.add_argument('--lambda_recons', type=float, default=0.0)
    
    # 表征类型
    parser.add_argument('--retrieval_repr_types', type=str, default='263')
    parser.add_argument('--use_unified_dim', action='store_true', default=True)
    parser.add_argument('--no_unified_dim', action='store_true', default=False)
    
    # 数据配置
    parser.add_argument('--use_retrieval_packed', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--flowmdm_datapath', type=str, default=None)
    parser.add_argument('--retrieval_packed_path', type=str, default=None,
                        help='Path to packed retrieval data directory (contains retrieval_train.pth, retrieval_val.pth, retrieval_test.pth)')
    
    # 其他
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--exp_note', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--checkpoint_prefix', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    
    # 断点续训/跳过阶段
    parser.add_argument('--resume_stage1_ckpt', type=str, default=None)
    parser.add_argument('--skip_stage1', action='store_true', default=False)
    parser.add_argument('--skip_stage2', action='store_true', default=False)
    parser.add_argument('--skip_stage3', action='store_true', default=False)
    
    # 全参训练模式
    parser.add_argument('--retrieval_full_finetune', action='store_true', default=False)
    parser.add_argument('--critic_full_finetune', action='store_true', default=False)
    parser.add_argument('--ai_full_finetune', action='store_true', default=False)
    
    # Critic 数据格式
    parser.add_argument('--use_new_critic_data', action='store_true', default=True)
    parser.add_argument('--use_old_critic_data', action='store_true', default=False)
    
    # 评估模式
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_retrieval', action='store_true', default=False)
    parser.add_argument('--eval_critic', action='store_true', default=False)
    parser.add_argument('--eval_ai_detection', action='store_true', default=False)
    parser.add_argument('--resume_critic_lora_ckpt', type=str, default=None)
    parser.add_argument('--resume_critic_head_ckpt', type=str, default=None)
    parser.add_argument('--resume_ai_detection_lora_ckpt', type=str, default=None)
    parser.add_argument('--resume_ai_detection_head_ckpt', type=str, default=None)
    
    # 跨表征对齐
    parser.add_argument('--use_cross_repr_align', action='store_true', default=False,
                        help='Enable cross-representation alignment in Stage 1')
    parser.add_argument('--lambda_cross_repr', type=float, default=0.1,
                        help='Weight for cross-representation alignment loss')
    
    # Retrieval 参数
    parser.add_argument('--NoiseThr', type=float, default=None)
    parser.add_argument('--step_aware', type=str, default=None)
    parser.add_argument('--maxT', type=int, default=None)
    
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = OmegaConf.load(args.cfg)
    
    # 应用模型规模配置
    model_cfg = get_model_config(args.model_size)
    cfg.MODEL = OmegaConf.create(model_cfg)
    cfg.model_size = args.model_size
    
    # LoRA 配置
    cfg.lora_rank = args.lora_rank
    cfg.lora_alpha = args.lora_alpha
    cfg.lora_dropout = args.lora_dropout
    
    # 三阶段配置
    cfg.stage1_epochs = args.stage1_epochs
    cfg.stage2_epochs = args.stage2_epochs
    cfg.stage3_epochs = args.stage3_epochs
    cfg.stage2_eval_freq = args.stage2_eval_freq
    cfg.skip_retrieval_eval_in_stage2 = args.skip_retrieval_eval_in_stage2
    cfg.stage1_lr = args.stage1_lr
    cfg.stage2_lr = args.stage2_lr
    cfg.stage3_lr = args.stage3_lr
    cfg.lambda_recons = args.lambda_recons
    
    # 命令行覆盖
    if args.NoiseThr is not None:
        cfg.Retrieval.NoiseThr = args.NoiseThr
    if args.maxT is not None:
        cfg.Retrieval.maxT = args.maxT
    if args.step_aware is not None:
        cfg.Retrieval.step_aware = args.step_aware
    if args.batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    
    cfg.retrieval_repr_types = args.retrieval_repr_types
    cfg.use_unified_dim = args.use_unified_dim and not args.no_unified_dim
    cfg.use_cross_repr_align = args.use_cross_repr_align
    cfg.lambda_cross_repr = args.lambda_cross_repr
    cfg.use_retrieval_packed = args.use_retrieval_packed
    cfg.flowmdm_datapath = args.flowmdm_datapath
    cfg.retrieval_packed_path = args.retrieval_packed_path
    
    # 如果指定了 retrieval_packed_path，自动设置打包数据路径
    if args.retrieval_packed_path:
        packed_dir = args.retrieval_packed_path
        cfg.retrieval_packed_train = os.path.join(packed_dir, 'retrieval_train.pth')
        cfg.retrieval_packed_val = os.path.join(packed_dir, 'retrieval_val.pth')
        cfg.retrieval_packed_test = os.path.join(packed_dir, 'retrieval_test.pth')
    
    cfg.exp_name = args.exp_name
    cfg.exp_note = args.exp_note
    cfg.SEED_VALUE = args.seed  # 随机种子
    
    # Checkpoint 保存路径配置
    if args.checkpoint_dir is not None:
        cfg.Retrieval_CHECKPOINT_DIR = args.checkpoint_dir
    cfg.checkpoint_prefix = args.checkpoint_prefix
    
    # 断点续训/跳过阶段配置
    cfg.resume_stage1_ckpt = args.resume_stage1_ckpt
    cfg.skip_stage1 = args.skip_stage1
    cfg.skip_stage2 = args.skip_stage2
    cfg.skip_stage3 = args.skip_stage3
    
    # 评估模式配置
    cfg.eval_only = args.eval_only
    cfg.eval_retrieval = args.eval_retrieval
    cfg.eval_critic = args.eval_critic
    cfg.eval_ai_detection = args.eval_ai_detection
    cfg.resume_critic_lora_ckpt = args.resume_critic_lora_ckpt
    cfg.resume_critic_head_ckpt = args.resume_critic_head_ckpt
    cfg.resume_ai_detection_lora_ckpt = args.resume_ai_detection_lora_ckpt
    cfg.resume_ai_detection_head_ckpt = args.resume_ai_detection_head_ckpt
    
    # 全参训练模式
    cfg.retrieval_full_finetune = args.retrieval_full_finetune
    cfg.critic_full_finetune = args.critic_full_finetune
    cfg.ai_full_finetune = args.ai_full_finetune
    
    # Debug 模式配置
    cfg.debug = args.debug
    cfg.debug_samples = args.debug_samples
    if args.debug:
        cfg.stage1_epochs = min(cfg.stage1_epochs, 2)
        cfg.stage2_epochs = min(cfg.stage2_epochs, 2)
        cfg.stage3_epochs = min(cfg.stage3_epochs, 2)
        cfg.TRAIN.BATCH_SIZE = min(cfg.TRAIN.BATCH_SIZE, 8)
        cfg.TRAIN.num_workers = 0
    
    # 默认数据路径
    cfg.retrieval_packed_train = os.path.join(PROJ_DIR, 'datasets/retrieval_packed', 'retrieval_train.pth')
    cfg.retrieval_packed_val = os.path.join(PROJ_DIR, 'datasets/retrieval_packed', 'retrieval_val.pth')
    
    # Critic 数据路径 (旧格式 - 目录格式)
    cfg.critic_train_data_263 = os.path.join(PROJ_DIR, 'datasets', 'critic_data', 'train_humanml3d.pth')
    cfg.critic_train_data_22x3 = os.path.join(PROJ_DIR, 'datasets', 'critic_data', 'train_22x3.pth')
    cfg.critic_val_data_263 = os.path.join(PROJ_DIR, 'datasets', 'critic_data', 'val_humanml3d.pth')
    cfg.critic_val_data_22x3 = os.path.join(PROJ_DIR, 'datasets', 'critic_data', 'val_22x3.pth')
    cfg.critic_data_dir_263 = os.path.join(PROJ_DIR, 'datasets', 'critic', 'datasets_humanml3d_263')
    cfg.critic_data_dir_66 = os.path.join(PROJ_DIR, 'datasets', 'critic', 'datasets_positions_66')
    cfg.critic_data_dir_135 = os.path.join(PROJ_DIR, 'datasets', 'critic', 'datasets_rot6d_135')
    cfg.use_new_critic_data = args.use_new_critic_data and not args.use_old_critic_data
    
    # Critic 数据路径 (从 MotionCritic 转换的单文件格式)
    cfg.critic_converted_train_263 = os.path.join(PROJ_DIR, 'datasets', 'critic', 'critic_train_263.pth')
    cfg.critic_converted_train_22x3 = os.path.join(PROJ_DIR, 'datasets', 'critic', 'critic_train_22x3.pth')
    cfg.critic_converted_train_135 = os.path.join(PROJ_DIR, 'datasets', 'critic', 'critic_train_135.pth')
    cfg.critic_converted_val_263 = os.path.join(PROJ_DIR, 'datasets', 'critic', 'critic_val_263.pth')
    cfg.critic_converted_val_22x3 = os.path.join(PROJ_DIR, 'datasets', 'critic', 'critic_val_22x3.pth')
    cfg.critic_converted_val_135 = os.path.join(PROJ_DIR, 'datasets', 'critic', 'critic_val_135.pth')
    
    cfg.ai_packed_train = os.path.join(PROJ_DIR, 'datasets/ai_detection_packed', 'ai_generated_train.pth')
    cfg.ai_packed_val = os.path.join(PROJ_DIR, 'datasets/ai_detection_packed', 'ai_generated_val.pth')
    cfg.ai_packed_test = os.path.join(PROJ_DIR, 'datasets/ai_detection_packed', 'ai_generated_test.pth')
    
    return cfg


def build_model_config_for_save(cfg, retrieval_repr_list):
    """构建用于保存到 checkpoint 的模型配置字典"""
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


def get_checkpoint_config_info(ckpt_path):
    """从 checkpoint 中提取模型配置信息"""
    if ckpt_path is None or not os.path.exists(ckpt_path):
        return {}
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'model_config' in ckpt:
            return ckpt['model_config']
        return {}
    except Exception:
        return {}


def print_checkpoint_info(ckpt_path, rank=0):
    """打印 checkpoint 的配置信息"""
    if rank != 0:
        return
    config_info = get_checkpoint_config_info(ckpt_path)
    print("\n" + "=" * 60)
    print("[Checkpoint Info]")
    print(f"Path: {ckpt_path}")
    if config_info:
        print("\n[Saved Model Config]")
        for key, value in config_info.items():
            print(f"  {key}: {value}")
    else:
        print("\n[Warning] No model_config saved in checkpoint (old format)")
    print("=" * 60 + "\n")


def check_critic_data(cfg, rank=0):
    """检查 Critic 训练数据是否存在"""
    if not cfg.use_critic:
        return True
    retrieval_repr_list = [r.strip() for r in cfg.retrieval_repr_types.split(',') if r.strip()]
    found_files = []
    if '263' in retrieval_repr_list:
        if cfg.critic_train_data_263 and os.path.exists(cfg.critic_train_data_263):
            found_files.append(f"263 train: {cfg.critic_train_data_263}")
    if '22x3' in retrieval_repr_list:
        if cfg.critic_train_data_22x3 and os.path.exists(cfg.critic_train_data_22x3):
            found_files.append(f"22x3 train: {cfg.critic_train_data_22x3}")
    return len(found_files) > 0
