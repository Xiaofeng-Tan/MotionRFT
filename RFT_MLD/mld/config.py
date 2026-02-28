import os
import importlib
from typing import Type, TypeVar
from argparse import ArgumentParser

from omegaconf import OmegaConf, DictConfig


def get_module_config(cfg_model: DictConfig, paths: list[str], cfg_root: str) -> DictConfig:
    files = [os.path.join(cfg_root, 'modules', p+'.yaml') for p in paths]
    for file in files:
        assert os.path.exists(file), f'{file} is not exists.'
        with open(file, 'r') as f:
            cfg_model.merge_with(OmegaConf.load(f))
    return cfg_model


def get_obj_from_str(string: str, reload: bool = False) -> Type:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: DictConfig) -> TypeVar:
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    
# def parse_args() -> DictConfig:
#     parser = ArgumentParser()
#     parser.add_argument("--cfg", type=str, required=True, help="The main config file")
#     parser.add_argument('--example', type=str, required=False, help="The input texts and lengths with txt format")
#     parser.add_argument('--example_hint', type=str, required=False, help="The input hint ids and lengths with txt format")
#     parser.add_argument('--no-plot', action="store_true", required=False, help="Whether to plot the skeleton-based motion")
#     parser.add_argument('--replication', type=int, default=1, help="The number of replications of sampling")
#     parser.add_argument('--vis', type=str, default="swanlab", choices=['tb', 'swanlab'], help="The visualization backends: tensorboard or swanlab")
#     parser.add_argument('--optimize', action='store_true', help="Enable optimization for motion control")
#     args = parser.parse_args()

#     cfg = OmegaConf.load(args.cfg)
#     cfg_root = os.path.dirname(args.cfg)
#     cfg_model = get_module_config(cfg.model, cfg.model.target, cfg_root)
#     cfg = OmegaConf.merge(cfg, cfg_model)

#     cfg.example = args.example
#     cfg.example_hint = args.example_hint
#     cfg.no_plot = args.no_plot
#     cfg.replication = args.replication
#     cfg.vis = args.vis
#     cfg.optimize = args.optimize
#     return cfg

def parse_args() -> DictConfig:
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="The main config file")
    parser.add_argument('--example', type=str, required=False, help="The input texts and lengths with txt format")
    parser.add_argument('--example_hint', type=str, required=False, help="The input hint ids and lengths with txt format")
    parser.add_argument('--ckpt', type=str, required=False, help="The input hint ids and lengths with txt format")
    parser.add_argument('--no-plot', action="store_true", required=False, help="Whether to plot the skeleton-based motion")
    parser.add_argument('--replication', type=int, default=1, help="The number of replications of sampling")
    parser.add_argument('--vis', type=str, default="swanlab", choices=['tb', 'swanlab'], help="The visualization backends: tensorboard or swanlab")
    parser.add_argument('--optimize', action='store_true', help="Enable optimization for motion control")
    
    parser.add_argument('--spm_path', type=str, required=False, help="")
    parser.add_argument('--mld_path', type=str, required=False, help="")
    parser.add_argument('--reward_model_size', type=str, default='base',
                        choices=['retrieval_original', 'tiny', 'small', 'base', 'large', 'xlarge', 'xxlarge', 'giant'],
                        help="MotionReward model size for configuration")
    parser.add_argument('--reward_lora_rank', type=int, default=128,
                        help="LoRA rank for MotionReward (must match checkpoint)")
    parser.add_argument('--reward_lora_alpha', type=int, default=256,
                        help="LoRA alpha for MotionReward (must match checkpoint)")

    parser.add_argument('--ft_m', type=int, default=None)
    parser.add_argument('--ft_prob', type=float, default=None)
    parser.add_argument('--ft_t', type=int, default=None)
    parser.add_argument('--ft_dy', type=int, default=2)
    parser.add_argument('--ft_k', type=int, default=None)
    parser.add_argument('--ft_skip', type=str2bool, default=False)
    parser.add_argument('--ft_reverse', type=str2bool, default=False)
    parser.add_argument('--ft_custom', type=str, default=None)
    parser.add_argument('--ft_type', type=str, default=None, choices=['ReFL', 'DRaFT', 'DRTune', 'AlignProp', 'NIPS', 'None'])
    parser.add_argument('--ft_lambda_reward', type=float, default=None)
    parser.add_argument('--ft_lambda_diff', type=float, default=None)
    parser.add_argument('--ft_lr', type=float, default=None)
    parser.add_argument('--ft_epochs', type=int, default=4, help='Number of training epochs (default: 4)')
    parser.add_argument('--ft_name', type=str, default='ft_mld_motionreward')
    
    # MotionReward 相关参数
    parser.add_argument('--reward_stage1_ckpt', type=str, default=None,
                        help="Stage 1 backbone checkpoint path for MotionReward")
    parser.add_argument('--eval_reward_before_ft', type=str2bool, default=True,
                        help="Whether to evaluate reward model before fine-tuning")
    parser.add_argument('--debug', action='store_true',
                        help="Debug mode: load only a small subset of data for quick testing")
    parser.add_argument('--no_debug_vis', action='store_true',
                        help="Disable debug visualization during RFT training")
    
    # Critic Reward 相关参数 (Stage 2)
    parser.add_argument('--critic_backbone_ckpt', type=str, default='',
                        help="Stage 1/2 backbone checkpoint for Critic reward")
    parser.add_argument('--critic_lora_ckpt', type=str, default='',
                        help="Stage 2 Critic LoRA checkpoint")
    parser.add_argument('--critic_head_ckpt', type=str, default='',
                        help="Stage 2 Critic Head checkpoint")
    parser.add_argument('--lambda_critic', type=float, default=1.0,
                        help="Weight for Critic reward (Stage 2)")
    parser.add_argument('--lambda_retrieval', type=float, default=0.0,
                        help="Weight for Retrieval reward (Stage 1)")
    parser.add_argument('--lambda_m2m', type=float, default=0.0,
                        help="Weight for M2M reward (GT vs Generated motion similarity)")
    parser.add_argument('--lambda_ai_detection', type=float, default=0.0,
                        help="Weight for AI Detection reward (Stage 3)")
    parser.add_argument('--ai_detection_lora_ckpt', type=str, default='',
                        help="Stage 3 AI Detection LoRA checkpoint")
    parser.add_argument('--ai_detection_head_ckpt', type=str, default='',
                        help="Stage 3 AI Detection Head checkpoint")
    parser.add_argument('--no_save', action='store_true',
                        help="Disable saving checkpoints")
    parser.add_argument('--validation_steps', type=int, default=100,
                        help="Validation frequency (steps)")
    parser.add_argument('--reward_max_t', type=int, default=500,
                        help="Max timestep seen during MotionReward training (used to clamp timestep in RFT reward)")
    parser.add_argument('--reward_t_switch', type=int, default=50,
                        help="NIPS reward strategy switch point (step index). "
                             "i < reward_t_switch: predict x_0, reward=R(x_0,0); "
                             "i >= reward_t_switch: use x_t directly, reward=R(x_t,t). "
                             "Default=50 means all steps use x_0 prediction.")
    parser.add_argument('--curriculum', action='store_true',
                        help="Enable Motion Reward timestep scheduling for NIPS. "
                             "First sweep_ratio%% training sweeps all steps (high→low noise), "
                             "remaining time focuses on final k steps (low noise).")
    parser.add_argument('--sweep_ratio', type=float, default=0.03,
                        help="Fraction of training for curriculum sweep phase (default: 0.03). "
                             "0.0 = 100%% time on final k steps (low noise only), "
                             "0.03 = 3%% sweep + 97%% final k steps.")
    parser.add_argument('--fid_save_threshold', type=float, default=0.15,
                        help="Only save checkpoints when FID < this threshold (default: 0.15)")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    cfg_root = os.path.dirname(args.cfg)
    cfg_model = get_module_config(cfg.model, cfg.model.target, cfg_root)
    cfg = OmegaConf.merge(cfg, cfg_model)

    cfg.example = args.example
    cfg.example_hint = args.example_hint
    cfg.no_plot = args.no_plot
    cfg.replication = args.replication
    cfg.vis = args.vis
    cfg.optimize = args.optimize
    cfg.mld_path = args.mld_path
    if args.spm_path is not None:
        # 绝对路径保持原样，相对路径去掉 .pth 扩展名（后续会加回来）
        if os.path.isabs(args.spm_path):
            cfg.spm_path = args.spm_path
        else:
            cfg.spm_path = args.spm_path.replace('.pth', '')
    if args.mld_path is not None:
        cfg.mld_path = args.mld_path.replace('.ckpt', '')
    cfg.ft_type = args.ft_type
    cfg.ft_m = args.ft_m

    cfg.ft_prob = args.ft_prob
    cfg.ft_t = args.ft_t
    cfg.ft_k = args.ft_k
    cfg.ft_skip = args.ft_skip
    cfg.ft_reverse = args.ft_reverse
    cfg.ft_custom = args.ft_custom
    cfg.ft_lambda_reward = args.ft_lambda_reward
    cfg.ft_lambda_diff = args.ft_lambda_diff
    cfg.ft_lr = args.ft_lr
    cfg.ft_dy = args.ft_dy
    cfg.ft_name = args.ft_name
    cfg.ft_epochs = args.ft_epochs
    cfg.reward_model_size = args.reward_model_size
    cfg.reward_stage1_ckpt = args.reward_stage1_ckpt
    cfg.eval_reward_before_ft = args.eval_reward_before_ft
    cfg.debug = args.debug
    cfg.no_debug_vis = args.no_debug_vis
    
    # Critic 相关配置
    cfg.critic_backbone_ckpt = args.critic_backbone_ckpt
    cfg.critic_lora_ckpt = args.critic_lora_ckpt
    cfg.critic_head_ckpt = args.critic_head_ckpt
    cfg.lambda_critic = args.lambda_critic
    cfg.lambda_retrieval = args.lambda_retrieval
    cfg.lambda_m2m = args.lambda_m2m
    cfg.lambda_ai_detection = args.lambda_ai_detection
    cfg.ai_detection_lora_ckpt = args.ai_detection_lora_ckpt
    cfg.ai_detection_head_ckpt = args.ai_detection_head_ckpt
    cfg.no_save = args.no_save
    cfg.validation_steps = args.validation_steps
    cfg.reward_max_t = args.reward_max_t
    cfg.reward_t_switch = args.reward_t_switch
    cfg.curriculum = args.curriculum
    cfg.sweep_ratio = args.sweep_ratio
    cfg.fid_save_threshold = args.fid_save_threshold
    return cfg



def parse_args_RM() -> DictConfig:
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="The main config file")

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate, default=1e-4')
    parser.add_argument('--NoiseThr', type=float, default=5, help='')
    parser.add_argument('--CLThr', type=float, default=0.9, help='')
    parser.add_argument('--CLTemp', type=float, default=0.1, help='')
    parser.add_argument('--step_aware', type=str, default='M0T0', help='')
    parser.add_argument('--maxT', type=int, default=1000, help='')
    
    parser.add_argument('--lambda_t2m', type=float, default=0, help='')
    parser.add_argument('--lambda_m2m', type=float, default=0, help='')
    parser.add_argument('--finetune', type=str, default=None, help='')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)
    cfg_root = os.path.dirname(args.cfg)
    cfg_model = get_module_config(cfg.model, cfg.model.target, cfg_root)
    cfg = OmegaConf.merge(cfg, cfg_model)
    
    cfg.lr = args.lr
    cfg.NoiseThr = args.NoiseThr
    cfg.CLThr = args.CLThr
    cfg.CLTemp = args.CLTemp
    cfg.maxT = args.maxT
    cfg.step_aware = args.step_aware
    cfg.lambda_t2m = args.lambda_t2m
    if 'pth' not in args.finetune:
        cfg.finetune = None
    else:
        cfg.finetune = args.finetune
    cfg.lambda_m2m = args.lambda_m2m
    assert cfg.step_aware in ['M0T0', 'M1T0', 'M0T1', 'M1T1'], 'Error Mode'
    assert cfg.maxT in [i for i in range(1001)], 'Error MaxT'
    return cfg

