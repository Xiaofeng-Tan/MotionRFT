"""
MotionReward Training Trainers Module

包含各阶段训练器函数
"""

from .retrieval_trainer import train_retrieval_epoch
from .critic_trainer import train_critic_phase
from .ai_detection_trainer import train_ai_detection_phase
from .multi_stage_trainer import train_multi_stage
from .lora_trainer import (
    train_stage1_retrieval,
    train_stage2_critic_lora,
    train_stage3_ai_detection_lora,
    get_training_timestamp,
    save_model_modular,
    load_model_modular,
)

__all__ = [
    # Retrieval 训练
    'train_retrieval_epoch',
    # Critic 训练
    'train_critic_phase',
    # AI Detection 训练
    'train_ai_detection_phase',
    # 多阶段训练
    'train_multi_stage',
    # LoRA 三阶段训练
    'train_stage1_retrieval',
    'train_stage2_critic_lora',
    'train_stage3_ai_detection_lora',
    # 工具函数
    'get_training_timestamp',
    'save_model_modular',
    'load_model_modular',
]
