"""
MotionReward Training Module

重构后的模块化结构：
- trainers/: 训练器函数

注意：以下模块已移动到 motionreward 根目录：
- datasets/ -> motionreward/datasets/
- models/ -> motionreward/models/（合并）
- evaluation/ -> motionreward/evaluation/
- utils/ -> motionreward/utils/（合并）
"""

from . import trainers

__all__ = [
    'trainers',
]
