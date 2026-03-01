"""
MotionReward Datasets Module

包含所有训练数据集类、collate 函数和 sampler
"""

from .retrieval_datasets import (
    Text2MotionDataset263,
    Text2MotionDataset135,
    JointLevelText2MotionDataset,
    PackedText2MotionDataset,
    PackedPairedReprDataset,
    PairedReprDataset,
    retrieval_collate_fn,
    paired_collate_fn,
    ReprTypeBatchSampler,
)

from .critic_datasets import (
    CriticPairDataset,
    critic_collate_fn,
    CriticReprTypeBatchSampler,
)

from .ai_detection_datasets import (
    AIDetectionDataset,
    ai_detection_collate_fn,
    AIDetectionReprTypeBatchSampler,
)

__all__ = [
    # Retrieval datasets
    'Text2MotionDataset263',
    'Text2MotionDataset135',
    'JointLevelText2MotionDataset',
    'PackedText2MotionDataset',
    'PackedPairedReprDataset',
    'PairedReprDataset',
    'retrieval_collate_fn',
    'paired_collate_fn',
    'ReprTypeBatchSampler',
    # Critic datasets
    'CriticPairDataset',
    'critic_collate_fn',
    'CriticReprTypeBatchSampler',
    # AI Detection datasets
    'AIDetectionDataset',
    'ai_detection_collate_fn',
    'AIDetectionReprTypeBatchSampler',
]
