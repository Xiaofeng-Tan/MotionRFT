"""
MotionReward - Motion Reward Modeling Framework

A unified framework for motion-to-text retrieval, motion quality scoring,
and AI-generated motion detection.
"""

__version__ = "0.1.0"

# Core models
from .models.retrieval import Retrieval, process_T5_outputs
from .models.utils import lengths_to_mask, mld_collate_motion_only

# Multi-representation models
from .models import MultiReprRetrieval, MultiReprRetrievalWithLoRA

# Utils
from .utils.common import set_seed, print_table, move_batch_to_device, count_parameters

__all__ = [
    # Core models
    "Retrieval",
    "process_T5_outputs",
    "lengths_to_mask",
    "mld_collate_motion_only",
    # Multi-representation models
    "MultiReprRetrieval",
    "MultiReprRetrievalWithLoRA",
    # Utils
    "set_seed",
    "print_table",
    "move_batch_to_device",
    "count_parameters",
]
