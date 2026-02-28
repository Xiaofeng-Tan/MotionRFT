"""MotionReward Configs - Configuration files."""

import os

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(CONFIG_DIR, "retrieval_multi_repr.yaml")

__all__ = ["CONFIG_DIR", "DEFAULT_CONFIG"]
