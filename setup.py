"""Setup script for motionreward package."""

from setuptools import setup, find_packages

setup(
    name="motionreward",
    version="0.1.0",
    description="Motion Reward Modeling Framework - Multi-representation motion learning",
    author="MotionReward Team",
    packages=find_packages(exclude=["tests", "datasets", "checkpoints", "deps", "logs"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9",
        "numpy",
        "tqdm",
        "omegaconf",
        "sentence-transformers",
        "diffusers",
        "transformers",
        "einops",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
        ],
        "logging": [
            "swanlab",
        ],
    },
    entry_points={
        "console_scripts": [
            "motionreward-train=motionreward.training.train_retrieval_multi_repr:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
