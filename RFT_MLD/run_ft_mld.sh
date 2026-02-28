#!/bin/bash
# RFT_MLD 训练脚本
# 使用 MotionReward 作为 reward model 对 MLD 进行微调

# 切换到 RFT_MLD 目录
cd "$(dirname "$0")"

# 设置环境变量
export TOKENIZERS_PARALLELISM=false

# 默认参数
FT_TYPE=${FT_TYPE:-"NIPS"}
FT_K=${FT_K:-10}
FT_LR=${FT_LR:-1e-5}
FT_LAMBDA_REWARD=${FT_LAMBDA_REWARD:-1.0}
FT_EPOCHS=${FT_EPOCHS:-4}  # 默认训练 4 个 epoch
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-"../checkpoints/motionreward/stage1_retrieval_backbone.pth"}
# MotionReward 模型规模: retrieval_original, tiny, small, base, large, xlarge, xxlarge, giant
REWARD_MODEL_SIZE=${REWARD_MODEL_SIZE:-"small"}

# 运行训练
python ft_mld.py \
    --cfg configs/ft_mld_t2m.yaml \
    --ft_type ${FT_TYPE} \
    --ft_k ${FT_K} \
    --ft_lr ${FT_LR} \
    --ft_lambda_reward ${FT_LAMBDA_REWARD} \
    --ft_epochs ${FT_EPOCHS} \
    --spm_path "${REWARD_MODEL_PATH}" \
    --reward_model_size "${REWARD_MODEL_SIZE}" \
    --vis swanlab \
    "$@"