#!/bin/bash
# RFT_MLD 训练脚本 (Critic Reward)
# 使用 Critic Head 分数作为 reward 对 MLD 进行微调
# 支持组合 Critic reward 和 Retrieval reward

# 切换到 RFT_MLD 目录
cd "$(dirname "$0")"

# 设置环境变量
export TOKENIZERS_PARALLELISM=false

# 默认参数
FT_TYPE=${FT_TYPE:-"NIPS"}
FT_M=${FT_M:-10}
FT_LR=${FT_LR:-1e-5}
FT_LAMBDA_REWARD=${FT_LAMBDA_REWARD:-1.0}
FT_EPOCHS=${FT_EPOCHS:-20}
FT_K=${FT_K:-10}

# Reward 权重配置
# lambda_critic: Critic reward (Stage 2) 权重
# lambda_retrieval: Retrieval reward (Stage 1) 权重
# 使用两个 reward 示例: LAMBDA_CRITIC=0.02 LAMBDA_RETRIEVAL=1.0
LAMBDA_CRITIC=${LAMBDA_CRITIC:-0.00000000000000002}
LAMBDA_RETRIEVAL=${LAMBDA_RETRIEVAL:-1.0}

# 是否保存模型 (设置为 true 禁用保存)
NO_SAVE=${NO_SAVE:-false}

# Critic Reward 模型配置
REWARD_MODEL_SIZE=${REWARD_MODEL_SIZE:-"small"}
CRITIC_BACKBONE_CKPT=${CRITIC_BACKBONE_CKPT:-"../checkpoints/motionreward/stage1_retrieval_backbone.pth"}
CRITIC_LORA_CKPT=${CRITIC_LORA_CKPT:-"../checkpoints/motionreward/stage2_critic_lora.pth"}
CRITIC_HEAD_CKPT=${CRITIC_HEAD_CKPT:-"../checkpoints/motionreward/stage2_critic_head.pth"}

# 打印配置信息
echo "=============================================="
echo "Reward Configuration:"
echo "  LAMBDA_CRITIC=${LAMBDA_CRITIC}"
echo "  LAMBDA_RETRIEVAL=${LAMBDA_RETRIEVAL}"
echo "  REWARD_MODEL_SIZE=${REWARD_MODEL_SIZE}"
echo "=============================================="

# 构建额外参数
EXTRA_ARGS=""
if [ "$NO_SAVE" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --no_save"
fi

# 运行训练
python motionrft_mld.py \
    --cfg configs/ft_mld_t2m.yaml \
    --ft_type ${FT_TYPE} \
    --ft_m ${FT_M} \
    --ft_k ${FT_K} \
    --ft_lr ${FT_LR} \
    --ft_lambda_reward ${FT_LAMBDA_REWARD} \
    --ft_epochs ${FT_EPOCHS} \
    --reward_model_size "${REWARD_MODEL_SIZE}" \
    --critic_backbone_ckpt "${CRITIC_BACKBONE_CKPT}" \
    --critic_lora_ckpt "${CRITIC_LORA_CKPT}" \
    --critic_head_ckpt "${CRITIC_HEAD_CKPT}" \
    --lambda_critic ${LAMBDA_CRITIC} \
    --lambda_retrieval ${LAMBDA_RETRIEVAL} \
    --vis swanlab \
    $EXTRA_ARGS \
    "$@"
