#!/bin/bash
#===============================================================================
#
#   MotionReward 微调 MLD - 示例配置脚本
#
#   展示不同 loss 权重配置的效果
#
#===============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          MotionReward 微调 MLD - 示例配置                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "本脚本展示不同的 loss 权重配置，请根据需求选择："
echo ""
echo "  1) 纯 Reward 微调      (LAMBDA_DIFF=0.0, LAMBDA_REWARD=1.0)"
echo "  2) 标准训练            (LAMBDA_DIFF=1.0, LAMBDA_REWARD=0.0)"
echo "  3) 混合训练 (平衡)     (LAMBDA_DIFF=1.0, LAMBDA_REWARD=1.0)"
echo "  4) 混合训练 (强 Reward) (LAMBDA_DIFF=0.5, LAMBDA_REWARD=2.0)"
echo "  5) 混合训练 (弱 Reward) (LAMBDA_DIFF=2.0, LAMBDA_REWARD=0.5)"
echo "  6) 自定义"
echo ""
read -p "请选择 (1-6): " choice

case $choice in
    1)
        LAMBDA_DIFF=0.0
        LAMBDA_REWARD=1.0
        CONFIG_NAME="pure_reward"
        echo "✓ 选择: 纯 Reward 微调"
        ;;
    2)
        LAMBDA_DIFF=1.0
        LAMBDA_REWARD=0.0
        CONFIG_NAME="standard_training"
        echo "✓ 选择: 标准训练"
        ;;
    3)
        LAMBDA_DIFF=1.0
        LAMBDA_REWARD=1.0
        CONFIG_NAME="balanced_mix"
        echo "✓ 选择: 混合训练 (平衡)"
        ;;
    4)
        LAMBDA_DIFF=0.5
        LAMBDA_REWARD=2.0
        CONFIG_NAME="strong_reward_mix"
        echo "✓ 选择: 混合训练 (强 Reward)"
        ;;
    5)
        LAMBDA_DIFF=2.0
        LAMBDA_REWARD=0.5
        CONFIG_NAME="weak_reward_mix"
        echo "✓ 选择: 混合训练 (弱 Reward)"
        ;;
    6)
        read -p "输入 LAMBDA_DIFF (例如 1.0): " LAMBDA_DIFF
        read -p "输入 LAMBDA_REWARD (例如 1.0): " LAMBDA_REWARD
        CONFIG_NAME="custom_${LAMBDA_DIFF}_${LAMBDA_REWARD}"
        echo "✓ 选择: 自定义配置"
        ;;
    *)
        echo "✗ 无效选择，退出"
        exit 1
        ;;
esac

echo ""
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│  Loss 权重配置"
echo "├──────────────────────────────────────────────────────────────────┤"
echo "│  LAMBDA_DIFF      : ${LAMBDA_DIFF}"
echo "│  LAMBDA_REWARD    : ${LAMBDA_REWARD}"
echo "│  CONFIG_NAME      : ${CONFIG_NAME}"
echo "└──────────────────────────────────────────────────────────────────┘"
echo ""

# ---------------------- 固定配置 ----------------------
export CUDA_VISIBLE_DEVICES=0
MLD_PRETRAINED="./checkpoints/mld_humanml3d.ckpt"
REWARD_MODEL_SIZE="small"
REWARD_STAGE1_CKPT="./checkpoints/motionreward/stage1_retrieval_backbone.pth"
BATCH_SIZE=32
VAL_BATCH_SIZE=32
LEARNING_RATE=1e-5
MAX_FT_EPOCHS=10
FT_TYPE="default"
FT_NAME="ft_mld_${CONFIG_NAME}"
DATASET_NAME="humanml3d"
SPLIT="val"
SEED=42
EVAL_REWARD_BEFORE_FT=true

# ---------------------- 检查文件 ----------------------
if [ ! -f "${MLD_PRETRAINED}" ]; then
    echo "✗ [错误] MLD 预训练权重不存在: ${MLD_PRETRAINED}"
    exit 1
fi

if [ ! -f "${REWARD_STAGE1_CKPT}" ]; then
    echo "✗ [错误] MotionReward Stage1 checkpoint 不存在: ${REWARD_STAGE1_CKPT}"
    exit 1
fi

echo "✓ 文件检查通过"
echo ""

# ---------------------- 确认启动 ----------------------
read -p "确认启动训练？(y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "取消训练"
    exit 0
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                      开始训练                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ---------------------- 启动训练 ----------------------
python ft_mld_motionreward.py \
    --cfg ./configs/config_mld_humanml3d.yaml \
    --cfg_assets ./configs/assets.yaml \
    --batch_size ${BATCH_SIZE} \
    --nodebug \
    TRAIN.PRETRAINED ${MLD_PRETRAINED} \
    TRAIN.BATCH_SIZE ${BATCH_SIZE} \
    TRAIN.learning_rate ${LEARNING_RATE} \
    TRAIN.max_ft_epochs ${MAX_FT_EPOCHS} \
    VAL.BATCH_SIZE ${VAL_BATCH_SIZE} \
    VAL.SPLIT ${SPLIT} \
    DATASET.NAME ${DATASET_NAME} \
    SEED_VALUE ${SEED} \
    reward_model_size ${REWARD_MODEL_SIZE} \
    reward_stage1_ckpt ${REWARD_STAGE1_CKPT} \
    ft_lambda_diff ${LAMBDA_DIFF} \
    ft_lambda_reward ${LAMBDA_REWARD} \
    ft_type ${FT_TYPE} \
    ft_name ${FT_NAME} \
    eval_reward_before_ft ${EVAL_REWARD_BEFORE_FT}

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                      训练完成                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
