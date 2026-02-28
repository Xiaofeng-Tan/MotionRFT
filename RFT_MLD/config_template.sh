#!/bin/bash
#===============================================================================
#
#   MotionReward 微调 MLD - 配置模板
#
#   复制此文件并修改配置，然后运行
#   
#   使用方法:
#     1. cp config_template.sh my_config.sh
#     2. vi my_config.sh  # 修改配置
#     3. chmod +x my_config.sh && ./my_config.sh
#
#===============================================================================

set -e

#===============================================================================
#                           🔧 配置区域 - 请修改这里
#===============================================================================

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPU 配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export CUDA_VISIBLE_DEVICES=0              # 使用的 GPU 编号，多卡用逗号分隔 (如 "0,1,2,3")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MLD 模型配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MLD_PRETRAINED="./checkpoints/mld_humanml3d.ckpt"  # MLD 预训练权重路径

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MotionReward 配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REWARD_MODEL_SIZE="small"                 # 模型规模: tiny/small/base/large/xlarge

# Stage1 Checkpoint 路径（主干模型）
REWARD_STAGE1_CKPT="./checkpoints/motionreward/stage1_retrieval_backbone.pth"

# (可选) Critic LoRA Checkpoint 路径，不用则留空
REWARD_CRITIC_LORA_CKPT=""                # 例如: "./checkpoints/motionreward/stage2_critic_lora.pth"

# Motion 表征类型
REPR_TYPE="263"                           # 可选: "263" / "22x3" / "135"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 训练超参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BATCH_SIZE=32                             # 训练批次大小
VAL_BATCH_SIZE=32                         # 验证批次大小
LEARNING_RATE=1e-5                        # 学习率
MAX_FT_EPOCHS=10                          # 最大微调轮数

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Loss 权重配置 ⚠️ 重要！
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAMBDA_DIFF=0.0                           # Diffusion Loss 权重 (MSE loss)
LAMBDA_REWARD=1.0                         # Reward Loss 权重 (MotionReward 相似度)

# 💡 常用配置:
#   纯 Reward:      LAMBDA_DIFF=0.0, LAMBDA_REWARD=1.0  (推荐)
#   标准训练:       LAMBDA_DIFF=1.0, LAMBDA_REWARD=0.0
#   混合 (平衡):    LAMBDA_DIFF=1.0, LAMBDA_REWARD=1.0
#   混合 (强 Reward): LAMBDA_DIFF=0.5, LAMBDA_REWARD=2.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 微调策略
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FT_TYPE="default"                         # 微调类型: default/NIPS/ReFL/DRaFT/DRTune
FT_NAME="my_experiment"                   # 实验名称（用于保存路径）

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据集配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET_NAME="humanml3d"                  # 数据集名称
SPLIT="val"                               # 验证集划分

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 其他配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED=42                                   # 随机种子
EVAL_REWARD_BEFORE_FT=true                # 微调前是否评估 reward model

#===============================================================================
#                      以下代码无需修改，自动运行
#===============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           MotionReward 微调 MLD - 配置确认                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│  🎯 实验配置"
echo "├──────────────────────────────────────────────────────────────────┤"
echo "│  实验名称          : ${FT_NAME}"
echo "│  GPU               : ${CUDA_VISIBLE_DEVICES}"
echo "│  Batch Size        : ${BATCH_SIZE}"
echo "│  Learning Rate     : ${LEARNING_RATE}"
echo "│  Max Epochs        : ${MAX_FT_EPOCHS}"
echo "├──────────────────────────────────────────────────────────────────┤"
echo "│  💎 MotionReward 配置"
echo "├──────────────────────────────────────────────────────────────────┤"
echo "│  Model Size        : ${REWARD_MODEL_SIZE}"
echo "│  Repr Type         : ${REPR_TYPE}"
echo "│  Stage1 Ckpt       : ${REWARD_STAGE1_CKPT}"
if [ -n "${REWARD_CRITIC_LORA_CKPT}" ]; then
echo "│  Critic LoRA Ckpt  : ${REWARD_CRITIC_LORA_CKPT}"
else
echo "│  Critic LoRA Ckpt  : (未使用)"
fi
echo "├──────────────────────────────────────────────────────────────────┤"
echo "│  ⚖️  Loss 权重"
echo "├──────────────────────────────────────────────────────────────────┤"
echo "│  Lambda Diff       : ${LAMBDA_DIFF}  (Diffusion MSE)"
echo "│  Lambda Reward     : ${LAMBDA_REWARD}  (MotionReward 相似度)"
echo "└──────────────────────────────────────────────────────────────────┘"
echo ""

# 检查文件
echo "📋 检查必需文件..."
missing_files=0

if [ ! -f "${MLD_PRETRAINED}" ]; then
    echo "  ✗ MLD 预训练权重不存在: ${MLD_PRETRAINED}"
    missing_files=$((missing_files + 1))
else
    echo "  ✓ MLD 预训练权重"
fi

if [ ! -f "${REWARD_STAGE1_CKPT}" ]; then
    echo "  ✗ MotionReward Stage1 checkpoint 不存在: ${REWARD_STAGE1_CKPT}"
    missing_files=$((missing_files + 1))
else
    echo "  ✓ MotionReward Stage1 checkpoint"
fi

if [ -n "${REWARD_CRITIC_LORA_CKPT}" ] && [ ! -f "${REWARD_CRITIC_LORA_CKPT}" ]; then
    echo "  ✗ Critic LoRA checkpoint 不存在: ${REWARD_CRITIC_LORA_CKPT}"
    missing_files=$((missing_files + 1))
elif [ -n "${REWARD_CRITIC_LORA_CKPT}" ]; then
    echo "  ✓ Critic LoRA checkpoint"
fi

if [ $missing_files -gt 0 ]; then
    echo ""
    echo "❌ 有 ${missing_files} 个文件缺失，请检查配置"
    exit 1
fi

echo ""
echo "✅ 所有文件检查通过"
echo ""

# 确认启动
read -p "是否开始训练？(y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "❌ 取消训练"
    exit 0
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                   🚀 开始训练                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# 构建训练命令
TRAIN_CMD="python ft_mld_motionreward.py \
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
    eval_reward_before_ft ${EVAL_REWARD_BEFORE_FT}"

# 如果有 Critic LoRA，添加到命令
if [ -n "${REWARD_CRITIC_LORA_CKPT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} reward_critic_lora_ckpt ${REWARD_CRITIC_LORA_CKPT}"
fi

# 执行训练
eval ${TRAIN_CMD}

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                   ✅ 训练完成                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 查看结果:"
echo "   日志文件: ./checkpoints/ft_mld_motionreward/${FT_NAME}_*/train.log"
echo "   TensorBoard: tensorboard --logdir ./checkpoints/ft_mld_motionreward/"
echo ""
