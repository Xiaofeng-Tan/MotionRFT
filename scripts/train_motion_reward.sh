#!/bin/bash
#===============================================================================
#
#   MotionReward 三阶段训练脚本（带跨表征对齐, LoRA rank=16）
#
#===============================================================================
#
#   训练流程概览:
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  Stage 1: Retrieval 检索任务        │  GPU 0      │  全参数训练 + 跨表征对齐 │
#   │  Stage 2: Critic 评分任务     │  GPU 0,1    │  LoRA 微调 (冻结主干)    │
#   │  Stage 3: AI Detection 任务   │  GPU 0      │  LoRA 微调 (冻结主干)    │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#   表征类型: 263, 22x3, 135 (三表征)
#   模型规模: tiny
#
#   使用方法:
#       chmod +x scripts/train_motion_reward.sh
#       ./scripts/train_motion_reward.sh
#
#===============================================================================

set -e  # 遇到错误立即退出

#===============================================================================
#                              配 置 区 域
#===============================================================================

# ---------------------- 模型配置 ----------------------
MODEL_SIZE="tiny"                    # 模型规模: tiny/small/base/large
REPR_TYPES="263,22x3,135"            # 表征类型 (逗号分隔)

# ---------------------- LoRA 配置 ----------------------
LORA_RANK=16                          # LoRA 秩
LORA_ALPHA=32                         # LoRA 缩放因子 (保持 alpha/rank = 2)

# ---------------------- 训练轮数 ----------------------
STAGE1_EPOCHS=50                      # Stage 1: Retrieval 训练轮数（带跨表征对齐）
STAGE2_EPOCHS=3000                      # Stage 2: Critic 训练轮数
STAGE3_EPOCHS=50                      # Stage 3: AI Detection 训练轮数

# ---------------------- 学习率 ----------------------
STAGE1_LR=1e-4                       # Stage 1 学习率
STAGE2_LR=1e-4                       # Stage 2 学习率
STAGE3_LR=1e-4                       # Stage 3 学习率

# ---------------------- 跨表征对齐配置 ----------------------
USE_CROSS_REPR_ALIGN=true            # 是否启用跨表征对齐
LAMBDA_CROSS_REPR=0.1                # 跨表征对齐损失权重

# ---------------------- 其他配置 ----------------------
BATCH_SIZE=128                        # 批量大小

# ---------------------- GPU 配置 ----------------------
STAGE1_GPUS="0"                      # Stage 1 使用的 GPU (单卡) - 跳过
STAGE2_GPUS="0,1,2,3"               # Stage 2 使用的 GPU (多卡 DDP)
STAGE2_NPROC=4                       # Stage 2 进程数 (与 GPU 数量一致)
STAGE3_GPUS="0"                      # Stage 3 使用的 GPU (单卡)

# ---------------------- Debug 配置 ----------------------
DEBUG_MODE=false                     # 设为 true 开启 debug 模式
DEBUG_SAMPLES=50                     # debug 模式下每个数据集使用的样本数

# ---------------------- 路径配置 ----------------------
CHECKPOINT_DIR="./checkpoints/motionreward"
EXP_NAME="motionreward"

# ---------------------- Stage 1 权重配置 ----------------------
# 跳过 Stage 1，直接使用现有的 Stage 1 权重
# 如果需要从头训练 Stage 1，请将此变量设为空字符串
EXISTING_STAGE1_CKPT="./checkpoints/motionreward/stage1_retrieval_backbone.pth"

#===============================================================================
#                           辅 助 函 数
#===============================================================================

# 构建 debug 参数
DEBUG_ARGS=""
if [ "${DEBUG_MODE}" = "true" ]; then
    DEBUG_ARGS="--debug --debug_samples ${DEBUG_SAMPLES}"
    echo ""
    echo "  ⚠ DEBUG 模式已开启: 仅使用 ${DEBUG_SAMPLES} 个样本"
    echo ""
fi

# 构建跨表征对齐参数
CROSS_REPR_ARGS=""
if [ "${USE_CROSS_REPR_ALIGN}" = "true" ]; then
    CROSS_REPR_ARGS="--use_cross_repr_align --lambda_cross_repr ${LAMBDA_CROSS_REPR}"
fi

# 打印分隔线
print_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  $1"
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

# 打印配置信息
print_config() {
    echo "  ├─ Model Size   : ${MODEL_SIZE}"
    echo "  ├─ Repr Types   : ${REPR_TYPES}"
    echo "  ├─ LoRA Rank    : ${LORA_RANK}"
    echo "  ├─ LoRA Alpha   : ${LORA_ALPHA}"
    echo "  ├─ Batch Size   : ${BATCH_SIZE}"
    echo "  └─ Checkpoint   : ${CHECKPOINT_DIR}"
}

# 打印阶段完成信息
print_stage_complete() {
    echo ""
    echo "  ✓ $1 完成"
    echo "    保存路径: $2"
    echo ""
}

# 打印数据路径信息
print_data_paths() {
    stage=$1
    PROJ_DIR=$(cd "$(dirname "$0")" && pwd)
    echo ""
    echo "┌──────────────────────────────────────────────────────────────────┐"
    echo "│  ${stage} 数据路径信息"
    echo "├──────────────────────────────────────────────────────────────────┤"
    echo "│  Retrieval 数据 (原始格式):"
    echo "│    263维: ${PROJ_DIR}/datasets/humanml3d/new_joint_vecs/"
    echo "│    22x3维: ${PROJ_DIR}/datasets/humanml3d/new_joints/"
    echo "│    135维: ${PROJ_DIR}/datasets/humanml3d/joints_6d/"
    echo "│    文本: ${PROJ_DIR}/datasets/humanml3d/texts/"
    echo "│  Critic 数据 (从 MotionCritic 转换):"
    echo "│    训练 263维: ${PROJ_DIR}/datasets/critic/critic_train_263.pth"
    echo "│    训练 22x3维: ${PROJ_DIR}/datasets/critic/critic_train_22x3.pth"
    echo "│    训练 135维: ${PROJ_DIR}/datasets/critic/critic_train_135.pth"
    echo "│    验证 263维: ${PROJ_DIR}/datasets/critic/critic_val_263.pth"
    echo "│    验证 22x3维: ${PROJ_DIR}/datasets/critic/critic_val_22x3.pth"
    echo "│    验证 135维: ${PROJ_DIR}/datasets/critic/critic_val_135.pth"
    echo "│  AI Detection 数据:"
    echo "│    ${PROJ_DIR}/datasets/ai_detection_packed/"
    echo "└──────────────────────────────────────────────────────────────────┘"
    echo ""
}

#===============================================================================
#                     Stage 1: Retrieval 检索任务
#===============================================================================

if [ -n "$EXISTING_STAGE1_CKPT" ] && [ -f "$EXISTING_STAGE1_CKPT" ]; then
    print_header "Stage 1: 跳过，使用现有 Retrieval 权重"
    echo "  ├─ 使用现有权重: ${EXISTING_STAGE1_CKPT}"
    STAGE1_CKPT="$EXISTING_STAGE1_CKPT"
    print_stage_complete "Stage 1 (使用现有权重)" "${STAGE1_CKPT}"
else
    print_header "Stage 1: Retrieval 检索任务训练 (全参数 + 跨表征对齐)"
    echo "  ├─ GPU         : ${STAGE1_GPUS}"
    echo "  ├─ Epochs      : ${STAGE1_EPOCHS}"
    echo "  ├─ Learning Rate: ${STAGE1_LR}"
    print_config

    CUDA_VISIBLE_DEVICES=${STAGE1_GPUS} python -m motionreward.training.train_retrieval_lora_new \
        --model_size ${MODEL_SIZE} \
        --retrieval_repr_types ${REPR_TYPES} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --stage1_epochs ${STAGE1_EPOCHS} \
        --stage2_epochs 0 \
        --stage3_epochs 0 \
        --stage1_lr ${STAGE1_LR} \
        --batch_size ${BATCH_SIZE} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        --exp_name "${EXP_NAME}_stage1" \
        --skip_stage2 \
        --skip_stage3 \
        ${CROSS_REPR_ARGS} \
        ${DEBUG_ARGS}

    STAGE1_CKPT=$(ls -t ${CHECKPOINT_DIR}/*_stage1_best_retrieval_backbone_*.pth 2>/dev/null | head -1)
    print_stage_complete "Stage 1 (Retrieval)" "${STAGE1_CKPT:-未生成}"
fi

#===============================================================================
#                   Stage 2: Critic 评分任务 (LoRA 微调)
#===============================================================================

print_header "Stage 2: Critic 评分任务训练 (LoRA)"
print_data_paths "Stage 2 (Critic LoRA)"
echo "  ├─ GPU         : ${STAGE2_GPUS} (DDP, ${STAGE2_NPROC} procs)"
echo "  ├─ Epochs      : ${STAGE2_EPOCHS}"
echo "  ├─ Learning Rate: ${STAGE2_LR}"
echo "  ├─ Resume From : ${STAGE1_CKPT}"
print_config

# 随机选择一个可用端口 (29500-29999)
MASTER_PORT=$((29500 + RANDOM % 500))

CUDA_VISIBLE_DEVICES=${STAGE2_GPUS} torchrun \
    --nproc_per_node=${STAGE2_NPROC} \
    --master_port=${MASTER_PORT} \
    -m motionreward.training.train_retrieval_lora_new \
    --model_size ${MODEL_SIZE} \
    --retrieval_repr_types ${REPR_TYPES} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --stage1_epochs 0 \
    --stage2_epochs ${STAGE2_EPOCHS} \
    --stage3_epochs 0 \
    --stage2_lr ${STAGE2_LR} \
    --batch_size ${BATCH_SIZE} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --exp_name "${EXP_NAME}_stage2" \
    --resume_stage1_ckpt ${STAGE1_CKPT} \
    --skip_stage1 \
    --skip_stage3 \
    --stage2_eval_freq 100 \
    --skip_retrieval_eval_in_stage2 \
    ${DEBUG_ARGS}

# 查找最新的 Stage 2 checkpoint
STAGE2_CKPT=$(ls -t ${CHECKPOINT_DIR}/*_critic_lora_*.pth 2>/dev/null | head -1)

print_stage_complete "Stage 2 (Critic LoRA)" "${STAGE2_CKPT:-未生成}"

#===============================================================================
#                 Stage 3: AI Detection 任务 (LoRA 微调)
#===============================================================================

print_header "Stage 3: AI Detection 任务训练 (LoRA)"
print_data_paths "Stage 3 (AI Detection LoRA)"
echo "  ├─ GPU         : ${STAGE3_GPUS}"
echo "  ├─ Epochs      : ${STAGE3_EPOCHS}"
echo "  ├─ Learning Rate: ${STAGE3_LR}"
echo "  ├─ Resume From : ${STAGE1_CKPT}"
print_config

CUDA_VISIBLE_DEVICES=${STAGE3_GPUS} python -m motionreward.training.train_retrieval_lora_new \
    --model_size ${MODEL_SIZE} \
    --retrieval_repr_types ${REPR_TYPES} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --stage1_epochs 0 \
    --stage2_epochs 0 \
    --stage3_epochs ${STAGE3_EPOCHS} \
    --stage3_lr ${STAGE3_LR} \
    --batch_size ${BATCH_SIZE} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --exp_name "${EXP_NAME}_stage3" \
    --resume_stage1_ckpt ${STAGE1_CKPT} \
    --skip_stage1 \
    --skip_stage2 \
    ${DEBUG_ARGS}

# 查找最新的 Stage 3 checkpoint
STAGE3_CKPT=$(ls -t ${CHECKPOINT_DIR}/*_ai_detection_lora_*.pth 2>/dev/null | head -1)

print_stage_complete "Stage 3 (AI Detection LoRA)" "${STAGE3_CKPT:-未生成}"

#===============================================================================
#                              训 练 完 成
#===============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                      全部训练阶段完成                            ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Checkpoint 目录: ${CHECKPOINT_DIR}"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Stage 1 (Retrieval 主干 + 跨表征对齐):                             ║"
echo "║    ${STAGE1_CKPT:-未找到}"
echo "║                                                                  ║"
echo "║  Stage 2 (Critic LoRA):                                          ║"
echo "║    ${STAGE2_CKPT:-未生成}"
echo "║                                                                  ║"
echo "║  Stage 3 (AI Detection LoRA):                                    ║"
echo "║    ${STAGE3_CKPT:-未生成}"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

#===============================================================================
#                              性 能 测 试
#===============================================================================

# ---------------------- 测试配置 ----------------------
EVAL_GPUS="0"                        # 测试使用的 GPU

print_header "性能测试: 一次性测试所有任务"
echo "  ├─ GPU             : ${EVAL_GPUS}"
echo "  ├─ Retrieval 主干权重    : ${STAGE1_CKPT}"
echo "  ├─ Critic LoRA     : ${STAGE2_CKPT:-未生成}"
echo "  └─ AI Detection LoRA: ${STAGE3_CKPT:-未生成}"

# 构建评估命令
EVAL_CMD="CUDA_VISIBLE_DEVICES=${EVAL_GPUS} python -m motionreward.training.train_retrieval_lora_new \
    --model_size ${MODEL_SIZE} \
    --retrieval_repr_types ${REPR_TYPES} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --stage1_epochs 0 \
    --stage2_epochs 0 \
    --stage3_epochs 0 \
    --batch_size ${BATCH_SIZE} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --exp_name ${EXP_NAME}_eval \
    --resume_stage1_ckpt ${STAGE1_CKPT} \
    --eval_only \
    --eval_critic \
    --eval_ai_detection \
    ${DEBUG_ARGS}"

# 添加 Critic LoRA + Head (如果存在)
if [ -n "$STAGE2_CKPT" ]; then
    EVAL_CMD="${EVAL_CMD} --resume_critic_lora_ckpt ${STAGE2_CKPT}"
    STAGE2_HEAD_CKPT=$(echo "$STAGE2_CKPT" | sed 's/_critic_lora_/_critic_head_/')
    if [ -f "$STAGE2_HEAD_CKPT" ]; then
        EVAL_CMD="${EVAL_CMD} --resume_critic_head_ckpt ${STAGE2_HEAD_CKPT}"
    fi
fi

# 添加 AI Detection LoRA + Head (如果存在)
if [ -n "$STAGE3_CKPT" ]; then
    EVAL_CMD="${EVAL_CMD} --resume_ai_detection_lora_ckpt ${STAGE3_CKPT}"
    STAGE3_HEAD_CKPT=$(echo "$STAGE3_CKPT" | sed 's/_ai_detection_lora_/_ai_detection_head_/')
    if [ -f "$STAGE3_HEAD_CKPT" ]; then
        EVAL_CMD="${EVAL_CMD} --resume_ai_detection_head_ckpt ${STAGE3_HEAD_CKPT}"
    fi
fi

# 执行评估
eval ${EVAL_CMD}

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                      全部测试完成                                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
