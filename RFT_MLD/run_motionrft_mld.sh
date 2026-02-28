#!/bin/bash
#===============================================================================
#
#   RFT_MLD 训练脚本 (Critic + Retrieval + M2M + AI Detection Reward)
#
#===============================================================================
#
#   Reward 组合:
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  - Critic reward: Critic Head 分数 (Stage 2)                           │
#   │  - Retrieval reward: text-motion 相似度 (Stage 1)                      │
#   │  - M2M reward: GT motion 和 Generated motion 的余弦相似度              │
#   │  - AI Detection reward: AI Detection Head 分数 (Stage 3)               │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#   Checkpoint 加载逻辑 (follow eval_motion_reward.sh):
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  - Stage 1 Backbone: 用于 Retrieval reward 和作为 Critic/AI Det 的基础 │
#   │  - Stage 2 Critic LoRA + Head: 用于 Critic reward                      │
#   │  - Stage 3 AI Detection LoRA + Head: 用于 AI Detection reward          │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#===============================================================================

# 切换到 RFT_MLD 目录
cd "$(dirname "$0")"

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

#===============================================================================
#                              配 置 区 域
#===============================================================================

# ---------------------- Finetune 配置 ----------------------
FT_TYPE=${FT_TYPE:-"NIPS"}
FT_M=${FT_M:-10}
FT_K=${FT_K:-10}
FT_LR=${FT_LR:-1e-5}
FT_LAMBDA_REWARD=${FT_LAMBDA_REWARD:-1.0}
FT_EPOCHS=${FT_EPOCHS:-20}

# ---------------------- Reward 权重配置 ----------------------
# lambda_critic: Critic reward (Stage 2) 权重
# lambda_retrieval: Retrieval reward (Stage 1) 权重
# lambda_m2m: M2M reward (GT vs Generated motion) 权重
# lambda_ai_detection: AI Detection reward (Stage 3) 权重
LAMBDA_CRITIC=${LAMBDA_CRITIC:-0.0002}
LAMBDA_RETRIEVAL=${LAMBDA_RETRIEVAL:-1.0}
LAMBDA_M2M=${LAMBDA_M2M:-1.0}
LAMBDA_AI_DETECTION=${LAMBDA_AI_DETECTION:-0.0002}

# ---------------------- 模型配置 ----------------------
REWARD_MODEL_SIZE=${REWARD_MODEL_SIZE:-"tiny"}   # 模型规模: tiny/small/base/large
LORA_RANK=${LORA_RANK:-128}                      # LoRA 秩
LORA_ALPHA=${LORA_ALPHA:-256}                    # LoRA 缩放因子

# ---------------------- 其他配置 ----------------------
VALIDATION_STEPS=${VALIDATION_STEPS:-20}         # 验证频率 (每 N 步验证一次)
NO_SAVE=${NO_SAVE:-false}                        # 是否禁止保存 checkpoint
FID_SAVE_THRESHOLD=${FID_SAVE_THRESHOLD:-0.15}   # FID 阈值：仅保存 FID < 该值的 checkpoint
REWARD_MAX_T=${REWARD_MAX_T:-500}                # MotionReward 训练时的 maxT，用于 clamp RFT 中的 timestep
REWARD_T_SWITCH=${REWARD_T_SWITCH:-0}           # Reward 策略分界点 (step index)
                                                  #   i < REWARD_T_SWITCH: 单步预测 x_0, reward=R(x_0,0)
                                                  #   i >= REWARD_T_SWITCH: 直接用 x_t, reward=R(x_t,t)
                                                  #   默认50=全部用 x_0 预测
CURRICULUM=${CURRICULUM:-true}                   # Motion Reward 时间步调度
                                                  #   true: 启用 curriculum 滑动窗口
                                                  #   false: 固定优化最后 k 步 (默认行为)
SWEEP_RATIO=${SWEEP_RATIO:-0.03}                 # Curriculum 扫过比例
                                                  #   0.0:  100% 时间固定最后 k 步 (纯低噪声)
                                                  #   0.03: 3% 扫过 + 97% 最后 k 步 (默认)

#===============================================================================
#                         Checkpoint 路径配置
#===============================================================================
# 根据 LORA_RANK 自动选择对应的 checkpoint 后缀
# rank=16 使用默认文件名，rank=128 使用 _r128 后缀
#===============================================================================

if [ "${LORA_RANK}" -eq 128 ]; then
    _SUFFIX="_r128"
else
    _SUFFIX=""
fi

# Stage 1: Backbone checkpoint (用于 Retrieval reward)
STAGE1_BACKBONE_CKPT=${STAGE1_BACKBONE_CKPT:-"../checkpoints/motionreward/stage1_retrieval_backbone${_SUFFIX}.pth"}

# Stage 2: Critic LoRA + Head checkpoints (分离加载，用于 Critic reward)
STAGE2_CRITIC_LORA_CKPT=${STAGE2_CRITIC_LORA_CKPT:-"../checkpoints/motionreward/stage2_critic_lora${_SUFFIX}.pth"}
STAGE2_CRITIC_HEAD_CKPT=${STAGE2_CRITIC_HEAD_CKPT:-"../checkpoints/motionreward/stage2_critic_head${_SUFFIX}.pth"}

# Stage 3: AI Detection LoRA + Head checkpoints (分离加载，用于 AI Detection reward)
STAGE3_AI_DETECTION_LORA_CKPT=${STAGE3_AI_DETECTION_LORA_CKPT:-"../checkpoints/motionreward/stage3_ai_detection_lora${_SUFFIX}.pth"}
STAGE3_AI_DETECTION_HEAD_CKPT=${STAGE3_AI_DETECTION_HEAD_CKPT:-"../checkpoints/motionreward/stage3_ai_detection_head${_SUFFIX}.pth"}

#===============================================================================
#                           辅 助 函 数
#===============================================================================

# 打印分隔线
print_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  $1"
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

#===============================================================================
#                              打 印 配 置
#===============================================================================

print_header "训练配置"
echo "  ├─ Finetune 配置:"
echo "  │   ├─ FT_TYPE           : ${FT_TYPE}"
echo "  │   ├─ FT_M              : ${FT_M}"
echo "  │   ├─ FT_K              : ${FT_K}"
echo "  │   ├─ FT_LR             : ${FT_LR}"
echo "  │   ├─ FT_LAMBDA_REWARD  : ${FT_LAMBDA_REWARD}"
echo "  │   └─ FT_EPOCHS         : ${FT_EPOCHS}"
echo "  ├─ Reward 权重配置:"
echo "  │   ├─ LAMBDA_CRITIC     : ${LAMBDA_CRITIC}"
echo "  │   ├─ LAMBDA_RETRIEVAL  : ${LAMBDA_RETRIEVAL}"
echo "  │   ├─ LAMBDA_M2M        : ${LAMBDA_M2M}"
echo "  │   └─ LAMBDA_AI_DETECTION: ${LAMBDA_AI_DETECTION}"
echo "  ├─ 模型配置:"
echo "  │   ├─ REWARD_MODEL_SIZE : ${REWARD_MODEL_SIZE}"
echo "  │   ├─ LORA_RANK         : ${LORA_RANK}"
echo "  │   └─ LORA_ALPHA        : ${LORA_ALPHA}"
echo "  ├─ Checkpoint 配置:"
echo "  │   ├─ Stage 1 Backbone  : ${STAGE1_BACKBONE_CKPT}"
echo "  │   ├─ Stage 2 Critic LoRA: ${STAGE2_CRITIC_LORA_CKPT}"
echo "  │   ├─ Stage 2 Critic Head: ${STAGE2_CRITIC_HEAD_CKPT}"
echo "  │   ├─ Stage 3 AI Det LoRA: ${STAGE3_AI_DETECTION_LORA_CKPT}"
echo "  │   └─ Stage 3 AI Det Head: ${STAGE3_AI_DETECTION_HEAD_CKPT}"
echo "  └─ 其他配置:"
echo "      ├─ VALIDATION_STEPS  : ${VALIDATION_STEPS}"
echo "      ├─ NO_SAVE           : ${NO_SAVE}"
echo "      ├─ FID_SAVE_THRESHOLD: ${FID_SAVE_THRESHOLD}"
echo "      ├─ REWARD_MAX_T      : ${REWARD_MAX_T}"
echo "      ├─ REWARD_T_SWITCH   : ${REWARD_T_SWITCH}"
echo "      ├─ CURRICULUM        : ${CURRICULUM}"
echo "      ├─ SWEEP_RATIO       : ${SWEEP_RATIO}"
echo "      └─ CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

#===============================================================================
#                              构 建 命 令
#===============================================================================

print_header "开始训练"

# 构建训练命令
TRAIN_CMD="python motionrft_mld.py \
    --cfg configs/ft_mld_t2m.yaml \
    --ft_type ${FT_TYPE} \
    --ft_m ${FT_M} \
    --ft_k ${FT_K} \
    --ft_lr ${FT_LR} \
    --ft_lambda_reward ${FT_LAMBDA_REWARD} \
    --ft_epochs ${FT_EPOCHS} \
    --reward_model_size ${REWARD_MODEL_SIZE} \
    --reward_lora_rank ${LORA_RANK} \
    --reward_lora_alpha ${LORA_ALPHA} \
    --critic_backbone_ckpt ${STAGE1_BACKBONE_CKPT} \
    --critic_lora_ckpt ${STAGE2_CRITIC_LORA_CKPT} \
    --critic_head_ckpt ${STAGE2_CRITIC_HEAD_CKPT} \
    --ai_detection_lora_ckpt ${STAGE3_AI_DETECTION_LORA_CKPT} \
    --ai_detection_head_ckpt ${STAGE3_AI_DETECTION_HEAD_CKPT} \
    --lambda_critic ${LAMBDA_CRITIC} \
    --lambda_retrieval ${LAMBDA_RETRIEVAL} \
    --lambda_m2m ${LAMBDA_M2M} \
    --lambda_ai_detection ${LAMBDA_AI_DETECTION} \
    --validation_steps ${VALIDATION_STEPS} \
    --reward_max_t ${REWARD_MAX_T} \
    --reward_t_switch ${REWARD_T_SWITCH} \
    --no_debug_vis \
    --vis swanlab"

# 添加 curriculum 选项
if [ "$CURRICULUM" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --curriculum --sweep_ratio ${SWEEP_RATIO}"
fi

# 添加 no_save 选项
if [ "$NO_SAVE" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --no_save"
fi

# 添加 FID 保存阈值
TRAIN_CMD="${TRAIN_CMD} --fid_save_threshold ${FID_SAVE_THRESHOLD}"

# 添加额外参数
TRAIN_CMD="${TRAIN_CMD} $@"

echo ""
echo "  执行命令:"
echo "  ${TRAIN_CMD}"
echo ""

# 执行训练
eval ${TRAIN_CMD}

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                      训练完成                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
