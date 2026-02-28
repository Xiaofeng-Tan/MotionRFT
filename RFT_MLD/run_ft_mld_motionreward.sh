#!/bin/bash
# ===============================================================================
#   使用 MotionReward 微调 MLD 的启动脚本
#   
#   功能:
#   1. 加载预训练 MLD 模型
#   2. 加载 MotionReward 作为 reward model
#   3. 训练前自动评测 reward model 性能
#   4. 使用 reward 信号微调 MLD
# ===============================================================================

set -e

# 切换到 RFT_MLD 目录
cd "$(dirname "$0")"

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ===============================================================================
#                           可配置参数
# ===============================================================================

# ===== 微调参数 =====
FT_TYPE=${FT_TYPE:-"NIPS"}              # 微调类型: ReFL, DRaFT, DRTune, AlignProp, NIPS, None
FT_LR=${FT_LR:-1e-5}                    # 学习率
FT_LAMBDA_DIFF=${FT_LAMBDA_DIFF:-0.0}   # Diffusion loss 权重
FT_LAMBDA_REWARD=${FT_LAMBDA_REWARD:-1.0}  # Reward loss 权重
FT_NAME=${FT_NAME:-"ft_mld_motionreward"}  # 实验名称

# ===== Reward Model 参数 =====
# 模型规模: tiny, small, base, large, xlarge, xxlarge, giant
REWARD_MODEL_SIZE=${REWARD_MODEL_SIZE:-"small"}

# Stage 1 checkpoint 路径 (留空则自动查找最新)
REWARD_STAGE1_CKPT=${REWARD_STAGE1_CKPT:-""}

# 是否在微调前评测 reward model (true/false)
EVAL_REWARD_BEFORE_FT=${EVAL_REWARD_BEFORE_FT:-true}

# Debug 模式：只加载少量数据快速测试
DEBUG=${DEBUG:-false}

# ===== 训练参数 (通过 YAML 配置) =====
# MAX_FT_EPOCHS 和 BATCH_SIZE 需在 configs/ft_mld_t2m.yaml 中修改
# MAX_FT_EPOCHS=${MAX_FT_EPOCHS:-10}
# BATCH_SIZE=${BATCH_SIZE:-64}

# ===============================================================================
#                           自动查找 Checkpoint
# ===============================================================================

# 如果未指定 REWARD_STAGE1_CKPT，自动查找最新的
if [ -z "$REWARD_STAGE1_CKPT" ]; then
    CKPT_DIR="../checkpoints/motionreward"
    if [ -d "$CKPT_DIR" ]; then
        # 查找最新的 stage1 backbone checkpoint
        REWARD_STAGE1_CKPT=$(ls -t ${CKPT_DIR}/*_stage1_*_retrieval_backbone_*.pth 2>/dev/null | head -1)
        if [ -n "$REWARD_STAGE1_CKPT" ]; then
            echo "[Auto-selected Stage1 Checkpoint]"
            echo "  $REWARD_STAGE1_CKPT"
        else
            echo "[Warning] No Stage 1 checkpoint found in $CKPT_DIR"
            echo "  Will use random weights for reward model"
        fi
    else
        echo "[Warning] Checkpoint directory not found: $CKPT_DIR"
    fi
fi

# ===============================================================================
#                           打印配置
# ===============================================================================

echo "==============================================================================="
echo "  Fine-tuning MLD with MotionReward"
echo "==============================================================================="
echo "  FT Type          : $FT_TYPE"
echo "  FT Name          : $FT_NAME"
echo "  Learning Rate    : $FT_LR"
echo "  Lambda Diff      : $FT_LAMBDA_DIFF"
echo "  Lambda Reward    : $FT_LAMBDA_REWARD"
echo "-------------------------------------------------------------------------------"
echo "  Reward Model Size: $REWARD_MODEL_SIZE"
echo "  Stage1 Checkpoint: ${REWARD_STAGE1_CKPT:-'(auto-select)'}"
echo "  Eval Before FT   : $EVAL_REWARD_BEFORE_FT"
echo "  Debug Mode       : $DEBUG"
echo "-------------------------------------------------------------------------------"
echo "  CUDA Devices     : $CUDA_VISIBLE_DEVICES"
echo "==============================================================================="

# ===============================================================================
#                           运行训练
# ===============================================================================

python ft_mld_motionreward.py \
    --cfg configs/ft_mld_t2m.yaml \
    --ft_type "$FT_TYPE" \
    --ft_name "$FT_NAME" \
    --ft_lr "$FT_LR" \
    --ft_lambda_diff "$FT_LAMBDA_DIFF" \
    --ft_lambda_reward "$FT_LAMBDA_REWARD" \
    --reward_model_size "$REWARD_MODEL_SIZE" \
    --reward_stage1_ckpt "$REWARD_STAGE1_CKPT" \
    --eval_reward_before_ft "$EVAL_REWARD_BEFORE_FT" \
    $( [ "$DEBUG" = "true" ] && echo "--debug" ) \
    "$@"
