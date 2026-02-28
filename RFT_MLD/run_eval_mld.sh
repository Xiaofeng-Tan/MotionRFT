#!/bin/bash
#===============================================================================
#
#   批量多次重复评估脚本 (支持多GPU并行)
#   对 rft_mld/ 下所有 checkpoint 进行 REPLICATION_TIMES 次评估
#
#   用法:
#     # 单GPU
#     CUDA_VISIBLE_DEVICES=4 bash run_eval_mld.sh
#
#     # 多GPU并行 (每个GPU独立评估不同的ckpt)
#     GPU_IDS="0,1,2,3" bash run_eval_mld.sh
#
#     # 其他选项
#     REPLICATION_TIMES=5 bash run_eval_mld.sh
#     CKPT_FILTER="*SW0.03*" bash run_eval_mld.sh
#
#===============================================================================

cd "$(dirname "$0")"

# 配置
export TOKENIZERS_PARALLELISM=false

REPLICATION_TIMES=${REPLICATION_TIMES:-20}
CFG_FILE=${CFG_FILE:-"configs/ft_mld_t2m.yaml"}
CKPT_BASE_DIR=${CKPT_BASE_DIR:-"checkpoints/rft_mld"}
CKPT_FILTER=${CKPT_FILTER:-"*"}
BATCH_SIZE=${BATCH_SIZE:-256}
NO_MM_TEST=${NO_MM_TEST:-true}

# 多GPU配置: GPU_IDS="0,1,2,3" 或者通过 CUDA_VISIBLE_DEVICES 指定单GPU
GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}

# Reward Model checkpoint 路径
REWARD_MODEL_SIZE=${REWARD_MODEL_SIZE:-"tiny"}
STAGE1_BACKBONE_CKPT=${STAGE1_BACKBONE_CKPT:-""}
STAGE2_CRITIC_LORA_CKPT=${STAGE2_CRITIC_LORA_CKPT:-"../checkpoints/motionreward/stage2_critic_lora_r128.pth"}
STAGE2_CRITIC_HEAD_CKPT=${STAGE2_CRITIC_HEAD_CKPT:-"../checkpoints/motionreward/stage2_critic_head_r128.pth"}
STAGE3_AI_DETECTION_LORA_CKPT=${STAGE3_AI_DETECTION_LORA_CKPT:-"../checkpoints/motionreward/stage3_ai_detection_lora_r128.pth"}
STAGE3_AI_DETECTION_HEAD_CKPT=${STAGE3_AI_DETECTION_HEAD_CKPT:-"../checkpoints/motionreward/stage3_ai_detection_head_r128.pth"}

# 辅助函数
print_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  $1"
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

# 解析 GPU 列表
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

print_header "Multi-Replication Evaluation for All MR* Checkpoints"
echo "  ├─ GPU_IDS              : ${GPU_IDS} (${NUM_GPUS} GPUs)"
echo "  ├─ REPLICATION_TIMES    : ${REPLICATION_TIMES}"
echo "  ├─ CFG_FILE             : ${CFG_FILE}"
echo "  ├─ CKPT_BASE_DIR        : ${CKPT_BASE_DIR}"
echo "  ├─ CKPT_FILTER          : ${CKPT_FILTER}"
echo "  ├─ BATCH_SIZE           : ${BATCH_SIZE}"
echo "  ├─ NO_MM_TEST           : ${NO_MM_TEST}"
echo "  ├─ REWARD_MODEL_SIZE    : ${REWARD_MODEL_SIZE}"
echo "  └─ STAGE1_BACKBONE_CKPT : ${STAGE1_BACKBONE_CKPT}"

# 收集所有 checkpoint 文件
CKPT_FILES=()
while IFS= read -r -d '' ckpt_file; do
    CKPT_FILES+=("$ckpt_file")
done < <(find "${CKPT_BASE_DIR}" -maxdepth 3 -path "*/${CKPT_FILTER}/checkpoints/*.ckpt" -print0 2>/dev/null | sort -z)

TOTAL_CKPTS=${#CKPT_FILES[@]}

if [ "$TOTAL_CKPTS" -eq 0 ]; then
    echo "Error: No .ckpt files found in ${CKPT_BASE_DIR}/${CKPT_FILTER}"
    exit 1
fi

echo ""
echo "Found ${TOTAL_CKPTS} checkpoints to evaluate:"
for i in "${!CKPT_FILES[@]}"; do
    echo "  [$((i+1))/${TOTAL_CKPTS}] ${CKPT_FILES[$i]}"
done
echo ""

# 构建额外参数
EXTRA_ARGS=""
if [ "$NO_MM_TEST" = "true" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --no_mm_test"
fi

# Reward model 参数
REWARD_ARGS=""
if [ -n "$STAGE1_BACKBONE_CKPT" ] && [ -f "$STAGE1_BACKBONE_CKPT" ]; then
    REWARD_ARGS="--reward_model_size ${REWARD_MODEL_SIZE}"
    REWARD_ARGS="${REWARD_ARGS} --critic_backbone_ckpt ${STAGE1_BACKBONE_CKPT}"
    REWARD_ARGS="${REWARD_ARGS} --critic_lora_ckpt ${STAGE2_CRITIC_LORA_CKPT}"
    REWARD_ARGS="${REWARD_ARGS} --critic_head_ckpt ${STAGE2_CRITIC_HEAD_CKPT}"
    REWARD_ARGS="${REWARD_ARGS} --ai_detection_lora_ckpt ${STAGE3_AI_DETECTION_LORA_CKPT}"
    REWARD_ARGS="${REWARD_ARGS} --ai_detection_head_ckpt ${STAGE3_AI_DETECTION_HEAD_CKPT}"
    echo "Reward model enabled"
else
    echo "WARNING: Reward model checkpoint not found, running without reward scoring"
fi

# 记录总结信息
SUMMARY_FILE="results/eval_summary_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p results
echo "Multi-Replication Evaluation Summary" > "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "REPLICATION_TIMES: ${REPLICATION_TIMES}" >> "$SUMMARY_FILE"
echo "GPU_IDS: ${GPU_IDS} (${NUM_GPUS} GPUs)" >> "$SUMMARY_FILE"
echo "Total checkpoints: ${TOTAL_CKPTS}" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"

# 过滤出需要评估的 checkpoint (跳过已有结果的)
PENDING_CKPTS=()
SKIP_COUNT=0

for i in "${!CKPT_FILES[@]}"; do
    CKPT="${CKPT_FILES[$i]}"
    IDX=$((i+1))

    CKPT_BASENAME=$(basename "${CKPT}" .ckpt)
    CKPT_DIR=$(dirname "${CKPT}")
    if [ "$(basename "${CKPT_DIR}")" = "checkpoints" ]; then
        EXP_NAME=$(basename "$(dirname "${CKPT_DIR}")")
    else
        EXP_NAME=$(basename "${CKPT_DIR}")
    fi

    EXISTING_RESULTS=$(find "results/${EXP_NAME}" -path "*/${CKPT_BASENAME}_rep*/metrics.json" 2>/dev/null | head -1)
    if [ -n "$EXISTING_RESULTS" ]; then
        SKIP_COUNT=$((SKIP_COUNT+1))
        echo "  [${IDX}/${TOTAL_CKPTS}] SKIP (already evaluated): ${EXP_NAME}/${CKPT_BASENAME}"
        echo "    Existing result: ${EXISTING_RESULTS}"
        echo "[${IDX}/${TOTAL_CKPTS}] SKIPPED (existing): ${CKPT}" >> "$SUMMARY_FILE"
    else
        PENDING_CKPTS+=("${CKPT}")
    fi
done

PENDING_COUNT=${#PENDING_CKPTS[@]}
echo ""
echo "Skipped: ${SKIP_COUNT}, Pending: ${PENDING_COUNT}"

if [ "$PENDING_COUNT" -eq 0 ]; then
    print_header "All checkpoints already evaluated!"
    exit 0
fi

# ========== 评估函数 ==========
eval_single_ckpt() {
    local CKPT="$1"
    local GPU_ID="$2"
    local IDX="$3"
    local TOTAL="$4"

    local CKPT_BASENAME=$(basename "${CKPT}" .ckpt)
    local CKPT_DIR=$(dirname "${CKPT}")
    local EXP_NAME
    if [ "$(basename "${CKPT_DIR}")" = "checkpoints" ]; then
        EXP_NAME=$(basename "$(dirname "${CKPT_DIR}")")
    else
        EXP_NAME=$(basename "${CKPT_DIR}")
    fi

    echo "[GPU ${GPU_ID}] [${IDX}/${TOTAL}] Evaluating: ${EXP_NAME}/$(basename ${CKPT})"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_mld.py \
        --cfg ${CFG_FILE} \
        --ckpt_path ${CKPT} \
        --replication_times ${REPLICATION_TIMES} \
        --batch_size ${BATCH_SIZE} \
        ${EXTRA_ARGS} \
        ${REWARD_ARGS} \
        > "results/.eval_gpu${GPU_ID}_${CKPT_BASENAME}.log" 2>&1

    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[GPU ${GPU_ID}] [${IDX}/${TOTAL}] SUCCESS: ${EXP_NAME}/$(basename ${CKPT})"
        echo "[${IDX}/${TOTAL}] SUCCESS (GPU ${GPU_ID}): ${CKPT}" >> "$SUMMARY_FILE"
    else
        echo "[GPU ${GPU_ID}] [${IDX}/${TOTAL}] FAILED: ${EXP_NAME}/$(basename ${CKPT}) (exit=${EXIT_CODE})"
        echo "[${IDX}/${TOTAL}] FAILED (GPU ${GPU_ID}, exit=${EXIT_CODE}): ${CKPT}" >> "$SUMMARY_FILE"
    fi
    return $EXIT_CODE
}

# ========== 多GPU并行 or 单GPU串行 ==========
EVAL_COUNT=0
FAIL_COUNT=0

if [ "$NUM_GPUS" -gt 1 ]; then
    print_header "Running ${PENDING_COUNT} evaluations on ${NUM_GPUS} GPUs in parallel"

    # 使用 GNU parallel 风格的 round-robin 分配
    PIDS=()
    for i in "${!PENDING_CKPTS[@]}"; do
        CKPT="${PENDING_CKPTS[$i]}"
        GPU_IDX=$((i % NUM_GPUS))
        GPU_ID="${GPU_ARRAY[$GPU_IDX]}"

        eval_single_ckpt "$CKPT" "$GPU_ID" "$((i+1))" "$PENDING_COUNT" &
        PIDS+=($!)

        # 当所有 GPU 都在忙时，等待任一完成
        if [ ${#PIDS[@]} -ge $NUM_GPUS ]; then
            # 等待最早的那个完成
            wait "${PIDS[0]}"
            if [ $? -eq 0 ]; then
                EVAL_COUNT=$((EVAL_COUNT+1))
            else
                FAIL_COUNT=$((FAIL_COUNT+1))
            fi
            PIDS=("${PIDS[@]:1}")  # 移除第一个
        fi
    done

    # 等待所有剩余进程
    for pid in "${PIDS[@]}"; do
        wait "$pid"
        if [ $? -eq 0 ]; then
            EVAL_COUNT=$((EVAL_COUNT+1))
        else
            FAIL_COUNT=$((FAIL_COUNT+1))
        fi
    done
else
    print_header "Running ${PENDING_COUNT} evaluations on GPU ${GPU_ARRAY[0]} sequentially"

    for i in "${!PENDING_CKPTS[@]}"; do
        CKPT="${PENDING_CKPTS[$i]}"
        print_header "Evaluating [$((i+1))/${PENDING_COUNT}]"

        CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} python eval_mld.py \
            --cfg ${CFG_FILE} \
            --ckpt_path ${CKPT} \
            --replication_times ${REPLICATION_TIMES} \
            --batch_size ${BATCH_SIZE} \
            ${EXTRA_ARGS} \
            ${REWARD_ARGS}

        if [ $? -eq 0 ]; then
            EVAL_COUNT=$((EVAL_COUNT+1))
            echo "[$((i+1))/${PENDING_COUNT}] SUCCESS: ${CKPT}" >> "$SUMMARY_FILE"
        else
            FAIL_COUNT=$((FAIL_COUNT+1))
            echo "[$((i+1))/${PENDING_COUNT}] FAILED: ${CKPT}" >> "$SUMMARY_FILE"
        fi
    done
fi

# 最终总结
print_header "Evaluation Complete"
echo "  ├─ Total checkpoints : ${TOTAL_CKPTS}"
echo "  ├─ Skipped (existing): ${SKIP_COUNT}"
echo "  ├─ Evaluated         : ${EVAL_COUNT}"
echo "  ├─ Failed            : ${FAIL_COUNT}"
echo "  └─ Summary file      : ${SUMMARY_FILE}"

echo "" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "Total: ${TOTAL_CKPTS}, Skipped: ${SKIP_COUNT}, Evaluated: ${EVAL_COUNT}, Failed: ${FAIL_COUNT}" >> "$SUMMARY_FILE"

# 汇总所有结果到一个 JSON
print_header "Collecting all metrics.json into summary"

python -c "
import os, json, glob

results_dir = 'results'
summary = {}

for metrics_file in sorted(glob.glob(os.path.join(results_dir, '**/metrics.json'), recursive=True)):
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        meta = data.get('_meta', {})
        ckpt = meta.get('checkpoint', metrics_file)
        exp = meta.get('experiment', 'unknown')

        key_metrics = {}
        for k in ['Metrics/FID/mean', 'Metrics/FID/conf_interval',
                   'Metrics/R_precision_top_1/mean', 'Metrics/R_precision_top_1/conf_interval',
                   'Metrics/R_precision_top_2/mean', 'Metrics/R_precision_top_2/conf_interval',
                   'Metrics/R_precision_top_3/mean', 'Metrics/R_precision_top_3/conf_interval',
                   'Metrics/Matching_score/mean', 'Metrics/Matching_score/conf_interval',
                   'Metrics/Diversity/mean', 'Metrics/Diversity/conf_interval',
                   'Metrics/MultiModality/mean', 'Metrics/MultiModality/conf_interval',
                   'Reward/critic/mean', 'Reward/critic/conf_interval',
                   'Reward/retrieval/mean', 'Reward/retrieval/conf_interval',
                   'Reward/m2m/mean', 'Reward/m2m/conf_interval',
                   'Reward/ai_detection/mean', 'Reward/ai_detection/conf_interval']:
            if k in data:
                key_metrics[k] = data[k]

        ckpt_name = os.path.basename(ckpt)
        summary[f'{exp}/{ckpt_name}'] = key_metrics

    except Exception as e:
        print(f'Warning: Failed to read {metrics_file}: {e}')

summary_file = os.path.join(results_dir, 'all_metrics_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=4)
print(f'Summary saved to {summary_file}')

# 打印表格
print(f'\n{\"=\"*150}')
print(f'{\"Experiment/Checkpoint\":<55} {\"FID\":>14} {\"R@1\":>14} {\"R@2\":>14} {\"R@3\":>14} {\"Div\":>14} {\"Critic\":>14} {\"AI_Det\":>14}')
print(f'{\"=\"*150}')
for name, m in summary.items():
    fid = f\"{m.get('Metrics/FID/mean', 0):.4f}±{m.get('Metrics/FID/conf_interval', 0):.4f}\" if 'Metrics/FID/mean' in m else 'N/A'
    r1 = f\"{m.get('Metrics/R_precision_top_1/mean', 0):.3f}±{m.get('Metrics/R_precision_top_1/conf_interval', 0):.3f}\" if 'Metrics/R_precision_top_1/mean' in m else 'N/A'
    r2 = f\"{m.get('Metrics/R_precision_top_2/mean', 0):.3f}±{m.get('Metrics/R_precision_top_2/conf_interval', 0):.3f}\" if 'Metrics/R_precision_top_2/mean' in m else 'N/A'
    r3 = f\"{m.get('Metrics/R_precision_top_3/mean', 0):.3f}±{m.get('Metrics/R_precision_top_3/conf_interval', 0):.3f}\" if 'Metrics/R_precision_top_3/mean' in m else 'N/A'
    div = f\"{m.get('Metrics/Diversity/mean', 0):.3f}±{m.get('Metrics/Diversity/conf_interval', 0):.3f}\" if 'Metrics/Diversity/mean' in m else 'N/A'
    critic = f\"{m.get('Reward/critic/mean', 0):.4f}±{m.get('Reward/critic/conf_interval', 0):.4f}\" if 'Reward/critic/mean' in m else 'N/A'
    ai_det = f\"{m.get('Reward/ai_detection/mean', 0):.4f}±{m.get('Reward/ai_detection/conf_interval', 0):.4f}\" if 'Reward/ai_detection/mean' in m else 'N/A'
    print(f'{name:<55} {fid:>14} {r1:>14} {r2:>14} {r3:>14} {div:>14} {critic:>14} {ai_det:>14}')
print(f'{\"=\"*150}')
"

echo ""
echo "Done!"
