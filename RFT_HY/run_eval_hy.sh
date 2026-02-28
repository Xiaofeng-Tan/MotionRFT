#!/bin/bash
cd "$(dirname "$0")"

# 激活 hymotion 环境 (Qwen3 tokenizer 需要新版 transformers)
# conda activate hymotion

MODEL=../pretrain/hymotion/HY-Motion-1.0-Lite
SPM=./t2m/evaluator.pth
T5=../deps/sentence-t5-large
DATA=../datasets/humanml3d
MOTION=../datasets/humanml3d/joints_6d
CKPT_DIR=${MODEL}/rl_finetune_v2_mr_only/checkpoints

LORA_PATHS=(
    "${CKPT_DIR}/lora_step000300.pth"
    "${CKPT_DIR}/lora_step001300.pth"
    "${CKPT_DIR}/lora_step002300.pth"
    "${CKPT_DIR}/lora_step004300.pth"
    "${CKPT_DIR}/best_r_precision.pth"
)
GPUS=(2 3 4 5 6)

for i in "${!LORA_PATHS[@]}"; do
    gpu=${GPUS[$i]}
    lora=${LORA_PATHS[$i]}
    name=$(basename "$lora" .pth)
    out="${MODEL}/rl_finetune_v2_mr_only/eval_results/${name}"

    echo "GPU ${gpu}: evaluating ${name}"
    CUDA_VISIBLE_DEVICES=${gpu} python eval_hy.py \
        --spm_path ${SPM} \
        --t5_path ${T5} \
        --data_root ${DATA} \
        --motion_dir ${MOTION} \
        --model_path ${MODEL} \
        --lora_path ${lora} \
        --eval_mode both \
        --repeat_times 1 \
        --split test \
        --output_dir ${out} \
        > "${MODEL}/rl_finetune_v2_mr_only/eval_results/.eval_gpu${gpu}_${name}.log" 2>&1 &
done

echo "All 5 eval jobs launched on GPU 2-6, logs in:"
echo "  ${MODEL}/rl_finetune_v2_mr_only/eval_results/"
echo ""
echo "Monitor: tail -f ${MODEL}/rl_finetune_v2_mr_only/eval_results/.eval_gpu*.log"

wait
echo "All done."
