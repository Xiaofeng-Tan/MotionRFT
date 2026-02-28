"""
Multi-replication evaluation for RFT-MLD checkpoints.

参考 MotionLCM test.py 的多次重复评估逻辑:
- 对每个 checkpoint 运行 REPLICATION_TIMES 次评估
- 计算 mean 和 95% 置信区间 (CI = 1.96 * std / sqrt(N))
- 支持 MultiModality (MM) 测试
- 支持 Reward Model (Critic / AI Detection / Retrieval / M2M) 评分
- 结果保存到 metrics.json

用法:
    python eval_mld.py \
        --cfg configs/ft_mld_t2m.yaml \
        --ckpt_path <path_to_checkpoint.ckpt> \
        --replication_times 20 \
        --reward_model_size tiny \
        --critic_backbone_ckpt <path> \
        --critic_lora_ckpt <path> \
        --critic_head_ckpt <path> \
        --ai_detection_lora_ckpt <path> \
        --ai_detection_head_ckpt <path>
"""

import os
import sys
import json
import random
import logging
import datetime
import os.path as osp
from typing import Union

import numpy as np
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mld.config import get_module_config
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.utils.utils import print_table, set_seed, move_batch_to_device

# 添加 motionreward 到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_metric_statistics(values: np.ndarray, replication_times: int) -> tuple:
    """计算 mean 和 95% 置信区间"""
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def load_reward_model(args, device, logger):
    """加载 CriticRewardAdapter (仅在需要时)"""
    if not args.critic_backbone_ckpt:
        return None

    from motionrft_mld import CriticRewardAdapter

    logger.info("Loading Reward Model...")
    logger.info(f"  Backbone: {args.critic_backbone_ckpt}")
    logger.info(f"  Critic LoRA: {args.critic_lora_ckpt}")
    logger.info(f"  Critic Head: {args.critic_head_ckpt}")
    logger.info(f"  AI Det LoRA: {args.ai_detection_lora_ckpt}")
    logger.info(f"  AI Det Head: {args.ai_detection_head_ckpt}")
    logger.info(f"  Model Size: {args.reward_model_size}")

    t5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'deps', 'sentence-t5-large')

    reward_model = CriticRewardAdapter(
        backbone_ckpt=args.critic_backbone_ckpt,
        critic_lora_ckpt=args.critic_lora_ckpt,
        critic_head_ckpt=args.critic_head_ckpt,
        ai_detection_lora_ckpt=args.ai_detection_lora_ckpt,
        ai_detection_head_ckpt=args.ai_detection_head_ckpt,
        t5_path=t5_path,
        repr_type='263',
        model_size=args.reward_model_size,
        lora_rank=args.reward_lora_rank,
        lora_alpha=args.reward_lora_alpha,
        device=str(device),
        lambda_critic=1.0,
        lambda_retrieval=1.0,
        lambda_m2m=1.0,
        lambda_ai_detection=1.0,
    ).to(device)
    reward_model.eval()
    reward_model.requires_grad_(False)
    logger.info("Reward Model loaded successfully!")
    return reward_model


@torch.no_grad()
def test_one_epoch(model: MLD, dataloader: DataLoader, device: torch.device,
                   reward_model=None) -> dict:
    """运行一次完整的测试 epoch，包括 reward 评分"""
    all_critic_scores = []
    all_retrieval_scores = []
    all_m2m_scores = []
    all_ai_detection_scores = []

    for batch in tqdm(dataloader, desc="  Eval batches"):
        batch = move_batch_to_device(batch, device)
        model.test_step(batch)

        # Reward model 打分
        if reward_model is not None:
            try:
                rs_set = model.t2m_eval(batch)
                feats_rst_raw = rs_set['feats_rst_raw']
                lengths = batch['length']
                if isinstance(lengths, torch.Tensor):
                    lengths = lengths.tolist()

                gt_motion_feats = batch.get('motion', None)
                gt_len = batch.get('length', None)
                if isinstance(gt_len, torch.Tensor):
                    gt_len = gt_len.tolist()

                _, critic_score, retrieval_score, m2m_score, ai_detection_score = \
                    reward_model.get_reward_t2m(
                        raw_texts=batch['text'],
                        motion_feats=feats_rst_raw,
                        m_len=lengths,
                        gt_motion_feats=gt_motion_feats,
                        gt_len=gt_len,
                        return_details=True
                    )
                if critic_score is not None:
                    all_critic_scores.append(critic_score.cpu())
                if retrieval_score is not None:
                    all_retrieval_scores.append(retrieval_score.cpu())
                if m2m_score is not None:
                    all_m2m_scores.append(m2m_score.cpu())
                if ai_detection_score is not None:
                    all_ai_detection_scores.append(ai_detection_score.cpu())
            except Exception as e:
                logging.getLogger(__name__).warning(f"Reward scoring failed: {e}")

    metrics = model.allsplit_epoch_end()

    # 添加 reward metrics
    if len(all_critic_scores) > 0:
        metrics["Reward/critic"] = torch.cat(all_critic_scores).mean().item()
    if len(all_retrieval_scores) > 0:
        metrics["Reward/retrieval"] = torch.cat(all_retrieval_scores).mean().item()
    if len(all_m2m_scores) > 0:
        metrics["Reward/m2m"] = torch.cat(all_m2m_scores).mean().item()
    if len(all_ai_detection_scores) > 0:
        metrics["Reward/ai_detection"] = torch.cat(all_ai_detection_scores).mean().item()

    return metrics


def main():
    # ========== 解析参数 ==========
    parser = ArgumentParser(description="Multi-replication evaluation for RFT-MLD checkpoints")
    parser.add_argument("--cfg", type=str, required=True, help="Config file path")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint path (.ckpt)")
    parser.add_argument("--replication_times", type=int, default=None,
                        help="Number of replication runs (default: use config TEST.REPLICATION_TIMES)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-generated under results/)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override test batch size")
    parser.add_argument("--no_mm_test", action="store_true",
                        help="Disable MultiModality test")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: use config SEED_VALUE)")

    # Reward model 参数
    parser.add_argument("--reward_model_size", type=str, default="tiny",
                        choices=['retrieval_original', 'tiny', 'small', 'base', 'large', 'xlarge', 'xxlarge', 'giant'],
                        help="Reward model size")
    parser.add_argument("--reward_lora_rank", type=int, default=128,
                        help="LoRA rank for MotionReward (must match checkpoint)")
    parser.add_argument("--reward_lora_alpha", type=int, default=256,
                        help="LoRA alpha for MotionReward (must match checkpoint)")
    parser.add_argument("--critic_backbone_ckpt", type=str, default="",
                        help="Stage 1 backbone checkpoint")
    parser.add_argument("--critic_lora_ckpt", type=str, default="",
                        help="Stage 2 Critic LoRA checkpoint")
    parser.add_argument("--critic_head_ckpt", type=str, default="",
                        help="Stage 2 Critic Head checkpoint")
    parser.add_argument("--ai_detection_lora_ckpt", type=str, default="",
                        help="Stage 3 AI Detection LoRA checkpoint")
    parser.add_argument("--ai_detection_head_ckpt", type=str, default="",
                        help="Stage 3 AI Detection Head checkpoint")

    args = parser.parse_args()

    # ========== 加载配置 ==========
    cfg = OmegaConf.load(args.cfg)
    cfg_root = os.path.dirname(args.cfg)
    cfg_model = get_module_config(cfg.model, cfg.model.target, cfg_root)
    cfg = OmegaConf.merge(cfg, cfg_model)

    # 覆盖配置
    cfg.TEST.CHECKPOINTS = args.ckpt_path
    if args.batch_size is not None:
        cfg.TEST.BATCH_SIZE = args.batch_size
        cfg.VAL.BATCH_SIZE = args.batch_size
    if args.no_mm_test:
        cfg.TEST.DO_MM_TEST = False

    replication_times = args.replication_times if args.replication_times is not None else cfg.TEST.REPLICATION_TIMES
    seed_value = args.seed if args.seed is not None else cfg.SEED_VALUE

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(seed_value)

    # ========== 设置输出目录 ==========
    ckpt_name = osp.splitext(osp.basename(args.ckpt_path))[0]
    ckpt_dir = osp.dirname(args.ckpt_path)
    exp_name = osp.basename(osp.dirname(ckpt_dir)) if osp.basename(ckpt_dir) == "checkpoints" else osp.basename(ckpt_dir)

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = osp.join("results", f"{exp_name}", f"{ckpt_name}_rep{replication_times}_{time_str}")

    os.makedirs(output_dir, exist_ok=True)
    cfg.output_dir = output_dir

    # ========== 设置 logging ==========
    steam_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(osp.join(output_dir, 'output.log'))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[steam_handler, file_handler]
    )
    logger = logging.getLogger(__name__)

    OmegaConf.save(cfg, osp.join(output_dir, 'config.yaml'))

    has_reward = bool(args.critic_backbone_ckpt)
    logger.info(f"="*70)
    logger.info(f"Multi-Replication Evaluation")
    logger.info(f"  Checkpoint: {args.ckpt_path}")
    logger.info(f"  Experiment: {exp_name}")
    logger.info(f"  Replication Times: {replication_times}")
    logger.info(f"  Seed: {seed_value}")
    logger.info(f"  Output Dir: {output_dir}")
    logger.info(f"  Batch Size: {cfg.TEST.BATCH_SIZE}")
    logger.info(f"  DO_MM_TEST: {cfg.TEST.DO_MM_TEST}")
    logger.info(f"  Reward Model: {'Yes' if has_reward else 'No'}")
    logger.info(f"="*70)

    # ========== 加载 checkpoint ==========
    state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
    logger.info(f"Loading checkpoint from {args.ckpt_path}")

    # Filter out reward_model keys (saved during training but not part of MLD)
    filtered_keys = [k for k in state_dict if k.startswith("reward_model.")]
    for k in filtered_keys:
        del state_dict[k]
    if filtered_keys:
        logger.info(f"Filtered out {len(filtered_keys)} reward_model.* keys from checkpoint")

    # ========== 加载数据集和模型 ==========
    dataset = get_dataset(cfg)
    test_dataloader = dataset.test_dataloader()

    model = MLD(cfg, dataset)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    logger.info(model.load_state_dict(state_dict, strict=False))

    # ========== 加载 Reward Model ==========
    reward_model = load_reward_model(args, device, logger)

    # ========== 多次重复评估 ==========
    all_metrics = {}
    metrics_type = ", ".join(cfg.METRIC.TYPE)

    for i in range(replication_times):
        logger.info(f"Evaluating {metrics_type} - Replication {i}/{replication_times}")
        metrics = test_one_epoch(model, test_dataloader, device, reward_model=reward_model)

        # MM (MultiModality) 测试
        if "TM2TMetrics" in metrics_type and cfg.TEST.DO_MM_TEST:
            logger.info(f"Evaluating MultiModality - Replication {i}/{replication_times}")
            dataset.mm_mode(True)
            test_mm_dataloader = dataset.test_dataloader()
            mm_metrics = test_one_epoch(model, test_mm_dataloader, device)
            metrics.update(mm_metrics)
            dataset.mm_mode(False)

        print_table(f"Metrics@Replication-{i}", metrics)
        logger.info(f"Replication {i} metrics: {metrics}")

        for key, item in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = [item]
            else:
                all_metrics[key] += [item]

    # ========== 计算统计量 ==========
    all_metrics_new = dict()
    for key, item in all_metrics.items():
        mean, conf_interval = get_metric_statistics(np.array(item), replication_times)
        all_metrics_new[key + "/mean"] = float(mean)
        all_metrics_new[key + "/conf_interval"] = float(conf_interval)

    logger.info(f"\n{'='*70}")
    logger.info(f"Final Results (Mean ± 95% CI over {replication_times} replications)")
    logger.info(f"{'='*70}")
    print_table(f"Mean Metrics ({replication_times} replications)", all_metrics_new)

    # 打印关键指标
    key_metrics = ["Metrics/FID", "Metrics/R_precision_top_1", "Metrics/R_precision_top_2",
                   "Metrics/R_precision_top_3", "Metrics/Matching_score", "Metrics/Diversity"]
    logger.info(f"\nKey Metrics Summary:")
    for km in key_metrics:
        if km + "/mean" in all_metrics_new:
            logger.info(f"  {km}: {all_metrics_new[km + '/mean']:.4f} ± {all_metrics_new[km + '/conf_interval']:.4f}")

    # Reward metrics
    reward_keys = ["Reward/critic", "Reward/retrieval", "Reward/m2m", "Reward/ai_detection"]
    for km in reward_keys:
        if km + "/mean" in all_metrics_new:
            logger.info(f"  {km}: {all_metrics_new[km + '/mean']:.4f} ± {all_metrics_new[km + '/conf_interval']:.4f}")

    # MM metrics
    for km in ["Metrics/MultiModality"]:
        if km + "/mean" in all_metrics_new:
            logger.info(f"  {km}: {all_metrics_new[km + '/mean']:.4f} ± {all_metrics_new[km + '/conf_interval']:.4f}")

    # 合并原始数据
    all_metrics_serializable = {}
    for key, item in all_metrics.items():
        all_metrics_serializable[key] = [float(v) for v in item]
    all_metrics_new.update(all_metrics_serializable)

    # 添加元信息
    all_metrics_new["_meta"] = {
        "checkpoint": args.ckpt_path,
        "experiment": exp_name,
        "replication_times": replication_times,
        "seed": seed_value,
        "batch_size": cfg.TEST.BATCH_SIZE,
        "do_mm_test": cfg.TEST.DO_MM_TEST,
        "has_reward_model": has_reward,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ========== 保存结果 ==========
    metric_file = osp.join(output_dir, "metrics.json")
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics_new, f, indent=4)
    logger.info(f"\nResults saved to {metric_file}")


if __name__ == "__main__":
    main()
