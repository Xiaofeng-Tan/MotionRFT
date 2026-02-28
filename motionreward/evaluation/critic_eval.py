"""
Critic 评估函数

支持任意 k 个表征类型的评估和记录
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

from motionreward.models.heads import pairwise_loss


def eval_critic(val_loader, model, device='cuda', rank=0):
    """Critic 评估
    
    支持任意 k 个表征类型，每个表征单独统计并返回
    
    对于 precision/recall/F1：采用随机 swap 策略构造平衡二分类问题。
    原始 pair 中 col0=better, col1=worse, label=0。
    随机 swap 约 50% 的 pair，swap 后 label=1。
    预测：score[0] > score[1] → predict 0，否则 predict 1。
    
    Args:
        val_loader: 验证数据加载器
        model: 模型
        device: 设备
        rank: 进程 rank
    
    Returns:
        dict: 评估结果
            - 'loss': float - 总体平均损失
            - 'acc': float - 总体平均准确率 (原始 pairwise acc)
            - 'precision': float - 总体精确率
            - 'recall': float - 总体召回率
            - 'f1': float - 总体 F1 分数
            - 'acc_{repr_type}': float - 各表征类型的准确率
            - 'precision_{repr_type}': float - 各表征类型的精确率
            - 'recall_{repr_type}': float - 各表征类型的召回率
            - 'f1_{repr_type}': float - 各表征类型的 F1 分数
            - 'loss_{repr_type}': float - 各表征类型的损失
            - 'count_{repr_type}': int - 各表征类型的样本数
            - 'repr_types': list - 所有评估的表征类型列表
    """
    actual_model = model.module if hasattr(model, 'module') else model
    
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    
    # 收集所有 critic 输出用于计算 precision/recall/F1
    all_critics = []
    all_repr_types = []
    
    # 使用 defaultdict 动态统计每个表征类型
    repr_stats = defaultdict(lambda: {'count': 0, 'acc_sum': 0.0, 'loss_sum': 0.0, 'critics': []})
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc='Critic Eval', disable=rank != 0):
            repr_type = batch_data['repr_type']
            batch_data = {k: v.float().to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
            critic = actual_model.forward_critic(batch_data)
            
            if critic is None:
                continue
            
            loss, _, acc = pairwise_loss(critic)
            loss_val = loss.item()
            acc_val = acc.item()
            
            total_loss += loss_val
            total_acc += acc_val
            n_batches += 1
            
            # 收集 critic 输出
            all_critics.append(critic.detach().cpu())
            all_repr_types.extend([repr_type] * critic.shape[0])
            
            # 动态统计每个表征类型
            repr_stats[repr_type]['count'] += 1
            repr_stats[repr_type]['acc_sum'] += acc_val
            repr_stats[repr_type]['loss_sum'] += loss_val
            repr_stats[repr_type]['critics'].append(critic.detach().cpu())
    
    if n_batches == 0:
        return {'loss': 0.0, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'repr_types': []}
    
    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    
    # === 计算 precision/recall/F1（随机 swap 构造平衡二分类） ===
    all_critics_cat = torch.cat(all_critics, dim=0).numpy()  # [N, 2]
    N = all_critics_cat.shape[0]
    
    np.random.seed(42)
    swap_mask = np.random.rand(N) < 0.5
    
    # swap 后的 scores
    shuffled = all_critics_cat.copy()
    shuffled[swap_mask] = shuffled[swap_mask][:, [1, 0]]
    
    # 真实标签：未 swap=0，已 swap=1
    y_true = np.zeros(N, dtype=int)
    y_true[swap_mask] = 1
    
    # 预测：score[0] > score[1] → predict 0，否则 predict 1
    y_pred = (shuffled[:, 0] <= shuffled[:, 1]).astype(int)
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    result = {
        'loss': avg_loss, 
        'acc': avg_acc,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'repr_types': list(repr_stats.keys())
    }
    
    # 添加每个表征类型的指标
    all_repr_types_arr = np.array(all_repr_types)
    for repr_type, stats in repr_stats.items():
        if stats['count'] > 0:
            result[f'acc_{repr_type}'] = stats['acc_sum'] / stats['count']
            result[f'loss_{repr_type}'] = stats['loss_sum'] / stats['count']
            result[f'count_{repr_type}'] = stats['count']
            
            # 每个 repr_type 的 precision/recall/F1
            mask_repr = all_repr_types_arr == repr_type
            if mask_repr.sum() > 0:
                y_true_r = y_true[mask_repr]
                y_pred_r = y_pred[mask_repr]
                
                tp_r = ((y_pred_r == 1) & (y_true_r == 1)).sum()
                fp_r = ((y_pred_r == 1) & (y_true_r == 0)).sum()
                fn_r = ((y_pred_r == 0) & (y_true_r == 1)).sum()
                
                prec_r = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0.0
                rec_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0.0
                f1_r = 2 * prec_r * rec_r / (prec_r + rec_r) if (prec_r + rec_r) > 0 else 0.0
                
                result[f'precision_{repr_type}'] = float(prec_r)
                result[f'recall_{repr_type}'] = float(rec_r)
                result[f'f1_{repr_type}'] = float(f1_r)
    
    return result
