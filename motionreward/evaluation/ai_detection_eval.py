"""
AI Detection 评估函数

支持任意 k 个表征类型的评估和记录
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm


def eval_ai_detection(val_loader, model, device='cuda', rank=0):
    """AI 检测评估
    
    支持任意 k 个表征类型，每个表征单独统计并返回
    
    Args:
        val_loader: 验证数据加载器
        model: 模型
        device: 设备
        rank: 进程 rank
    
    Returns:
        dict: 包含 loss, acc, precision, recall, f1 等指标
            - 'loss': float - 总体平均损失
            - 'acc': float - 总体准确率
            - 'precision': float - 总体精确率
            - 'recall': float - 总体召回率
            - 'f1': float - 总体 F1 分数
            - 'acc_{repr_type}': float - 各表征类型的准确率
            - 'f1_{repr_type}': float - 各表征类型的 F1 分数
            - 'count_{repr_type}': int - 各表征类型的样本数
            - 'repr_types': list - 所有评估的表征类型列表
    """
    actual_model = model.module if hasattr(model, 'module') else model
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    n_batches = 0
    
    # 使用 defaultdict 动态统计每个表征类型
    repr_stats = defaultdict(lambda: {'preds': [], 'labels': [], 'loss_sum': 0.0, 'n_batches': 0})
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc='AI Detection Eval', disable=rank != 0):
            repr_type = batch_data['repr_type']
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
            
            logits, labels = actual_model.forward_ai_detection(batch_data)
            
            if logits is None:
                continue
            
            loss = F.cross_entropy(logits, labels)
            loss_val = loss.item()
            total_loss += loss_val
            
            preds = logits.argmax(dim=-1)
            preds_list = preds.cpu().tolist()
            labels_list = labels.cpu().tolist()
            
            all_preds.extend(preds_list)
            all_labels.extend(labels_list)
            
            # 动态统计每个表征类型
            repr_stats[repr_type]['preds'].extend(preds_list)
            repr_stats[repr_type]['labels'].extend(labels_list)
            repr_stats[repr_type]['loss_sum'] += loss_val
            repr_stats[repr_type]['n_batches'] += 1
            
            n_batches += 1
    
    if n_batches == 0:
        return {'loss': 0.0, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'repr_types': []}
    
    avg_loss = total_loss / n_batches
    
    # 计算整体指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = (all_preds == all_labels).mean()
    
    # 计算 precision, recall, f1 (对于 AI 生成类，label=1)
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    result = {
        'loss': avg_loss,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'repr_types': list(repr_stats.keys())
    }
    
    # 添加每个表征类型的指标
    for repr_type, stats in repr_stats.items():
        if len(stats['preds']) > 0:
            preds_arr = np.array(stats['preds'])
            labels_arr = np.array(stats['labels'])
            
            # 准确率
            repr_acc = (preds_arr == labels_arr).mean()
            result[f'acc_{repr_type}'] = repr_acc
            
            # F1 分数
            tp_r = ((preds_arr == 1) & (labels_arr == 1)).sum()
            fp_r = ((preds_arr == 1) & (labels_arr == 0)).sum()
            fn_r = ((preds_arr == 0) & (labels_arr == 1)).sum()
            
            prec_r = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0.0
            rec_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0.0
            f1_r = 2 * prec_r * rec_r / (prec_r + rec_r) if (prec_r + rec_r) > 0 else 0.0
            result[f'f1_{repr_type}'] = f1_r
            result[f'precision_{repr_type}'] = prec_r
            result[f'recall_{repr_type}'] = rec_r
            
            # 损失
            if stats['n_batches'] > 0:
                result[f'loss_{repr_type}'] = stats['loss_sum'] / stats['n_batches']
            
            # 样本数
            result[f'count_{repr_type}'] = len(stats['preds'])
    
    return result
