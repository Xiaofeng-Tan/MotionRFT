"""
检索指标计算函数

独立于 mld 的检索指标计算实现
"""

import numpy as np


def _normalize_L2(x):
    """L2 归一化"""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # 避免除零
    return x / norms


def calculate_retrieval_metrics(test_result, verbose=True, epoch=0, fptr=None):
    """计算完整数据集的检索指标
    
    Args:
        test_result: [text_list, text_latents, motion_latents]
        verbose: 是否打印结果
        epoch: 当前 epoch
        fptr: 日志文件指针
    
    Returns:
        dict: 包含 R1, R2, R3, R5, R10 的字典
    """
    text_list, text_latents, motion_latents = test_result
    text_latents = np.array(text_latents).astype('float32')
    motion_latents = np.array(motion_latents).astype('float32')

    # 数据校验
    assert len(text_latents) == len(motion_latents), "潜在向量数量不匹配"
    
    # 归一化处理（余弦相似度）
    text_latents = _normalize_L2(text_latents)
    motion_latents = _normalize_L2(motion_latents)

    # 计算相似度矩阵并获取 top-k 索引
    # sim_matrix: [N, N]，使用内积（归一化后等于余弦相似度）
    sim_matrix = np.dot(text_latents, motion_latents.T)
    
    # 获取 top-10 索引（降序排列）
    k = 10
    indices = np.argsort(-sim_matrix, axis=1)[:, :k]  # [N, k]

    # 向量化计算指标
    total = len(text_latents)
    target_ids = np.arange(total).reshape(-1, 1)  # 每个文本对应的正确目标 ID
    
    # 计算各层准确率
    r1 = np.mean(np.any(indices[:, :1] == target_ids, axis=1))
    r2 = np.mean(np.any(indices[:, :2] == target_ids, axis=1))
    r3 = np.mean(np.any(indices[:, :3] == target_ids, axis=1))
    r5 = np.mean(np.any(indices[:, :5] == target_ids, axis=1))
    r10 = np.mean(np.any(indices[:, :10] == target_ids, axis=1))

    # 结果格式化
    results = {
        'R1': round(r1 * 100, 3),
        'R2': round(r2 * 100, 3),
        'R3': round(r3 * 100, 3),
        'R5': round(r5 * 100, 3),
        'R10': round(r10 * 100, 3)
    }

    # 控制台输出
    if verbose:
        print(f"FULL | Epoch {epoch} | TMR R@k | R@1: {results['R1']}% | R@2: {results['R2']}% | R@3: {results['R3']}% | R@5: {results['R5']}% | R@10: {results['R10']}% | DB: {total} pairs")
        
        if fptr:
            print(f"FULL | Epoch {epoch} | TMR R@k | R@1: {results['R1']}% | R@2: {results['R2']}% | R@3: {results['R3']}% | R@5: {results['R5']}% | R@10: {results['R10']}% | DB: {total} pairs", file=fptr)
    
    return results


def calculate_retrieval_metrics_small_batches(test_result, batch_size=32, epoch=0, fptr=None):
    """计算小批次检索指标（BS32 协议）
    
    Args:
        test_result: [text_list, text_latents, motion_latents]
        batch_size: 批次大小
        epoch: 当前 epoch
        fptr: 日志文件指针
    
    Returns:
        dict: 包含 R1, R2, R3, R5, R10 的字典
    """
    text_list, text_latents, motion_latents = test_result
    text_latents = np.array(text_latents).astype('float32')
    motion_latents = np.array(motion_latents).astype('float32')
    total_samples = len(text_latents)
    r1_sum, r5_sum, r10_sum = 0.0, 0.0, 0.0
    r2_sum, r3_sum = 0, 0
    num_batches = 0
    batch_size = 32
    
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        
        # 构造当前批次的数据
        batch_text_latents = text_latents[start_idx:end_idx]
        batch_motion_latents = motion_latents[start_idx:end_idx]
        
        # 如果剩余样本不足，随机采样补全
        if len(batch_text_latents) < batch_size:
            remaining = batch_size - len(batch_text_latents)
            random_indices = np.random.choice(total_samples, remaining, replace=False)
            batch_text_latents = np.concatenate([batch_text_latents, text_latents[random_indices]])
            batch_motion_latents = np.concatenate([batch_motion_latents, motion_latents[random_indices]])
        
        # 调用原始函数计算当前批次的指标
        batch_result = calculate_retrieval_metrics(
            [text_list, batch_text_latents, batch_motion_latents],
            verbose=False,
            epoch=epoch
        )
        
        # 累加指标
        r1_sum += batch_result['R1']
        r2_sum += batch_result['R2']
        r3_sum += batch_result['R3']
        r5_sum += batch_result['R5']
        r10_sum += batch_result['R10']
        num_batches += 1

    # 计算平均指标
    avg_r1 = r1_sum / num_batches
    avg_r2 = r2_sum / num_batches
    avg_r3 = r3_sum / num_batches
    avg_r5 = r5_sum / num_batches
    avg_r10 = r10_sum / num_batches

    # 结果格式化
    results = {
        'R1': round(avg_r1, 3),
        'R2': round(avg_r2, 3),
        'R3': round(avg_r3, 3),
        'R5': round(avg_r5, 3),
        'R10': round(avg_r10, 3)
    }

    # 控制台输出
    total = len(text_latents)
    print(f"BS32 | Epoch {epoch} | TMR R@k | R@1: {results['R1']}% | R@2: {results['R2']}% | R@3: {results['R3']}% | R@5: {results['R5']}% | R@10: {results['R10']}% | DB: {total} pairs")
    
    if fptr:
        print(f"BS32 | Epoch {epoch} | TMR R@k | R@1: {results['R1']}% | R@2: {results['R2']}% | R@3: {results['R3']}% | R@5: {results['R5']}% | R@10: {results['R10']}% | DB: {total} pairs", file=fptr)

    return results
