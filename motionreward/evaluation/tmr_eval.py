"""
TMR (Text-Motion Retrieval) 评估函数

独立于 mld 的评估实现
"""

import os
import random
import torch
import numpy as np
from tqdm import tqdm

from .retrieval_metrics import calculate_retrieval_metrics, calculate_retrieval_metrics_small_batches


# 日志文件
os.makedirs('./logs', exist_ok=True)
fptr = open('./logs/tmr_eval.log', 'a')


def process_T5_outputs(raw_texts, T5_model):
    """处理 T5 模型输出
    
    与 train_retrieval_multi_repr.py 保持完全一致
    """
    with torch.no_grad():
        T5_outputs = T5_model.encode(raw_texts, show_progress_bar=False, batch_size=len(raw_texts), output_value=None)
        attn_masks = torch.stack([item['attention_mask'] for item in T5_outputs])
        token_embeddings = torch.stack([item['token_embeddings'] for item in T5_outputs])
        sentence_embeddings = torch.stack([item['sentence_embedding'] for item in T5_outputs])
        t_length = attn_masks.sum(1).tolist()
        return t_length, token_embeddings.float(), sentence_embeddings.unsqueeze(1).float()


def eval_tmr(test_loader, model, epoch=0, repr_type='263', mode='M1T0', writer=None, device='cuda', rank=0):
    """TMR Retrieval 评估
    
    独立于 mld 的评估实现
    
    Args:
        test_loader: 测试数据加载器
        model: 模型
        epoch: 当前 epoch
        repr_type: 表征类型 ('263' 或 '22x3')
        mode: 评估模式 ('M1T0', 'M0T1', 'M1T1', etc.)
        writer: SwanLab writer
        device: 设备
        rank: 进程 rank
    
    Returns:
        dict: 包含 BS32 指标的字典，格式为:
            {
                'bs32': {'t2m': {...}, 'm2t': {...}}
            }
    """
    
    actual_model = model.module if hasattr(model, 'module') else model
    
    test_result = [[], [], []]
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Eval TMR [{repr_type}]', disable=rank != 0):
        feats_ref, text, m_len = batch['motion'], batch['text'], batch['length']
        assert len(text) <= 32, "Please follow Small Batch protocol"
        
        timestep = torch.zeros(len(text),).long().to(device)
        with torch.no_grad():
            t_len, token_emb, cls_token = process_T5_outputs(text, actual_model.clip)
        
        feats_ref = feats_ref.float().to(device)
        if isinstance(m_len, torch.Tensor):
            m_len = m_len.tolist() if m_len.numel() > 1 else [m_len.item()] * len(text)
        elif not isinstance(m_len, list):
            m_len = list(m_len)
        
        m_latent = actual_model.encode_motion(feats_ref, m_len, repr_type=repr_type, timestep=timestep if mode[3] == '1' else None)[0].squeeze().detach().cpu().numpy()
        t_latent = actual_model.encode_text(token_emb, t_len, timestep=timestep if mode[1] == '1' else None)[0].squeeze().detach().cpu().numpy()
        
        for j in range(len(text)):
            test_result[0].append(text[j])
            test_result[1].append(t_latent[j])
            test_result[2].append(m_latent[j])
        
        del timestep, feats_ref, t_len, token_emb, cls_token, m_latent, t_latent
        torch.cuda.empty_cache()
    
    eval_results = {'bs32': {'t2m': None, 'm2t': None}}
    
    if rank == 0:
        random.seed(42)
        shuffle_index = [i for i in range(len(test_result[2]))]
        random.shuffle(shuffle_index)
        
        test_result[0] = [test_result[0][i] for i in shuffle_index]
        test_result[1] = [test_result[1][i] for i in shuffle_index]
        test_result[2] = [test_result[2][i] for i in shuffle_index]

        print(f'==================T2M Retrieval Results ({repr_type})====================')
        t2m_metrics = calculate_retrieval_metrics_small_batches(test_result, epoch=epoch, fptr=fptr)
        temp = test_result[2]
        test_result[2] = test_result[1]
        test_result[1] = temp
        print(f'==================M2T Retrieval Results ({repr_type})====================')
        m2t_metrics = calculate_retrieval_metrics_small_batches(test_result, epoch=epoch, fptr=fptr)
        fptr.flush()
        
        # 存储评估结果（仅 BS32）
        eval_results['bs32']['t2m'] = t2m_metrics
        eval_results['bs32']['m2t'] = m2t_metrics
        
        if writer is not None:
            # BS32 metrics only
            if t2m_metrics:
                for key, val in t2m_metrics.items():
                    writer.log({f"Val/{repr_type}/T2M_{key}": val}, step=epoch)
            if m2t_metrics:
                for key, val in m2t_metrics.items():
                    writer.log({f"Val/{repr_type}/M2T_{key}": val}, step=epoch)
    
    return eval_results
