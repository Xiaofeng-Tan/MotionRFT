"""
通用工具函数
从 mld/utils/utils.py 提取并整合
"""

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rich import get_console
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def set_seed(seed: int) -> None:
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_table(title: str, metrics: dict) -> None:
    """打印格式化的表格"""
    if HAS_RICH:
        table = Table(title=title)
        table.add_column("Metrics", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        for key, value in metrics.items():
            table.add_row(key, str(value))
        console = get_console()
        console.print(table, justify="center")
    else:
        print(f"\n{title}")
        print("-" * 40)
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("-" * 40)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """将 batch 中的 tensor 移动到指定设备"""
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


def count_parameters(module: nn.Module) -> float:
    """统计模块参数量（单位：百万）"""
    num_params = sum(p.numel() for p in module.parameters())
    return round(num_params / 1e6, 3)


def get_guidance_scale_embedding(
    w: torch.Tensor, 
    embedding_dim: int = 512,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """获取 guidance scale 嵌入"""
    assert len(w.shape) == 1
    w = w * 1000.0
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def extract_into_tensor(
    a: torch.Tensor, 
    t: torch.Tensor, 
    x_shape: torch.Size
) -> torch.Tensor:
    """从 tensor 中按索引提取并 reshape"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def sum_flat(tensor: torch.Tensor) -> torch.Tensor:
    """对除 batch 维度外的所有维度求和"""
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def control_loss_calculate(
    vaeloss_type: str, 
    loss_func: str, 
    src: torch.Tensor,
    tgt: torch.Tensor, 
    mask: torch.Tensor
) -> torch.Tensor:
    """计算控制损失"""
    if loss_func == 'l1':
        loss = F.l1_loss(src, tgt, reduction='none')
    elif loss_func == 'l1_smooth':
        loss = F.smooth_l1_loss(src, tgt, reduction='none')
    elif loss_func == 'l2':
        loss = F.mse_loss(src, tgt, reduction='none')
    else:
        raise ValueError(f'Unknown loss func: {loss_func}')

    if vaeloss_type == 'sum':
        loss = loss.sum(-1, keepdims=True) * mask
        loss = loss.sum() / mask.sum()
    elif vaeloss_type == 'sum_mask':
        loss = loss.sum(-1, keepdims=True) * mask
        loss = sum_flat(loss) / sum_flat(mask)
        loss = loss.mean()
    elif vaeloss_type == 'mask':
        loss = sum_flat(loss * mask)
        n_entries = src.shape[-1]
        non_zero_elements = sum_flat(mask) * n_entries
        loss = loss / non_zero_elements
        loss = loss.mean()
    else:
        raise ValueError(f'Unsupported vaeloss_type: {vaeloss_type}')

    return loss
