"""
DDP (Distributed Data Parallel) 工具函数
"""

import os
import torch
import torch.distributed as dist


def setup_ddp():
    """设置 DDP 环境
    
    Returns:
        tuple: (rank, world_size, local_rank, is_distributed)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_ddp():
    """清理 DDP 环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """判断是否为主进程
    
    Args:
        rank: 进程 rank
    
    Returns:
        bool: 是否为主进程
    """
    return rank == 0
