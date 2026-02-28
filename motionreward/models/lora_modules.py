"""
LoRA (Low-Rank Adaptation) 模块

提供 LoRA 层实现和注入函数，用于高效微调大模型
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """LoRA 低秩适配层
    
    将原始权重 W 分解为 W + BA，其中 B 和 A 是低秩矩阵
    只训练 B 和 A，冻结原始 W
    
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        dropout: Dropout 概率
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 低秩矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量 [*, in_features]
            
        Returns:
            LoRA 增量: BA * x * scaling [*, out_features]
        """
        result = self.lora_dropout(x)
        result = F.linear(result, self.lora_A)  # [*, rank]
        result = F.linear(result, self.lora_B)  # [*, out_features]
        return result * self.scaling


class LoRALinear(nn.Module):
    """带 LoRA 的线性层
    
    output = W @ x + b + LoRA(x)
    
    Args:
        original_linear: 原始 nn.Linear 层
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        dropout: Dropout 概率
    """
    def __init__(self, original_linear, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.disabled = False  # LoRA 禁用开关
        
        # 冻结原始权重
        for param in self.original_linear.parameters():
            param.requires_grad = False
        
        # LoRA 层
        self.lora = LoRALayer(
            self.in_features, self.out_features,
            rank=rank, alpha=alpha, dropout=dropout
        )
    
    def forward(self, x):
        original_output = self.original_linear(x)
        if self.disabled:
            return original_output
        lora_output = self.lora(x)
        return original_output + lora_output


class LoRAMultiheadAttention(nn.Module):
    """带 LoRA 的 MultiheadAttention
    
    对 Q, K, V 投影层添加 LoRA
    
    Args:
        original_mha: 原始 nn.MultiheadAttention 层
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        dropout: Dropout 概率
        lora_q: 是否对 Q 添加 LoRA
        lora_k: 是否对 K 添加 LoRA
        lora_v: 是否对 V 添加 LoRA
        lora_o: 是否对输出投影添加 LoRA
    """
    def __init__(self, original_mha, rank=8, alpha=16, dropout=0.1, 
                 lora_q=True, lora_k=True, lora_v=True, lora_o=True):
        super().__init__()
        self.original_mha = original_mha
        self.embed_dim = original_mha.embed_dim
        self.num_heads = original_mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.disabled = False  # LoRA 禁用开关
        
        # 冻结原始参数
        for param in self.original_mha.parameters():
            param.requires_grad = False
        
        # 为 Q, K, V, O 添加 LoRA
        self.lora_q = LoRALayer(self.embed_dim, self.embed_dim, rank, alpha, dropout) if lora_q else None
        self.lora_k = LoRALayer(self.embed_dim, self.embed_dim, rank, alpha, dropout) if lora_k else None
        self.lora_v = LoRALayer(self.embed_dim, self.embed_dim, rank, alpha, dropout) if lora_v else None
        self.lora_o = LoRALayer(self.embed_dim, self.embed_dim, rank, alpha, dropout) if lora_o else None
    
    def forward(self, query, key, value, key_padding_mask=None, 
                need_weights=True, attn_mask=None, average_attn_weights=True):
        # 如果 LoRA 被禁用，直接调用原始 MHA
        if self.disabled:
            return self.original_mha(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights
            )
        
        # 添加 LoRA 增量
        if self.lora_q is not None:
            query = query + self.lora_q(query)
        if self.lora_k is not None:
            key = key + self.lora_k(key)
        if self.lora_v is not None:
            value = value + self.lora_v(value)
        
        # 调用原始 MHA
        output, attn_weights = self.original_mha(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights
        )
        
        # 输出投影 LoRA
        if self.lora_o is not None:
            output = output + self.lora_o(output)
        
        return output, attn_weights


def inject_lora_to_encoder(encoder, rank=8, alpha=16, dropout=0.1, prefix=''):
    """向 SkipTransformerEncoder 注入 LoRA
    
    Args:
        encoder: SkipTransformerEncoder 实例
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        dropout: LoRA dropout
        prefix: 参数名前缀（用于区分不同任务的 LoRA）
    
    Returns:
        lora_modules: LoRA 模块字典
        lora_params: LoRA 参数列表
    """
    lora_modules = {}
    lora_params = []
    
    # 获取 encoder 所在的设备
    device = next(encoder.parameters()).device
    
    # 处理 input_blocks
    for i, block in enumerate(encoder.input_blocks):
        # 注入 self_attn LoRA
        lora_mha = LoRAMultiheadAttention(
            block.self_attn, rank=rank, alpha=alpha, dropout=dropout
        ).to(device)  # 移到正确的设备
        block.self_attn = lora_mha
        lora_modules[f'{prefix}input_block_{i}_attn'] = lora_mha
        lora_params.extend(lora_mha.parameters())
        
        # 注入 FFN LoRA
        lora_linear1 = LoRALinear(block.linear1, rank=rank, alpha=alpha, dropout=dropout).to(device)
        lora_linear2 = LoRALinear(block.linear2, rank=rank, alpha=alpha, dropout=dropout).to(device)
        block.linear1 = lora_linear1
        block.linear2 = lora_linear2
        lora_modules[f'{prefix}input_block_{i}_ffn1'] = lora_linear1
        lora_modules[f'{prefix}input_block_{i}_ffn2'] = lora_linear2
        lora_params.extend(lora_linear1.parameters())
        lora_params.extend(lora_linear2.parameters())
    
    # 处理 middle_block
    lora_mha = LoRAMultiheadAttention(
        encoder.middle_block.self_attn, rank=rank, alpha=alpha, dropout=dropout
    ).to(device)
    encoder.middle_block.self_attn = lora_mha
    lora_modules[f'{prefix}middle_block_attn'] = lora_mha
    lora_params.extend(lora_mha.parameters())
    
    lora_linear1 = LoRALinear(encoder.middle_block.linear1, rank=rank, alpha=alpha, dropout=dropout).to(device)
    lora_linear2 = LoRALinear(encoder.middle_block.linear2, rank=rank, alpha=alpha, dropout=dropout).to(device)
    encoder.middle_block.linear1 = lora_linear1
    encoder.middle_block.linear2 = lora_linear2
    lora_modules[f'{prefix}middle_block_ffn1'] = lora_linear1
    lora_modules[f'{prefix}middle_block_ffn2'] = lora_linear2
    lora_params.extend(lora_linear1.parameters())
    lora_params.extend(lora_linear2.parameters())
    
    # 处理 output_blocks
    for i, block in enumerate(encoder.output_blocks):
        lora_mha = LoRAMultiheadAttention(
            block.self_attn, rank=rank, alpha=alpha, dropout=dropout
        ).to(device)
        block.self_attn = lora_mha
        lora_modules[f'{prefix}output_block_{i}_attn'] = lora_mha
        lora_params.extend(lora_mha.parameters())
        
        lora_linear1 = LoRALinear(block.linear1, rank=rank, alpha=alpha, dropout=dropout).to(device)
        lora_linear2 = LoRALinear(block.linear2, rank=rank, alpha=alpha, dropout=dropout).to(device)
        block.linear1 = lora_linear1
        block.linear2 = lora_linear2
        lora_modules[f'{prefix}output_block_{i}_ffn1'] = lora_linear1
        lora_modules[f'{prefix}output_block_{i}_ffn2'] = lora_linear2
        lora_params.extend(lora_linear1.parameters())
        lora_params.extend(lora_linear2.parameters())
    
    return lora_modules, lora_params


def get_lora_state_dict(lora_modules):
    """获取 LoRA 模块的 state_dict
    
    Args:
        lora_modules: LoRA 模块字典
        
    Returns:
        state_dict: LoRA 参数字典
    """
    state_dict = {}
    for name, module in lora_modules.items():
        for param_name, param in module.named_parameters():
            if 'lora' in param_name:
                state_dict[f'{name}.{param_name}'] = param.data.clone()
    return state_dict


def load_lora_state_dict(lora_modules, state_dict):
    """加载 LoRA state_dict
    
    Args:
        lora_modules: LoRA 模块字典
        state_dict: LoRA 参数字典
    """
    for name, module in lora_modules.items():
        for param_name, param in module.named_parameters():
            if 'lora' in param_name:
                key = f'{name}.{param_name}'
                if key in state_dict:
                    param.data.copy_(state_dict[key])


def count_lora_params(lora_modules):
    """统计 LoRA 参数量
    
    Args:
        lora_modules: LoRA 模块字典
        
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = 0
    trainable_params = 0
    for module in lora_modules.values():
        for param in module.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
    return total_params, trainable_params


def freeze_non_lora_params(model):
    """冻结非 LoRA 参数
    
    Args:
        model: 模型实例
    """
    for name, param in model.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = False


def unfreeze_lora_params(model):
    """解冻 LoRA 参数
    
    Args:
        model: 模型实例
    """
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True


def disable_lora(lora_modules):
    """禁用 LoRA 模块（评估 Retrieval 时使用）
    
    Args:
        lora_modules: LoRA 模块字典
    """
    if lora_modules is None:
        return
    for module in lora_modules.values():
        if hasattr(module, 'disabled'):
            module.disabled = True


def enable_lora(lora_modules):
    """启用 LoRA 模块
    
    Args:
        lora_modules: LoRA 模块字典
    """
    if lora_modules is None:
        return
    for module in lora_modules.values():
        if hasattr(module, 'disabled'):
            module.disabled = False


class LoRADisabled:
    """LoRA 禁用上下文管理器
    
    用于在评估 Retrieval 时临时禁用 LoRA
    
    Usage:
        with LoRADisabled(model.critic_lora_modules):
            eval_tmr(...)
    """
    def __init__(self, lora_modules):
        self.lora_modules = lora_modules
    
    def __enter__(self):
        disable_lora(self.lora_modules)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        enable_lora(self.lora_modules)
        return False
