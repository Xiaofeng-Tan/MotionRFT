"""
Reward Model Adapter - 使用 MotionReward 作为 reward 信号

将 motionreward 的 retrieval 模型封装为与原 SPM 兼容的接口
通过 model_size 参数自动获取模型配置，确保与训练时完全一致
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加 motionreward 到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motionreward.models import MultiReprRetrievalWithLoRA
from motionreward.utils.config_utils import get_model_config


class MotionRewardAdapter(nn.Module):
    """
    MotionReward 适配器 - 封装 MultiReprRetrievalWithLoRA 作为 reward model
    
    与原 SPM 的 get_reward_t2m 接口保持兼容
    通过 model_size 参数控制模型配置，确保与训练时一致
    """
    
    def __init__(
        self,
        ckpt_path: str = None,
        t5_path: str = '../deps/sentence-t5-large',
        repr_type: str = '263',  # 支持: '263', '22x3', '135'
        model_size: str = 'base',  # 模型规模，控制所有层数配置
        temp: float = 0.1,
        thr: float = 0.8,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_unified_dim: bool = True,
        device: str = 'cuda',
        **kwargs
    ):
        super().__init__()
        
        self.repr_type = repr_type
        self.device = device
        self.model_size = model_size
        
        # 通过 model_size 获取模型配置
        model_cfg = get_model_config(model_size)
        print(f"[MotionRewardAdapter] Using model_size='{model_size}' config:")
        print(f"  latent_dim={model_cfg['latent_dim']}, unified_dim={model_cfg['unified_dim']}")
        print(f"  encoder_num_layers={model_cfg['encoder_num_layers']}, text_num_layers={model_cfg['text_num_layers']}")
        
        # 创建 MultiReprRetrievalWithLoRA 模型
        self.model = MultiReprRetrievalWithLoRA(
            t5_path=t5_path,
            temp=temp,
            thr=thr,
            latent_dim=model_cfg['latent_dim'],
            unified_dim=model_cfg['unified_dim'],
            encoder_num_layers=model_cfg['encoder_num_layers'],
            encoder_num_heads=model_cfg['encoder_num_heads'],
            encoder_ff_size=model_cfg['encoder_ff_size'],
            text_num_layers=model_cfg['text_num_layers'],
            text_num_heads=model_cfg['text_num_heads'],
            text_ff_size=model_cfg['text_ff_size'],
            proj_hidden_dim=model_cfg['proj_hidden_dim'],
            proj_num_layers=model_cfg['proj_num_layers'],
            use_unified_dim=use_unified_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        # 加载 checkpoint
        if ckpt_path is not None and os.path.exists(ckpt_path):
            print(f"[MotionRewardAdapter] Loading checkpoint from: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            print(f"[MotionRewardAdapter] Checkpoint loaded successfully")
        else:
            print(f"[MotionRewardAdapter] Warning: No checkpoint loaded, using random weights")
        
        self.model.to(device)
        self.model.eval()
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
    
    def get_reward_t2m(
        self,
        raw_texts: list,
        motion_feats: torch.Tensor,
        m_len: list,
        t_len: list = None,
        sent_emb: torch.Tensor = None,
        timestep: torch.Tensor = None,
        return_m: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        计算 text-motion 相似度作为 reward
        
        与原 SPM.get_reward_t2m 接口兼容
        
        Args:
            raw_texts: 文本列表
            motion_feats: motion 特征 [B, T, D]
            m_len: motion 长度列表
            t_len: (unused) 文本长度
            sent_emb: (unused) 句子嵌入
            timestep: (unused) 时间步
            return_m: 是否返回 motion latent
            
        Returns:
            reward: cosine similarity [B]
        """
        with torch.enable_grad():
            # 获取 text embedding
            t_latent = self.model.get_text_embedding(raw_texts)
            
            # 获取 motion embedding
            m_latent = self.model.get_motion_embedding(
                motion_feats, 
                m_len, 
                repr_type=self.repr_type
            )
            
            # 计算 cosine similarity 作为 reward
            reward = F.cosine_similarity(
                t_latent.squeeze(), 
                m_latent.squeeze(), 
                dim=-1
            )
        
        if return_m:
            return reward, m_latent
        return reward
    
    def forward(self, *args, **kwargs):
        """Forward 直接调用 get_reward_t2m"""
        return self.get_reward_t2m(*args, **kwargs)


def create_reward_model(
    ckpt_path: str = None,
    t5_path: str = '../deps/sentence-t5-large',
    repr_type: str = '263',
    model_size: str = 'base',
    device: str = 'cuda',
    **kwargs
) -> MotionRewardAdapter:
    """
    创建 reward model 的便捷函数
    
    Args:
        ckpt_path: checkpoint 路径
        t5_path: T5 模型路径
        repr_type: motion 表征类型
        model_size: 模型规模 ('retrieval_original', 'tiny', 'small', 'base', 'large', 'xlarge', 'xxlarge', 'giant')
        device: 设备
        
    Returns:
        MotionRewardAdapter 实例
    """
    return MotionRewardAdapter(
        ckpt_path=ckpt_path,
        t5_path=t5_path,
        repr_type=repr_type,
        model_size=model_size,
        device=device,
        **kwargs
    )


# 为了兼容原代码中的 process_T5_outputs 函数
def process_T5_outputs(raw_texts, T5_model):
    """
    处理 T5 输出 - 与原 GradGuidance.spm 中的函数兼容
    """
    with torch.no_grad():
        T5_outputs = T5_model.encode(
            raw_texts, 
            show_progress_bar=False, 
            batch_size=len(raw_texts),
            output_value=None
        )
        attn_masks = torch.stack([item['attention_mask'] for item in T5_outputs])
        token_embeddings = torch.stack([item['token_embeddings'] for item in T5_outputs])
        sentence_embeddings = torch.stack([item['sentence_embedding'] for item in T5_outputs])
        t_length = attn_masks.sum(1).tolist()
        return t_length, token_embeddings.float(), sentence_embeddings.unsqueeze(1).float()
