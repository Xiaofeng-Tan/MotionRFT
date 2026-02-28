"""
支持 LoRA 的多表征 Retrieval 模型

主干网络（motion encoder, text encoder）使用全参数
Critic 和 AI Detection 任务使用独立的 LoRA 适配器
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .projections import Repr263Projection, Repr22x3Projection, Repr135Projection
from .heads import CriticMLP, AIDetectionHead
from .lora_modules import inject_lora_to_encoder, LoRALayer, LoRALinear


def process_T5_outputs(raw_texts, T5_model, device=None):
    """处理 T5 模型输出
    
    与 multi_repr_retrieval.py 保持完全一致
    
    Args:
        raw_texts: 原始文本列表
        T5_model: SentenceTransformer T5 模型
        device: 目标设备（可选）
    
    Returns:
        t_length: 文本长度列表
        token_embeddings: token 级别嵌入 [B, T, 1024]
        sentence_embeddings: 句子级别嵌入 [B, 1, 1024]
    """
    with torch.no_grad():
        # 强制将 SentenceTransformer 内部模型移动到目标设备
        if device is not None:
            try:
                T5_model._first_module().auto_model.to(device)
            except Exception:
                pass
        
        # 如果指定了设备，在 encode 时传递
        encode_kwargs = {
            'show_progress_bar': False, 
            'batch_size': len(raw_texts), 
            'output_value': None
        }
        if device is not None:
            encode_kwargs['device'] = str(device)
        
        T5_outputs = T5_model.encode(raw_texts, **encode_kwargs)
        
        attn_masks = torch.stack([item['attention_mask'] for item in T5_outputs])
        token_embeddings = torch.stack([item['token_embeddings'] for item in T5_outputs])
        sentence_embeddings = torch.stack([item['sentence_embedding'] for item in T5_outputs])
        t_length = attn_masks.sum(1).tolist()
        return t_length, token_embeddings.float(), sentence_embeddings.unsqueeze(1).float()


class MultiReprRetrievalWithLoRA(nn.Module):
    """支持 LoRA 的多表征 Retrieval 模型
    
    主干网络（motion encoder, text encoder）使用全参数
    Critic 和 AI Detection 任务使用独立的 LoRA 适配器
    
    Args:
        t5_path: T5 模型路径
        temp: 对比学习温度
        thr: 对比学习相似度阈值
        latent_dim: 潜在空间维度
        unified_dim: 统一表征维度
        encoder_num_layers: 编码器层数
        encoder_num_heads: 编码器注意力头数
        encoder_ff_size: 编码器 FFN 维度
        text_num_layers: 文本编码器层数
        text_num_heads: 文本编码器注意力头数
        text_ff_size: 文本编码器 FFN 维度
        proj_hidden_dim: 投影层隐藏维度
        proj_num_layers: 投影层数
        proj_dropout: 投影层 dropout
        use_unified_dim: 是否使用统一维度
        lora_rank: LoRA 秩
        lora_alpha: LoRA 缩放因子
        lora_dropout: LoRA dropout
    """
    def __init__(self, t5_path, temp=0.1, thr=0.9, 
                 latent_dim=256, unified_dim=512,
                 encoder_num_layers=9, encoder_num_heads=4, encoder_ff_size=1024,
                 text_num_layers=9, text_num_heads=4, text_ff_size=1024,
                 proj_hidden_dim=512, proj_num_layers=3, proj_dropout=0.1,
                 use_unified_dim=False,
                 lora_rank=16, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.unified_dim = unified_dim
        self.use_unified_dim = use_unified_dim
        self.temp = temp
        self.thr = thr
        
        # 保存投影层配置（用于后续创建独立投影层，如 AI Detection）
        self.proj_hidden_dim = proj_hidden_dim
        self.proj_num_layers = proj_num_layers
        self.proj_dropout = proj_dropout
        
        # LoRA 配置
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # 编码器工作维度
        encoder_dim = unified_dim if use_unified_dim else latent_dim
        
        # 表征投影层
        self.proj_263 = Repr263Projection(
            input_dim=263, unified_dim=encoder_dim,
            hidden_dim=proj_hidden_dim, num_layers=proj_num_layers, dropout=proj_dropout
        )
        self.proj_22x3 = Repr22x3Projection(
            num_joints=22, unified_dim=encoder_dim,
            hidden_dim=proj_hidden_dim, num_layers=proj_num_layers, dropout=proj_dropout
        )
        self.proj_135 = Repr135Projection(
            input_dim=135, unified_dim=encoder_dim,
            hidden_dim=proj_hidden_dim, num_layers=proj_num_layers, dropout=proj_dropout
        )
        
        # 加载 T5 编码器
        from sentence_transformers import SentenceTransformer
        self.clip = SentenceTransformer(t5_path)
        
        from motionreward.models.opt.attention import SkipTransformerEncoder, TransformerEncoderLayer
        from motionreward.models.opt.position_encoding import build_position_encoding
        from motionreward.models.opt.embeddings import TimestepEmbedding, Timesteps
        
        # Motion Encoder
        self.query_pos_encoder = build_position_encoding(encoder_dim, position_embedding='learned')
        encoder_layer = TransformerEncoderLayer(
            encoder_dim, encoder_num_heads, encoder_ff_size, 0.1, 'gelu', False
        )
        encoder_norm = nn.LayerNorm(encoder_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, encoder_num_layers, encoder_norm)
        self.global_motion_token = nn.Parameter(torch.randn(1, encoder_dim))
        self.dist_layer = nn.Linear(encoder_dim, 2 * latent_dim)
        
        # Text Encoder
        self.text_embedding = nn.Linear(1024, latent_dim)
        self.query_text_pos_encoder = build_position_encoding(latent_dim, position_embedding='learned')
        text_encoder_layer = TransformerEncoderLayer(
            latent_dim, text_num_heads, text_ff_size, 0.1, 'gelu', False
        )
        text_encoder_norm = nn.LayerNorm(latent_dim)
        self.text_encoder = SkipTransformerEncoder(text_encoder_layer, text_num_layers, text_encoder_norm)
        self.global_text_token = nn.Parameter(torch.randn(1, latent_dim))
        self.dist_text_layer = nn.Linear(latent_dim, 2 * latent_dim)
        
        # Time Embedding
        self.time_proj = Timesteps(512, True, 0)
        self.time_embedding = TimestepEmbedding(512, encoder_dim)
        self.time_embedding_text = TimestepEmbedding(512, latent_dim)
        
        # Decoder
        from motionreward.models.opt.attention import SkipTransformerDecoder, TransformerDecoderLayer
        self.query_pos_decoder = build_position_encoding(latent_dim, position_embedding='learned')
        decoder_layer = TransformerDecoderLayer(
            latent_dim, encoder_num_heads, encoder_ff_size, 0.1, 'gelu', False
        )
        decoder_norm = nn.LayerNorm(latent_dim)
        self.decoder = SkipTransformerDecoder(decoder_layer, encoder_num_layers, decoder_norm)
        
        # Final layers
        self.final_layer_263 = nn.Linear(latent_dim, 263)
        self.final_layer_22x3 = nn.Linear(latent_dim, 22 * 3)
        self.final_layer_135 = nn.Linear(latent_dim, 135)
        
        # Loss functions
        self.recons_loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.latent_loss_fn = nn.SmoothL1Loss(reduction='mean')
        
        # Task heads (初始化为 None)
        self.critic_head = None
        self.ai_detection_head = None
        
        # LoRA 模块 (初始化为 None，在需要时注入)
        self.critic_lora_modules = None
        self.ai_detection_lora_modules = None
        self.ai_detection_proj_lora_modules = None  # AI Detection 投影层 LoRA
        
        # AI Detection 独立 encoder (用于避免与 Critic LoRA 冲突)
        self.ai_detection_encoder = None
        self.ai_detection_query_pos_encoder = None
        self.ai_detection_dist_layer = None
        self.ai_detection_global_motion_token = None
        
        # AI Detection 独立投影层
        self.ai_detection_proj_263 = None
        self.ai_detection_proj_22x3 = None
        self.ai_detection_proj_135 = None
        
        # 当前激活的 LoRA 任务
        self._active_lora_task = None  # 'critic' or 'ai_detection' or None
    
    def init_critic_head(self, hidden_dim=None):
        """初始化 Critic Head（用于全参训练模式，不使用 LoRA）
        
        Args:
            hidden_dim: 隐藏层维度，默认使用 latent_dim
        
        Returns:
            critic_head: Critic Head 模块
        """
        if self.critic_head is not None:
            print("Critic head already initialized")
            return self.critic_head
        
        hidden_dim = hidden_dim or self.latent_dim
        device = next(self.parameters()).device
        
        self.critic_head = CriticMLP(
            in_features=self.latent_dim,
            hidden_features=hidden_dim,
            out_features=1,
            drop=0.1
        ).to(device)
        
        print(f"Critic head initialized: {sum(p.numel() for p in self.critic_head.parameters()):,} params")
        return self.critic_head
    
    def init_ai_detection_head(self, hidden_dim=None, num_classes=2):
        """初始化 AI Detection Head（用于全参训练模式，不使用 LoRA）
        
        Args:
            hidden_dim: 隐藏层维度，默认使用 latent_dim
            num_classes: 分类数量，默认 2（真实/AI生成）
        
        Returns:
            ai_detection_head: AI Detection Head 模块
        """
        if self.ai_detection_head is not None:
            print("AI Detection head already initialized")
            return self.ai_detection_head
        
        hidden_dim = hidden_dim or self.latent_dim
        device = next(self.parameters()).device
        
        self.ai_detection_head = AIDetectionHead(
            in_features=self.latent_dim,
            hidden_features=hidden_dim,
            num_classes=num_classes,
            drop=0.1
        ).to(device)
        
        print(f"AI Detection head initialized: {sum(p.numel() for p in self.ai_detection_head.parameters()):,} params")
        return self.ai_detection_head
    
    def _inject_projection_lora(self, device, prefix='', lora_modules_dict=None):
        """为表征投影层注入 LoRA
        
        为 proj_263, proj_22x3, proj_135 的线性层添加 LoRA 适配器
        
        Args:
            device: 设备
            prefix: 参数名前缀（用于区分不同任务的 LoRA）
            lora_modules_dict: 用于存储 LoRA 模块的字典（可选）
        
        Returns:
            lora_params: LoRA 参数列表
        """
        lora_params = []
        
        # 为每个投影层的线性层注入 LoRA
        for proj_name, proj_module in [('proj_263', self.proj_263), 
                                        ('proj_22x3', self.proj_22x3), 
                                        ('proj_135', self.proj_135)]:
            if hasattr(proj_module, 'proj'):
                proj_layer = proj_module.proj
                if isinstance(proj_layer, nn.Sequential):
                    # 多层 MLP：为每个线性层注入 LoRA
                    for i, layer in enumerate(proj_layer):
                        if isinstance(layer, nn.Linear):
                            lora_linear = LoRALinear(
                                layer, 
                                rank=self.lora_rank, 
                                alpha=self.lora_alpha, 
                                dropout=self.lora_dropout
                            ).to(device)
                            proj_layer[i] = lora_linear
                            lora_params.extend(lora_linear.lora.parameters())
                            
                            # 记录到 lora_modules
                            if lora_modules_dict is not None:
                                lora_modules_dict[f'{prefix}{proj_name}_layer{i}'] = lora_linear
                elif isinstance(proj_layer, nn.Linear):
                    # 单层线性
                    lora_linear = LoRALinear(
                        proj_layer, 
                        rank=self.lora_rank, 
                        alpha=self.lora_alpha, 
                        dropout=self.lora_dropout
                    ).to(device)
                    proj_module.proj = lora_linear
                    lora_params.extend(lora_linear.lora.parameters())
                    
                    if lora_modules_dict is not None:
                        lora_modules_dict[f'{prefix}{proj_name}'] = lora_linear
        
        proj_lora_count = sum(p.numel() for p in lora_params)
        print(f"Projection LoRA ({prefix}): {proj_lora_count:,} params")
        
        return lora_params
    
    def inject_critic_lora(self):
        """为 Critic 任务注入 LoRA（包括表征投影层）
        
        Returns:
            critic_lora_params: Critic LoRA 参数列表
        """
        if self.critic_lora_modules is not None:
            print("Critic LoRA already injected")
            return []
        
        # 获取模型所在的设备
        device = next(self.parameters()).device
        
        # 为 encoder 注入 LoRA
        self.critic_lora_modules, critic_lora_params = inject_lora_to_encoder(
            self.encoder, 
            rank=self.lora_rank, 
            alpha=self.lora_alpha, 
            dropout=self.lora_dropout,
            prefix='critic_'
        )
        
        # 为表征投影层注入 LoRA
        proj_lora_params = self._inject_projection_lora(device, prefix='critic_', lora_modules_dict=self.critic_lora_modules)
        critic_lora_params.extend(proj_lora_params)
        
        # 初始化 Critic head（也移到正确的设备）
        self.critic_head = CriticMLP(
            in_features=self.latent_dim,
            hidden_features=self.latent_dim,
            out_features=1,
            drop=0.1
        ).to(device)
        
        print(f"Critic LoRA injected (encoder + projections): {sum(p.numel() for p in critic_lora_params):,} params")
        print(f"Critic head: {sum(p.numel() for p in self.critic_head.parameters()):,} params")
        
        return critic_lora_params
    
    def inject_ai_detection_lora(self, backbone_ckpt=None):
        """为 AI Detection 任务注入 LoRA
        
        创建独立的 encoder 副本，避免与 Critic LoRA 冲突
        
        Args:
            backbone_ckpt: Stage 1 backbone checkpoint 路径，用于加载原始 encoder 权重
                          当 self.encoder 已注入 Critic LoRA 时必须提供
        
        Returns:
            ai_lora_params: AI Detection LoRA 参数列表
        """
        if self.ai_detection_lora_modules is not None:
            print("AI Detection LoRA already injected")
            return []
        
        # 获取模型所在的设备
        device = next(self.parameters()).device
        
        # 编码器工作维度
        encoder_dim = self.unified_dim if self.use_unified_dim else self.latent_dim
        
        # 导入必要的模块
        from motionreward.models.opt.attention import SkipTransformerEncoder, TransformerEncoderLayer
        from motionreward.models.opt.position_encoding import build_position_encoding
        import copy
        
        # 创建独立的 encoder 副本 (深拷贝结构，但参数独立)
        # 获取原始 encoder 的配置
        original_layer = self.encoder.input_blocks[0]
        num_layers = self.encoder.num_layers
        
        # 检测是否已注入 LoRA，获取正确的 num_heads 和 ff_size
        if hasattr(original_layer.self_attn, 'original_mha'):
            # 已注入 LoRA，从 original_mha 获取
            num_heads = original_layer.self_attn.original_mha.num_heads
        else:
            num_heads = original_layer.self_attn.num_heads
        
        if hasattr(original_layer.linear1, 'original_linear'):
            # 已注入 LoRA，从 original_linear 获取
            ff_size = original_layer.linear1.original_linear.out_features
        else:
            ff_size = original_layer.linear1.out_features
        
        # 创建新的 encoder
        encoder_layer = TransformerEncoderLayer(
            encoder_dim, num_heads, ff_size, 0.1, 'gelu', False
        )
        encoder_norm = nn.LayerNorm(encoder_dim)
        self.ai_detection_encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm).to(device)
        
        # 从 backbone checkpoint 加载原始 encoder 权重（避免 LoRA 结构不匹配）
        if backbone_ckpt is not None and os.path.exists(backbone_ckpt):
            print(f"[inject_ai_detection_lora] Loading encoder weights from backbone: {backbone_ckpt}")
            import torch
            checkpoint = torch.load(backbone_ckpt, map_location='cpu', weights_only=False)
            if 'state_dict' in checkpoint:
                full_state_dict = checkpoint['state_dict']
            else:
                full_state_dict = checkpoint
            
            # 提取 encoder 相关的权重
            encoder_state_dict = {}
            for k, v in full_state_dict.items():
                if k.startswith('encoder.'):
                    encoder_state_dict[k[len('encoder.'):]] = v
            
            self.ai_detection_encoder.load_state_dict(encoder_state_dict)
            print(f"[inject_ai_detection_lora] Loaded {len(encoder_state_dict)} encoder weights from backbone")
        else:
            # 尝试从当前 encoder 复制（仅当未注入 LoRA 时有效）
            try:
                self.ai_detection_encoder.load_state_dict(self.encoder.state_dict())
                print("[inject_ai_detection_lora] Loaded encoder weights from current encoder")
            except RuntimeError as e:
                print(f"[inject_ai_detection_lora] Warning: Failed to load from current encoder: {e}")
                print("[inject_ai_detection_lora] Using randomly initialized encoder (please provide backbone_ckpt)")
        
        # 创建独立的位置编码
        self.ai_detection_query_pos_encoder = build_position_encoding(encoder_dim, position_embedding='learned').to(device)
        self.ai_detection_query_pos_encoder.load_state_dict(self.query_pos_encoder.state_dict())
        
        # 创建独立的 dist_layer
        self.ai_detection_dist_layer = nn.Linear(encoder_dim, 2 * self.latent_dim).to(device)
        self.ai_detection_dist_layer.load_state_dict(self.dist_layer.state_dict())
        
        # 创建独立的 global_motion_token
        self.ai_detection_global_motion_token = nn.Parameter(self.global_motion_token.data.clone())
        
        # 创建独立的投影层副本（需要处理可能已注入 LoRA 的情况）
        # 从 backbone checkpoint 加载投影层权重
        if backbone_ckpt is not None and os.path.exists(backbone_ckpt):
            # 重新创建投影层，使用保存的配置以确保与各种 model_size 兼容
            from .projections import Repr263Projection, Repr22x3Projection, Repr135Projection
            
            self.ai_detection_proj_263 = Repr263Projection(
                input_dim=263, unified_dim=encoder_dim,
                hidden_dim=self.proj_hidden_dim, num_layers=self.proj_num_layers, dropout=self.proj_dropout
            ).to(device)
            self.ai_detection_proj_22x3 = Repr22x3Projection(
                num_joints=22, unified_dim=encoder_dim,
                hidden_dim=self.proj_hidden_dim, num_layers=self.proj_num_layers, dropout=self.proj_dropout
            ).to(device)
            self.ai_detection_proj_135 = Repr135Projection(
                input_dim=135, unified_dim=encoder_dim,
                hidden_dim=self.proj_hidden_dim, num_layers=self.proj_num_layers, dropout=self.proj_dropout
            ).to(device)
            
            # 加载投影层权重
            checkpoint = torch.load(backbone_ckpt, map_location='cpu', weights_only=False)
            if 'state_dict' in checkpoint:
                full_state_dict = checkpoint['state_dict']
            else:
                full_state_dict = checkpoint
            
            for proj_name, proj_module in [('proj_263', self.ai_detection_proj_263),
                                            ('proj_22x3', self.ai_detection_proj_22x3),
                                            ('proj_135', self.ai_detection_proj_135)]:
                proj_state_dict = {}
                for k, v in full_state_dict.items():
                    if k.startswith(f'{proj_name}.'):
                        proj_state_dict[k[len(f'{proj_name}.'):]] = v
                if proj_state_dict:
                    proj_module.load_state_dict(proj_state_dict)
            print(f"[inject_ai_detection_lora] Loaded projection weights from backbone")
        else:
            # 深拷贝当前投影层
            self.ai_detection_proj_263 = copy.deepcopy(self.proj_263).to(device)
            self.ai_detection_proj_22x3 = copy.deepcopy(self.proj_22x3).to(device)
            self.ai_detection_proj_135 = copy.deepcopy(self.proj_135).to(device)
        
        # 为独立的 encoder 注入 LoRA
        self.ai_detection_lora_modules, ai_lora_params = inject_lora_to_encoder(
            self.ai_detection_encoder, 
            rank=self.lora_rank, 
            alpha=self.lora_alpha, 
            dropout=self.lora_dropout,
            prefix='ai_detection_'
        )
        
        # 为独立的投影层注入 LoRA
        proj_lora_params = self._inject_ai_detection_projection_lora(device)
        ai_lora_params.extend(proj_lora_params)
        
        # 初始化 AI Detection head
        self.ai_detection_head = AIDetectionHead(
            in_features=self.latent_dim,
            hidden_features=self.latent_dim,
            num_classes=2,
            drop=0.1
        ).to(device)
        
        total_lora_params = sum(p.numel() for p in ai_lora_params)
        print(f"AI Detection LoRA injected (independent encoder + projections): {total_lora_params:,} params")
        print(f"AI Detection head: {sum(p.numel() for p in self.ai_detection_head.parameters()):,} params")
        
        return ai_lora_params
    
    def _inject_ai_detection_projection_lora(self, device):
        """为 AI Detection 独立投影层注入 LoRA
        
        Args:
            device: 设备
        
        Returns:
            lora_params: LoRA 参数列表
        """
        lora_params = []
        
        # 为每个独立投影层的线性层注入 LoRA
        for proj_name, proj_module in [('ai_proj_263', self.ai_detection_proj_263), 
                                        ('ai_proj_22x3', self.ai_detection_proj_22x3), 
                                        ('ai_proj_135', self.ai_detection_proj_135)]:
            if hasattr(proj_module, 'proj'):
                proj_layer = proj_module.proj
                if isinstance(proj_layer, nn.Sequential):
                    # 多层 MLP：为每个线性层注入 LoRA
                    for i, layer in enumerate(proj_layer):
                        if isinstance(layer, nn.Linear):
                            lora_linear = LoRALinear(
                                layer, 
                                rank=self.lora_rank, 
                                alpha=self.lora_alpha, 
                                dropout=self.lora_dropout
                            ).to(device)
                            proj_layer[i] = lora_linear
                            lora_params.extend(lora_linear.lora.parameters())
                            
                            # 记录到 lora_modules
                            self.ai_detection_lora_modules[f'{proj_name}_layer{i}'] = lora_linear
                elif isinstance(proj_layer, nn.Linear):
                    # 单层线性
                    lora_linear = LoRALinear(
                        proj_layer, 
                        rank=self.lora_rank, 
                        alpha=self.lora_alpha, 
                        dropout=self.lora_dropout
                    ).to(device)
                    proj_module.proj = lora_linear
                    lora_params.extend(lora_linear.lora.parameters())
                    
                    self.ai_detection_lora_modules[f'{proj_name}'] = lora_linear
        
        proj_lora_count = sum(p.numel() for p in lora_params)
        print(f"AI Detection Projection LoRA: {proj_lora_count:,} params")
        
        return lora_params
    
    def set_active_lora(self, task=None):
        """设置当前激活的 LoRA 任务
        
        Args:
            task: 'critic', 'ai_detection', or None (不使用 LoRA)
        """
        self._active_lora_task = task
    
    def project_motion(self, motion, repr_type):
        """投影 motion 到统一维度"""
        if repr_type == '263':
            return self.proj_263(motion)
        elif repr_type == '22x3':
            return self.proj_22x3(motion)
        elif repr_type == '135':
            return self.proj_135(motion)
        else:
            raise ValueError(f"Unknown repr_type: {repr_type}")
    
    def _lengths_to_mask(self, lengths, device, max_len=None):
        """将长度列表转换为 mask"""
        # 兼容 list 和 tensor 输入
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths, device=device)
        else:
            lengths = lengths.to(device)
        max_len = max_len if max_len else max(lengths)
        mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        return mask
    
    def encode_motion(self, motion, lengths, repr_type='263', timestep=None):
        """编码 motion
        
        Args:
            motion: motion 张量
            lengths: 长度列表
            repr_type: 表征类型
            timestep: 时间步（可选）
            
        Returns:
            latent: 潜在表示
            dist: 分布
        """
        device = motion.device
        bs = motion.shape[0]
        
        if lengths is None:
            lengths = [motion.shape[1]] * bs
        
        x = self.project_motion(motion, repr_type)
        
        mask = self._lengths_to_mask(lengths, device, max_len=x.shape[1])
        x = x.permute(1, 0, 2)  # [L, B, D]
        
        dist = self.global_motion_token.unsqueeze(1).expand(-1, bs, -1)
        dist_masks = torch.ones((bs, 1), dtype=bool, device=device)
        
        if timestep is not None:
            timesteps = timestep.expand(bs).clone()
            time_emb = self.time_proj(timesteps)
            time_emb = time_emb.to(dtype=x.dtype)
            time_emb = self.time_embedding(time_emb).unsqueeze(0)
            xseq = torch.cat((dist, time_emb, x), 0)
            time_masks = torch.ones((bs, 1), dtype=bool, device=device)
            aug_mask = torch.cat((dist_masks, time_masks, mask), 1)
        else:
            xseq = torch.cat((dist, x), 0)
            aug_mask = torch.cat((dist_masks, mask), 1)
        
        xseq = self.query_pos_encoder(xseq)
        
        # 使用 encoder（可能包含 LoRA）
        dist_out = self.encoder(xseq, src_key_padding_mask=~aug_mask)[:1]
        
        tokens_dist = self.dist_layer(dist_out)
        mu = tokens_dist[:, :, :self.latent_dim]
        logvar = tokens_dist[:, :, self.latent_dim:]
        
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        
        return latent, dist
    
    def encode_motion_ai_detection(self, motion, lengths, repr_type='263', timestep=None):
        """使用 AI Detection 独立 encoder 编码 motion
        
        Args:
            motion: motion 张量
            lengths: 长度列表
            repr_type: 表征类型
            timestep: 时间步（可选）
            
        Returns:
            latent: 潜在表示
            dist: 分布
        """
        device = motion.device
        bs = motion.shape[0]
        
        if lengths is None:
            lengths = [motion.shape[1]] * bs
        
        # 使用 AI Detection 独立投影层
        if repr_type == '263':
            x = self.ai_detection_proj_263(motion)
        elif repr_type == '22x3':
            x = self.ai_detection_proj_22x3(motion)
        elif repr_type == '135':
            x = self.ai_detection_proj_135(motion)
        else:
            raise ValueError(f"Unknown repr_type: {repr_type}")
        
        mask = self._lengths_to_mask(lengths, device, max_len=x.shape[1])
        x = x.permute(1, 0, 2)  # [L, B, D]
        
        dist = self.ai_detection_global_motion_token.unsqueeze(1).expand(-1, bs, -1)
        dist_masks = torch.ones((bs, 1), dtype=bool, device=device)
        
        if timestep is not None:
            timesteps = timestep.expand(bs).clone()
            time_emb = self.time_proj(timesteps)
            time_emb = time_emb.to(dtype=x.dtype)
            time_emb = self.time_embedding(time_emb).unsqueeze(0)
            xseq = torch.cat((dist, time_emb, x), 0)
            time_masks = torch.ones((bs, 1), dtype=bool, device=device)
            aug_mask = torch.cat((dist_masks, time_masks, mask), 1)
        else:
            xseq = torch.cat((dist, x), 0)
            aug_mask = torch.cat((dist_masks, mask), 1)
        
        xseq = self.ai_detection_query_pos_encoder(xseq)
        
        # 使用 AI Detection 独立 encoder（包含 AI Detection LoRA）
        dist_out = self.ai_detection_encoder(xseq, src_key_padding_mask=~aug_mask)[:1]
        
        tokens_dist = self.ai_detection_dist_layer(dist_out)
        mu = tokens_dist[:, :, :self.latent_dim]
        logvar = tokens_dist[:, :, self.latent_dim:]
        
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        
        return latent, dist
    
    def encode_text(self, features, lengths=None, timestep=None):
        """编码文本特征
        
        与 train_retrieval_lora.py 完全对齐
        
        Args:
            features: 文本特征张量 [B, T, 1024]（已经过 T5 编码）
            lengths: 长度列表（可选）
            timestep: 时间步（可选）
        
        Returns:
            latent: 潜在表示
            dist: 分布
        """
        device = features.device
        bs, nbtokens, nfeats = features.shape
        
        if lengths is None:
            lengths = [nbtokens] * bs
        
        mask = self._lengths_to_mask(lengths, device, max_len=nbtokens)
        
        x = self.text_embedding(features)
        x = x.permute(1, 0, 2)
        
        dist = self.global_text_token.unsqueeze(1).expand(-1, bs, -1)
        dist_masks = torch.ones((bs, 1), dtype=bool, device=device)
        
        if timestep is not None:
            timesteps = timestep.expand(bs).clone()
            time_emb = self.time_proj(timesteps)
            time_emb = time_emb.to(dtype=x.dtype)
            time_emb = self.time_embedding_text(time_emb).unsqueeze(0)
            xseq = torch.cat((dist, time_emb, x), 0)
            time_masks = torch.ones((bs, 1), dtype=bool, device=device)
            aug_mask = torch.cat((dist_masks, time_masks, mask), 1)
        else:
            xseq = torch.cat((dist, x), 0)
            aug_mask = torch.cat((dist_masks, mask), 1)
        
        xseq = self.query_text_pos_encoder(xseq)
        dist_out = self.text_encoder(xseq, src_key_padding_mask=~aug_mask)[:1]
        
        tokens_dist = self.dist_text_layer(dist_out)
        mu = tokens_dist[:, :, :self.latent_dim]
        logvar = tokens_dist[:, :, self.latent_dim:]
        
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        
        return latent, dist
    
    def decode(self, z, lengths, max_motion_length, repr_type='263'):
        """解码潜在表示"""
        device = z.device
        bs = z.shape[1]
        max_len = max_motion_length
        
        mask = self._lengths_to_mask(lengths, device, max_len=max_len)
        
        queries = torch.zeros(max_len, bs, self.latent_dim, device=device, requires_grad=True)
        queries = self.query_pos_decoder(queries)
        
        output = self.decoder(tgt=queries, memory=z, tgt_key_padding_mask=~mask)
        
        if repr_type == '263':
            output = self.final_layer_263(output)
            output = output.permute(1, 0, 2)
        elif repr_type == '135':
            output = self.final_layer_135(output)
            output = output.permute(1, 0, 2)
        else:  # 22x3
            output = self.final_layer_22x3(output)
            output = output.permute(1, 0, 2)
            output = output.view(output.shape[0], output.shape[1], -1, 3)
        
        for i, length in enumerate(lengths):
            output[i, length:] = 0
        
        return output
    
    def _kl_loss(self, q, p):
        """KL 散度损失"""
        mu_q, logvar_q = q
        mu_p, logvar_p = p
        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()
    
    def _infonce_loss(self, x, y, sent_emb=None):
        """InfoNCE 对比损失
        
        与 train_retrieval_lora.py 完全对齐
        """
        # 确保输入至少是 2D: (bs, dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        if sent_emb is not None and sent_emb.dim() == 1:
            sent_emb = sent_emb.unsqueeze(0)
        
        bs, device = x.shape[0], x.device
        
        x_logits = F.normalize(x, dim=-1)
        y_logits = F.normalize(y, dim=-1)
        sim_matrix = (x_logits @ y_logits.mT) / self.temp
        
        if sent_emb is not None and self.thr and bs > 1:
            real_threshold_selfsim = 2 * self.thr - 1
            sent_emb_normalized = F.normalize(sent_emb, dim=-1)
            selfsim = sent_emb_normalized @ sent_emb_normalized.mT
            selfsim_nodiag = selfsim - torch.diag(torch.diag(selfsim))
            idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            sim_matrix[idx] = -torch.inf
        
        labels = torch.arange(bs, device=device)
        total_loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.mT, labels)) / 2
        return total_loss
    
    def forward_cross_repr(self, motion_263, motion_22x3, m_len, timestep=None, motion_135=None):
        """跨表征对齐前向传播
        
        对同一 motion 的不同表征进行对齐，使其在潜在空间中接近。
        支持 263↔22x3 对齐，以及可选的 135 维对齐。
        
        Args:
            motion_263: 263维 motion 张量
            motion_22x3: 22x3维 motion 张量
            m_len: 长度列表
            timestep: 时间步（可选）
            motion_135: 135维 motion 张量（可选）
            
        Returns:
            cross_repr_loss: 跨表征对齐损失
            details: 损失详情字典
        """
        # 编码 263 和 22x3
        m_latent_263, m_dist_263 = self.encode_motion(motion_263, m_len, repr_type='263', timestep=timestep)
        m_latent_22x3, m_dist_22x3 = self.encode_motion(motion_22x3, m_len, repr_type='22x3', timestep=timestep)
        
        # 263 ↔ 22x3 对齐
        latent_align_loss = self.latent_loss_fn(m_latent_263, m_latent_22x3)
        dist_263_params = [m_dist_263.loc, 2 * torch.log(m_dist_263.scale)]
        dist_22x3_params = [m_dist_22x3.loc, 2 * torch.log(m_dist_22x3.scale)]
        kl_align_loss = self._kl_loss(dist_263_params, dist_22x3_params) + self._kl_loss(dist_22x3_params, dist_263_params)
        cl_align_loss = self._infonce_loss(m_latent_263.squeeze(0), m_latent_22x3.squeeze(0))
        
        cross_repr_loss = latent_align_loss * 1e-1 + kl_align_loss * 1e-5 + cl_align_loss * 1e-1
        details = {
            'latent_align': latent_align_loss.item(),
            'kl_align': kl_align_loss.item(),
            'cl_align': cl_align_loss.item(),
        }
        
        # 可选：135 维对齐
        if motion_135 is not None:
            m_latent_135, m_dist_135 = self.encode_motion(motion_135, m_len, repr_type='135', timestep=timestep)
            dist_135_params = [m_dist_135.loc, 2 * torch.log(m_dist_135.scale)]
            
            # 263 ↔ 135
            latent_align_135 = self.latent_loss_fn(m_latent_263, m_latent_135)
            kl_align_135 = self._kl_loss(dist_263_params, dist_135_params) + self._kl_loss(dist_135_params, dist_263_params)
            cl_align_135 = self._infonce_loss(m_latent_263.squeeze(0), m_latent_135.squeeze(0))
            
            cross_repr_loss += latent_align_135 * 1e-1 + kl_align_135 * 1e-5 + cl_align_135 * 1e-1
            details['latent_align_135'] = latent_align_135.item()
            details['kl_align_135'] = kl_align_135.item()
            details['cl_align_135'] = cl_align_135.item()
        
        return cross_repr_loss, details
    
    def forward(self, text, motion_feature, m_len, repr_type='263', timestep=None, mode='M1T0'):
        """Retrieval 前向传播
        
        Args:
            text: 文本列表
            motion_feature: motion 张量
            m_len: 长度列表
            repr_type: 表征类型
            timestep: 时间步
            mode: 训练模式
            
        Returns:
            total_loss: 总损失
        """
        with torch.no_grad():
            # 获取当前模型所在设备
            device = next(self.parameters()).device
            t_len, sent_emb, cls_token = process_T5_outputs(text, self.clip, device=device)
        
        if mode == 'M1T0':
            t_latent, t_dist = self.encode_text(sent_emb, t_len)
            m_latent, m_dist = self.encode_motion(motion_feature, m_len, repr_type=repr_type, timestep=timestep)
        elif mode == 'M0T1':
            t_latent, t_dist = self.encode_text(sent_emb, t_len, timestep=timestep)
            m_latent, m_dist = self.encode_motion(motion_feature, m_len, repr_type=repr_type)
        elif mode == 'M1T1':
            t_latent, t_dist = self.encode_text(sent_emb, t_len, timestep=timestep)
            m_latent, m_dist = self.encode_motion(motion_feature, m_len, repr_type=repr_type, timestep=timestep)
        else:
            t_latent, t_dist = self.encode_text(sent_emb, t_len)
            m_latent, m_dist = self.encode_motion(motion_feature, m_len, repr_type=repr_type)
        
        max_motion_length = motion_feature.shape[1]
        m_rst = self.decode(m_latent, m_len, max_motion_length, repr_type=repr_type)
        t_rst = self.decode(t_latent, m_len, max_motion_length, repr_type=repr_type)
        
        # 计算 loss
        recons_loss = self.recons_loss_fn(t_rst, motion_feature) + self.recons_loss_fn(m_rst, motion_feature) + self.recons_loss_fn(m_rst, t_rst)
        
        m_dist_params = [m_dist.loc, 2 * torch.log(m_dist.scale)]
        t_dist_params = [t_dist.loc, 2 * torch.log(t_dist.scale)]
        ref_mus = torch.zeros_like(m_dist_params[0])
        ref_logvar = torch.zeros_like(t_dist_params[1])
        ref_dist = (ref_mus, ref_logvar)
        
        kl_loss = (
            self._kl_loss(t_dist_params, m_dist_params) +
            self._kl_loss(m_dist_params, t_dist_params) +
            self._kl_loss(m_dist_params, ref_dist) +
            self._kl_loss(t_dist_params, ref_dist)
        )
        
        latent_loss = self.latent_loss_fn(m_latent, t_latent)
        cl_loss = self._infonce_loss(t_latent.squeeze(), m_latent.squeeze(), cls_token.squeeze())
        
        total_loss = recons_loss * 1e0 + kl_loss * 1e-5 + latent_loss * 1e-5 + cl_loss * 1e-1
        
        return total_loss
    
    def forward_critic(self, batch_data, return_aux_loss=False):
        """Critic 前向传播
        
        Args:
            batch_data: 批次数据字典
            return_aux_loss: 是否返回辅助损失（重建 + KL，用于辅助训练）
            
        Returns:
            critic: Critic 输出 [B, 2]
            aux_loss: 辅助损失（如果 return_aux_loss=True），与 Stage 1 保持一致的比例
        """
        if self.critic_head is None:
            return (None, None) if return_aux_loss else None
        
        motion_better = batch_data['motion_better']
        motion_worse = batch_data['motion_worse']
        repr_type = batch_data['repr_type']
        
        bs = motion_better.shape[0]
        lengths = [motion_better.shape[1]] * bs
        
        latent_better, dist_better = self.encode_motion(motion_better, lengths, repr_type=repr_type)
        latent_worse, dist_worse = self.encode_motion(motion_worse, lengths, repr_type=repr_type)
        
        latent_better_squeezed = latent_better.squeeze(0)
        latent_worse_squeezed = latent_worse.squeeze(0)
        
        score_better = self.critic_head(latent_better_squeezed)
        score_worse = self.critic_head(latent_worse_squeezed)
        
        critic = torch.cat([score_better, score_worse], dim=1)
        
        if return_aux_loss:
            # 计算辅助损失，与 Stage 1 保持一致的比例
            max_motion_length = motion_better.shape[1]
            recons_better = self.decode(latent_better, lengths, max_motion_length, repr_type=repr_type)
            recons_worse = self.decode(latent_worse, lengths, max_motion_length, repr_type=repr_type)
            
            # 重建损失 (比例 1e0)
            recons_loss = self.recons_loss_fn(recons_better, motion_better) + \
                          self.recons_loss_fn(recons_worse, motion_worse)
            
            # KL 损失 (比例 1e-5)，正则化潜在空间
            dist_better_params = [dist_better.loc, 2 * torch.log(dist_better.scale)]
            dist_worse_params = [dist_worse.loc, 2 * torch.log(dist_worse.scale)]
            ref_mus = torch.zeros_like(dist_better_params[0])
            ref_logvar = torch.zeros_like(dist_better_params[1])
            ref_dist = (ref_mus, ref_logvar)
            
            kl_loss = self._kl_loss(dist_better_params, ref_dist) + \
                      self._kl_loss(dist_worse_params, ref_dist)
            
            # 总辅助损失，与 Stage 1 保持一致的比例
            aux_loss = recons_loss * 1e0 + kl_loss * 1e-5
            return critic, aux_loss
        
        return critic
    
    def forward_ai_detection(self, batch_data, return_aux_loss=False):
        """AI Detection 前向传播
        
        使用 AI Detection 独立 encoder（包含 AI Detection LoRA）
        
        Args:
            batch_data: 批次数据字典
            return_aux_loss: 是否返回辅助损失（重建 + KL，用于辅助训练）
            
        Returns:
            logits: 分类 logits
            labels: 标签
            aux_loss: 辅助损失（如果 return_aux_loss=True），与 Stage 1 保持一致的比例
        """
        if self.ai_detection_head is None:
            return (None, None, None) if return_aux_loss else (None, None)
        
        motion = batch_data['motion']
        labels = batch_data['label']
        repr_type = batch_data['repr_type']
        lengths = batch_data['length']
        
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.tolist()
        
        # 使用 AI Detection 独立 encoder
        latent, dist = self.encode_motion_ai_detection(motion, lengths, repr_type=repr_type)
        latent_squeezed = latent.squeeze(0)
        
        logits = self.ai_detection_head(latent_squeezed)
        
        if return_aux_loss:
            # 计算辅助损失，与 Stage 1 保持一致的比例
            max_motion_length = motion.shape[1]
            recons = self.decode(latent, lengths, max_motion_length, repr_type=repr_type)
            
            # 重建损失 (比例 1e0)
            recons_loss = self.recons_loss_fn(recons, motion)
            
            # KL 损失 (比例 1e-5)，正则化潜在空间
            dist_params = [dist.loc, 2 * torch.log(dist.scale)]
            ref_mus = torch.zeros_like(dist_params[0])
            ref_logvar = torch.zeros_like(dist_params[1])
            ref_dist = (ref_mus, ref_logvar)
            
            kl_loss = self._kl_loss(dist_params, ref_dist)
            
            # 总辅助损失，与 Stage 1 保持一致的比例
            aux_loss = recons_loss * 1e0 + kl_loss * 1e-5
            return logits, labels, aux_loss
        
        return logits, labels
    
    def get_motion_embedding(self, motion, lengths, repr_type='263', timestep=None):
        """获取 motion 的嵌入表示
        
        Args:
            motion: motion 张量
            lengths: 长度列表
            repr_type: 表征类型
            timestep: 时间步（可选，用于噪声感知编码）
            
        Returns:
            embedding: motion 嵌入
        """
        # 确保 motion 是 float32
        motion = motion.float()
        latent, _ = self.encode_motion(motion, lengths, repr_type=repr_type, timestep=timestep)
        return latent.squeeze(0)
    
    def get_text_embedding(self, text):
        """获取文本的嵌入表示
        
        Args:
            text: 文本列表
            
        Returns:
            embedding: 文本嵌入
        """
        with torch.no_grad():
            # 获取当前模型所在设备
            device = next(self.parameters()).device
            t_len, sent_emb, _ = process_T5_outputs(text, self.clip, device=device)
        latent, _ = self.encode_text(sent_emb, t_len)
        return latent.squeeze(0)
