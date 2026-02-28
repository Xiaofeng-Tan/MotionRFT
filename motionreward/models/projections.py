"""
表征投影模块

包含:
- Repr263Projection: 263维表征投影层 (HumanML3D)
- Repr22x3Projection: 22x3表征投影层 (关节位置)
- Repr135Projection: 135维表征投影层 (FlowMDM/BABEL rot6d)
- Repr201Projection: 201维表征投影层 (FlowMDM 22x3 joints + velocities)
"""

import torch
import torch.nn as nn


class Repr263Projection(nn.Module):
    """263维表征投影层 - 支持多层MLP
    
    将 263 维 motion 表征投影到统一维度空间
    """
    def __init__(self, input_dim=263, unified_dim=256, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        if num_layers == 1:
            self.proj = nn.Linear(input_dim, unified_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(hidden_dim, unified_dim))
            self.proj = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, 263] motion 表征
        
        Returns:
            [B, T, unified_dim] 投影后的表征
        """
        return self.proj(x)


class Repr22x3Projection(nn.Module):
    """22x3表征投影层 - 支持多层MLP
    
    将 22x3 (关节位置) motion 表征投影到统一维度空间
    """
    def __init__(self, num_joints=22, unified_dim=256, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        input_dim = num_joints * 3
        if num_layers == 1:
            self.proj = nn.Linear(input_dim, unified_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(hidden_dim, unified_dim))
            self.proj = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, 22, 3] 或 [B, T, 66] motion 表征
        
        Returns:
            [B, T, unified_dim] 投影后的表征
        """
        if len(x.shape) == 4:
            # [B, T, 22, 3] -> [B, T, 66]
            B, L, J, C = x.shape
            x = x.reshape(B, L, J * C)
        # 否则已经是 [B, T, 66] 格式
        return self.proj(x)


class Repr135Projection(nn.Module):
    """135维表征投影层 - FlowMDM/BABEL rot6d 表征
    
    135维 = 1 (root_y) + 2 (vel_trajectory) + 22 joints × 6 (rot6d)
    
    将 135 维 motion 表征投影到统一维度空间
    """
    def __init__(self, input_dim=135, unified_dim=256, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        if num_layers == 1:
            self.proj = nn.Linear(input_dim, unified_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(hidden_dim, unified_dim))
            self.proj = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, 135] motion 表征
        
        Returns:
            [B, T, unified_dim] 投影后的表征
        """
        return self.proj(x)


class Repr201Projection(nn.Module):
    """201维表征投影层 - FlowMDM xyz joints + velocities
    
    201维 = 1 (root_y) + 2 (vel_trajectory) + 22 joints × 3 (xyz) + 22 joints × 3 (velocity) + 4 (foot contact)
    或简化版: 22 joints × 3 (xyz) + 22 joints × 3 (velocity) + 3 (root velocity)
    
    将 201 维 motion 表征投影到统一维度空间
    """
    def __init__(self, input_dim=201, unified_dim=256, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        if num_layers == 1:
            self.proj = nn.Linear(input_dim, unified_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(hidden_dim, unified_dim))
            self.proj = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, 201] motion 表征
        
        Returns:
            [B, T, unified_dim] 投影后的表征
        """
        return self.proj(x)
