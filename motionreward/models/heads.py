"""
任务头模块

包含:
- CriticMLP: 评分头（用于 pairwise ranking）
- AIDetectionHead: AI 检测分类头
- pairwise_loss: Pairwise ranking loss 函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticMLP(nn.Module):
    """评分头 MLP - 用于 Critic 评分任务
    
    输入 motion latent，输出评分
    """
    def __init__(self, in_features, hidden_features=None, out_features=1, drop=0.1):
        super().__init__()
        hidden_features = hidden_features or in_features
        hidden_1 = hidden_features * 2
        hidden_2 = hidden_features
        hidden_3 = hidden_features // 2
        
        self.fc1 = nn.Linear(in_features, hidden_1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.act2 = nn.GELU()
        
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.act3 = nn.GELU()
        
        self.fc4 = nn.Linear(hidden_3, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.drop(x)
        x = self.fc4(x)
        return x


class AIDetectionHead(nn.Module):
    """AI 检测分类头 - 判断 motion 是否为 AI 生成
    
    二分类：0=真实, 1=AI生成
    """
    def __init__(self, in_features, hidden_features=None, num_classes=2, drop=0.1):
        super().__init__()
        hidden_features = hidden_features or in_features
        hidden_1 = hidden_features * 2
        hidden_2 = hidden_features
        hidden_3 = hidden_features // 2
        
        self.fc1 = nn.Linear(in_features, hidden_1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.act2 = nn.GELU()
        
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.act3 = nn.GELU()
        
        self.fc4 = nn.Linear(hidden_3, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.drop(x)
        x = self.fc4(x)
        return x


def pairwise_loss(critic):
    """Pairwise ranking loss: better 应该得分更高
    
    Args:
        critic: [B, 2] tensor，第一列是 better 的分数，第二列是 worse 的分数
    
    Returns:
        loss: 交叉熵损失
        loss_list: 每个样本的损失
        acc: 准确率（better 分数 > worse 分数的比例）
    """
    target = torch.zeros(critic.shape[0], dtype=torch.long).to(critic.device)
    loss_list = F.cross_entropy(critic, target, reduction='none')
    loss = torch.mean(loss_list)
    
    critic_diff = critic[:, 0] - critic[:, 1]
    acc = torch.mean((critic_diff > 0).clone().detach().float())
    
    return loss, loss_list, acc
