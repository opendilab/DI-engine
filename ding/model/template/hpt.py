from typing import Union, Optional, Dict, Callable, List
import torch
import torch.nn as nn

from ding.model.common.head import DuelingHead
from ding.utils.registry_factory import MODEL_REGISTRY
from ding.model.template.policy_stem import PolicyStem
@MODEL_REGISTRY.register('hpt')
class HPT(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(HPT, self).__init__()
        # 初始化 Policy Stem
        self.policy_stem = PolicyStem()
        self.policy_stem.init_cross_attn()
        
        # Dueling Head，输入为 16*128，输出为动作维度
        self.head = DuelingHead(hidden_size=16*128, output_size=action_dim)
    def forward(self, x):
        # Policy Stem 输出 [B, 16, 128]
        tokens = self.policy_stem.compute_latent(x)
        # Flatten 操作
        tokens_flattened = tokens.view(tokens.size(0), -1)  # [B, 16*128]
        # 输入到 Dueling Head
        q_values = self.head(tokens_flattened)
        return q_values