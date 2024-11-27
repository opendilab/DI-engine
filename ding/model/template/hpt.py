from typing import Union, Optional, Dict, Callable, List
from einops import rearrange, repeat
import torch
import torch.nn as nn
from ding.model.common.head import DuelingHead
from ding.utils.registry_factory import MODEL_REGISTRY


INIT_CONST = 0.02

@MODEL_REGISTRY.register('hpt')
class HPT(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(HPT, self).__init__()
        # Initialise Policy Stem
        self.policy_stem = PolicyStem()
        self.policy_stem.init_cross_attn()
        
        # Dueling Head, input is 16*128, output is action dimension
        self.head = DuelingHead(hidden_size=16*128, output_size=action_dim)
    def forward(self, x):
        # Policy Stem Outputs [B, 16, 128]
        tokens = self.policy_stem.compute_latent(x)
        # Flatten Operation
        tokens_flattened = tokens.view(tokens.size(0), -1)  # [B, 16*128]
        # Enter to Dueling Head
        q_values = self.head(tokens_flattened)
        return q_values



class PolicyStem(nn.Module):
    """policy stem
        Overview:
             The reference uses PolicyStem from
            <https://github.com/liruiw/HPT/blob/main/hpt/models/policy_stem.py>
    """
    def __init__(self, feature_dim: int = 8, token_dim: int = 128, **kwargs):
        super().__init__()
        # Initialise the feature extraction module
        self.feature_extractor = nn.Linear(feature_dim, token_dim)
        # Initialise CrossAttention
        self.init_cross_attn()

    def init_cross_attn(self):
        """Initialize cross attention module and learnable tokens."""
        token_num = 16
        self.tokens = nn.Parameter(torch.randn(1, token_num, 128) * INIT_CONST)
        self.cross_attention = CrossAttention(128, heads=8, dim_head=64, dropout=0.1)

    def compute_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute latent representations of input data using attention.

        Args:
            x (torch.Tensor): Input tensor with shape [B, T, ..., F].

        Returns:
            torch.Tensor: Latent tokens, shape [B, 16, 128].
        """
        # Using the Feature Extractor
        stem_feat = self.feature_extractor(x)  
        stem_feat = stem_feat.reshape(stem_feat.shape[0], -1, stem_feat.shape[-1])  # (B, N, 128)
        # Calculating latent tokens using CrossAttention
        stem_tokens = self.tokens.repeat(len(stem_feat), 1, 1)  # (B, 16, 128)
        stem_tokens = self.cross_attention(stem_tokens, stem_feat)  # (B, 16, 128)
        return stem_tokens
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute latent tokens.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Latent tokens tensor.
        """
        return self.compute_latent(x)
        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path : str):
        torch.save(self.state_dict(), path)

    @property
    def device(self):
        return next(self.parameters()).device

class CrossAttention(nn.Module):
    """
    CrossAttention module used in the Perceiver IO model.

    Args:
        query_dim (int): The dimension of the query input.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        dropout (float, optional): The dropout probability. Defaults to 0.0.
    """

    def __init__(self, query_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the CrossAttention module.

        Args:
            x (torch.Tensor): The query input tensor.
            context (torch.Tensor): The context input tensor.
            mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            # fill in the masks with negative values
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)