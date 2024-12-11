from typing import Union, Optional, Dict, Callable, List
from einops import rearrange, repeat
import torch
import torch.nn as nn
from ding.model.common.head import DuelingHead
from ding.utils import MODEL_REGISTRY, squeeze


@MODEL_REGISTRY.register('hpt')
class HPT(nn.Module):
    """
    Overview:
        The HPT model for reinforcement learning, which consists of a Policy Stem and a Dueling Head.
        The Policy Stem utilizes cross-attention to process input data,
        and the Dueling Head computes Q-values for discrete action spaces.

    Interfaces:
        ``__init__``, ``forward``

    GitHub: [https://github.com/liruiw/HPT/blob/main/hpt/models/policy_stem.py]

    """

    def __init__(self, state_dim: int, action_dim: int):
        """
        Overview:
            Initialize the HPT model, including the Policy Stem and the Dueling Head.

        Arguments:
            - state_dim (:obj:`int`): The dimension of the input state.
            - action_dim (:obj:`int`): The dimension of the action space.

        .. note::
            The Policy Stem is initialized with cross-attention,
            and the Dueling Head is set to process the resulting tokens.
        """
        super(HPT, self).__init__()
        # Initialise Policy Stem
        self.policy_stem = PolicyStem(state_dim, 128)
        self.policy_stem.init_cross_attn()

        action_dim = squeeze(action_dim)
        # Dueling Head, input is 16*128, output is action dimension
        self.head = DuelingHead(hidden_size=16 * 128, output_size=action_dim)

    def forward(self, x: torch.Tensor):
        """
        Overview:
            Forward pass of the HPT model.
            Computes latent tokens from the input state and passes them through the Dueling Head.

        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor representing the state.

        Returns:
            - q_values (:obj:`torch.Tensor`): The predicted Q-values for each action.
        """
        # Policy Stem Outputs [B, 16, 128]
        tokens = self.policy_stem.compute_latent(x)
        # Flatten Operation
        tokens_flattened = tokens.view(tokens.size(0), -1)  # [B, 16*128]
        # Enter to Dueling Head
        q_values = self.head(tokens_flattened)
        return q_values


class PolicyStem(nn.Module):
    """
    Overview:
        The Policy Stem module is responsible for processing input features
        and generating latent tokens using a cross-attention mechanism.
        It extracts features from the input and then applies cross-attention
        to generate a set of latent tokens.

    Interfaces:
        ``__init__``, ``init_cross_attn``, ``compute_latent``, ``forward``

    .. note::
        This module is inspired by the implementation in the Perceiver IO model
        and uses attention mechanisms for feature extraction.
    """
    INIT_CONST = 0.02

    def __init__(self, feature_dim: int = 8, token_dim: int = 128):
        """
        Overview:
            Initialize the Policy Stem module with a feature extractor and cross-attention mechanism.

        Arguments:
            - feature_dim (:obj:`int`): The dimension of the input features.
            - token_dim (:obj:`int`): The dimension of the latent tokens generated
            by the attention mechanism.
        """
        super().__init__()
        # Initialise the feature extraction module
        self.feature_extractor = nn.Linear(feature_dim, token_dim)
        # Initialise CrossAttention
        self.init_cross_attn()

    def init_cross_attn(self):
        """Initialize cross-attention module and learnable tokens."""
        token_num = 16
        self.tokens = nn.Parameter(torch.randn(1, token_num, 128) * self.INIT_CONST)
        self.cross_attention = CrossAttention(128, heads=8, dim_head=64, dropout=0.1)

    def compute_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Compute latent representations of the input data using
            the feature extractor and cross-attention.

        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor with shape [B, T, ..., F].

        Returns:
            - stem_tokens (:obj:`torch.Tensor`): Latent tokens with shape [B, 16, 128].
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
        Overview:
            Forward pass to compute latent tokens.

        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor.

        Returns:
            - torch.Tensor: Latent tokens tensor.
        """
        return self.compute_latent(x)

    @property
    def device(self) -> torch.device:
        """Returns the device on which the model parameters are located."""
        return next(self.parameters()).device


class CrossAttention(nn.Module):

    def __init__(self, query_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        """
        Overview:
            CrossAttention module used in the Perceiver IO model.
            It computes the attention between the query and context tensors,
            and returns the output tensor after applying attention.

        Arguments:
            - query_dim (:obj:`int`): The dimension of the query input.
            - heads (:obj:`int`, optional): The number of attention heads. Defaults to 8.
            - dim_head (:obj:`int`, optional): The dimension of each attention head. Defaults to 64.
            - dropout (:obj:`float`, optional): The dropout probability. Defaults to 0.0.
        """
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim
        # Scaling factor for the attention logits to ensure stable gradients.
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Overview:
            Forward pass of the CrossAttention module.
            Computes the attention between the query and context tensors.

        Arguments:
            - x (:obj:`torch.Tensor`): The query input tensor.
            - context (:obj:`torch.Tensor`): The context input tensor.
            - mask (:obj:`Optional[torch.Tensor]`): The attention mask tensor. Defaults to None.

        Returns:
            - torch.Tensor: The output tensor after applying attention.
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
