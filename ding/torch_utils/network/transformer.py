import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple

from .nn_module import fc_block, build_normalization


class Attention(nn.Module):
    """
    Overview:
        For each entry embedding, compute individual attention across all entries, add them up to get output attention.
    Interfaces:
        ``__init__``, ``split``, ``forward``
    """

    def __init__(self, input_dim: int, head_dim: int, output_dim: int, head_num: int, dropout: nn.Module) -> None:
        """
        Overview:
            Initialize the Attention module with the provided dimensions and dropout layer.
        Arguments:
            - input_dim (:obj:`int`): The dimension of the input.
            - head_dim (:obj:`int`): The dimension of each head in the multi-head attention mechanism.
            - output_dim (:obj:`int`): The dimension of the output.
            - head_num (:obj:`int`): The number of heads in the multi-head attention mechanism.
            - dropout (:obj:`nn.Module`): The dropout layer used in the attention mechanism.
        """
        super(Attention, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_pre = fc_block(input_dim, head_dim * head_num * 3)  # query, key, value
        self.project = fc_block(head_dim * head_num, output_dim)

    def split(self, x: torch.Tensor, T: bool = False) -> List[torch.Tensor]:
        """
        Overview:
            Split the input to get multi-head queries, keys, and values.
        Arguments:
            - x (:obj:`torch.Tensor`): The tensor to be split, which could be a query, key, or value.
            - T (:obj:`bool`, optional): If True, transpose the output tensors. Defaults to False.
        Returns:
            - x (:obj:`List[torch.Tensor]`): A list of output tensors for each head.
        """
        B, N = x.shape[:2]
        x = x.view(B, N, self.head_num, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # B, head_num, N, head_dim
        if T:
            x = x.permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Overview:
            Compute the attention from the input tensor.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor for the forward computation.
            - mask (:obj:`Optional[torch.Tensor]`, optional): Optional mask to exclude invalid entries.
                Defaults to None.
        Returns:
            - attention (:obj:`torch.Tensor`): The computed attention tensor.
        """
        assert (len(x.shape) == 3)
        B, N = x.shape[:2]
        x = self.attention_pre(x)
        query, key, value = torch.chunk(x, 3, dim=2)
        query, key, value = self.split(query), self.split(key, T=True), self.split(value)

        score = torch.matmul(query, key)  # B, head_num, N, N
        score /= math.sqrt(self.head_dim)
        if mask is not None:
            # inplace modification for reasonable softmax
            score.masked_fill_(~mask, value=-1e9)

        score = F.softmax(score, dim=-1)
        score = self.dropout(score)
        attention = torch.matmul(score, value)  # B, head_num, N, head_dim

        attention = attention.permute(0, 2, 1, 3).contiguous()  # B, N, head_num, head_dim
        attention = self.project(attention.view(B, N, -1))  # B, N, output_dim
        return attention


class TransformerLayer(nn.Module):
    """
    Overview:
        In transformer layer, first computes entries's attention and applies a feedforward layer.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self, input_dim: int, head_dim: int, hidden_dim: int, output_dim: int, head_num: int, mlp_num: int,
            dropout: nn.Module, activation: nn.Module
    ) -> None:
        """
        Overview:
            Initialize the TransformerLayer with the provided dimensions, dropout layer, and activation function.
        Arguments:
            - input_dim (:obj:`int`): The dimension of the input.
            - head_dim (:obj:`int`): The dimension of each head in the multi-head attention mechanism.
            - hidden_dim (:obj:`int`): The dimension of the hidden layer in the MLP (Multi-Layer Perceptron).
            - output_dim (:obj:`int`): The dimension of the output.
            - head_num (:obj:`int`): The number of heads in the multi-head attention mechanism.
            - mlp_num (:obj:`int`): The number of layers in the MLP.
            - dropout (:obj:`nn.Module`): The dropout layer used in the attention mechanism.
            - activation (:obj:`nn.Module`): The activation function used in the MLP.
        """
        super(TransformerLayer, self).__init__()
        self.attention = Attention(input_dim, head_dim, output_dim, head_num, dropout)
        self.layernorm1 = build_normalization('LN')(output_dim)
        self.dropout = dropout
        layers = []
        dims = [output_dim] + [hidden_dim] * (mlp_num - 1) + [output_dim]
        for i in range(mlp_num):
            layers.append(fc_block(dims[i], dims[i + 1], activation=activation))
            if i != mlp_num - 1:
                layers.append(self.dropout)
        layers.append(self.dropout)
        self.mlp = nn.Sequential(*layers)
        self.layernorm2 = build_normalization('LN')(output_dim)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Compute the forward pass through the Transformer layer.
        Arguments:
            - inputs (:obj:`Tuple[torch.Tensor, torch.Tensor]`): A tuple containing the input tensor `x` and
                the mask tensor.
        Returns:
            - output (:obj:`Tuple[torch.Tensor, torch.Tensor]`): A tuple containing the predicted value tensor and
                the mask tensor.
        """
        x, mask = inputs
        a = self.dropout(self.attention(x, mask))
        x = self.layernorm1(x + a)
        m = self.dropout(self.mlp(x))
        x = self.layernorm2(x + m)
        return x, mask


class Transformer(nn.Module):
    """
    Overview:
        Implementation of the Transformer model.

    .. note::
        For more details, refer to "Attention is All You Need": http://arxiv.org/abs/1706.03762.

    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        input_dim: int,
        head_dim: int = 128,
        hidden_dim: int = 1024,
        output_dim: int = 256,
        head_num: int = 2,
        mlp_num: int = 2,
        layer_num: int = 3,
        dropout_ratio: float = 0.,
        activation: nn.Module = nn.ReLU(),
    ):
        """
        Overview:
            Initialize the Transformer with the provided dimensions, dropout layer, activation function,
            and layer numbers.
        Arguments:
            - input_dim (:obj:`int`): The dimension of the input.
            - head_dim (:obj:`int`): The dimension of each head in the multi-head attention mechanism.
            - hidden_dim (:obj:`int`): The dimension of the hidden layer in the MLP (Multi-Layer Perceptron).
            - output_dim (:obj:`int`): The dimension of the output.
            - head_num (:obj:`int`): The number of heads in the multi-head attention mechanism.
            - mlp_num (:obj:`int`): The number of layers in the MLP.
            - layer_num (:obj:`int`): The number of Transformer layers.
            - dropout_ratio (:obj:`float`): The dropout ratio for the dropout layer.
            - activation (:obj:`nn.Module`): The activation function used in the MLP.
        """
        super(Transformer, self).__init__()
        self.embedding = fc_block(input_dim, output_dim, activation=activation)
        self.act = activation
        layers = []
        dims = [output_dim] + [output_dim] * layer_num
        self.dropout = nn.Dropout(dropout_ratio)
        for i in range(layer_num):
            layers.append(
                TransformerLayer(dims[i], head_dim, hidden_dim, dims[i + 1], head_num, mlp_num, self.dropout, self.act)
            )
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Overview:
            Perform the forward pass through the Transformer.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor, with shape `(B, N, C)`, where `B` is batch size, \
                `N` is the number of entries, and `C` is the feature dimension.
            - mask (:obj:`Optional[torch.Tensor]`, optional): The mask tensor (bool), used to mask out invalid \
                entries in attention. It has shape `(B, N)`, where `B` is batch size and `N` is number of \
                entries. Defaults to None.
        Returns:
            - x (:obj:`torch.Tensor`): The output tensor from the Transformer.
        """
        if mask is not None:
            mask = mask.unsqueeze(dim=1).repeat(1, mask.shape[1], 1).unsqueeze(dim=1)
        x = self.embedding(x)
        x = self.dropout(x)
        x, mask = self.main((x, mask))
        return x


class ScaledDotProductAttention(nn.Module):
    """
    Overview:
        Implementation of Scaled Dot Product Attention, a key component of Transformer models.
        This class performs the dot product of the query, key and value tensors, scales it with the square root of the
        dimension of the key vector (d_k) and applies dropout for regularization.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, d_k: int, dropout: float = 0.0) -> None:
        """
        Overview:
            Initialize the ScaledDotProductAttention module with the dimension of the key vector and the dropout rate.
        Arguments:
            - d_k (:obj:`int`): The dimension of the key vector. This will be used to scale the dot product of the \
                query and key.
            - dropout (:obj:`float`, optional): The dropout rate to be applied after the softmax operation. \
                Defaults to 0.0.
        """
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Overview:
            Perform the Scaled Dot Product Attention operation on the query, key and value tensors.
        Arguments:
            - q (:obj:`torch.Tensor`): The query tensor.
            - k (:obj:`torch.Tensor`): The key tensor.
            - v (:obj:`torch.Tensor`): The value tensor.
            - mask (:obj:`Optional[torch.Tensor]`): An optional mask tensor to be applied on the attention scores.
                Defaults to None.
        Returns:
            - output (:obj:`torch.Tensor`): The output tensor after the attention operation.
        """
        attn = torch.matmul(q / (self.d_k ** 0.5), k.transpose(2, 3))
        if mask is not None:
            # inplace modification for reasonable softmax
            attn.masked_fill_(~mask, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output
