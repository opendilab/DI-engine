import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .nn_module import fc_block, build_normalization


class Attention(nn.Module):
    r"""
    Overview:
        For each entry embedding, compute individual attention across all entries, add them up to get output attention
    """

    def __init__(self, input_dim, head_dim, output_dim, head_num, dropout):
        r"""
        Overview:
            Init attention
        Arguments:
            - input_dim (:obj:`int`): dimension of input
            - head_dim (:obj:`int`): dimension of each head
            - output_dim (:obj:`int`): dimension of output
            - head_num (:obj:`int`): head num for multihead attention
            - dropout (:obj:`nn.Module`): dropout layer
        """
        super(Attention, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_pre = fc_block(input_dim, head_dim * head_num * 3)  # query, key, value
        self.project = fc_block(head_dim * head_num, output_dim)

    def split(self, x, T=False):
        r"""
        Overview:
            Split input to get multihead queries, keys, values
        Arguments:
            - x (:obj:`tensor`): query or key or value
            - T (:obj:`bool`): whether to transpose output
        Returns:
            - x (:obj:`list`): list of output tensors for each head
        """
        B, N = x.shape[:2]
        x = x.view(B, N, self.head_num, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # B, head_num, N, head_dim
        if T:
            x = x.permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self, x, mask=None):
        r"""
        Overview:
           Compute attention
        Arguments:
            - x (:obj:`tensor`): input tensor
            - mask (:obj:`tensor`): mask out invalid entries
        Returns:
            - attention (:obj:`tensor`): attention tensor
        """
        assert (len(x.shape) == 3)
        B, N = x.shape[:2]
        x = self.attention_pre(x)
        query, key, value = torch.chunk(x, 3, dim=2)
        query, key, value = self.split(query), self.split(key, T=True), self.split(value)

        score = torch.matmul(query, key)  # B, head_num, N, N
        score /= math.sqrt(self.head_dim)
        if mask is not None:
            score.masked_fill(~mask, value=-1e9)

        score = F.softmax(score, dim=-1)
        score = self.dropout(score)
        attention = torch.matmul(score, value)  # B, head_num, N, head_dim

        attention = attention.permute(0, 2, 1, 3).contiguous()  # B, N, head_num, head_dim
        attention = self.project(attention.view(B, N, -1))  # B, N, output_dim
        return attention


class TransformerLayer(nn.Module):
    r"""
    Overview:
        In transformer layer, first computes entries's attention and applies a feedforward layer
    """

    def __init__(self, input_dim, head_dim, hidden_dim, output_dim, head_num, mlp_num, dropout, activation):
        r"""
        Overview:
            Init transformer layer
        Arguments:
            - input_dim (:obj:`int`): dimension of input
            - head_dim (:obj:`int`): dimension of each head
            - hidden_dim (:obj:`int`): dimension of hidden layer in mlp
            - output_dim (:obj:`int`): dimension of output
            - head_num (:obj:`int`): number of heads for multihead attention
            - mlp_num (:obj:`int`): number of mlp layers
            - dropout (:obj:`nn.Module`): dropout layer
            - activation (:obj:`nn.Module`): activation function
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

    def forward(self, inputs):
        """
        Overview:
            transformer layer forward
        Arguments:
            - inputs (:obj:`tuple`): x and mask
        Returns:
            - output (:obj:`tuple`): x and mask
        """
        x, mask = inputs
        a = self.dropout(self.attention(x, mask))
        x = self.layernorm1(x + a)
        m = self.dropout(self.mlp(x))
        x = self.layernorm2(x + m)
        return (x, mask)


class Transformer(nn.Module):
    '''
    Overview:
        Transformer implementation

        Note:
            For details refer to Attention is all you need: http://arxiv.org/abs/1706.03762
    '''

    def __init__(
        self,
        input_dim,
        head_dim=128,
        hidden_dim=1024,
        output_dim=256,
        head_num=2,
        mlp_num=2,
        layer_num=3,
        dropout_ratio=0.0,
        activation=nn.ReLU(),
    ):
        r"""
        Overview:
            Init transformer
        Arguments:
            - input_dim (:obj:`int`): dimension of input
            - head_dim (:obj:`int`): dimension of each head
            - hidden_dim (:obj:`int`): dimension of hidden layer in mlp
            - output_dim (:obj:`int`): dimension of output
            - head_num (:obj:`int`): number of heads for multihead attention
            - mlp_num (:obj:`int`): number of mlp layers
            - layer_num (:obj:`int`): number of transformer layers
            - dropout_ratio (:obj:`float`): dropout ratio
            - activation (:obj:`nn.Module`): activation function
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

    def forward(self, x, mask=None):
        r"""
        Overview:
            Transformer forward
        Arguments:
            - x (:obj:`tensor`): input tensor, shape (B, N, C), B is batch size, N is number of entries,
                C is feature dimension
            - mask (:obj:`tensor` or :obj:`None`): bool tensor, can be used to mask out invalid entries in attention,
                shape (B, N), B is batch size, N is number of entries
        Returns:
            - x (:obj:`tensor`): transformer output
        """
        if mask is not None:
            mask = mask.unsqueeze(dim=1).repeat(1, mask.shape[1], 1).unsqueeze(dim=1)
        x = self.embedding(x)
        x = self.dropout(x)
        x, mask = self.main((x, mask))
        return x


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k: int, dropout: float = 0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / (self.d_k ** 0.5), k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(~mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output
