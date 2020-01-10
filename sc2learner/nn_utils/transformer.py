import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from .nn_module import fc_block, build_normalization


class Attention(nn.Module):
    def __init__(self, input_dim, head_dim, output_dim, head_num, dropout_ratio):
        super(Attention, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.attention_pre = fc_block(input_dim, head_dim*head_num*3)  # query, key, value
        self.dropout = nn.Dropout(dropout_ratio)
        self.project = fc_block(head_dim, output_dim)

    def split(self, x, T=False):
        B = x.shape[0]
        x = x.view(B, self.head_num, self.head_dim)
        if T:
            x = x.permute(0, 2, 1).contiguous()
        return x

    def forward(self, x):
        assert(len(x.shape) >= 2)
        old_shape = x.shape[:-1]
        N = x.shape[-1]
        B = reduce(lambda x, y: x*y, old_shape)
        x = x.reshape(B, N)
        x = self.attention_pre(x)
        query, key, value = torch.chunk(x, 3, dim=1)
        query, key, value = self.split(query), self.split(key, T=True), self.split(value)

        score = torch.matmul(query, key)
        score /= math.sqrt(value.shape[2])
        score = F.softmax(score, dim=2)
        score = self.dropout(score)
        attention = torch.matmul(score, value)

        attention = attention.view(B*self.head_num, self.head_dim)
        attention = self.project(attention)
        attention = attention.view(B, self.head_num, -1)
        attention = attention.sum(dim=1)
        attention = self.dropout(attention)
        attention = attention.view(*old_shape, -1)
        return attention


class TransformerLayer(nn.Module):
    def __init__(self, input_dim, head_dim, hidden_dim, output_dim, head_num, mlp_num, dropout_ratio, activation):
        super(TransformerLayer, self).__init__()
        self.attention = Attention(input_dim, head_dim, output_dim, head_num, dropout_ratio)
        self.layernorm1 = build_normalization('LN')(output_dim)
        layers = []
        dims = [output_dim] + [hidden_dim]*(mlp_num-1) + [output_dim]
        for i in range(mlp_num):
            layers.append(fc_block(dims[i], dims[i+1], activation=activation))
        layers.append(nn.Dropout(dropout_ratio))
        self.mlp = nn.Sequential(*layers)
        self.layernorm2 = build_normalization('LN')(output_dim)

    def forward(self, x):
        a = self.attention(x)
        x = self.layernorm1(x + a)
        m = self.mlp(x)
        x = self.layernorm2(x + m)
        return x


class Transformer(nn.Module):
    '''
        Note:
          Input has passed through embedding
    '''
    def __init__(self, input_dim, head_dim=128, hidden_dim=1024, output_dim=256,
                 head_num=2, mlp_num=2, layer_num=3, dropout_ratio=0.1, activation=nn.ReLU()):
        super(Transformer, self).__init__()
        self.embedding = fc_block(input_dim, output_dim, activation=activation)
        self.act = activation
        layers = []
        dims = [output_dim] + [output_dim] * layer_num
        for i in range(layer_num):
            layers.append(TransformerLayer(dims[i], head_dim, hidden_dim, dims[i+1],
                          head_num, mlp_num, dropout_ratio, self.act))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x)
        return self.main(x)
