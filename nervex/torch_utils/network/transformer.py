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
        self.attention_pre = fc_block(input_dim, head_dim * head_num * 3)  # query, key, value
        self.dropout = nn.Dropout(dropout_ratio)
        self.project = fc_block(head_dim * head_num, output_dim)

    def split(self, x, T=False):
        B, N = x.shape[:2]
        x = x.view(B, N, self.head_num, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # B, head_num, N, head_dim
        if T:
            x = x.permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self, x, vaild_num=None):
        """
        Overview:
            x: [batch_size, seq_len, embeddding_size]
        """
        assert (len(x.shape) == 3)
        B, N = x.shape[:2]
        x = self.attention_pre(x)
        query, key, value = torch.chunk(x, 3, dim=2)
        query, key, value = self.split(query), self.split(key, T=True), self.split(value)

        score = torch.matmul(query, key)  # B, head_num, N, N
        score /= math.sqrt(self.head_dim)
        if vaild_num is not None:
            for idx, v in enumerate(vaild_num):
                score[idx, :, v:, :] = -1e9
                score[idx, :, :, v:] = -1e9

        score = F.softmax(score, dim=-1)
        score = self.dropout(score)
        attention = torch.matmul(score, value)  # B, head_num, N, head_dim

        attention = attention.permute(0, 2, 1, 3).contiguous()  # B, N, head_num, head_dim
        attention = self.project(attention.view(B, N, -1))  # B, N, output_dim
        attention = self.dropout(attention)
        return attention


class AttentionEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, head_dim=2, head_num=16, dropout_ratio=0.1, activation=nn.ReLU()):
        super(AttentionEmbedding, self).__init__()
        self.attention = Attention(1, head_dim, 1, head_num, dropout_ratio)
        self.embedding = fc_block(input_dim, embedding_dim, activation=activation)

    def forward(self, x):
        B, S = x.shape[:2]
        x = x.view(B * S, -1, 1)
        x = self.attention(x).view(B, S, -1)
        return self.embedding(x)


class TransformerLayer(nn.Module):
    def __init__(self, input_dim, head_dim, hidden_dim, output_dim, head_num, mlp_num, dropout_ratio, activation):
        super(TransformerLayer, self).__init__()
        self.attention = Attention(input_dim, head_dim, output_dim, head_num, dropout_ratio)
        self.layernorm1 = build_normalization('LN')(output_dim)
        layers = []
        dims = [output_dim] + [hidden_dim] * (mlp_num - 1) + [output_dim]
        for i in range(mlp_num):
            layers.append(fc_block(dims[i], dims[i + 1], activation=activation))
        layers.append(nn.Dropout(dropout_ratio))
        self.mlp = nn.Sequential(*layers)
        self.layernorm2 = build_normalization('LN')(output_dim)

    def forward(self, inputs):
        x, valid_num = inputs['data'], inputs['vaild_num']
        a = self.attention(x, valid_num)
        x = self.layernorm1(x + a)
        m = self.mlp(x)
        x = self.layernorm2(x + m)
        return {'data': x, 'vaild_num': valid_num}


class Transformer(nn.Module):
    '''
        Note:
          Input has passed through embedding
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
        pad_val=0,
        dropout_ratio=0.1,
        activation=nn.ReLU()
    ):
        super(Transformer, self).__init__()
        self.embedding = fc_block(input_dim, output_dim, activation=activation)
        #self.embedding = AttentionEmbedding(input_dim, output_dim, activation=activation)
        self.pad_val = pad_val
        self.act = activation
        layers = []
        dims = [output_dim] + [output_dim] * layer_num
        for i in range(layer_num):
            layers.append(
                TransformerLayer(
                    dims[i], head_dim, hidden_dim, dims[i + 1], head_num, mlp_num, dropout_ratio, self.act
                )
            )
        self.main = nn.Sequential(*layers)

    def forward(self, x, tensor_output=False):
        if isinstance(x, list):
            x, valid_num = self._pack_inputs(x)  # batch_size, seq_len, input_dim
            x = self.embedding(x)
            x = self.main({'data': x, 'vaild_num': valid_num})['data']
            if tensor_output:
                return x, valid_num
            else:
                return self._filter_outputs(x, valid_num)
        elif isinstance(x, torch.Tensor):
            x = self.embedding(x)
            x = self.main({'data': x, 'vaild_num': None})['data']
            return x
        else:
            raise TypeError("invalid type: {}".format(type(x)))

    def _pack_inputs(self, x):
        assert isinstance(x, list)
        self.max_seq_len = max([t.shape[0] for t in x])
        aligned_x = []
        valid_num = []
        for item in x:
            assert len(item.shape) == 2  # seq_len, embeddding_size
            N, M = item.shape
            if N >= self.max_seq_len:
                aligned_x.append(item[:self.max_seq_len])
                valid_num.append(self.max_seq_len)
            else:
                pad_tensor = torch.full(size=(self.max_seq_len - N, M), fill_value=self.pad_val).to(item.device)
                aligned_x.append(torch.cat([item, pad_tensor], dim=0))
                valid_num.append(N)
        aligned_x = torch.stack(aligned_x, dim=0)
        return aligned_x, valid_num

    def _filter_outputs(self, x, valid_num):
        ret = []
        for item, v in zip(x, valid_num):
            ret.append(item[:v])
        return ret
