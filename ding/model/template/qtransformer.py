import copy
import math
import os
import time
import warnings
from functools import wraps
from os.path import exists
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from packaging import version
from sympy import numer

from torch import Tensor, einsum, nn
from torch.cuda.amp import autocast
from torch.nn import Module, ModuleList
from torch.nn.functional import log_softmax, pad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # return log_softmax(self.proj(x), dim=-1)
        return self.proj(x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, tgt_mask):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return self.sublayer[1](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class stateEncode(nn.Module):
    def __init__(self, num_timesteps, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(num_timesteps * state_dim, 256)
        self.fc2 = nn.Linear(256, 256)  # Corrected the input size
        self.fc3 = nn.Linear(256, 512)

    def forward(self, x):
        batch_size = x.size(0)
        # Reshape from (Batch, 8, 256) to (Batch, 2048)
        x = x.view(batch_size, -1)
        # Pass through the layers with activation functions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.unsqueeze(1)


class actionEncode(nn.Module):
    def __init__(self, action_dim, action_bin):
        super().__init__()
        self.actionbin = action_bin
        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.actionbin, 512) for _ in range(action_dim)]
        )

    def forward(self, x):
        x = x.to(dtype=torch.float)
        b, n, _ = x.shape
        slices = torch.unbind(x, dim=1)
        layer_outputs = torch.empty(b, n, 512, device=x.device)
        for i, layer in enumerate(self.linear_layers[:n]):
            slice_output = layer(slices[i])
            layer_outputs[:, i, :] = slice_output
        return layer_outputs


class DecoderOnly(nn.Module):
    def __init__(self, action_bin, N=8, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(DecoderOnly, self).__init__()
        c = copy.deepcopy
        self_attn = MultiHeadedAttention(h, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.position = PositionalEncoding(d_model, dropout)
        self.model = Decoder(
            DecoderLayer(d_model, c(self_attn), c(feed_forward), dropout), N
        )
        self.Generator = Generator(d_model, vocab=action_bin)


    def forward(self, x):
        x = self.position(x)
        x = self.model(x, subsequent_mask(x.size(1)).to(x.device))
        x = self.Generator(x)
        return x


class QTransformer(nn.Module):
    def __init__(self, num_timesteps, state_dim, action_dim, action_bin):
        super().__init__()
        self.stateEncode = stateEncode(num_timesteps, state_dim)
        self.actionEncode = actionEncode(action_dim, action_bin)
        self.Transormer = DecoderOnly(action_bin)

    def forward(
        self,
        state: Tensor,
        action: Optional[Tensor] = None,
    ):
        stateEncode = self.stateEncode(state)
        if action is not None:
            actionEncode = self.actionEncode(action)
            return self.Transormer(torch.cat((stateEncode, actionEncode), dim=1))
        return self.Transormer(stateEncode)


# def get_optimal_actions(
#         self,
#         encoded_state,
#         actions: Optional[Tensor] = None,
#     ):
#         batch_size = encoded_state.shape[0]
#         action_bins = torch.empty(
#             batch_size, self.num_actions, device=encoded_state.device, dtype=torch.long
#         )
#         cache = None
#         tokens = self.state_append_actions(encoded_state, actions=actions)

#         for action_idx in range(self.num_actions):
#             embed, cache = self.transformer(
#                 tokens, context=encoded_state, cache=cache, return_cache=True
#             )
#             q_values = self.get_q_value_fuction(embed[:, 1:, :])
#             if action_idx == 0:
#                 special_idx = action_idx
#             else:
#                 special_idx = action_idx - 1
#             _, selected_action_indices = q_values[:, special_idx, :].max(dim=-1)
#             action_bins[:, action_idx] = selected_action_indices
#             now_actions = action_bins[:, 0 : action_idx + 1]
#             tokens = self.state_append_actions(encoded_state, actions=now_actions)
#         return action_bins
