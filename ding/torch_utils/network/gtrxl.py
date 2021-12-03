from typing import Union, Optional, Dict, Callable, List
import torch
import torch.nn as nn
from ding.torch_utils.network.nn_module import *

from ding.torch_utils import get_lstm
from ding.utils import MODEL_REGISTRY, SequenceType, squeeze


class PositionalEmbedding(nn.Module):
    """
    Overview:
        Positional Embedding used in vanilla Transformer
    .. note::
        Adapted from https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
    """
    def __init__(self, embedding_dim):
        """
        Arguments:
            - embedding_dim: (:obj:`int`): dimension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        inv_freq = 1 / (10000 ** (torch.arange(0.0, embedding_dim, 2.0) / embedding_dim))  # (embedding_dim / 2.0)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        """
        Overview:
            Compute positional embedding
        Arguments:
            - pos_seq: (:obj:`torch.Tensor`): positional sequence,
             usually a 1D integer sequence as [N-1, N-2, ..., 1, 0], N = embedding_dim
        Returns:
            - pos_embedding: (:obj:`torch.Tensor`): positional embedding. Shape (N, 1, N)
        """
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        # For position embedding, the order of sin/cos is negligible.
        # This is because tokens are consumed by the matrix multiplication which is permutation-invariant.
        pos_embedding = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_embedding[:, None, :]


class GRUGatingUnit(torch.nn.Module):
    """
    Overview:
        GRU Gating Unit used in GTrXL
    """
    def __init__(self, input_dim, bg=0.2):
        """
        Arguments:
            - input_dim: (:obj:`int`): dimension of input
        """
        super(GRUGatingUnit, self).__init__()
        self.Wr = torch.nn.Linear(input_dim, input_dim)
        self.Ur = torch.nn.Linear(input_dim, input_dim)
        self.Wz = torch.nn.Linear(input_dim, input_dim)
        self.Uz = torch.nn.Linear(input_dim, input_dim)
        self.Wg = torch.nn.Linear(input_dim, input_dim)
        self.Ug = torch.nn.Linear(input_dim, input_dim)
        self.bg = bg  # constant bias
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        """
        Overview:
            Compute output value with gating mechanism
        Arguments:
            - x: (:obj:`torch.Tensor`): first input.
            - y: (:obj:`torch.Tensor`): first input.
            x and y have same shape and last shape is input_dim.
        Returns:
            - g: (:obj:`torch.Tensor`): output of GRU. Same shape of x and y.
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))  # element wise multiplication
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g  # x.shape == y.shape == g.shape


class Memory:
    """
    Overview:
        Stores the hidden states computed in the previous segments
    .. note::
        For details refer to Transformer-XL: https://arxiv.org/abs/1901.02860
    """
    def __init__(
            self,
            memory_len: int = 20,
            batch_size: int = 64,
            embedding_dim: int = 256,
            layer_num: int = 3,
    ) -> None:
        super(Memory, self).__init__()
        self.embedding_dim = embedding_dim
        self.bs = batch_size
        self.layer_num = layer_num
        self.memory_len = memory_len
        self.memory = None
        self.init()

    def init(self, memory: List[torch.Tensor] = None):
        if memory:
            self.memory = memory
        else:
            self.memory = [
                torch.zeros(self.memory_len, self.bs, self.embedding_dim, dtype=torch.float)
                for _ in range(self.layer_num + 1)
            ]

    def update(self, hidden_state: List[torch.Tensor]):
        """
        + Arguments
            - hidden_states: List[torch.FloatTensor]
        """
        if self.memory is None or hidden_state is None:
            return None
        sequence_len = hidden_state[0].shape[0]
        with torch.no_grad():
            new_memory = []
            end = self.memory_len + sequence_len
            beg = max(0, end - self.memory_len)
            for m, h in zip(self.memory, hidden_state):
                cat = torch.cat([m, h], dim=0)
                new_memory.append(cat[beg:end].detach())
        self.memory = new_memory
        return new_memory

    def get(self):
        return self.memory


class AttentionXL(torch.nn.Module):
    def __init__(self, input_dim: int, head_dim: int, head_num: int, dropout: nn.Module) -> None:
        super(AttentionXL, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_kv = fc_block(input_dim, head_dim * head_num * 2)  # key, value
        self.attention_q = fc_block(input_dim, head_dim * head_num)  # query (not computed with past hidden states)
        self.project = fc_block(head_dim * head_num, input_dim)
        self.project_pos = fc_block(input_dim, head_dim * head_num)  # project the positional embedding
        self.scale = 1 / (head_dim ** 0.5)  # for scaled dot product attention

    def _rel_shift(self, x, zero_triu=False):
        """
        Overview:
            Relatively shift the attention score matrix
        Example:
            a00 a01 a02      0 a00 a01 a02       0  a00 a01      a02  0  a10     a02  0   0
            a10 a11 a12  =>  0 a10 a11 a12  =>  a02  0  a10  =>  a11 a12  0  =>  a11 a12  0
            a20 a21 a22      0 a20 a21 a22      a11 a12  0       a20 a21 a22     a20 a21 a22
                                                a20 a21 a22
            1) Append one "column" of zeros to the left
            2) Reshape the matrix from [3 x 4] into [4 x 3]
            3) Remove the first "row"
            4) Mask out the upper triangle
        Note:
            See the following material for better understanding:
                https://github.com/kimiyoung/transformer-xl/issues/8
                https://arxiv.org/pdf/1901.02860.pdf (Appendix B)
        :param x:
        :param zero_triu:
        :return:
        """
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)
        return x

    def forward(self,
                inputs: torch.Tensor,
                pos_embedding: torch.Tensor,
                full_input: torch.Tensor,
                u: torch.nn.Parameter,
                v: torch.nn.Parameter,
                mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        + pos_embs: positional embeddings passed separately to handle relative positions.
        + Arguments
            - input: torch.FloatTensor, shape - (seq, bs, self.d_input) = (20, 5, 8)
            - pos_embs: torch.FloatTensor, shape - (seq + prev_seq, bs, self.d_input) = (40, 1, 8)
            - memory: torch.FloatTensor, shape - (prev_seq, b, d_in) = (20, 5, 8)
            - u: torch.FloatTensor, shape - (num_heads, inner_dim) = (3 x )
            - v: torch.FloatTensor, shape - (num_heads, inner_dim)
            - mask: torch.FloatTensor, Optional = (20, 40, 1)
        + Returns
            - output: torch.FloatTensor, shape - (seq, bs, self.d_input)
        + symbols representing shape of the tensors
            - cs: current sequence length, b: batch, H: no. of heads
            - d: inner dimension, ps: previous sequence length
        """
        bs, cur_seq, full_seq = inputs.shape[1], inputs.shape[0], full_input.shape[0]
        prev_seq = full_seq - cur_seq

        kv = self.attention_kv(full_input)
        key, value = torch.chunk(kv, 2, dim=-1)  # full_seq x bs x num_head*dim_head
        query = self.attention_q(inputs)  # cur_seq x bs x num_head*dim_head
        #print('pos_embedding:', pos_embedding.shape)
        r = self.project_pos(pos_embedding)  # full_seq x 1 x num_head*dim_head

        # x_i * W^q * (W^k)^T * (x_j)^T
        content_attn = torch.einsum(
            "ibhd,jbhd->ijbh",
            (
                (query.view(cur_seq, bs, self.head_num, self.head_dim) + u),
                key.view(full_seq, bs, self.head_num, self.head_dim),
            ),
        )  # cur_seq x full_seq x bs x head_num

        position_attn = torch.einsum(
            "ibhd,jhd->ijbh",
            (
                (query.view(cur_seq, bs, self.head_num, self.head_dim) + v),
                r.view(cur_seq + prev_seq, self.head_num, self.head_dim),
            ),
        )  # cur_seq x full_seq x bs x head_num
        position_attn = self._rel_shift(position_attn)
        attn = content_attn + position_attn  # cur_seq x full_seq x bs x head_num
        attn.mul_(self.scale)

        if mask is not None and mask.any().item():
            # fills float('-inf') where mask is True.
            attn = attn.masked_fill(mask[..., None], -float("inf")).type_as(attn)
        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)

        # attn_weighted_values = [curr x B x n_heads.d_inner] = [20 x 5 x 96]
        attn_vec = torch.einsum(
                "ijbh,jbhd->ibhd",
                (
                    attn,
                    value.view(cur_seq + prev_seq, bs, self.head_num, self.head_dim),
                ),
            )  # cur_seq x bs x head_num x head_dim
        attn_vec = attn_vec.contiguous().view(cur_seq, bs, self.head_num * self.head_dim)
        # cur_seq x bs x head_num * head_dim

        output = self.dropout(self.project(attn_vec))  # cur_seq x bs x input_dim
        return output


class GatedTransformerXLLayer(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            head_dim: int,
            hidden_dim: int,
            head_num: int,
            mlp_num: int,
            dropout: nn.Module,
            activation: nn.Module,
            gating: bool = True,
    ) -> None:
        super(GatedTransformerXLLayer, self).__init__()
        self.dropout = dropout
        self.gating = gating
        self.gate1 = GRUGatingUnit(input_dim)
        self.gate2 = GRUGatingUnit(input_dim)
        self.attention = AttentionXL(
            input_dim,
            head_dim,
            head_num,
            dropout,
        )
        layers = []
        dims = [input_dim] + [hidden_dim] * (mlp_num - 1) + [input_dim]
        for i in range(mlp_num):
            layers.append(fc_block(dims[i], dims[i + 1], activation=activation))
            if i != mlp_num - 1:
                layers.append(self.dropout)
        layers.append(self.dropout)
        self.mlp = nn.Sequential(*layers)
        self.layernorm1 = build_normalization('LN')(input_dim)
        self.layernorm1 = build_normalization('LN')(input_dim)
        self.activation = activation

    def forward(self,
                inputs: torch.Tensor,
                pos_embedding: torch.Tensor,
                u: torch.nn.Parameter,
                v: torch.nn.Parameter,
                memory: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        # concat memory with input across sequence dimension
        full_input = torch.cat([memory.detach(), inputs], dim=0)  # full_seq x bs x input_dim
        x1 = self.layernorm1(full_input)
        a1 = self.dropout(self.attention(inputs, pos_embedding, x1, u, v, mask=mask))
        a1 = self.activation(a1)  # RELU after attention
        o1 = self.gate1(inputs, a1) if self.gating else inputs + a1
        x2 = self.layernorm1(o1)
        m2 = self.dropout(self.mlp(x2))
        o2 = self.gate2(o1, m2) if self.gating else o1 + m2
        return o2


@MODEL_REGISTRY.register('gtrxl')
class GTrXL(nn.Module):
    """
    Overview:
        GTrCL Transformer implementation
    .. note::
        For details refer to Stabilizing Transformer for Reinforcement Learning: https://arxiv.org/abs/1910.06764
    """
    def __init__(
        self,
            input_dim: int,
            head_dim: int = 128,
            embedding_dim: int = 256,
            head_num: int = 2,
            mlp_num: int = 2,
            layer_num: int = 3,
            memory_len: int = 64,
            dropout_ratio: float = 0.,
            activation: nn.Module = nn.ReLU(),
    ) -> None:
        """Overview:
            Init GTrXL Model
        Arguments:
            - input_dim (:obj:`int`): dimension of input (dimension of a single observation)
            - head_dim (:obj:`int`): dimension of each head
            - hidden_dim (:obj:`int`): dimension of hidden layer in mlp
            - embedding_dim (:obj:`int`): dimension of embedding (dimension of a single observation after embedding)
            - head_num (:obj:`int`): number of heads for multihead attention
            - mlp_num (:obj:`int`): number of mlp layers in attention layer
            - layer_num (:obj:`int`): number of transformer layers
            - dropout_ratio (:obj:`float`): dropout ratio
            - activation (:obj:`nn.Module`): activation function
        """

        super(GTrXL, self).__init__()
        assert embedding_dim % 2 == 0, 'embedding_dim={} should be even'.format(input_dim)
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.embedding = fc_block(input_dim, embedding_dim, activation=activation)
        self.activation = activation
        self.pos_embedding = PositionalEmbedding(embedding_dim)
        # memory to save hidden states of past segments
        # it will be initialized in the forward method to get its size dynamically
        self.memory = None
        self.memory_len = memory_len
        layers = []
        dims = [embedding_dim] + [embedding_dim] * layer_num
        self.dropout = nn.Dropout(dropout_ratio)
        for i in range(layer_num):
            layers.append(
                GatedTransformerXLLayer(dims[i], head_dim, embedding_dim, head_num, mlp_num, self.dropout,
                                        self.activation)
            )
        self.layers = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim
        # u and v are the parameters to compute global content bias and global positional bias
        self.u, self.v = (
            torch.nn.Parameter(torch.Tensor(self.head_num, self.head_dim)),
            torch.nn.Parameter(torch.Tensor(self.head_num, self.head_dim)),
        )

    def forward(self, x: torch.Tensor,) -> torch.Tensor:
        r"""
        Overview:
            GTrXL forward
        Arguments:
            - x (:obj:`torch.Tensor`): input tensor. Shape (B, N, C), B is batch size, \
                N is number of entries, C is feature dimension
            - mask (:obj:`Optional[torch.Tensor]`): bool tensor, can be used to mask out invalid entries in attention. \
                Shape (B, N), B is batch size, N is number of entries
        Returns:
            - x (:obj:`torch.Tensor`): transformer output
        """
        cur_seq, bs = x.shape[:2]
        memory = None if self.memory is None else self.memory.get()
        if memory is None:
            self.memory = Memory(self.memory_len, bs, self.embedding_dim, self.layer_num + 1)
            # (layer_num+1) x memory_len x batch_size x embedding_dim
            memory = self.memory.get()
        #print('memory:', memory[0].shape)

        x = self.dropout(self.embedding(x))
        prev_seq = memory[0].size(0)
        full_seq = cur_seq + prev_seq

        dec_attn_mask = (
            torch.triu(
                torch.ones((cur_seq, cur_seq + prev_seq)),
                diagonal=1 + prev_seq,
            ).bool()[..., None].to(x.device)
        )  # cur_seq x full_seq x 1

        pos_ips = torch.arange(full_seq - 1, -1, -1.0, dtype=torch.float)  # full_seq
        pos_embedding = self.dropout(self.pos_embedding(pos_ips))  # full_seq x 1 x embedding_dim

        hidden_state = [x]
        out = x
        for memory, layer in zip(memory, self.layers):
            out = layer(
                out,
                pos_embedding,
                self.u,
                self.v,
                mask=dec_attn_mask,
                memory=memory,
            )   # cur_seq x bs x embedding_dim
            hidden_state.append(out)
        #print('out:', out.shape)

        out = self.dropout(out)
        memory = self.memory.update(hidden_state)
        output = {"logits": out, "memory": memory}
        return output


if __name__ == "__main__":
    dim_size = 128
    seq_len = 64
    bs = 32
    # input shape: cur_seq x bs x input_dim
    a = torch.rand(seq_len, bs, dim_size)
    print('input:', a.shape)
    m = GTrXL(128, memory_len=50)
    res = m(a)
    o, mem = res['logits'], res['memory']
    print(o.shape)
    print(mem[0].shape)
