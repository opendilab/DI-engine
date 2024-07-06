"""
Overview:
    This file implements the core modules of GTrXL Transformer as described in
    "Stabilizing Transformer for Reinforcement Learning" (https://arxiv.org/abs/1910.06764).
"""
from typing import Optional, Dict, List
import warnings
import numpy as np
import torch
import torch.nn as nn
from ding.torch_utils.network.nn_module import fc_block, build_normalization, F


class PositionalEmbedding(nn.Module):
    """
    Overview:
        The PositionalEmbedding module implements the positional embedding used in the vanilla Transformer model.
    Interfaces:
        ``__init__``, ``forward``

    .. note::
        This implementation is adapted from https://github.com/kimiyoung/transformer-xl/blob/ \
            master/pytorch/mem_transformer.py
    """

    def __init__(self, embedding_dim: int):
        """
        Overview:
            Initialize the PositionalEmbedding module.
        Arguments:
            - embedding_dim: (:obj:`int`): The dimensionality of the embeddings.
        """

        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        inv_freq = 1 / (10000 ** (torch.arange(0.0, embedding_dim, 2.0) / embedding_dim))  # (embedding_dim / 2)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Compute positional embedding given a sequence of positions.
        Arguments:
             - pos_seq (:obj:`torch.Tensor`): The positional sequence, \
                typically a 1D tensor of integers in the form of [seq_len-1, seq_len-2, ..., 1, 0],
        Returns:
            - pos_embedding (:obj:`torch.Tensor`): The computed positional embeddings. \
                The shape of the tensor is (seq_len, 1, embedding_dim).
        """

        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        # For position embedding, the order of sin/cos is negligible.
        # This is because tokens are consumed by the matrix multiplication which is permutation-invariant.
        pos_embedding = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_embedding.unsqueeze(1)


class GRUGatingUnit(torch.nn.Module):
    """
    Overview:
        The GRUGatingUnit module implements the GRU gating mechanism used in the GTrXL model.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, input_dim: int, bg: float = 2.):
        """
        Overview:
            Initialize the GRUGatingUnit module.
        Arguments:
            - input_dim (:obj:`int`): The dimensionality of the input.
            - bg (:obj:`bg`): The gate bias. By setting bg > 0 we can explicitly initialize the gating mechanism to \
                be close to the identity map. This can greatly improve the learning speed and stability since it \
                initializes the agent close to a Markovian policy (ignore attention at the beginning).
        """

        super(GRUGatingUnit, self).__init__()
        self.Wr = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Overview:
            Compute the output value using the GRU gating mechanism.
        Arguments:
            - x: (:obj:`torch.Tensor`): The first input tensor.
            - y: (:obj:`torch.Tensor`): The second input tensor. \
                x and y should have the same shape and their last dimension should match the input_dim.
        Returns:
            - g: (:obj:`torch.Tensor`): The output of the GRU gating mechanism. \
                The shape of g matches the shapes of x and y.
        """

        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))  # element wise multiplication
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g  # x.shape == y.shape == g.shape


class Memory:
    """
    Overview:
        A class that stores the context used to add memory to Transformer.
    Interfaces:
        ``__init__``, ``init``, ``update``, ``get``, ``to``

    .. note::
        For details, refer to Transformer-XL: https://arxiv.org/abs/1901.02860
    """

    def __init__(
            self,
            memory_len: int = 20,
            batch_size: int = 64,
            embedding_dim: int = 256,
            layer_num: int = 3,
            memory: Optional[torch.Tensor] = None
    ) -> None:
        """
        Overview:
            Initialize the Memory module.
        Arguments:
            - memory_len (:obj:`int`): The dimension of memory, i.e., how many past observations to use as memory.
            - batch_size (:obj:`int`): The dimension of each batch.
            - embedding_dim (:obj:`int`): The dimension of embedding, which is the dimension of a single observation \
                after embedding.
            - layer_num (:obj:`int`): The number of transformer layers.
            - memory (:obj:`Optional[torch.Tensor]`): The initial memory. Default is None.
        """

        super(Memory, self).__init__()
        self.embedding_dim = embedding_dim
        self.bs = batch_size
        self.layer_num = layer_num
        self.memory_len = memory_len
        self.memory = None
        self.init(memory)

    def init(self, memory: Optional[torch.Tensor] = None):
        """
        Overview:
            Initialize memory with an input list of tensors or create it automatically given its dimensions.
        Arguments:
            - memory (:obj:`Optional[torch.Tensor]`): Input memory tensor with shape \
                (layer_num, memory_len, bs, embedding_dim). Its shape is (layer_num, memory_len, bs, embedding_dim), \
                where memory_len is length of memory, bs is batch size and embedding_dim is the dimension of embedding.
        """

        if memory is not None:
            self.memory = memory
            layer_num_plus1, self.memory_len, self.bs, self.embedding_dim = memory.shape
            self.layer_num = layer_num_plus1 - 1
        else:
            self.memory = torch.zeros(
                self.layer_num + 1, self.memory_len, self.bs, self.embedding_dim, dtype=torch.float
            )

    def update(self, hidden_state: List[torch.Tensor]):
        """
        Overview:
            Update the memory given a sequence of hidden states.
            Example for single layer: (memory_len=3, hidden_size_len=2, bs=3)

                    m00 m01 m02      h00 h01 h02              m20 m21 m22
                m = m10 m11 m12  h = h10 h11 h12  => new_m =  h00 h01 h02
                    m20 m21 m22                               h10 h11 h12
        Arguments:
            - hidden_state: (:obj:`List[torch.Tensor]`): The hidden states to update the memory. \
                Each tensor in the list has shape (cur_seq, bs, embedding_dim), where cur_seq \
                is the length of the sequence.
        Returns:
            - memory: (:obj:`Optional[torch.Tensor]`): The updated memory, with shape \
                (layer_num, memory_len, bs, embedding_dim).
        """

        if self.memory is None or hidden_state is None:
            raise ValueError('Failed to update memory! Memory would be None')  # TODO add support of no memory
        sequence_len = hidden_state[0].shape[0]
        with torch.no_grad():
            new_memory = []
            end = self.memory_len + sequence_len
            beg = max(0, end - self.memory_len)
            for i in range(self.layer_num + 1):
                m = self.memory[i]
                h = hidden_state[i]
                cat = torch.cat([m, h], dim=0)
                new_memory.append(cat[beg:end].detach())
        new_memory = torch.stack(new_memory, dim=0)
        self.memory = new_memory
        return new_memory

    def get(self):
        """
        Overview:
            Get the current memory.
        Returns:
            - memory: (:obj:`Optional[torch.Tensor]`): The current memory, \
                with shape (layer_num, memory_len, bs, embedding_dim).
        """

        return self.memory

    def to(self, device: str = 'cpu'):
        """
        Overview:
            Move the current memory to the specified device.
        Arguments:
            device (:obj:`str`): The device to move the memory to. Default is 'cpu'.
        """

        self.memory = self.memory.to(device)


class AttentionXL(torch.nn.Module):
    """
    Overview:
         An implementation of the Attention mechanism used in the TransformerXL model.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, input_dim: int, head_dim: int, head_num: int, dropout: nn.Module) -> None:
        """
        Overview:
            Initialize the AttentionXL module.
        Arguments:
            - input_dim (:obj:`int`): The dimensionality of the input features.
            - head_dim (:obj:`int`): The dimensionality of each attention head.
            - head_num (:obj:`int`): The number of attention heads.
            - dropout (:obj:`nn.Module`): The dropout layer to use
        """

        super(AttentionXL, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_kv = fc_block(input_dim, head_dim * head_num * 2)  # key, value
        self.attention_q = fc_block(input_dim, head_dim * head_num)  # query (not computed with past hidden states)
        self.project = fc_block(head_dim * head_num, input_dim)  # project attention output back to input_dim
        self.project_pos = fc_block(input_dim, head_dim * head_num)  # project the positional embedding
        self.scale = 1 / (head_dim ** 0.5)  # for scaled dot product attention

    def _rel_shift(self, x: torch.Tensor, zero_upper: bool = False) -> torch.Tensor:
        """
        Overview:
            Perform a relative shift operation on the attention score matrix.
            Example:
                a00 a01 a02      0 a00 a01 a02       0  a00 a01      a02  0  a10     a02  0   0
                a10 a11 a12  =>  0 a10 a11 a12  =>  a02  0  a10  =>  a11 a12  0  =>  a11 a12  0
                a20 a21 a22      0 a20 a21 a22      a11 a12  0       a20 a21 a22     a20 a21 a22
                                                    a20 a21 a22
                1) Append one "column" of zeros to the left
                2) Reshape the matrix from [3 x 4] into [4 x 3]
                3) Remove the first "row"
                4) Mask out the upper triangle (optional)

        .. note::
            See the following material for better understanding: https://github.com/kimiyoung/transformer-xl/issues/8 \
            https://arxiv.org/pdf/1901.02860.pdf (Appendix B)
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor with shape (cur_seq, full_seq, bs, head_num).
            - zero_upper (:obj:`bool`): If True, the upper-right triangle of the matrix is set to zero.
        Returns:
            - x (:obj:`torch.Tensor`): The input tensor after the relative shift operation, \
                with shape (cur_seq, full_seq, bs, head_num).
        """

        x_padded = F.pad(x, [1, 0])  # step 1
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))  # step 2
        x = x_padded[:, :, 1:].view_as(x)  # step 3
        if zero_upper:
            ones = torch.ones((x.size(2), x.size(3))).unsqueeze(0).unsqueeze(0)
            x = x * torch.tril(ones.to(x.device), x.size(3) - x.size(2))  # step 4
        return x

    def forward(
            self,
            inputs: torch.Tensor,
            pos_embedding: torch.Tensor,
            full_input: torch.Tensor,
            u: torch.nn.Parameter,
            v: torch.nn.Parameter,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Compute the forward pass for the AttentionXL module.
        Arguments:
            - inputs (:obj:`torch.Tensor`): The attention input with shape (cur_seq, bs, input_dim).
            - pos_embedding (:obj:`torch.Tensor`): The positional embedding with shape (full_seq, 1, full_seq).
            - full_input (:obj:`torch.Tensor`): The concatenated memory and input tensor with shape \
                (full_seq, bs, input_dim).
            - u (:obj:`torch.nn.Parameter`): The content parameter with shape (head_num, head_dim).
            - v (:obj:`torch.nn.Parameter`): The position parameter with shape (head_num, head_dim).
            - mask (:obj:`Optional[torch.Tensor]`): The attention mask with shape (cur_seq, full_seq, 1). \
                If None, no masking is applied.
        Returns:
            - output (:obj:`torch.Tensor`): The output of the attention mechanism with shape (cur_seq, bs, input_dim).
        """

        bs, cur_seq, full_seq = inputs.shape[1], inputs.shape[0], full_input.shape[0]
        prev_seq = full_seq - cur_seq

        kv = self.attention_kv(full_input)
        key, value = torch.chunk(kv, 2, dim=-1)  # full_seq x bs x num_head*dim_head
        query = self.attention_q(inputs)  # cur_seq x bs x num_head*dim_head
        r = self.project_pos(pos_embedding)  # full_seq x 1 x num_head*dim_head

        key = key.view(full_seq, bs, self.head_num, self.head_dim)
        query = query.view(cur_seq, bs, self.head_num, self.head_dim)
        value = value.view(cur_seq + prev_seq, bs, self.head_num, self.head_dim)
        r = r.view(full_seq, self.head_num, self.head_dim)

        # (query + u) * key^T
        q_u = query + u
        content_attn = q_u.permute(1, 2, 0, 3) @ key.permute(1, 2, 3, 0)  # bs x head_num x cur_seq x full_seq

        # (query + v) * R^T
        q_v = query + v
        position_attn = q_v.permute(1, 2, 0, 3) @ r.permute(1, 2, 0)  # bs x head_num x cur_seq x full_seq
        position_attn = self._rel_shift(position_attn)

        attn = content_attn + position_attn  # bs x head_num x cur_seq x full_seq
        attn.mul_(self.scale)

        # fills float('-inf') where mask is True to let softmax ignore those positions.
        if mask is not None and mask.any().item():
            mask = mask.permute(2, 0, 1).unsqueeze(1)  # 1 x 1 x cur_seq x full_seq
            assert mask.shape[2:] == attn.shape[2:]  # check shape of mask
            attn = attn.masked_fill(mask, -float("inf")).type_as(attn)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # multiply softmax output by value
        attn_vec = attn @ value.permute(1, 2, 0, 3)
        attn_vec = attn_vec.permute(2, 0, 1, 3)

        attn_vec = attn_vec.contiguous().view(cur_seq, bs, self.head_num * self.head_dim)
        # cur_seq x bs x head_num * head_dim
        output = self.dropout(self.project(attn_vec))  # cur_seq x bs x input_dim
        return output


class GatedTransformerXLLayer(torch.nn.Module):
    """
    Overview:
        This class implements the attention layer of GTrXL (Gated Transformer-XL).
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self,
            input_dim: int,
            head_dim: int,
            hidden_dim: int,
            head_num: int,
            mlp_num: int,
            dropout: nn.Module,
            activation: nn.Module,
            gru_gating: bool = True,
            gru_bias: float = 2.
    ) -> None:
        """
        Overview:
            Initialize GatedTransformerXLLayer.
        Arguments:
            - input_dim (:obj:`int`): The dimension of the input tensor.
            - head_dim (:obj:`int`): The dimension of each head in the multi-head attention.
            - hidden_dim (:obj:`int`): The dimension of the hidden layer in the MLP.
            - head_num (:obj:`int`): The number of heads for the multi-head attention.
            - mlp_num (:obj:`int`): The number of MLP layers in the attention layer.
            - dropout (:obj:`nn.Module`): The dropout module used in the MLP and attention layers.
            - activation (:obj:`nn.Module`): The activation function to be used in the MLP layers.
            - gru_gating (:obj:`bool`, optional): Whether to use GRU gates. If False, replace GRU gates with \
                residual connections. Default is True.
            - gru_bias (:obj:`float`, optional): The bias of the GRU gate. Default is 2.
        """

        super(GatedTransformerXLLayer, self).__init__()
        self.dropout = dropout
        self.gating = gru_gating
        if self.gating is True:
            self.gate1 = GRUGatingUnit(input_dim, gru_bias)
            self.gate2 = GRUGatingUnit(input_dim, gru_bias)
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
        self.layernorm2 = build_normalization('LN')(input_dim)
        self.activation = activation

    def forward(
            self,
            inputs: torch.Tensor,
            pos_embedding: torch.Tensor,
            u: torch.nn.Parameter,
            v: torch.nn.Parameter,
            memory: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Compute forward pass of GTrXL layer.
        Arguments:
            - inputs (:obj:`torch.Tensor`): The attention input tensor of shape (cur_seq, bs, input_dim).
            - pos_embedding (:obj:`torch.Tensor`): The positional embedding tensor of shape (full_seq, 1, full_seq).
            - u (:obj:`torch.nn.Parameter`): The content parameter tensor of shape (head_num, head_dim).
            - v (:obj:`torch.nn.Parameter`): The position parameter tensor of shape (head_num, head_dim).
            - memory (:obj:`torch.Tensor`): The memory tensor of shape (prev_seq, bs, input_dim).
            - mask (:obj:`Optional[torch.Tensor]`): The attention mask tensor of shape (cur_seq, full_seq, 1).
                Default is None.
        Returns:
            - output (:obj:`torch.Tensor`): layer output of shape (cur_seq, bs, input_dim)
        """

        # concat memory with input across sequence dimension
        full_input = torch.cat([memory, inputs], dim=0)  # full_seq x bs x input_dim
        x1 = self.layernorm1(full_input)
        a1 = self.dropout(self.attention(inputs, pos_embedding, x1, u, v, mask=mask))
        a1 = self.activation(a1)  # RELU after attention
        o1 = self.gate1(inputs, a1) if self.gating else inputs + a1
        x2 = self.layernorm2(o1)
        m2 = self.dropout(self.mlp(x2))
        o2 = self.gate2(o1, m2) if self.gating else o1 + m2
        return o2


class GTrXL(nn.Module):
    """
    Overview:
        GTrXL Transformer implementation as described in "Stabilizing Transformer for Reinforcement Learning"
        (https://arxiv.org/abs/1910.06764).
    Interfaces:
        ``__init__``, ``forward``, ``reset_memory``, ``get_memory``
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
        gru_gating: bool = True,
        gru_bias: float = 2.,
        use_embedding_layer: bool = True,
    ) -> None:
        """Overview:
            Init GTrXL Model.
        Arguments:
            - input_dim (:obj:`int`): The dimension of the input observation.
            - head_dim (:obj:`int`, optional): The dimension of each head. Default is 128.
            - embedding_dim (:obj:`int`, optional): The dimension of the embedding. Default is 256.
            - head_num (:obj:`int`, optional): The number of heads for multi-head attention. Default is 2.
            - mlp_num (:obj:`int`, optional): The number of MLP layers in the attention layer. Default is 2.
            - layer_num (:obj:`int`, optional): The number of transformer layers. Default is 3.
            - memory_len (:obj:`int`, optional): The length of memory. Default is 64.
            - dropout_ratio (:obj:`float`, optional): The dropout ratio. Default is 0.
            - activation (:obj:`nn.Module`, optional): The activation function. Default is nn.ReLU().
            - gru_gating (:obj:`bool`, optional): If False, replace GRU gates with residual connections. \
                Default is True.
            - gru_bias (:obj:`float`, optional): The GRU gate bias. Default is 2.0.
            - use_embedding_layer (:obj:`bool`, optional): If False, don't use input embedding layer. Default is True.
        Raises:
            - AssertionError: If `embedding_dim` is not an even number.
        """

        super(GTrXL, self).__init__()
        assert embedding_dim % 2 == 0, 'embedding_dim={} should be even'.format(input_dim)
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        if isinstance(input_dim, list):
            input_dim = np.prod(input_dim)
        self.use_embedding_layer = use_embedding_layer
        if use_embedding_layer:
            self.embedding = fc_block(input_dim, embedding_dim, activation=activation)
        self.activation = activation
        self.pos_embedding = PositionalEmbedding(embedding_dim)
        # memory to save hidden states of past segments
        # it will be initialized in the forward method to get its size dynamically
        self.memory = None
        self.memory_len = memory_len
        layers = []
        dims = [embedding_dim] + [embedding_dim] * layer_num
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        for i in range(layer_num):
            layers.append(
                GatedTransformerXLLayer(
                    dims[i], head_dim, embedding_dim, head_num, mlp_num, self.dropout, self.activation, gru_gating,
                    gru_bias
                )
            )
        self.layers = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim
        # u and v are the parameters to compute global content bias and global positional bias
        self.u, self.v = (
            torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)),
            torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)),
        )
        self.att_mask = {}  # create an attention mask for each different seq_len, in this way we don't need to create a
        # new one each time we call the forward method
        self.pos_embedding_dict = {}  # create a pos embedding for each different seq_len

    def reset_memory(self, batch_size: Optional[int] = None, state: Optional[torch.Tensor] = None):
        """
        Overview:
            Clear or set the memory of GTrXL.
        Arguments:
            - batch_size (:obj:`Optional[int]`): The batch size. Default is None.
            - state (:obj:`Optional[torch.Tensor]`): The input memory with shape \
                (layer_num, memory_len, bs, embedding_dim). Default is None.
        """

        self.memory = Memory(memory_len=self.memory_len, layer_num=self.layer_num, embedding_dim=self.embedding_dim)
        if batch_size is not None:
            self.memory = Memory(self.memory_len, batch_size, self.embedding_dim, self.layer_num)
        elif state is not None:
            self.memory.init(state)

    def get_memory(self):
        """
        Overview:
            Returns the memory of GTrXL.
        Returns:
            - memory (:obj:`Optional[torch.Tensor]`): The output memory or None if memory has not been initialized. \
                The shape is (layer_num, memory_len, bs, embedding_dim).
        """

        if self.memory is None:
            return None
        else:
            return self.memory.get()

    def forward(self, x: torch.Tensor, batch_first: bool = False, return_mem: bool = True) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Performs a forward pass on the GTrXL.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor with shape (seq_len, bs, input_size).
            - batch_first (:obj:`bool`, optional): If the input data has shape (bs, seq_len, input_size), \
                set this parameter to True to transpose along the first and second dimension and obtain shape \
                (seq_len, bs, input_size). This does not affect the output memory. Default is False. \
            - return_mem (:obj:`bool`, optional): If False, return only the output tensor without dict. Default is True.
        Returns:
            - x (:obj:`Dict[str, torch.Tensor]`): A dictionary containing the transformer output of shape \
             (seq_len, bs, embedding_size) and memory of shape (layer_num, seq_len, bs, embedding_size).
        """

        if batch_first:
            x = torch.transpose(x, 1, 0)  # bs x cur_seq x input_dim -> cur_seq x bs x input_dim
        cur_seq, bs = x.shape[:2]
        memory = None if self.memory is None else self.memory.get()
        if memory is None:
            self.reset_memory(bs)  # (layer_num+1) x memory_len x batch_size x embedding_dim
        elif memory.shape[-2] != bs or memory.shape[-1] != self.embedding_dim:
            warnings.warn(
                "Memory {} and Input {} dimensions don't match,"
                " this will cause the memory to be initialized to fit your input!".format(
                    list(memory.shape[-2:]), [x.shape[-2]] + [self.embedding_dim]
                )
            )
            self.reset_memory(bs)
        self.memory.to(x.device)
        memory = self.memory.get()

        if self.use_embedding_layer:
            x = self.dropout(self.embedding(x))
        prev_seq = self.memory_len
        full_seq = cur_seq + prev_seq

        if cur_seq in self.att_mask.keys():
            attn_mask = self.att_mask[cur_seq]
        else:
            attn_mask = (
                torch.triu(
                    torch.ones((cur_seq, full_seq)),
                    diagonal=1 + prev_seq,  # fixed in train, eval, collect
                ).bool().unsqueeze(-1).to(x.device)
            )  # cur_seq x full_seq x 1
            self.att_mask[cur_seq] = attn_mask

        if cur_seq in self.pos_embedding_dict.keys():
            pos_embedding = self.pos_embedding_dict[cur_seq]
        else:
            pos_ips = torch.arange(full_seq - 1, -1, -1.0, dtype=torch.float)  # full_seq
            pos_embedding = self.pos_embedding(pos_ips.to(x.device))
            self.pos_embedding_dict[cur_seq] = pos_embedding
        pos_embedding = self.dropout(pos_embedding)  # full_seq x 1 x embedding_dim

        hidden_state = [x]
        out = x
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(
                out,
                pos_embedding,
                self.u,
                self.v,
                mask=attn_mask,
                memory=memory[i],  # (layer_num+1) x memory_len x batch_size x embedding_dim
            )  # cur_seq x bs x embedding_dim
            hidden_state.append(out.clone())

        out = self.dropout(out)
        self.memory.update(hidden_state)  # (layer_num+1) x memory_len x batch_size x embedding_dim

        if batch_first:
            out = torch.transpose(out, 1, 0)  # cur_seq x bs x embedding_dim -> bs x cur_seq x embedding_dim
        if return_mem:
            output = {"logit": out, "memory": memory}  # return the content of the memory before the last update
        else:
            output = {"logit": out}
        return output
