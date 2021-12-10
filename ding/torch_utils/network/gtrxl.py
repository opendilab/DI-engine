from typing import Union, Optional, Dict, Callable, List
from ding.torch_utils.network.nn_module import *
from ding.utils import MODEL_REGISTRY


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
             usually a 1D integer sequence as [seq_len-1, seq_len-2, ..., 1, 0],
        Returns:
            - pos_embedding: (:obj:`torch.Tensor`): positional embedding. Shape (seq_len, 1, embedding_dim)
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
        self.bg = nn.Parameter(torch.zeros(input_dim).fill_(bg))  # bias
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        """
        Overview:
            Compute output value with gating mechanism
        Arguments:
            - x: (:obj:`torch.Tensor`): first input.
            - y: (:obj:`torch.Tensor`): second input.
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
        Stores the context used to add memory to Transformer.
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
        """
        Arguments:
            - memory_len (:obj:`int`): dimension of memory (how many past observations to use as memory)
            - batch_size (:obj:`int`): dimension of each batch
            - embedding_dim (:obj:`int`): dimension of embedding (dimension of a single observation after embedding)
            - layer_num (:obj:`int`): number of transformer layers
        """
        super(Memory, self).__init__()
        self.embedding_dim = embedding_dim
        self.bs = batch_size
        self.layer_num = layer_num
        self.memory_len = memory_len
        self.memory = None
        self.init()

    def init(self, memory: Optional[List[torch.Tensor]] = None):
        """
        Overview:
            Init memory with an input list of tensors or create it automatically given its dimensions.
        Arguments:
            - memory: (:obj:`Optional[List[torch.Tensor]]`): memory input.
            Shape is (memory_len, bs, embedding_dim) for each layer.
            memory_len is length of memory, bs is batch size and embedding_dim is the dimension of embedding.
        """
        if memory:
            self.memory = memory
        else:
            self.memory = [
                torch.zeros(self.memory_len, self.bs, self.embedding_dim, dtype=torch.float)
                for _ in range(self.layer_num + 1)
            ]

    def update(self, hidden_state: List[torch.Tensor]):
        """
        Overview:
            Update the memory given a sequence of hidden states.
        Example for single layer:

            memory_len=3, hidden_size_len=2, bs=3

                m00 m01 m02      h00 h01 h02              m20 m21 m22
            m = m10 m11 m12  h = h10 h11 h12  => new_m =  h00 h01 h02
                m20 m21 m22                               h10 h11 h12
        Arguments:
            - hidden_state: (:obj:`List[torch.Tensor]`): hidden states to update the memory.
            Shape is (cur_seq, bs, embedding_dim) for each layer. cur_seq is length of sequence.
        Returns:
            - memory: (:obj:`Optional[List[torch.Tensor]]`): output memory.
            Shape is (memory_len, bs, embedding_dim) for each layer.
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
        """
        Overview:
            Memory getter method.
        Returns:
            - memory: (:obj:`Optional[List[torch.Tensor]]`): output memory.
            Shape is (memory_len, bs, embedding_dim) for each layer.
        """
        return self.memory


class AttentionXL(torch.nn.Module):
    """
    Overview:
        Attention of TransformerXL.
    """
    def __init__(self, input_dim: int, head_dim: int, head_num: int, dropout: nn.Module) -> None:
        """Overview:
            Init AttentionXL.
        Arguments:
            - input_dim (:obj:`int`): dimension of input
            - head_dim (:obj:`int`): dimension of each head
            - head_num (:obj:`int`): number of heads for multihead attention
            - dropout (:obj:`nn.Module`): dropout function
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

    def _rel_shift(self, x: torch.Tensor):
        """
        Overview:
            Relatively shift the attention score matrix.
        Example:
            a00 a01 a02      0 a00 a01 a02       0  a00 a01      a02  0  a10     a02  0   0
            a10 a11 a12  =>  0 a10 a11 a12  =>  a02  0  a10  =>  a11 a12  0  =>  a11 a12  0
            a20 a21 a22      0 a20 a21 a22      a11 a12  0       a20 a21 a22     a20 a21 a22
                                                a20 a21 a22
            1) Append one "column" of zeros to the left
            2) Reshape the matrix from [3 x 4] into [4 x 3]
            3) Remove the first "row"
            4) Mask out the upper triangle
        .. note::
            See the following material for better understanding:
                https://github.com/kimiyoung/transformer-xl/issues/8
                https://arxiv.org/pdf/1901.02860.pdf (Appendix B)
        Arguments:
            - x (:obj:`torch.Tensor`): input tensor of shape (cur_seq, full_seq, bs, head_num).
        Returns:
            - x (:obj:`torch.Tensor`): input after relative shift. Shape (cur_seq, full_seq, bs, head_num).
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
        """Overview:
            Compute AttentionXL.
        Arguments:
            - inputs (:obj:`torch.Tensor`): attention input of shape (cur_seq, bs, input_dim)
            - pos_embedding (:obj:`torch.Tensor`): positional embedding of shape (full_seq, 1, full_seq)
            - full_input (:obj:`torch.Tensor`): memory + input concatenation of shape (full_seq, bs, input_dim)
            - u (:obj:`torch.nn.Parameter`): content parameter of shape (head_num, head_dim)
            - v (:obj:`torch.nn.Parameter`): position parameter of shape (head_num, head_dim)
            - mask (:obj:`Optional[torch.Tensor]`): attention mask of shape (cur_seq, full_seq, 1)
            full_seq = prev_seq + cur_seq
        Returns:
            - output (:obj:`torch.Tensor`): attention output of shape (cur_seq, bs, input_dim)
        """
        bs, cur_seq, full_seq = inputs.shape[1], inputs.shape[0], full_input.shape[0]
        prev_seq = full_seq - cur_seq

        kv = self.attention_kv(full_input)
        key, value = torch.chunk(kv, 2, dim=-1)  # full_seq x bs x num_head*dim_head
        query = self.attention_q(inputs)  # cur_seq x bs x num_head*dim_head
        r = self.project_pos(pos_embedding)  # full_seq x 1 x num_head*dim_head

        # (query + u) * key^T
        content_attn = torch.einsum(
            "ibhd,jbhd->ijbh",
            (
                (query.view(cur_seq, bs, self.head_num, self.head_dim) + u),
                key.view(full_seq, bs, self.head_num, self.head_dim),
            ),
        )  # cur_seq x full_seq x bs x head_num

        # (query + v) * R^T
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
            # fills float('-inf') where mask is True to let softmax ignore those positions.
            attn = attn.masked_fill(mask[..., None], -float("inf")).type_as(attn)
        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)

        # multiply softmax output by value
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
    """
    Overview:
        Attention layer of GTrXL
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
        gating: bool = True,
    ) -> None:
        """
        Arguments:
            - input_dim (:obj:`int`): dimension of input
            - head_dim (:obj:`int`): dimension of each head
            - hidden_dim (:obj:`int`): dimension of hidden layer in mlp
            - head_num (:obj:`int`): number of heads for multihead attention
            - mlp_num (:obj:`int`): number of mlp layers in attention layer
            - dropout (:obj:`nn.Module`): dropout
            - activation (:obj:`nn.Module`): activation function
            - gating (:obj:`bool`): whether to use gating mechanism or not
        """
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

    def forward(
        self,
        inputs: torch.Tensor,
        pos_embedding: torch.Tensor,
        u: torch.nn.Parameter,
        v: torch.nn.Parameter,
        memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Overview:
            Compute forward pass of GTrXL layer.
        Arguments:
            - inputs (:obj:`torch.Tensor`): attention input of shape (cur_seq, bs, input_dim)
            - pos_embedding (:obj:`torch.Tensor`): positional embedding of shape (full_seq, 1, full_seq)
            - u (:obj:`torch.nn.Parameter`): content parameter of shape (head_num, head_dim)
            - v (:obj:`torch.nn.Parameter`): position parameter of shape (head_num, head_dim)
            - memory (:obj:`Optional[torch.Tensor]`): memory of shape (prev_seq, bs, input_dim)
            - mask (:obj:`Optional[torch.Tensor]`): attention mask of shape (cur_seq, full_seq, 1)
            full_seq = prev_seq + cur_seq
        Returns:
            - output (:obj:`torch.Tensor`): layer output of shape (cur_seq, bs, input_dim)
        """
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


class GTrXL(nn.Module):
    """
    Overview:
        GTrXL Transformer
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

    def reset(self):
        r"""
        Overview:
            Clear the memory of GTrXL
        """
        self.memory = None

    def forward(self, x: torch.Tensor, batch_first: bool = False, return_mem: bool = True) -> Dict[str, torch.Tensor]:
        r"""
        Overview:
            GTrXL forward pass.
        Arguments:
            - x (:obj:`torch.Tensor`): input tensor. Shape (seq_len, bs, input_size).
            - batch_first (:obj:`bool`): if the input data has shape (bs, seq_len, input_size), set this param to 'True'
            in order to transpose along the first and second dimension and obtain shape (seq_len, bs, input_size). This
            param doesn't affects the output memory
            - return_mem (:obj:`bool`): if this param is False, return only the output tensor without dict.
        Returns:
            - x (:obj:`Dict[str, torch.Tensor]`): dict containing transformer output of shape
             (seq_len, bs, embedding_size) and memory of shape (seq_len, bs, embedding_size)
        """
        if batch_first:
            x = torch.transpose(x, 1, 0)  # bs x cur_seq x input_dim -> cur_seq x bs x input_dim

        cur_seq, bs = x.shape[:2]
        memory = None if self.memory is None else self.memory.get()
        if memory is None:
            self.memory = Memory(self.memory_len, bs, self.embedding_dim, self.layer_num + 1)
            # (layer_num+1) x memory_len x batch_size x embedding_dim
            memory = self.memory.get()
            memory = [mem.to(x.device) for mem in memory]

        x = self.dropout(self.embedding(x))
        prev_seq = memory[0].size(0)
        full_seq = cur_seq + prev_seq

        # TODO: add padding to attention mask, https://huggingface.co/docs/transformers/preprocessing
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

        out = self.dropout(out)
        memory = self.memory.update(hidden_state)

        if batch_first:
            out = torch.transpose(out, 1, 0)  # cur_seq x bs x embedding_dim -> bs x cur_seq x embedding_dim

        if return_mem:
            output = {"logit": out, "memory": memory}
        else:
            output = {"logit": out}
        return output


if __name__ == "__main__":
    dim_size = 128
    seq_len = 64
    bs = 32
    action_dim = 4
    embedding_dim = 256
    # input shape: cur_seq x bs x input_dim
    a = torch.rand(bs, seq_len, dim_size)
    print('input:', a.shape)
    m = GTrXL(128, memory_len=50, embedding_dim=embedding_dim)
    o = m(a)
    print('output', o['logit'].shape)
    #print('memory', mem[0].shape)