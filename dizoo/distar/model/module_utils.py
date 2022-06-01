import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import kaiming_normal_, kaiming_uniform_
from distar.ctools.torch_utils import conv2d_block, sequence_mask, fc_block, build_normalization
from typing import Optional, Dict, Tuple
from torch import Tensor


def scatter_connection(shape, project_embeddings, entity_location, scatter_dim, scatter_type='add'):
    B, H, W = shape
    device = entity_location.device
    entity_num = entity_location.shape[1]
    index = entity_location.view(-1, 2).long()
    bias = torch.arange(B).unsqueeze(1).repeat(1, entity_num).view(-1).to(device)
    bias *= H * W
    index[:, 0].clamp_(0, W - 1)
    index[:, 1].clamp_(0, H - 1)
    index = index[:, 1] * W + index[:, 0]  # entity_location: (x, y), spatial_info: (y, x)
    index += bias
    index = index.repeat(scatter_dim, 1)
    # flat scatter map and project embeddings
    scatter_map = torch.zeros(scatter_dim, B * H * W, device=device)
    project_embeddings = project_embeddings.view(-1, scatter_dim).permute(1, 0)
    if scatter_type == 'cover':
        scatter_map.scatter_(dim=1, index=index, src=project_embeddings)
    elif scatter_type == 'add':
        scatter_map.scatter_add_(dim=1, index=index, src=project_embeddings)
    else:
        raise NotImplementedError
    scatter_map = scatter_map.reshape(scatter_dim, B, H, W)
    scatter_map = scatter_map.permute(1, 0, 2, 3)
    return scatter_map


class AttentionPool(nn.Module):

    def __init__(self, key_dim, head_num, output_dim, max_num=None):
        super(AttentionPool, self).__init__()
        self.queries = torch.nn.Parameter(torch.zeros(1, 1, head_num, key_dim))
        torch.nn.init.xavier_uniform_(self.queries)
        self.head_num = head_num
        self.add_num = False
        if max_num is not None:
            self.add_num = True
            self.num_ebed = torch.nn.Embedding(num_embeddings=max_num, embedding_dim=output_dim)
        self.embed_fc = fc_block(key_dim * self.head_num, output_dim)

    def forward(self, x, num=None, mask=None):
        assert len(x.shape) == 3  # batch size, tokens, channels
        x_with_head = x.unsqueeze(dim=2)  # add head dim
        score = x_with_head * self.queries
        score = score.sum(dim=3)  # b, t, h
        if mask is not None:
            assert len(mask.shape) == 3 and mask.shape[-1] == 1
            mask = mask.repeat(1, 1, self.head_num)
            score.masked_fill_(~mask.bool(), value=-1e9)
        score = F.softmax(score, dim=1)
        x = x.unsqueeze(dim=3).repeat(1, 1, 1, self.head_num)  # b, t, c, h
        score = score.unsqueeze(dim=2)  # b, t, 1, h
        x = x * score
        x = x.sum(dim=1)  # b, c, h
        x = x.view(x.shape[0], -1)  # b, c * h
        x = self.embed_fc(x)  # b, c
        if self.add_num:
            x = x + F.relu(self.num_ebed(num.long()))
        x = F.relu(x)
        return x


class Attention(nn.Module):

    def __init__(self, input_dim, head_dim, output_dim, head_num, dropout):
        super(Attention, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_pre = fc_block(input_dim, head_dim * head_num * 3)  # query, key, value
        self.project = fc_block(head_dim * head_num, output_dim)

    def split(self, x, T: bool = False):
        B, N = x.shape[:2]
        x = x.view(B, N, self.head_num, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # B, head_num, N, head_dim
        if T:
            x = x.permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self, x, mask: Optional[torch.Tensor] = None):
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
        if mask is not None:
            score.masked_fill_(~mask, value=-1e9)

        score = F.softmax(score, dim=-1)
        if self.dropout:
            score = self.dropout(score)
        attention = torch.matmul(score, value)  # B, head_num, N, head_dim

        attention = attention.permute(0, 2, 1, 3).contiguous()  # B, N, head_num, head_dim
        attention = self.project(attention.view(B, N, -1))  # B, N, output_dim
        return attention


class TransformerLayer(nn.Module):

    def __init__(self, input_dim, head_dim, hidden_dim, output_dim, head_num, mlp_num, dropout, activation, ln_type):
        super(TransformerLayer, self).__init__()
        self.attention = Attention(input_dim, head_dim, output_dim, head_num, dropout)
        self.layernorm1 = build_normalization('LN')(output_dim)
        self.dropout = dropout
        layers = []
        dims = [output_dim] + [hidden_dim] * (mlp_num - 1) + [output_dim]
        for i in range(mlp_num):
            layers.append(fc_block(dims[i], dims[i + 1], activation=activation))
        if self.dropout:
            layers.append(self.dropout)
        self.mlp = nn.Sequential(*layers)
        self.layernorm2 = build_normalization('LN')(output_dim)
        self.ln_type = ln_type

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        if self.ln_type == 'post':
            a = self.attention(x, mask)
            if self.dropout:
                a = self.dropout(a)
            x = self.layernorm1(x + a)
            m = self.mlp(x)
            if self.dropout:
                m = self.dropout(m)
            x = self.layernorm2(x + m)
        elif self.ln_type == 'pre':
            a = self.attention(self.layernorm1(x), mask)
            if self.dropout:
                a = self.dropout(a)
            x = x + a
            m = self.mlp(self.layernorm2(x))
            if self.dropout:
                m = self.dropout(m)
            x = x + m
        else:
            raise NotImplementedError(self.ln_type)
        return x, mask


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
        dropout_ratio=0.0,
        activation=nn.ReLU(),
        ln_type='pre'
    ):
        super(Transformer, self).__init__()
        self.embedding = fc_block(input_dim, output_dim, activation=activation)
        self.pad_val = pad_val
        self.act = activation
        layers = []
        dims = [output_dim] + [output_dim] * layer_num
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None
        for i in range(layer_num):
            layers.append(
                TransformerLayer(
                    dims[i], head_dim, hidden_dim, dims[i + 1], head_num, mlp_num, self.dropout, self.act, ln_type
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            mask = mask.unsqueeze(dim=1).repeat(1, mask.shape[1], 1).unsqueeze(dim=1)
        x = self.embedding(x)
        if self.dropout:
            x = self.dropout(x)
        for m in self.layers:
            x, mask = m(x, mask)
        return x


class GatedResBlock(nn.Module):
    '''
    Gated Residual Block with conv2d_block by songgl at 2020.10.23
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.ReLU(), norm_type='BN'):
        super(GatedResBlock, self).__init__()
        assert (stride == 1)
        assert (in_channels == out_channels)
        self.act = activation
        self.conv1 = conv2d_block(in_channels, out_channels, 3, 1, 1, activation=self.act, norm_type=norm_type)
        self.conv2 = conv2d_block(out_channels, out_channels, 3, 1, 1, activation=None, norm_type=norm_type)
        self.GateWeightG = nn.Sequential(
            conv2d_block(out_channels, out_channels, 1, 1, 0, activation=self.act, norm_type=None),
            conv2d_block(out_channels, out_channels, 1, 1, 0, activation=self.act, norm_type=None),
            conv2d_block(out_channels, out_channels, 1, 1, 0, activation=self.act, norm_type=None),
            conv2d_block(out_channels, out_channels, 1, 1, 0, activation=None, norm_type=None)
        )
        self.UpdateSP = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.UpdateSP, 0.1)

    def forward(self, x, NoiseMap):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.tanh(x * torch.sigmoid(self.GateWeightG(NoiseMap))) * self.UpdateSP
        x = self.act(x + residual)
        #x = x + residual
        return x


class FiLM(nn.Module):
    """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """

    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas


class FiLMedResBlock(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim=None,
        with_residual=True,
        with_batchnorm=False,
        with_cond=[False],
        dropout=0,
        num_extra_channels=0,
        extra_channel_freq=1,
        with_input_proj=3,
        num_cond_maps=0,
        kernel_size=3,
        batchnorm_affine=False,
        num_layers=1,
        condition_method='conv-film',
        debug_every=float('inf')
    ):
        if out_dim is None:
            out_dim = in_dim
        super(FiLMedResBlock, self).__init__()
        self.with_residual = with_residual
        self.with_batchnorm = with_batchnorm
        self.with_cond = with_cond
        self.dropout = dropout
        self.extra_channel_freq = 0 if num_extra_channels == 0 else extra_channel_freq
        self.with_input_proj = with_input_proj  # Kernel size of input projection
        self.num_cond_maps = num_cond_maps
        self.kernel_size = kernel_size
        self.batchnorm_affine = batchnorm_affine
        self.num_layers = num_layers
        self.condition_method = condition_method
        self.debug_every = debug_every
        self.film = None
        self.bn1 = None
        self.drop = None

        if self.with_input_proj % 2 == 0:
            raise (NotImplementedError)
        if self.kernel_size % 2 == 0:
            raise (NotImplementedError)
        if self.num_layers >= 2:
            raise (NotImplementedError)

        if self.condition_method == 'block-input-film' and self.with_cond[0]:
            self.film = FiLM()
        if self.with_input_proj:
            self.input_proj = nn.Conv2d(
                in_dim + (num_extra_channels if self.extra_channel_freq >= 1 else 0),
                in_dim,
                kernel_size=self.with_input_proj,
                padding=self.with_input_proj // 2
            )

        self.conv1 = nn.Conv2d(
            in_dim + self.num_cond_maps + (num_extra_channels if self.extra_channel_freq >= 2 else 0),
            out_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2
        )
        if self.condition_method == 'conv-film' and self.with_cond[0]:
            self.film = FiLM()
        if self.with_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_dim, affine=((not self.with_cond[0]) or self.batchnorm_affine))
        if self.condition_method == 'bn-film' and self.with_cond[0]:
            self.film = FiLM()
        if dropout > 0:
            self.drop = nn.Dropout2d(p=self.dropout)
        if ((self.condition_method == 'relu-film' or self.condition_method == 'block-output-film')
                and self.with_cond[0]):
            self.film = FiLM()

        self.init_modules(self.modules())

    def init_modules(self, modules, init='normal'):
        if init.lower() == 'normal':
            init_params = kaiming_normal_
        elif init.lower() == 'uniform':
            init_params = kaiming_uniform_
        else:
            return
        for m in modules:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_params(m.weight)

    def forward(
        self,
        x,
        gammas: Optional[torch.Tensor] = None,
        betas: Optional[torch.Tensor] = None,
        extra_channels: Optional[torch.Tensor] = None,
        cond_maps: Optional[torch.Tensor] = None
    ):

        if self.film is not None:
            if self.condition_method == 'block-input-film' and self.with_cond[0]:
                x = self.film(x, gammas, betas)

        # ResBlock input projection
        if self.with_input_proj:
            if extra_channels is not None and self.extra_channel_freq >= 1:
                x = torch.cat([x, extra_channels], 1)
            x = F.relu(self.input_proj(x))
        out = x

        # ResBlock body
        if cond_maps is not None:
            out = torch.cat([out, cond_maps], 1)
        if extra_channels is not None and self.extra_channel_freq >= 2:
            out = torch.cat([out, extra_channels], 1)
        out = self.conv1(out)
        if self.film is not None:
            if self.condition_method == 'conv-film' and self.with_cond[0]:
                out = self.film(out, gammas, betas)
        if self.bn1 is not None:
            if self.with_batchnorm:
                out = self.bn1(out)
        if self.film is not None:
            if self.condition_method == 'bn-film' and self.with_cond[0]:
                out = self.film(out, gammas, betas)
        if self.drop is not None:
            if self.dropout > 0:
                out = self.drop(out)
        # out = F.relu(out)
        if self.film is not None:
            if self.condition_method == 'relu-film' and self.with_cond[0]:
                out = self.film(out, gammas, betas)

        # ResBlock remainder
        if self.with_residual:
            out = F.relu(x + out)
        if self.film is not None:
            if self.condition_method == 'block-output-film' and self.with_cond[0]:
                out = self.film(out, gammas, betas)
        return out


class LSTMForwardWrapper(object):

    def _before_forward(self, inputs, prev_state):
        assert hasattr(self, 'num_layers')
        assert hasattr(self, 'hidden_size')
        seq_len, batch_size = inputs.shape[:2]
        if prev_state is None:
            num_directions = 1
            zeros = torch.zeros(
                num_directions * self.num_layers,
                batch_size,
                self.hidden_size,
                dtype=inputs.dtype,
                device=inputs.device
            )
            prev_state = (zeros, zeros)
        elif isinstance(prev_state, list) and len(prev_state) == 2 and isinstance(prev_state[0], torch.Tensor):
            pass
        elif isinstance(prev_state, list) and len(prev_state) == batch_size:
            num_directions = 1
            zeros = torch.zeros(
                num_directions * self.num_layers, 1, self.hidden_size, dtype=inputs.dtype, device=inputs.device
            )
            state = []
            for prev in prev_state:
                if prev is None:
                    state.append([zeros, zeros])
                else:
                    state.append(prev)
            state = list(zip(*state))
            prev_state = [torch.cat(t, dim=1) for t in state]
        return prev_state

    def _after_forward(self, next_state, list_next_state: bool = False):
        if list_next_state:
            h, c = [torch.stack(t, dim=0) for t in zip(*next_state)]
            batch_size = h.shape[1]
            next_state = [torch.chunk(h, batch_size, dim=1), torch.chunk(c, batch_size, dim=1)]
            next_state = list(zip(*next_state))
        else:
            next_state = [torch.stack(t, dim=0) for t in zip(*next_state)]
        return next_state


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, norm_type=None, bias=True, dropout=0.):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        norm_func = build_normalization(norm_type)
        #self.norm = nn.ModuleList([norm_func(hidden_size) for _ in range(4 * num_layers)])
        self.norm_A = nn.ModuleList([norm_func(hidden_size * 4) for _ in range(2 * num_layers)])
        self.norm_B = nn.ModuleList([norm_func(hidden_size) for _ in range(1 * num_layers)])
        self.wx = nn.ParameterList()
        self.wh = nn.ParameterList()
        dims = [input_size] + [hidden_size] * 3
        for l in range(num_layers):
            self.wx.append(nn.Parameter(torch.zeros(dims[l], dims[l + 1] * 4)))
            self.wh.append(nn.Parameter(torch.zeros(hidden_size, hidden_size * 4)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_layers, hidden_size * 4))
        else:
            self.bias = None
        self.use_dropout = dropout > 0.
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)
        self._init()

    def _init(self):
        gain = math.sqrt(1. / self.hidden_size)
        for l in range(self.num_layers):
            torch.nn.init.uniform_(self.wx[l], -gain, gain)
            torch.nn.init.uniform_(self.wh[l], -gain, gain)
            if self.bias is not None:
                torch.nn.init.uniform_(self.bias[l], -gain, gain)

        # for i in range(len(self.norm_A)):
        #     torch.nn.init.constant_(self.norm_A[i].weight, 0)
        #     torch.nn.init.constant_(self.norm_A[i].bias, 1)
        # for i in range(len(self.norm_B)):
        #     torch.nn.init.constant_(self.norm_A[i].weight, 0)
        #     torch.nn.init.constant_(self.norm_A[i].bias, 1)

    def forward(
        self, inputs, prev_state: Tuple[Tensor, Tensor], list_next_state: bool = False, forget_bias: float = 1.0
    ):
        '''
        Input:
            inputs: tensor of size [seq_len, batch_size, input_size]
            prev_state: None or tensor of size [num_directions*num_layers, batch_size, hidden_size]
            list_next_state: whether return next_state with list format
        '''
        seq_len, batch_size = inputs.shape[:2]

        H, C = prev_state
        x = inputs
        next_state = []
        for l in range(self.num_layers):
            h, c = H[l], C[l]
            new_x = []
            for s in range(seq_len):
                if self.use_dropout:
                    gate = self.norm_A[l * 2](torch.matmul(self.dropout(x[s]), self.wx[l])
                                              ) + self.norm_A[l * 2 + 1](torch.matmul(h, self.wh[l]))
                else:
                    gate = self.norm_A[l * 2](torch.matmul(x[s], self.wx[l])
                                              ) + self.norm_A[l * 2 + 1](torch.matmul(h, self.wh[l]))
                if self.bias is not None:
                    gate += self.bias[l]
                gate = list(torch.chunk(gate, 4, dim=1))
                # for i in range(4):
                #     gate[i] = self.norm[l * 4 + i](gate[i])
                i, f, o, u = gate
                i = torch.sigmoid(i)
                f = torch.sigmoid(f + forget_bias)
                o = torch.sigmoid(o)
                u = torch.tanh(u)
                c = f * c + i * u
                cc = self.norm_B[l](c)
                h = o * torch.tanh(cc)
                # if self.use_dropout and l != self.num_layers - 1:  # layer input dropout
                #     h = self.dropout(h)
                new_x.append(h)
            next_state.append((h, c))
            x = torch.stack(new_x, dim=0)
        return x, next_state


class PytorchLSTM(nn.LSTM, LSTMForwardWrapper):

    def forward(self, inputs, prev_state, list_next_state: bool = False):
        prev_state = self._before_forward(inputs, prev_state)
        output, next_state = nn.LSTM.forward(self, inputs, prev_state)
        next_state = self._after_forward(next_state, list_next_state)
        return output, next_state

    def _after_forward(self, next_state, list_next_state: bool = False):
        if list_next_state:
            h, c = next_state
            batch_size = h.shape[1]
            next_state = [torch.chunk(h, batch_size, dim=1), torch.chunk(c, batch_size, dim=1)]
            return list(zip(*next_state))
        else:
            return next_state


def get_lstm(lstm_type, input_size, hidden_size, num_layers, norm_type, dropout=0.):
    assert lstm_type in ['normal', 'pytorch']
    if lstm_type == 'normal':
        return LSTM(input_size, hidden_size, num_layers, norm_type, dropout=dropout)
    elif lstm_type == 'pytorch':
        return PytorchLSTM(input_size, hidden_size, num_layers, dropout=dropout)


class GLU(nn.Module):

    def __init__(self, input_dim, output_dim, context_dim, input_type='fc'):
        super(GLU, self).__init__()
        assert (input_type in ['fc', 'conv2d'])
        if input_type == 'fc':
            self.layer1 = fc_block(context_dim, input_dim)
            self.layer2 = fc_block(input_dim, output_dim)
        elif input_type == 'conv2d':
            self.layer1 = conv2d_block(context_dim, input_dim, 1, 1, 0)
            self.layer2 = conv2d_block(input_dim, output_dim, 1, 1, 0)

    def forward(self, x, context):
        gate = self.layer1(context)
        gate = torch.sigmoid(gate)
        x = gate * x
        x = self.layer2(x)
        return x


def build_activation(activation):
    act_func = {'relu': nn.ReLU(inplace=True), 'glu': GLU, 'prelu': nn.PReLU(init=0.0)}
    if activation in act_func.keys():
        return act_func[activation]
    else:
        raise KeyError("invalid key for activation: {}".format(activation))
