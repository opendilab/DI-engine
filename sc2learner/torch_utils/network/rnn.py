import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalization import build_normalization


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

    def _after_forward(self, next_state, list_next_state=False):
        if list_next_state:
            h, c = [torch.stack(t, dim=0) for t in zip(*next_state)]
            batch_size = h.shape[1]
            next_state = [torch.chunk(h, batch_size, dim=1), torch.chunk(c, batch_size, dim=1)]
            next_state = list(zip(*next_state))
        else:
            next_state = [torch.stack(t, dim=0) for t in zip(*next_state)]
        return next_state


class LSTM(nn.Module, LSTMForwardWrapper):
    def __init__(self, input_size, hidden_size, num_layers, norm_type=None, bias=True, dropout=0.):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        norm_func = build_normalization(norm_type)
        self.norm = nn.ModuleList([norm_func(hidden_size) for _ in range(4 * num_layers)])
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

    def forward(self, inputs, prev_state, list_next_state=False):
        '''
        Input:
            inputs: tensor of size [seq_len, batch_size, input_size]
            prev_state: None or tensor of size [num_directions*num_layers, batch_size, hidden_size]
            list_next_state: whether return next_state with list format
        '''
        seq_len, batch_size = inputs.shape[:2]
        prev_state = self._before_forward(inputs, prev_state)

        H, C = prev_state
        x = inputs
        next_state = []
        for l in range(self.num_layers):
            h, c = H[l], C[l]
            if self.use_dropout:  # layer input dropout
                x = self.dropout(x)
            new_x = []
            for s in range(seq_len):
                gate = torch.matmul(x[s], self.wx[l]) + torch.matmul(h, self.wh[l])
                if self.bias is not None:
                    gate += self.bias[l]
                gate = list(torch.chunk(gate, 4, dim=1))
                for i in range(4):
                    gate[i] = self.norm[l * 4 + i](gate[i])
                i, f, o, u = gate
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                o = torch.sigmoid(o)
                u = torch.tanh(u)
                c = f * c + i * u
                h = o * torch.tanh(c)
                new_x.append(h)
            next_state.append((h, c))
            x = torch.stack(new_x, dim=0)

        next_state = self._after_forward(next_state)
        return x, next_state


class PytorchLSTM(nn.LSTM, LSTMForwardWrapper):
    def forward(self, inputs, prev_state, list_next_state=False):
        prev_state = self._before_forward(inputs, prev_state)
        output, next_state = nn.LSTM.forward(self, inputs, prev_state)
        next_state = self._after_forward(next_state, list_next_state)
        return output, next_state

    def _after_forward(self, next_state, list_next_state=False):
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
