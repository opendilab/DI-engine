import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalization import build_normalization


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, norm_type=None, bias=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        norm_func = build_normalization(norm_type)
        self.norm = nn.ModuleList([norm_func(hidden_size) for _ in range(4*num_layers)])
        self.wx = nn.ParameterList()
        self.wh = nn.ParameterList()
        dims = [input_size] + [hidden_size] * 3
        for l in range(num_layers):
            self.wx.append(nn.Parameter(torch.zeros(dims[l], dims[l+1]*4)))
            self.wh.append(nn.Parameter(torch.zeros(hidden_size, hidden_size*4)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_layers, hidden_size*4))
        else:
            self.bias = None
        self._init()

    def _init(self):
        gain = math.sqrt(1./self.hidden_size)
        for l in range(self.num_layers):
            torch.nn.init.uniform_(self.wx[l], -gain, gain)
            torch.nn.init.uniform_(self.wh[l], -gain, gain)
            if self.bias is not None:
                torch.nn.init.uniform_(self.bias[l], -gain, gain)

    def forward(self, inputs, prev_state, list_next_state=False):
        '''
        Input:
            inputs: [seq_len, batch_size, input_size]
            prev_state: [num_directions*num_layers, batch_size, hidden_size]
        '''
        seq_len, batch_size = inputs.shape[:2]
        if prev_state is None:
            num_directions = 1
            zeros = torch.zeros(num_directions*self.num_layers, batch_size, self.hidden_size,
                                dtype=inputs.dtype, device=inputs.device)
            prev_state = (zeros, zeros)
        elif isinstance(prev_state, list) and len(prev_state) == batch_size:
            num_directions = 1
            zeros = torch.zeros(num_directions*self.num_layers, 1, self.hidden_size,
                                dtype=inputs.dtype, device=inputs.device)
            state = []
            for prev in prev_state:
                if prev is None:
                    state.append([zeros, zeros])
                else:
                    state.append(prev)
            state = list(zip(*state))
            prev_state = [torch.cat(t, dim=1) for t in state]

        H, C = prev_state
        x = inputs
        next_state = []
        for l in range(self.num_layers):
            h, c = H[l], C[l]
            new_x = []
            for s in range(seq_len):
                gate = torch.matmul(x[s], self.wx[l]) + torch.matmul(h, self.wh[l])
                if self.bias is not None:
                    gate += self.bias[l]
                gate = list(torch.chunk(gate, 4, dim=1))
                for i in range(4):
                    gate[i] = self.norm[l*4+i](gate[i])
                i, f, o, u = gate
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                o = torch.sigmoid(o)
                u = torch.tanh(u)
                c = f * c + i * u
                h = o * torch.tanh(c)
                # new_x.append(self.dropout(h))
                new_x.append(h)
            next_state.append((h, c))
            x = torch.stack(new_x, dim=0)
        if list_next_state:
            h, c = [torch.stack(t, dim=0) for t in zip(*next_state)]
            next_state = [torch.chunk(h, batch_size, dim=1), torch.chunk(c, batch_size, dim=1)]
            next_state = list(zip(*next_state))
        else:
            next_state = [torch.stack(t, dim=0) for t in zip(*next_state)]
        return x, next_state
