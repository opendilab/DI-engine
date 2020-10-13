"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. build LSTM: you can use build_LSTM to build the lstm module
"""
import math

import torch
import torch.nn as nn

from nervex.torch_utils.network.normalization import build_normalization


def is_sequence(data):
    return isinstance(data, list) or isinstance(data, tuple)


class LSTMForwardWrapper(object):
    r"""
    Overview:
        abstract class used to wrap the LSTM forward method
    Interface:
        _before_forward, _after_forward
    """

    def _before_forward(self, inputs, prev_state):
        r"""
        Overview:
            preprocess the inputs and previous states
        Arguments:
            - inputs (:obj:`tensor`): input vector of cell, tensor of size [seq_len, batch_size, input_size]
            - prev_state (:obj:`tensor`): None or tensor of size [num_directions*num_layers, batch_size, hidden_size],
                                          if None then prv_state will be initialized to all zeros.
        """
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
        elif is_sequence(prev_state) and len(prev_state) == 2:
            if isinstance(prev_state[0], torch.Tensor):
                pass
            else:
                prev_state = [torch.cat(t, dim=1) for t in prev_state]
        elif is_sequence(prev_state) and len(prev_state) == batch_size:
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
        else:
            raise Exception()
        return prev_state

    def _after_forward(self, next_state, list_next_state=False):
        r"""
        Overview:
            post process the next_state, return list or tensor type next_states
        Arguments:
            - next_state (:obj:`list` :obj:`Tuple` of :obj:`tensor`): list of Tuple contains the next (h, c)
            - list_next_state (:obj:`bool`): whether return next_state with list format, default set to False
        Returns:
            - next_state(:obj:`list` of :obj:`tensor` or :obj:`tensor`): the formated next_state
        """
        if list_next_state:
            h, c = [torch.stack(t, dim=0) for t in zip(*next_state)]
            batch_size = h.shape[1]
            next_state = [torch.chunk(h, batch_size, dim=1), torch.chunk(c, batch_size, dim=1)]
            next_state = list(zip(*next_state))
        else:
            next_state = [torch.stack(t, dim=0) for t in zip(*next_state)]
        return next_state


class LSTM(nn.Module, LSTMForwardWrapper):
    r"""
    Overview:
        Implimentation of LSTM cell

        Notes: for begainners, you can reference <https://zhuanlan.zhihu.com/p/32085405> to learn the basics about lstm

    Interface:
        __init__, forward
    """

    def __init__(self, input_size, hidden_size, num_layers, norm_type=None, bias=True, dropout=0.):
        r"""
        Overview:
            initializate the LSTM cell

        Arguments:
            - input_size (:obj:`int`): size of the input vector
            - hidden_size (:obj:`int`): size of the hidden state vector
            - num_layers (:obj:`int`): number of lstm layers
            - norm_type (:obj:`str`): type of the normaliztion, (default: None)
            - bias (:obj:`bool`): whether to use bias, default set to True
            - dropout (:obj:float):  dropout rate, default set to .0
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        norm_func = build_normalization(norm_type)
        self.norm = nn.ModuleList([norm_func(hidden_size) for _ in range(4 * num_layers)])
        self.wx = nn.ParameterList()
        self.wh = nn.ParameterList()
        dims = [input_size] + [hidden_size] * num_layers
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
        r"""
        Overview:
            Take the previous state and the input and calculate the output and the nextstate

        Arguments:
            - inputs (:obj:`tensor`): input vector of cell, tensor of size [seq_len, batch_size, input_size]
            - prev_state (:obj:`tensor`): None or tensor of size [num_directions*num_layers, batch_size, hidden_size]
            - list_next_state (:obj:`bool`): whether return next_state with list format, default set to False
        """
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

        next_state = self._after_forward(next_state, list_next_state)
        return x, next_state


class PytorchLSTM(nn.LSTM, LSTMForwardWrapper):
    r"""
    Overview:
        Wrap the nn.LSTM , format the input and output
        Notes:
            you can reference the <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM>
    Interface:
        forward
    """

    def forward(self, inputs, prev_state, list_next_state=False):
        r"""
        Overview:
            wrapped nn.LSTM.forward
        Arguments
            - inputs (:obj:`tensor`): input vector of cell, tensor of size [seq_len, batch_size, input_size]
            - prev_state (:obj:`tensor`): None or tensor of size [num_directions*num_layers, batch_size, hidden_size]
            - list_next_state (:obj:`bool`): whether return next_state with list format, default set to False
        """
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
    r"""
    Overview:
        build and return the corresponding LSTM cell
    Arguments:
        - lstm_type (:obj:`str`): version of lstm cell, now support ['normal', 'pytorch']
        - input_size (:obj:`int`): size of the input vector
        - hidden_size (:obj:`int`): size of the hidden state vector
        - num_layers (:obj:`int`): number of lstm layers
        - norm_type (:obj:`str`): type of the normaliztion, (default: None)
        - bias (:obj:`bool`): whether to use bias, default set to True
        - dropout (:obj:float):  dropout rate, default set to .0
    Returns:
        - lstm (:obj:`LSTM` or :obj:`PytorchLSTM`): the corresponding lstm cell
    """
    assert lstm_type in ['normal', 'pytorch']
    if lstm_type == 'normal':
        return LSTM(input_size, hidden_size, num_layers, norm_type, dropout=dropout)
    elif lstm_type == 'pytorch':
        return PytorchLSTM(input_size, hidden_size, num_layers, dropout=dropout)
