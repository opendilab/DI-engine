from typing import Optional, Union, List, Tuple, Dict
import math
import torch
import torch.nn as nn
import treetensor.torch as ttorch

import ding
from ding.torch_utils.network.normalization import build_normalization
if ding.enable_hpc_rl:
    from hpc_rll.torch_utils.network.rnn import LSTM as HPCLSTM
else:
    HPCLSTM = None


def is_sequence(data):
    """
    Overview:
        Judege whether input ``data`` is instance ``list`` or ``tuple``.
    """
    return isinstance(data, list) or isinstance(data, tuple)


def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.BoolTensor:
    r"""
    Overview:
        create a mask for a batch sequences with different lengths
    Arguments:
        - lengths (:obj:`torch.Tensor`): lengths in each different sequences, shape could be (n, 1) or (n)
        - max_len (:obj:`int`): the padding size, if max_len is None, the padding size is the \
            max length of sequences
    Returns:
        - masks (:obj:`torch.BoolTensor`): mask has the same device as lengths
    """
    if len(lengths.shape) == 1:
        lengths = lengths.unsqueeze(dim=1)
    bz = lengths.numel()
    if max_len is None:
        max_len = lengths.max()
    else:
        max_len = min(max_len, lengths.max())
    return torch.arange(0, max_len).type_as(lengths).repeat(bz, 1).lt(lengths).to(lengths.device)


class LSTMForwardWrapper(object):
    r"""
    Overview:
        A class which provides methods to use before and after `forward`, in order to wrap the LSTM `forward` method.
    Interfaces:
        _before_forward, _after_forward
    """

    def _before_forward(self, inputs: torch.Tensor, prev_state: Union[None, List[Dict]]) -> torch.Tensor:
        """
        Overview:
            Preprocess the inputs and previous states
        Arguments:
            - inputs (:obj:`torch.Tensor`): input vector of cell, tensor of size [seq_len, batch_size, input_size]
            - prev_state (:obj:`Union[None, List[Dict]]`): None or tensor of size \
                [num_directions*num_layers, batch_size, hidden_size]. \
                If None then prv_state will be initialized to all zeros.
        Returns:
            - prev_state (:obj:`torch.Tensor`): batch previous state in lstm
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
        elif is_sequence(prev_state):
            if len(prev_state) != batch_size:
                raise RuntimeError(
                    "prev_state number is not equal to batch_size: {}/{}".format(len(prev_state), batch_size)
                )
            num_directions = 1
            zeros = torch.zeros(
                num_directions * self.num_layers, 1, self.hidden_size, dtype=inputs.dtype, device=inputs.device
            )
            state = []
            for prev in prev_state:
                if prev is None:
                    state.append([zeros, zeros])
                else:
                    if isinstance(prev, (Dict, ttorch.Tensor)):
                        state.append([v for v in prev.values()])
                    else:
                        state.append(prev)
            state = list(zip(*state))
            prev_state = [torch.cat(t, dim=1) for t in state]
        elif isinstance(prev_state, dict):
            prev_state = list(prev_state.values())
        else:
            raise TypeError("not support prev_state type: {}".format(type(prev_state)))
        return prev_state

    def _after_forward(self,
                       next_state: Tuple[torch.Tensor],
                       list_next_state: bool = False) -> Union[List[Dict], Dict[str, torch.Tensor]]:
        r"""
        Overview:
            Post-process the next_state, return list or tensor type next_states
        Arguments:
            - next_state (:obj:`Tuple[torch.Tensor]`): Tuple which contains next state (h, c).
            - list_next_state (:obj:`bool`): Whether to return next_state with list format, default set to False
        Returns:
            - next_state(:obj:`Union[List[Dict], Dict[str, torch.Tensor]]`): The formatted next_state.
        """
        if list_next_state:
            h, c = next_state
            batch_size = h.shape[1]
            next_state = [torch.chunk(h, batch_size, dim=1), torch.chunk(c, batch_size, dim=1)]
            next_state = list(zip(*next_state))
            next_state = [{k: v for k, v in zip(['h', 'c'], item)} for item in next_state]
        else:
            next_state = {k: v for k, v in zip(['h', 'c'], next_state)}
        return next_state


class LSTM(nn.Module, LSTMForwardWrapper):
    r"""
    Overview:
        Implimentation of LSTM cell with LN
    Interface:
        forward

    .. note::

        For beginners, you can refer to <https://zhuanlan.zhihu.com/p/32085405> to learn the basics about lstm
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            norm_type: Optional[str] = None,
            dropout: float = 0.
    ) -> None:
        r"""
        Overview:
            Initializate the LSTM cell arguments and parameters
        Arguments:
            - input_size (:obj:`int`): size of the input vector
            - hidden_size (:obj:`int`): size of the hidden state vector
            - num_layers (:obj:`int`): number of lstm layers
            - norm_type (:obj:`Optional[str]`): type of the normaliztion, (default: None)
            - dropout (:obj:`float`): dropout rate, default to 0
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        norm_func = build_normalization(norm_type)
        self.norm = nn.ModuleList([norm_func(hidden_size * 4) for _ in range(2 * num_layers)])
        self.wx = nn.ParameterList()
        self.wh = nn.ParameterList()
        dims = [input_size] + [hidden_size] * num_layers
        for l in range(num_layers):
            self.wx.append(nn.Parameter(torch.zeros(dims[l], dims[l + 1] * 4)))
            self.wh.append(nn.Parameter(torch.zeros(hidden_size, hidden_size * 4)))
        self.bias = nn.Parameter(torch.zeros(num_layers, hidden_size * 4))
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

    def forward(self,
                inputs: torch.Tensor,
                prev_state: torch.Tensor,
                list_next_state: bool = True) -> Tuple[torch.Tensor, Union[torch.Tensor, list]]:
        """
        Overview:
            Take the previous state and the input and calculate the output and the nextstate
        Arguments:
            - inputs (:obj:`torch.Tensor`): input vector of cell, tensor of size [seq_len, batch_size, input_size]
            - prev_state (:obj:`torch.Tensor`): None or tensor of size \
                [num_directions*num_layers, batch_size, hidden_size]
            - list_next_state (:obj:`bool`): whether return next_state with list format, default set to False
        Returns:
            - x (:obj:`torch.Tensor`): output from lstm
            - next_state (:obj:`Union[torch.Tensor, list]`): hidden state from lstm
        """
        seq_len, batch_size = inputs.shape[:2]
        prev_state = self._before_forward(inputs, prev_state)

        H, C = prev_state
        x = inputs
        next_state = []
        for l in range(self.num_layers):
            h, c = H[l], C[l]
            new_x = []
            for s in range(seq_len):
                gate = self.norm[l * 2](torch.matmul(x[s], self.wx[l])
                                        ) + self.norm[l * 2 + 1](torch.matmul(h, self.wh[l]))
                if self.bias is not None:
                    gate += self.bias[l]
                gate = list(torch.chunk(gate, 4, dim=1))
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
            if self.use_dropout and l != self.num_layers - 1:
                x = self.dropout(x)
        next_state = [torch.stack(t, dim=0) for t in zip(*next_state)]

        next_state = self._after_forward(next_state, list_next_state)
        return x, next_state


class PytorchLSTM(nn.LSTM, LSTMForwardWrapper):
    r"""
    Overview:
        Wrap the PyTorch nn.LSTM, format the input and output
    Interface:
        forward

    .. note::

        you can reference the <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM>
    """

    def forward(self,
                inputs: torch.Tensor,
                prev_state: torch.Tensor,
                list_next_state: bool = True) -> Tuple[torch.Tensor, Union[torch.Tensor, list]]:
        """
        Overview:
            Wrapped nn.LSTM.forward.
        Arguments:
            - inputs (:obj:`torch.Tensor`): input vector of cell, tensor of size \
                [seq_len, batch_size, input_size]
            - prev_state (:obj:`torch.Tensor`): None or tensor of size \
                [num_directions*num_layers, batch_size, hidden_size]
            - list_next_state (:obj:`bool`): whether return next_state with list format, default set to False
        Returns:
            - output (:obj:`torch.Tensor`): output from lstm
            - next_state (:obj:`Union[torch.Tensor, list]`): hidden state from lstm
        """
        prev_state = self._before_forward(inputs, prev_state)
        output, next_state = nn.LSTM.forward(self, inputs, prev_state)
        next_state = self._after_forward(next_state, list_next_state)
        return output, next_state


class GRU(nn.GRUCell, LSTMForwardWrapper):
    r"""
    Overview:
        Wrap the torch.nn.GRUCell, format the input and output
    Interface:
        forward

    .. note::
        you can reference the <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU>
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU, self).__init__(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, inputs, prev_state, list_next_state=True):
        r"""
        Overview:
            Wrapped nn.GRU.forward.
        Arguments:
            - inputs (:obj:`tensor`): input vector of cell, tensor of size [seq_len, batch_size, input_size]
            - prev_state (:obj:`tensor`): None or tensor of size [num_directions*num_layers, batch_size, hidden_size]
            - list_next_state (:obj:`bool`): whether return next_state with list format, default set to False
        Returns:
            - output (:obj:`tensor`): output from GRU
            - next_state (:obj:`tensor` or :obj:`list`): hidden state from GRU
        """
        # for compatibility
        prev_state, _ = self._before_forward(inputs, prev_state)
        inputs, prev_state = inputs.squeeze(0), prev_state.squeeze(0)
        next_state = nn.GRUCell.forward(self, inputs, prev_state)
        next_state = next_state.unsqueeze(0)
        x = next_state
        # for compatibility
        next_state = self._after_forward([next_state, next_state.clone()], list_next_state)
        return x, next_state


def get_lstm(
        lstm_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        norm_type: str = 'LN',
        dropout: float = 0.,
        seq_len: Optional[int] = None,
        batch_size: Optional[int] = None
) -> Union[LSTM, PytorchLSTM]:
    r"""
    Overview:
        Build and return the corresponding LSTM cell
    Arguments:
        - lstm_type (:obj:`str`): version of rnn cell, now support ['normal', 'pytorch', 'hpc', 'gru']
        - input_size (:obj:`int`): size of the input vector
        - hidden_size (:obj:`int`): size of the hidden state vector
        - num_layers (:obj:`int`): number of lstm layers
        - norm_type (:obj:`str`): type of the normaliztion, (default: None)
        - dropout (:obj:float):  dropout rate, default set to .0
        - seq_len (:obj:`Optional[int]`): seq len, default set to None
        - batch_size (:obj:`Optional[int]`): batch_size len, default set to None
    Returns:
        - lstm (:obj:`Union[LSTM, PytorchLSTM]`): the corresponding lstm cell
    """
    assert lstm_type in ['normal', 'pytorch', 'hpc', 'gru']
    if lstm_type == 'normal':
        return LSTM(input_size, hidden_size, num_layers, norm_type, dropout=dropout)
    elif lstm_type == 'pytorch':
        return PytorchLSTM(input_size, hidden_size, num_layers, dropout=dropout)
    elif lstm_type == 'hpc':
        return HPCLSTM(seq_len, batch_size, input_size, hidden_size, num_layers, norm_type, dropout).cuda()
    elif lstm_type == 'gru':
        assert num_layers == 1
        return GRU(input_size, hidden_size, num_layers)
