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
        Determines if the input data is of type list or tuple.
    Arguments:
        - data: The input data to be checked.
    Returns:
        - boolean: True if the input is a list or a tuple, False otherwise.
    """
    return isinstance(data, list) or isinstance(data, tuple)


def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.BoolTensor:
    """
    Overview:
        Generates a boolean mask for a batch of sequences with differing lengths.
    Arguments:
        - lengths (:obj:`torch.Tensor`): A tensor with the lengths of each sequence. Shape could be (n, 1) or (n).
        - max_len (:obj:`int`, optional): The padding size. If max_len is None, the padding size is the max length of \
            sequences.
    Returns:
        - masks (:obj:`torch.BoolTensor`): A boolean mask tensor. The mask has the same device as lengths.
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
    """
    Overview:
        Class providing methods to use before and after the LSTM `forward` method.
        Wraps the LSTM `forward` method.
    Interfaces:
        ``_before_forward``, ``_after_forward``
    """

    def _before_forward(self, inputs: torch.Tensor, prev_state: Union[None, List[Dict]]) -> torch.Tensor:
        """
        Overview:
            Preprocesses the inputs and previous states before the LSTM `forward` method.
        Arguments:
            - inputs (:obj:`torch.Tensor`): Input vector of the LSTM cell. Shape: [seq_len, batch_size, input_size]
            - prev_state (:obj:`Union[None, List[Dict]]`): Previous state tensor. Shape: [num_directions*num_layers, \
                batch_size, hidden_size]. If None, prv_state will be initialized to all zeros.
        Returns:
            - prev_state (:obj:`torch.Tensor`): Preprocessed previous state for the LSTM batch.
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
        """
        Overview:
            Post-processes the next_state after the LSTM `forward` method.
        Arguments:
            - next_state (:obj:`Tuple[torch.Tensor]`): Tuple containing the next state (h, c).
            - list_next_state (:obj:`bool`, optional): Determines the format of the returned next_state. \
                If True, returns next_state in list format. Default is False.
        Returns:
            - next_state(:obj:`Union[List[Dict], Dict[str, torch.Tensor]]`): The post-processed next_state.
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
    """
    Overview:
        Implementation of an LSTM cell with Layer Normalization (LN).
    Interfaces:
        ``__init__``, ``forward``

    .. note::

        For a primer on LSTM, refer to https://zhuanlan.zhihu.com/p/32085405.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            norm_type: Optional[str] = None,
            dropout: float = 0.
    ) -> None:
        """
        Overview:
            Initialize LSTM cell parameters.
        Arguments:
            - input_size (:obj:`int`): Size of the input vector.
            - hidden_size (:obj:`int`): Size of the hidden state vector.
            - num_layers (:obj:`int`): Number of LSTM layers.
            - norm_type (:obj:`Optional[str]`): Normalization type, default is None.
            - dropout (:obj:`float`): Dropout rate, default is 0.
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
        """
        Overview:
            Initialize the parameters of the LSTM cell.
        """

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
            Compute output and next state given previous state and input.
        Arguments:
            - inputs (:obj:`torch.Tensor`): Input vector of cell, size [seq_len, batch_size, input_size].
            - prev_state (:obj:`torch.Tensor`): Previous state, \
                size [num_directions*num_layers, batch_size, hidden_size].
            - list_next_state (:obj:`bool`): Whether to return next_state in list format, default is True.
        Returns:
            - x (:obj:`torch.Tensor`): Output from LSTM.
            - next_state (:obj:`Union[torch.Tensor, list]`): Hidden state from LSTM.
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
    """
    Overview:
        Wrapper class for PyTorch's nn.LSTM, formats the input and output. For more details on nn.LSTM,
        refer to https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
    Interfaces:
        ``forward``
    """

    def forward(self,
                inputs: torch.Tensor,
                prev_state: torch.Tensor,
                list_next_state: bool = True) -> Tuple[torch.Tensor, Union[torch.Tensor, list]]:
        """
        Overview:
            Executes nn.LSTM.forward with preprocessed input.
        Arguments:
            - inputs (:obj:`torch.Tensor`): Input vector of cell, size [seq_len, batch_size, input_size].
            - prev_state (:obj:`torch.Tensor`): Previous state, size [num_directions*num_layers, batch_size, \
                hidden_size].
            - list_next_state (:obj:`bool`): Whether to return next_state in list format, default is True.
        Returns:
            - output (:obj:`torch.Tensor`): Output from LSTM.
            - next_state (:obj:`Union[torch.Tensor, list]`): Hidden state from LSTM.
        """
        prev_state = self._before_forward(inputs, prev_state)
        output, next_state = nn.LSTM.forward(self, inputs, prev_state)
        next_state = self._after_forward(next_state, list_next_state)
        return output, next_state


class GRU(nn.GRUCell, LSTMForwardWrapper):
    """
    Overview:
        This class extends the `torch.nn.GRUCell` and `LSTMForwardWrapper` classes, and formats inputs and outputs
        accordingly.
    Interfaces:
        ``__init__``, ``forward``
    Properties:
        hidden_size, num_layers

    .. note::
        For further details, refer to the official PyTorch documentation:
        <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU>
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        """
        Overview:
            Initialize the GRU class with input size, hidden size, and number of layers.
        Arguments:
            - input_size (:obj:`int`): The size of the input vector.
            - hidden_size (:obj:`int`): The size of the hidden state vector.
            - num_layers (:obj:`int`): The number of GRU layers.
        """
        super(GRU, self).__init__(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self,
                inputs: torch.Tensor,
                prev_state: Optional[torch.Tensor] = None,
                list_next_state: bool = True) -> Tuple[torch.Tensor, Union[torch.Tensor, List]]:
        """
        Overview:
            Wrap the `nn.GRU.forward` method.
        Arguments:
            - inputs (:obj:`torch.Tensor`): Input vector of cell, tensor of size [seq_len, batch_size, input_size].
            - prev_state (:obj:`Optional[torch.Tensor]`): None or tensor of \
                size [num_directions*num_layers, batch_size, hidden_size].
            - list_next_state (:obj:`bool`): Whether to return next_state in list format (default is True).
        Returns:
            - output (:obj:`torch.Tensor`): Output from GRU.
            - next_state (:obj:`torch.Tensor` or :obj:`list`): Hidden state from GRU.
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
    """
    Overview:
        Build and return the corresponding LSTM cell based on the provided parameters.
    Arguments:
        - lstm_type (:obj:`str`): Version of RNN cell. Supported options are ['normal', 'pytorch', 'hpc', 'gru'].
        - input_size (:obj:`int`): Size of the input vector.
        - hidden_size (:obj:`int`): Size of the hidden state vector.
        - num_layers (:obj:`int`): Number of LSTM layers (default is 1).
        - norm_type (:obj:`str`): Type of normalization (default is 'LN').
        - dropout (:obj:`float`): Dropout rate (default is 0.0).
        - seq_len (:obj:`Optional[int]`): Sequence length (default is None).
        - batch_size (:obj:`Optional[int]`): Batch size (default is None).
    Returns:
        - lstm (:obj:`Union[LSTM, PytorchLSTM]`): The corresponding LSTM cell.
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
