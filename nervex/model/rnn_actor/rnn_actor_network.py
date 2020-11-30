import torch
import torch.nn as nn

from typing import Tuple, Dict, Union
from nervex.torch_utils import get_lstm
from nervex.utils import squeeze
import torch.nn.functional as F
from nervex.torch_utils import to_tensor, tensor_to_list


class RnnActorNetwork(nn.Module):

    def __init__(
        self,
        obs_dim: Union[int, tuple],
        action_dim: tuple,
        embedding_dim: int = 64,
        rnn_type: str = 'lstm',
        **kwargs,
    ):

        super(RnnActorNetwork, self).__init__()
        # rnn_kwargs = get_kwargs(kwargs)
        self._obs_dim = obs_dim
        self._act_dim = action_dim
        self._embedding_dim = embedding_dim
        self._fc1 = nn.Linear(squeeze(obs_dim), embedding_dim)
        self._act = F.relu
        # self._rnn_type = rnn_kwargs['rnn_type']
        self._rnn_type = rnn_type
        assert self._rnn_type in ['lstm', 'gru']
        self._rnn = get_lstm('normal', embedding_dim,
                             embedding_dim) if self._rnn_type == 'lstm' else nn.GRUCell(embedding_dim, embedding_dim)
        self._fc2 = nn.Linear(embedding_dim, action_dim)

    def forward(self, inputs: Dict) -> Dict:
        if isinstance(inputs, torch.Tensor):
            inputs = {'obs': inputs}
        x = self._fc1(inputs['obs'])
        x = self._act(x)
        if self._rnn_type == 'lstm':
            x = x.unsqueeze(0)
            x, next_state = self._rnn(x, inputs['prev_state'])
            x = x.squeeze(0)
            x = self._fc2(x)
            return {'logit': x, 'next_state': next_state}
        elif self._rnn_type == 'gru':
            num_directions = 1
            num_layers = 1
            for i in range(len(inputs['prev_state'])):
                if inputs['prev_state'][i] is None:
                    seq_len, batch_size = x.unsqueeze(0).shape[:2]
                    zeros = torch.zeros(
                        num_directions * num_layers,
                        self._embedding_dim,
                        dtype=x.dtype,
                        device=x.device
                    )
                    inputs['prev_state'][i] = zeros
            if isinstance(inputs['prev_state'], list):
                inputs['prev_state'] = torch.cat(inputs['prev_state'])
            next_state = self._rnn(x, inputs['prev_state'])
            x = self._fc2(next_state)
            return {'logit': x, 'next_state': next_state}


# def get_kwargs(kwargs: dict) -> Tuple[dict]:
#     if 'rnn_kwargs' in kwargs:
#         rnn_kwargs = kwargs['rnn_kwargs']
#         assert rnn_kwargs['rnn_type'] in ['lstm', 'gru']
#     else:
#         rnn_kwargs = {'rnn_type': kwargs.get('rnn_type', 'lstm'), 'lstm_type': kwargs.get('lstm_type', 'normal')}
#     return rnn_kwargs
