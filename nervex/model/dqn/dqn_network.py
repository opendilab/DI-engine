from functools import partial
from typing import Union, List, Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn

from nervex.model import DuelingHead, ConvEncoder
from nervex.torch_utils import get_lstm
from nervex.utils import squeeze


class DQNBase(nn.Module):

    def __init__(
            self,
            obs_dim: Union[int, tuple],
            action_dim: tuple,
            embedding_dim: int = 64,
            **kwargs,
    ) -> None:
        super(DQNBase, self).__init__()
        encoder_kwargs, lstm_kwargs, head_kwargs = get_kwargs(kwargs)
        self._encoder = Encoder(obs_dim, embedding_dim, **encoder_kwargs)
        if lstm_kwargs['lstm_type'] != 'none':
            lstm_kwargs['input_size'] = embedding_dim
            lstm_kwargs['hidden_size'] = embedding_dim
            self._lstm = get_lstm(**lstm_kwargs)
        self._head = Head(action_dim, embedding_dim, **head_kwargs)

    def forward(self, inputs: Dict) -> Dict:
        if isinstance(inputs, torch.Tensor):
            inputs = {'obs': inputs}
        if inputs.get('enable_fast_timestep', False):
            return self.fast_timestep_forward(inputs)
        x = self._encoder(inputs['obs'])
        if hasattr(self, '_lstm'):
            x = x.unsqueeze(0)
            x, next_state = self._lstm(x, inputs['prev_state'])
            x = x.squeeze(0)
            x = self._head(x)
            return {'logit': x, 'next_state': next_state}
        else:
            x = self._head(x)
            return {'logit': x}

    def fast_timestep_forward(self, inputs: Dict) -> Dict:
        assert hasattr(self, '_lstm')
        x, prev_state = inputs['obs'], inputs['prev_state']
        assert len(x.shape) in [3, 5], x.shape
        x = parallel_wrapper(self._encoder)(x)
        lstm_embedding = []
        for t in range(x.shape[0]):
            output, prev_state = self._lstm(x[t:t + 1], prev_state)
            lstm_embedding.append(output)
        x = torch.cat(lstm_embedding, 0)
        x = parallel_wrapper(self._head)(x)
        return {'logit': x, 'next_state': prev_state}


class Encoder(nn.Module):

    def __init__(
            self,
            obs_dim: Union[int, tuple],
            embedding_dim: int,
            encoder_type: str,
    ) -> None:
        super(Encoder, self).__init__()
        self.act = nn.ReLU()
        assert encoder_type in ['fc', 'conv2d'], encoder_type
        if encoder_type == 'fc':
            input_dim = squeeze(obs_dim)
            hidden_dim_list = [128, 128] + [embedding_dim]
            layers = []
            for dim in hidden_dim_list:
                layers.append(nn.Linear(input_dim, dim))
                layers.append(self.act)
                input_dim = dim
            self.main = nn.Sequential(*layers)
        elif encoder_type == 'conv2d':
            self.main = ConvEncoder(obs_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class Head(nn.Module):

    def __init__(
            self,
            action_dim: tuple,
            input_dim: int,
            dueling: bool = True,
            a_layer_num: int = 1,
            v_layer_num: int = 1
    ) -> None:
        super(Head, self).__init__()
        self.action_dim = squeeze(action_dim)
        self.dueling = dueling

        head_fn = partial(DuelingHead, a_layer_num=a_layer_num, v_layer_num=v_layer_num) if dueling else nn.Linear
        if isinstance(self.action_dim, tuple):
            self.pred = nn.ModuleList()
            for dim in self.action_dim:
                self.pred.append(head_fn(input_dim, dim))
        else:
            self.pred = head_fn(input_dim, self.action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.action_dim, tuple):
            x = [m(x) for m in self.pred]
        else:
            x = self.pred(x)
        return x


FCDQN = partial(
    DQNBase, encoder_kwargs={'encoder_type': 'fc'}, lstm_kwargs={'lstm_type': 'none'}, head_kwargs={'dueling': True}
)
ConvDQN = partial(
    DQNBase,
    encoder_kwargs={'encoder_type': 'conv2d'},
    lstm_kwargs={'lstm_type': 'none'},
    head_kwargs={'dueling': True}
)
FCDRQN = partial(
    DQNBase, encoder_kwargs={'encoder_type': 'fc'}, lstm_kwargs={'lstm_type': 'normal'}, head_kwargs={'dueling': True}
)
ConvDRQN = partial(
    DQNBase,
    encoder_kwargs={'encoder_type': 'conv2d'},
    lstm_kwargs={'lstm_type': 'normal'},
    head_kwargs={'dueling': True}
)


def parallel_wrapper(forward_fn: Callable) -> Callable:

    def wrapper(x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        T, B = x.shape[:2]
        x = x.reshape(T * B, *x.shape[2:])
        x = forward_fn(x)
        if isinstance(x, list):
            x = [t.reshape(T, B, *t.shape[1:]) for t in x]
        else:
            x = x.reshape(T, B, *x.shape[1:])
        return x

    return wrapper


def get_kwargs(kwargs: dict) -> Tuple[dict]:
    if 'encoder_kwargs' in kwargs:
        encoder_kwargs = kwargs['encoder_kwargs']
    else:
        encoder_kwargs = {
            'encoder_type': kwargs.get('encoder_type', None),
        }
    if 'lstm_kwargs' in kwargs:
        lstm_kwargs = kwargs['lstm_kwargs']
    else:
        lstm_kwargs = {
            'lstm_type': kwargs.get('lstm_type', 'normal'),
        }
    if 'head_kwargs' in kwargs:
        head_kwargs = kwargs['head_kwargs']
    else:
        head_kwargs = {
            'dueling': kwargs.get('dueling', True),
            'a_layer_num': kwargs.get('a_layer_num', 1),
            'v_layer_num': kwargs.get('v_layer_num', 1),
        }
    return encoder_kwargs, lstm_kwargs, head_kwargs
