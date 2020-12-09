from functools import partial
from typing import Union, List, Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn

from nervex.model import DuelingHead, ConvEncoder
from nervex.torch_utils import get_lstm
from nervex.utils import squeeze


class DiscreteNet(nn.Module):
    r"""
    Overview:
        Base class for DQN based models.
    Interface:
        __init__, forward, fast_timestep_forward
    """

    def __init__(
            self,
            obs_dim: Union[int, tuple],
            action_dim: tuple,
            embedding_dim: int = 64,
            **kwargs,
    ) -> None:
        r"""
        Overview:
            Init the DiscreteNet according to arguments, including encoder, lstm(if needed) and head.
        Arguments:
            - obs_dim (:obj:`Union[int, tuple]`): a tuple of observation dim
            - action_dim (:obj:`int`): the num of action dim, \
                note that it can be a tuple containing more than one element
            - embedding_dim (:obj:`int`): encoder's embedding dim (output dim)
        """
        super(DiscreteNet, self).__init__()
        encoder_kwargs, lstm_kwargs, head_kwargs = get_kwargs(kwargs)
        self._encoder = Encoder(obs_dim, embedding_dim, **encoder_kwargs)
        if lstm_kwargs['lstm_type'] != 'none':
            lstm_kwargs['input_size'] = embedding_dim
            lstm_kwargs['hidden_size'] = embedding_dim
            self._lstm = get_lstm(**lstm_kwargs)
        self._head = Head(action_dim, embedding_dim, **head_kwargs)

    def forward(self, inputs: Dict) -> Dict:
        r"""
        Overview:
            Normal forward. Would use lstm between encoder and head if needed
        Arguments:
            - inputs (:obj:`Dict`): a dict containing all raw input tensors, e.g. 'obs', 'prev_state' (if use lstm), \
                'enable_fast_timestep' (if use fast timestep forward)
        Returns:
            - return (:obj:`Dict`): a dict containing output tensors of this DQN model
        """
        if isinstance(inputs, torch.Tensor):
            inputs = {'obs': inputs}
        # fast_timestep_forward if enabled
        if inputs.get('enable_fast_timestep', False):
            return self.fast_timestep_forward(inputs)
        # normal forward
        x = self._encoder(inputs['obs'])
        if hasattr(self, '_lstm'):
            x = x.unsqueeze(0)  # unsqueeze the first dim (only 1 timestep) to fit lstm's input shape
            x, next_state = self._lstm(x, inputs['prev_state'])
            x = x.squeeze(0)
            x = self._head(x)
            return {'logit': x, 'next_state': next_state}
        else:
            x = self._head(x)
            return {'logit': x}

    def fast_timestep_forward(self, inputs: Dict) -> Dict:
        r"""
        Overview:
            A simple implementation of timestep forward.
            Multiply timstep T and batch size B and get a larger batch size T*B。
        Arguments:
            - inputs (:obj:`Dict`): a dict containing all raw input tensors, e.g. 'obs', 'prev_state' (if use lstm), \
                'enable_fast_timestep' (if use fast timestep forward)
        Returns:
            - return (:obj:`Dict`): a dict containing output tensors of this DQN model
        """
        assert hasattr(self, '_lstm')
        x, prev_state = inputs['obs'], inputs['prev_state']
        assert len(x.shape) in [3, 5], x.shape
        x = parallel_wrapper(self._encoder)(x)
        lstm_embedding = []
        for t in range(x.shape[0]):  # T timesteps
            output, prev_state = self._lstm(x[t:t + 1], prev_state)
            lstm_embedding.append(output)
        x = torch.cat(lstm_embedding, 0)
        x = parallel_wrapper(self._head)(x)
        return {'logit': x, 'next_state': prev_state}


class Encoder(nn.Module):
    r"""
    Overview:
        The Encoder used in DQN models. Encode env state into a tensor for further operation.
    Interfaces:
        __init__, forward
    """

    def __init__(
            self,
            obs_dim: Union[int, tuple],
            embedding_dim: int,
            encoder_type: str,
    ) -> None:
        r"""
        Overview:
            Init the Encoder according to arguments.
        Arguments:
            - obs_dim (:obj:`Union[int, tuple]`): a tuple of observation dim
            - embedding_dim (:obj:`int`): encoder's embedding dim (output dim)
            - encoder_type (:obj:`str`): type of encoder, now supports ['fc', 'conv2d']
        """
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
        r"""
        Overview:
            Transform from raw observation into encoded tensor.
        Arguments:
            - x (:obj:`Dict`): raw observation
        Returns:
            - return (:obj:`Dict`): encoded embedding tensor
        """
        return self.main(x)


class Head(nn.Module):
    r"""
    Overview:
        The Head used in DQN models. Receive encoded embedding tensor and use it to predict the action.
    Interfaces:
        __init__, forward
    """

    def __init__(
            self,
            action_dim: tuple,
            input_dim: int,
            dueling: bool = True,
            a_layer_num: int = 1,
            v_layer_num: int = 1
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - action_dim (:obj:`tuple`): the num of action dim, \
                note that it can be a tuple containing more than one element
            - input_dim (:obj:`int`): input tensor dim of the head
            - dueling (:obj:`bool`): whether to use ``DuelingHead`` or ``nn.Linear``
            - a_layer_num (:obj:`int`): the num of layers in ``DuelingHead`` to compute action output
            - v_layer_num (:obj:`int`): the num of layers in ``DuelingHead`` to compute value output
        """
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
        r"""
        Overview:
            Use encoded tensor to predict the action.
        Arguments:
            - x (:obj:`Dict`): encoded tensor
        Returns:
            - return (:obj:`Dict`): action in logits
        """
        if isinstance(self.action_dim, tuple):
            x = [m(x) for m in self.pred]
        else:
            x = self.pred(x)
        return x


FCDiscreteNet = partial(
    DiscreteNet,
    encoder_kwargs={'encoder_type': 'fc'},
    lstm_kwargs={'lstm_type': 'none'},
    head_kwargs={'dueling': True}
)
ConvDiscreteNet = partial(
    DiscreteNet,
    encoder_kwargs={'encoder_type': 'conv2d'},
    lstm_kwargs={'lstm_type': 'none'},
    head_kwargs={'dueling': True}
)
FCRDiscreteNet = partial(
    DiscreteNet,
    encoder_kwargs={'encoder_type': 'fc'},
    lstm_kwargs={'lstm_type': 'normal'},
    head_kwargs={'dueling': True}
)
ConvRDiscreteNet = partial(
    DiscreteNet,
    encoder_kwargs={'encoder_type': 'conv2d'},
    lstm_kwargs={'lstm_type': 'normal'},
    head_kwargs={'dueling': True}
)


def parallel_wrapper(forward_fn: Callable) -> Callable:
    r"""
    Overview:
        Process timestep T and batch_size B at the same time, in other words, treat different timestep data as
        different trajectories in a batch. Used in ``DiscreteNet``'s ``fast_timestep_forward``.
    Arguments:
        - forward_fn (:obj:`Callable`): normal nn.Module's forward function
    Returns:
        - wrapper (:obj:`Callable`): wrapped function
    """

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


def get_kwargs(kwargs: Dict) -> Tuple[Dict]:
    r"""
    Overview:
        Get kwargs of encoder, lstm and head, according to input model kwargs.
    Arguments:
        - kwargs (:obj:`Dict`): model kwargs dict, might have keys ['encoder_kwargs', 'lstm_kwargs', 'head_kwargs']
    Returns:
        - ret (:obj:`Tuple[Dict]`): (encoder kwargs, lstm kwargs, head kwargs)
    """
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
