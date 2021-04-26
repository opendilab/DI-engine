from functools import partial
from typing import Union, List, Dict, Optional, Tuple, Callable
from copy import deepcopy

import torch
import torch.nn as nn
from nervex.model import head_fn_map, ConvEncoder
from nervex.torch_utils import get_lstm
from nervex.utils import squeeze, MODEL_REGISTRY


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
            hidden_dim_list: list = [128, 128, 64],
            **kwargs,
    ) -> None:
        r"""
        Overview:
            Init the DiscreteNet according to arguments, including encoder, lstm(if needed) and head.
        Arguments:
            - obs_dim (:obj:`Union[int, tuple]`): a tuple of observation dim
            - action_dim (:obj:`int`): the num of action dim, \
                note that it can be a tuple containing more than one element
            - hidden_dim_list (:obj:`list`): encoder's hidden layer dimension list
        """
        super(DiscreteNet, self).__init__()
        # parse arguments
        encoder_kwargs, lstm_kwargs, head_kwargs = get_kwargs(kwargs)
        embedding_dim = hidden_dim_list[-1]
        action_dim = squeeze(action_dim) # if action_dim is '(n,)', transform it in to 'n'.
        use_multi_discrete = isinstance(action_dim, tuple)
        head_fn = head_fn_map[head_kwargs.pop('head_type')]

        # build encoder: the encoder encodes different formats of observation into 1d vector.
        self._encoder = Encoder(obs_dim, hidden_dim_list=hidden_dim_list, **encoder_kwargs)

        # build lstm: the lstm encodes the history of the observations.
        if lstm_kwargs['lstm_type'] != 'none':
            lstm_kwargs['input_size'] = embedding_dim
            lstm_kwargs['hidden_size'] = embedding_dim
            self._lstm = get_lstm(**lstm_kwargs)

        # build head: the head encodes the observations into q-value of actions.
        if use_multi_discrete:
            self._head = MultiDiscreteHead(embedding_dim, action_dim, head_fn, **head_kwargs)
        else:
            self._head = head_fn(embedding_dim, action_dim, **head_kwargs)

    def forward(self, inputs: Dict, num_quantiles: int = None) -> Dict:
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
            x['next_state'] = next_state
            return x
        else:
            if num_quantiles is not None:
                x = self._head(x, num_quantiles)
            else:
                x = self._head(x)
            return x

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
        x['next_state'] = prev_state
        return x


MODEL_REGISTRY.register('discrete_net', DiscreteNet)


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
            encoder_type: str,
            hidden_dim_list: list = [128, 128, 64],
    ) -> None:
        r"""
        Overview:
            Init the Encoder according to arguments.
        Arguments:
            - obs_dim (:obj:`Union[int, tuple]`): a tuple of observation dim
            - encoder_type (:obj:`str`): type of encoder, now supports ['fc', 'conv2d']
            - hidden_dim_list (:obj:`list`): encoder's hidden layer dimension list
        """
        super(Encoder, self).__init__()
        self.act = nn.ReLU()
        assert encoder_type in ['fc', 'conv2d'], encoder_type
        embedding_dim = hidden_dim_list[-1]
        if encoder_type == 'fc':
            input_dim = squeeze(obs_dim)
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


class MultiDiscreteHead(nn.Module):
    r"""
    Overview:
        The Head used in DQN models. Receive encoded embedding tensor and use it to predict the action.
    Interfaces:
        __init__, forward
    """

    def __init__(
            self,
            input_dim: int,
            action_dim: tuple,
            head_fn: nn.Module,
            **head_kwargs,
    ) -> None:
        r"""
        Overview:
            Init the Head according to arguments.
        Arguments:
            - action_dim (:obj:`tuple`): the num of action dim, \
                note that it can be a tuple containing more than one element
            - input_dim (:obj:`int`): input tensor dim of the head
            - head_fn: class of head, like dueling_head, distribution_head, quatile_head, etc
            - head_kwargs: class-specific arguments
        """
        super(MultiDiscreteHead, self).__init__()
        self.pred = nn.ModuleList()
        for dim in action_dim:
            self.pred.append(head_fn(input_dim, dim, **head_kwargs))

    def _collate(self, x: list) -> Dict:
        r"""
        Overview:
            Translate a list of dicts to a dict of lists.
        Arguments:
            - x (:obj:`list`): list of dict
        Returns:
            - return (:obj:`Dict`): dict of list
        """
        return {k:[_[k] for _ in x] for k in x[0].keys()}

    def forward(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            Use encoded tensor to predict the action.
        Arguments:
            - x (:obj:`torch.Tensor`): encoded tensor
        Returns:
            - return (:obj:`Dict`): action in logits
        """
        return self._collate([m(x) for m in self.pred])


FCDiscreteNet = partial(
    DiscreteNet,
    encoder_kwargs={'encoder_type': 'fc'},
    lstm_kwargs={'lstm_type': 'none'},
    head_kwargs={
        'head_type': 'dueling',
        'a_layer_num': 1,
        'v_layer_num': 1,
    }
)
MODEL_REGISTRY.register('fc_discrete_net', FCDiscreteNet)


SQNDiscreteNet = partial(
    DiscreteNet,
    hidden_dim_list=[512, 64],
    encoder_kwargs={'encoder_type': 'fc'},
    lstm_kwargs={'lstm_type': 'none'},
    head_kwargs={
        'head_type': 'base',
        'layer_num': 1
    }
)
MODEL_REGISTRY.register('sqn_discrete_net', SQNDiscreteNet)


class SQNModel(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(SQNModel, self).__init__()
        self.q0 = SQNDiscreteNet(*args, **kwargs)
        self.q1 = SQNDiscreteNet(*args, **kwargs)

    def forward(self, data: dict) -> dict:
        output0 = self.q0(data)
        output1 = self.q1(data)
        return {
            'q_value': [output0['logit'], output1['logit']],
            'logit': output0['logit'],
        }
MODEL_REGISTRY.register('sqn_model', SQNModel)

NoiseDistributionFCDiscreteNet = partial(
    DiscreteNet,
    encoder_kwargs={'encoder_type': 'fc'},
    lstm_kwargs={'lstm_type': 'none'},
    head_kwargs={
        'head_type': 'distribution',
        'layer_num': 1,
        'noise': True
    }
)
MODEL_REGISTRY.register('noise_dist_fc', NoiseDistributionFCDiscreteNet)


NoiseFCDiscreteNet = partial(
    DiscreteNet,
    encoder_kwargs={'encoder_type': 'fc'},
    lstm_kwargs={'lstm_type': 'none'},
    head_kwargs={
        'head_type': 'dueling',
        'a_layer_num': 1,
        'v_layer_num': 1,
        'noise': True
    }
)


ConvDiscreteNet = partial(
    DiscreteNet,
    encoder_kwargs={'encoder_type': 'conv2d'},
    lstm_kwargs={'lstm_type': 'none'},
    head_kwargs={
        'head_type': 'dueling',
        'a_layer_num': 1,
        'v_layer_num': 1
    }
)


FCRDiscreteNet = partial(
    DiscreteNet,
    encoder_kwargs={'encoder_type': 'fc'},
    lstm_kwargs={'lstm_type': 'normal'},
    head_kwargs={
        'head_type': 'dueling',
        'a_layer_num': 1,
        'v_layer_num': 1
    }
)
MODEL_REGISTRY.register('fcr_discrete_net', FCRDiscreteNet)


ConvRDiscreteNet = partial(
    DiscreteNet,
    encoder_kwargs={'encoder_type': 'conv2d'},
    lstm_kwargs={'lstm_type': 'normal'},
    head_kwargs={
        'head_type': 'dueling',
        'a_layer_num': 1,
        'v_layer_num': 1
    }
)
MODEL_REGISTRY.register('convr_discrete_net', ConvRDiscreteNet)


NoiseQuantileFCDiscreteNet = partial(
    DiscreteNet,
    encoder_kwargs={'encoder_type': 'fc'},
    lstm_kwargs={'lstm_type': 'none'},
    head_kwargs={
        'head_type': 'quantile',
        'layer_num': 1,
        'noise': True,
    }
)
MODEL_REGISTRY.register('noise_quantile_fc', NoiseQuantileFCDiscreteNet)


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

        def reshape(d):
            if isinstance(d, list):
                d = [reshape(t) for t in d]
            elif isinstance(d, dict):
                d = {k: reshape(v) for k, v in d.items()}
            else:
                d = d.reshape(T, B, *d.shape[1:])
            return d

        x = x.reshape(T * B, *x.shape[2:])
        x = forward_fn(x)
        x = reshape(x)
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
    head_kwargs_keys = ['v_max', 'v_min', 'n_atom', 'beta_function_type', 'num_quantiles', 'quantile_embedding_dim']
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
        head_kwargs = deepcopy(kwargs['head_kwargs'])
        for k in kwargs:
            if k in head_kwargs_keys:
                head_kwargs[k] = kwargs[k]
    else:
        head_kwargs = {
            'head_type': kwargs.get('head_type', 'base'),
            'a_layer_num': kwargs.get('a_layer_num', 1),
            'v_layer_num': kwargs.get('v_layer_num', 1),
            'noise': kwargs.get('noise', False),
            'v_max': kwargs.get('v_max', 10),
            'v_min': kwargs.get('v_min', -10),
            'n_atom': kwargs.get('n_atom', 51),
            'beta_function_type': kwargs.get('beta_function_type', 'uniform'),
            'num_quantiles': kwargs.get('num_quantiles', 32),
            'quantile_embedding_dim': kwargs.get('quantile_embedding_dim', 128),
        }
    return encoder_kwargs, lstm_kwargs, head_kwargs
