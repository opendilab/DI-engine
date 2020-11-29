import copy
from abc import ABC, abstractmethod, abstractclassmethod
from collections import OrderedDict
from typing import Any, Tuple, Callable, Union, Optional

import numpy as np
import torch
from nervex.torch_utils import get_tensor_data
from nervex.rl_utils import create_noise_generator


class IAgentPlugin(ABC):

    @abstractclassmethod
    def register(cls: type, agent: Any, **kwargs) -> None:
        """inplace modify agent"""
        raise NotImplementedError


IAgentStatelessPlugin = IAgentPlugin


class IAgentStatefulPlugin(IAgentPlugin):

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class GradHelper(IAgentStatelessPlugin):

    @classmethod
    def register(cls: type, agent: Any, enable_grad: bool) -> None:

        def grad_wrapper(fn):
            context = torch.enable_grad() if enable_grad else torch.no_grad()

            def wrapper(*args, **kwargs):
                with context:
                    return fn(*args, **kwargs)

            return wrapper

        agent.forward = grad_wrapper(agent.forward)


class HiddenStateHelper(IAgentStatefulPlugin):
    """
    Overview:
        maintain the hidden state for RNN-base model, each sample in a batch has its own state

    Note:
        1. this helper must deal with a actual batch with some parts of samples(e.g: 6 samples of state_num 8)
        2. this helper must deal with the single sample state reset
    """

    @classmethod
    def register(
            cls: type,
            agent: Any,
            state_num: int,
            save_prev_state: bool = False,
            init_fn: Callable = lambda: None
    ) -> None:
        state_manager = cls(state_num, init_fn=init_fn)
        agent._state_manager = state_manager

        def forward_state_wrapper(forward_fn):

            def wrapper(data, **kwargs):
                state_id = kwargs.pop('state_id', None)
                data, state_info = agent._state_manager.before_forward(data, state_id)
                output = forward_fn(data, **kwargs)
                h = output.pop('next_state')
                agent._state_manager.after_forward(h, state_info)
                if save_prev_state:
                    prev_state = get_tensor_data(data['prev_state'])
                    output['prev_state'] = prev_state
                return output

            return wrapper

        def reset_state_wrapper(reset_fn):

            def wrapper(*args, **kwargs):
                state = kwargs.pop('state', None)
                state_id = kwargs.pop('state_id', None)
                agent._state_manager.reset(state, state_id)
                return reset_fn(*args, **kwargs)

            return wrapper

        agent.forward = forward_state_wrapper(agent.forward)
        agent.reset = reset_state_wrapper(agent.reset)

    def __init__(self, state_num: int, init_fn: Callable) -> None:
        self._state_num = state_num
        self._state = {i: init_fn() for i in range(state_num)}
        self._init_fn = init_fn

    def reset(self, state: Optional[list] = None, state_id: Optional[list] = None) -> None:
        if state_id is None:
            state_id = [i for i in range(self._state_num)]
        if state is None:
            state = [self._init_fn() for i in range(len(state_id))]
        assert len(state) == len(state_id), '{}/{}'.format(len(state), len(state_id))
        for idx, s in zip(state_id, state):
            self._state[idx] = s

    def before_forward(self, data: dict, state_id: Optional[list]) -> Tuple[dict, dict]:
        if state_id is None:
            state_id = [i for i in range(self._state_num)]

        state_info = {idx: self._state[idx] for idx in state_id}
        data['prev_state'] = list(state_info.values())
        return data, state_info

    def after_forward(self, h: Any, state_info: dict) -> None:
        assert len(h) == len(state_info), '{}/{}'.format(len(h), len(state_info))
        for i, idx in enumerate(state_info.keys()):
            self._state[idx] = h[i]


class ArgmaxSampleHelper(IAgentStatelessPlugin):

    @classmethod
    def register(cls: type, agent: Any) -> None:

        def sample_wrapper(forward_fn: Callable) -> Callable:

            def wrapper(*args, **kwargs):
                output = forward_fn(*args, **kwargs)
                assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
                logit = output['logit']
                assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
                if isinstance(logit, torch.Tensor):
                    logit = [logit]
                action = [l.argmax(dim=-1) for l in logit]
                if len(action) == 1:
                    action, logit = action[0], logit[0]
                output['action'] = action
                return output

            return wrapper

        agent.forward = sample_wrapper(agent.forward)


class MultinomialSampleHelper(IAgentStatelessPlugin):

    @classmethod
    def register(cls: type, agent: Any) -> None:

        def sample_wrapper(forward_fn: Callable) -> Callable:

            def wrapper(*args, **kwargs):
                output = forward_fn(*args, **kwargs)
                assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
                logit = output['logit']
                assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
                if isinstance(logit, torch.Tensor):
                    logit = [logit]
                action = [torch.multinomial(torch.softmax(l, dim=1), 1) for l in logit]
                if len(action) == 1:
                    action, logit = action[0], logit[0]
                output['action'] = action
                return output

            return wrapper

        agent.forward = sample_wrapper(agent.forward)


class EpsGreedySampleHelper(IAgentStatelessPlugin):

    @classmethod
    def register(cls: type, agent: Any) -> None:

        def sample_wrapper(forward_fn: Callable) -> Callable:

            def wrapper(*args, **kwargs):
                eps = kwargs.pop('eps')
                output = forward_fn(*args, **kwargs)
                assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
                logit = output['logit']
                assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
                if isinstance(logit, torch.Tensor):
                    logit = [logit]
                action = []
                for l in logit:
                    # TODO batch-wise e-greedy exploration
                    if np.random.random() > eps:
                        action.append(l.argmax(dim=-1))
                    else:
                        action.append(torch.randint(0, l.shape[-1], size=(l.shape[0], )))
                if len(action) == 1:
                    action, logit = action[0], logit[0]
                output['action'] = action
                return output

            return wrapper

        agent.forward = sample_wrapper(agent.forward)


global noise_generator
noise_generator = None


class ActionNoiseHelper(IAgentStatelessPlugin):

    @classmethod
    def register(cls: type, agent: Any) -> None:

        def noise_wrapper(forward_fn: Callable) -> Callable:

            def wrapper(*args, **kwargs):
                global noise_generator
                noise_type = kwargs.pop('noise_type')
                noise_kwargs = kwargs.pop('noise_kwargs')
                noise_range = noise_kwargs.pop('range')
                action_range = kwargs.pop('action_range')
                if noise_generator is None:
                    noise_generator = create_noise_generator(noise_type, noise_kwargs)
                output = forward_fn(*args, **kwargs)
                assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
                action = output['action']
                assert isinstance(action, torch.Tensor)
                noise = noise_generator(action.shape, action.device)
                action += noise.clamp(-noise_range, noise_range)  # noise clip
                action = action.clamp(action_range['min'], action_range['max'])  # action clip
                output['action'] = action
                return output

            return wrapper

        agent.forward = noise_wrapper(agent.forward)


class TargetNetworkHelper(IAgentStatefulPlugin):

    @classmethod
    def register(cls: type, agent: Any, update_cfg: dict):
        target_network = cls(agent.model, update_cfg)
        agent._target_network = target_network
        setattr(agent, 'update', getattr(agent._target_network, 'update'))

    def __init__(self, model: torch.nn.Module, update_cfg: dict) -> None:
        self._model = model
        update_type = update_cfg['type']
        assert update_type in ['momentum', 'assign']
        self._update_type = update_type
        self._update_kwargs = update_cfg['kwargs']
        self._update_count = 0

    def update(self, state_dict: dict, direct: bool = False) -> None:
        if direct:
            self._model.load_state_dict(state_dict, strict=True)
            self._update_count = 0
        elif self._update_type == 'assign':
            if (self._update_count + 1) % self._update_kwargs['freq'] == 0:
                self._model.load_state_dict(state_dict, strict=True)
            self._update_count += 1
        elif self._update_type == 'momentum':
            theta = self._update_kwargs['theta']
            for name, p in self._model.named_parameters():
                # default theta = 0.001
                p = (1 - theta) * p + theta * state_dict[name]

    def reset(self) -> None:
        self._update_count = 0


plugin_name_map = {
    'grad': GradHelper,
    'hidden_state': HiddenStateHelper,
    'argmax_sample': ArgmaxSampleHelper,
    'eps_greedy_sample': EpsGreedySampleHelper,
    'multinomial_sample': MultinomialSampleHelper,
    'action_noise': ActionNoiseHelper,
    # model plugin
    'target': TargetNetworkHelper,
}


def register_plugin(agent: Any, plugin_cfg: Union[OrderedDict, None]) -> None:
    if plugin_cfg is None:
        return
    assert isinstance(plugin_cfg, OrderedDict), "plugin_cfg muse be ordered dict"
    for k, v in plugin_cfg.items():
        if k not in plugin_name_map.keys():
            raise KeyError("invalid agent plugin name: {}".format(k))
        else:
            plugin_name_map[k].register(agent, **v)


def add_plugin(name, plugin_type):
    assert isinstance(name, str)
    assert issubclass(plugin_type, IAgentPlugin)
    plugin_name_map[name] = plugin_type
