from typing import Any, Tuple, Callable, Union, Optional
import copy
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod, abstractclassmethod


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
    def register(cls: type, agent: Any, state_num: int) -> None:
        state_manager = cls(state_num, init_fn=lambda: None)
        agent._state_manager = state_manager

        def forward_state_wrapper(forward_fn):
            def wrapper(data, **kwargs):
                data, state_info = agent._state_manager.before_forward(data)
                output, h = forward_fn(data, **kwargs)
                agent._state_manager.after_forward(h, state_info)
                return output

            return wrapper

        def reset_state_wrapper(reset_fn):
            def wrapper(*args, **kwargs):
                agent._state_manager.reset()
                return reset_fn(*args, **kwargs)

            return wrapper

        agent.forward = forward_state_wrapper(agent.forward)
        agent.reset = reset_state_wrapper(agent.reset)

    def __init__(self, state_num: int, init_fn: Callable) -> None:
        self._state_num = state_num
        self._state = {i: init_fn() for i in range(state_num)}
        self._init_fn = init_fn

    def reset(self) -> None:
        self._state = {i: self._init_fn() for i in range(self._state_num)}

    def before_forward(self, inputs: dict) -> Tuple[dict, dict]:
        if 'state_info' in inputs.keys():
            state_info = inputs['state_info']
        else:
            state_info = {i: False for i in range(self._state_num)}
        for idx, is_reset in state_info.items():
            if is_reset:
                self._state[idx] = self._init_fn()
        state = [self._state[idx] for idx in state_info.keys()]
        data = inputs['data']
        assert all([isinstance(d, dict) for d in data]), [type(d) for d in data]
        for d, s in zip(data, state):
            d['prev_state'] = s
        return data, state_info

    def after_forward(self, h: Any, state_info: dict) -> None:
        assert len(h) == len(state_info), '{}/{}'.format(len(h), len(state_info))
        for i, idx in enumerate(state_info.keys()):
            self._state[idx] = h[i]


class ArgmaxSampleHelper(IAgentStatelessPlugin):
    @classmethod
    def register(cls: type, agent: Any):
        def sample_wrapper(forward_fn):
            def wrapper(*args, **kwargs):
                logit = forward_fn(*args, **kwargs)
                action = logit.argmax(dim=-1)
                return action, logit

            return wrapper

        agent.forward = sample_wrapper(agent.forward)


class EpsGreedySampleHelper(IAgentStatelessPlugin):
    @classmethod
    def register(cls: type, agent: Any):
        def sample_wrapper(forward_fn):
            def wrapper(*args, **kwargs):
                eps = kwargs.pop('eps')
                logits = forward_fn(*args, **kwargs)
                assert isinstance(logits, torch.Tensor) or isinstance(logits, list)
                if isinstance(logits, torch.Tensor):
                    logits = [logits]
                action = []
                for logit in logits:
                    # TODO batch-wise e-greedy exploration
                    if np.random.random() > eps:
                        action.append(logit.argmax(dim=-1))
                    else:
                        action.append(torch.randint(0, logit.shape[-1], size=(logit.shape[0], )))
                if len(action) == 1:
                    action, logits = action[0], logits[0]
                return action, logits

            return wrapper

        agent.forward = sample_wrapper(agent.forward)


class TargetNetworkHelper(IAgentStatefulPlugin):
    @classmethod
    def register(cls: type, agent: Any, update_cfg: dict):
        target_network = cls(agent.model, update_cfg)
        agent._target_network = target_network
        for method_name in ['update_target_network', 'target_forward', 'target_mode']:
            setattr(agent, method_name, getattr(agent._target_network, method_name))

    def __init__(self, model: torch.nn.Module, update_cfg: dict) -> None:
        self._model = copy.deepcopy(model)
        self.target_mode(train=True)
        update_type = update_cfg['type']
        assert update_type in ['momentum', 'assign']
        self._update_type = update_type
        self._update_kwargs = update_cfg['kwargs']
        self._update_count = 0

    def update_target_network(self, state_dict: dict, direct: bool = False) -> None:
        if self._update_type == 'assign':
            if direct or (self._update_count + 1) % self._update_kwargs['freq'] == 0:
                self._model.load_state_dict(state_dict, strict=True)
            self._update_count += 1
        elif self._update_type == 'momentum':
            theta = self._update_kwargs['theta']
            for name, p in self._model.named_parameters():
                # default theta = 0.001
                p = (1 - theta) * p + theta * state_dict[name]

    def target_mode(self, train: bool) -> None:
        if train:
            self._model.train()
        else:
            self._model.eval()

    def target_forward(self, data: Any, param: Optional[dict] = None) -> Any:
        with torch.no_grad():
            if param is not None:
                return self._model(data, **param)
            else:
                return self._model(data)

    def reset(self, state_dict: dict) -> None:
        self.update_target_network(state_dict)
        self._update_count = 0


plugin_name_map = {
    'grad': GradHelper,
    'hidden_state': HiddenStateHelper,
    'target_network': TargetNetworkHelper,
    'argmax_sample': ArgmaxSampleHelper,
    'eps_greedy_sample': EpsGreedySampleHelper
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
