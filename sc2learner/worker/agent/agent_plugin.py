from typing import Any, Tuple, Callable, Union
from collections import OrderedDict
import torch
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


def register_plugin(agent: Any, plugin_cfg: Union[OrderedDict, None]) -> None:
    if plugin_cfg is None:
        return
    assert isinstance(plugin_cfg, OrderedDict), "plugin_cfg muse be ordered dict"
    plugin_name_map = {'grad': GradHelper, 'hidden_state': HiddenStateHelper}
    for k, v in plugin_cfg.items():
        if k not in plugin_name_map.keys():
            raise KeyError("invalid agent plugin name: {}".format(k))
        else:
            plugin_name_map[k].register(agent, **v)


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
                output, h = forward_fn(data)
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

    def before_forward(self, data: dict) -> Tuple[dict, dict]:
        assert 'state_info' in data.keys()
        state_info = data['state_info']
        for idx, is_reset in state_info.items():
            if is_reset:
                self._state[idx] = self._init_fn()
        state = [self._state[idx] for idx in state_info.keys()]
        data['prev_state'] = state
        return data, state_info

    def after_forward(self, h: Any, state_info: dict) -> None:
        assert len(h) == len(state_info), '{}/{}'.format(len(h), len(state_info))
        for i, idx in enumerate(state_info.keys()):
            self._state[idx] = h[i]
