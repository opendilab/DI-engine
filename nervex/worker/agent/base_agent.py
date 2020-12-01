from abc import ABC
import copy
from collections import OrderedDict
from typing import Any, Union, Optional, Dict, List

import torch

from .agent_plugin import register_plugin


class BaseAgent(ABC):
    r"""
    Overview:
        the base agent class

    Interfaces:
        __init__, forward, mode, state_dict, load_state_dict, reset
    """

    def __init__(self, model: torch.nn.Module, plugin_cfg: Union[OrderedDict, None]) -> None:
        r"""
        Overview:
            init the model and register plugins

        Arguments:
            - model (:obj:`torch.nn.Module`): the model of the agent
            - pulgin_cfg (:obj:`Union[OrderedDict, None]`): the plugin config to register
        """
        self._model = model
        self._plugin_cfg = plugin_cfg
        register_plugin(self, plugin_cfg)

    def forward(self, data: Any, param: Optional[dict] = None) -> Any:
        r"""
        Overview:
            forward method will call the foward method of the agent's model

        Arguments:
            - data (:obj:`Any`): the input data
            - param (:obj:`dict` or None): the optinal parameters, default set to None

        Returns:
            - output (:obj:`Any`): the output calculated by model
        """
        if param is not None:
            return self._model(data, **param)
        else:
            return self._model(data)

    def mode(self, train: bool) -> None:
        r"""
        Overview:
            call the model's function accordingly

        Arguments:
            - train (:obj:`bool`): whether to call the train method or eval method
        """
        if train:
            self._model.train()
        else:
            self._model.eval()

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, _model: torch.nn.Module) -> None:
        self._model = _model

    def state_dict(self) -> dict:
        r"""
        Overview:
            return the state_dict

        Returns:
            - ret (:obj:`dict`): the returned state_dict, while the ret['model'] is the model's state_dict
        """
        return {'model': self._model.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Overview:
            load the state_dict to model

        Arguments:
            - state_dict (:obj:`dict`): the input state_dict the model will load
        """
        self._model.load_state_dict(state_dict['model'])

    def reset(self) -> None:
        pass


model_plugin_cfg_set = set(['main', 'target', 'teacher'])


class AgentAggregator(object):
    r"""
    Overview:
        the AgentAggregator helps to build an agent according to the given input

    Interfaces:
        __init__, __getattr__
    """

    def __init__(self, agent_type: type, model: Union[torch.nn.Module, List[torch.nn.Module]], plugin_cfg: Dict[str, OrderedDict]) -> None:
        r"""
        Overview:
            __init__ of the AgentAggregator will get a class with multi agents in ._agent

        Arguments:
            - agent_type (:obj:`type`): the based class type of the agents in ._agent
            - model (:obj:`torch.nn.Module`): the model of agents
            - plugin_cfg (:obj:`Dict[str, OrderedDict])`): the plugin configs of agents
        """
        assert issubclass(agent_type, BaseAgent)
        assert set(plugin_cfg.keys()
                   ).issubset(model_plugin_cfg_set), '{}-{}'.format(set(plugin_cfg.keys()), model_plugin_cfg_set)
        if isinstance(model, torch.nn.Module):
            if len(plugin_cfg) == 1:
                model = [model]
            else:
                model = [model] + [copy.deepcopy(model) for _ in range(len(plugin_cfg) - 1)]
        self._agent = {}
        for i, k in enumerate(plugin_cfg):
            self._agent[k] = agent_type(model[i], plugin_cfg[k])

    def __getattr__(self, key: str) -> Any:
        r"""
        Overview:
            get the attrbute in key

        Arguments:
            - key (:obj:`str`): the key to query

        Returns:
            - ret (:obj:`Any`): the return attribute

        .. note::
            in usage, if you want to get the attribute "attr" in agent[k], you should query k + "_" + "attr"
        """
        if len(self._agent) == 1:
            return getattr(self._agent['main'], key)
        else:
            name = 'main'
            for k in self._agent:
                if key.startswith(k):
                    name = k
                    key = key.split(k + '_')[1]
                    break
            return getattr(self._agent[name], key)
