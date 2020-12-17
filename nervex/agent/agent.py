from abc import ABC
import copy
from collections import OrderedDict
from typing import Any, Union, Optional, Dict, List

import torch

from .agent_plugin import add_plugin


class BaseAgent(ABC):
    r"""
    Overview:
        the base agent class

    Interfaces:
        __init__, forward, mode, state_dict, load_state_dict, reset, add_model, add_plugin
    """

    def __init__(self, model: torch.nn.Module) -> None:
        r"""
        Overview:
            init the agent with model
        Arguments:
            - model (:obj:`torch.nn.Module`): the model of the agent
        """
        self._model = model

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

    def reset(self, data_id: List[int] = None) -> None:
        pass


model_plugin_cfg_set = set(['main', 'target', 'teacher'])


class Agent(object):
    r"""
    Overview:
        the Agent use model and plugin mechanism to implement different runtime demand for RL algorithm

    Interfaces:
        __init__, __getattr_, add_model, add_plugin_
    """

    def __init__(self, model: torch.nn.Module, agent_type: type = BaseAgent) -> None:
        self._agent = {}
        self._agent['main'] = agent_type(model)
        self._agent_type = agent_type

    def add_model(self, name: str, model: Optional[torch.nn.Module] = None, **kwargs) -> None:
        assert name in model_plugin_cfg_set
        if model is None:
            model = copy.deepcopy(self._agent['main'].model)
        self._agent[name] = self._agent_type(model)
        self.add_plugin(name, name, **kwargs)

    def remove_model(self, name: str) -> None:
        if name not in self._agent:
            raise KeyError("agent doesn't have model named {}".format(name))
        self._agent.pop(name)

    def add_plugin(self, agent_name: str, plugin_name: str, **kwargs) -> None:
        add_plugin(self._agent[agent_name], plugin_name, **kwargs)

    def remove_plugin(self, agent_name: str, plugin_name: str) -> None:
        raise NotImplementedError

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
