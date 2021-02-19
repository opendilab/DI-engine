from abc import ABC
import copy
from collections import OrderedDict
from typing import Any, Union, Optional, Dict, List

import torch

from .armor_plugin import add_plugin


class BaseArmor(ABC):
    r"""
    Overview:
        Base armor class. Can be wrapped by different armor plugins.
        Use model and plugin mechanism to implement different runtime demand for RL algorithm.
    Interfaces:
        __init__, forward, mode, state_dict, load_state_dict, reset, add_model, add_plugin
    """

    def __init__(self, model: torch.nn.Module) -> None:
        r"""
        Overview:
            Init the armor with model.
        Arguments:
            - model (:obj:`torch.nn.Module`): An ``torch.nn.Module`` model.
        """
        self._model = model

    def forward(self, data: Any, param: Optional[dict] = None) -> Any:
        r"""
        Overview:
            Call ``model.foward``. Usually, will be wrapped by armor plugins.
        Arguments:
            - data (:obj:`Any`): Input data.
            - param (:obj:`Optional[dict]`): Optional parameters. Default set to None.
        Returns:
            - output (:obj:`Any`): The output calculated by model.
        """
        if param is not None:
            return self._model(data, **param)
        else:
            return self._model(data)

    def mode(self, train: bool) -> None:
        r"""
        Overview:
            Call the model's corresponding method.
        Arguments:
            - train (:obj:`bool`): Whether to call the train method or eval method.
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
            Return the model_state_dict
        Returns:
            - ret (:obj:`dict`): Model's state_dict. ``ret['model']`` is the model's state_dict.
        """
        return {'model': self._model.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Overview:
            Load the state_dict to model.
        Arguments:
            - state_dict (:obj:`dict`): Input state_dict that the model will load.
        """
        self._model.load_state_dict(state_dict['model'])

    def reset(self, data_id: List[int] = None) -> None:
        pass


model_plugin_cfg_set = set(['main', 'target', 'teacher'])


class Armor(object):
    r"""
    Overview:
        Manage multiple armors, e.g. main, target, teacher.
    Interfaces:
        __init__, __getattr_, add_model, add_plugin_
    """

    def __init__(self, model: torch.nn.Module, armor_type: type = BaseArmor) -> None:
        self._armor = {}
        self._armor['main'] = armor_type(model)
        self._armor_type = armor_type

    def add_model(self, name: str, model: Optional[torch.nn.Module] = None, **kwargs) -> None:
        assert name in model_plugin_cfg_set
        if model is None:
            model = copy.deepcopy(self._armor['main'].model)
        self._armor[name] = self._armor_type(model)
        self.add_plugin(name, name, **kwargs)

    def remove_model(self, name: str) -> None:
        if name not in self._armor:
            raise KeyError("armor doesn't have model named {}".format(name))
        self._armor.pop(name)

    def add_plugin(self, armor_name: str, plugin_name: str, **kwargs) -> None:
        add_plugin(self._armor[armor_name], plugin_name, **kwargs)

    def remove_plugin(self, armor_name: str, plugin_name: str) -> None:
        raise NotImplementedError

    def __getattr__(self, key: str) -> Any:
        r"""
        Overview:
            Get the attrbute in armor(model).
        Arguments:
            - key (:obj:`str`): The key to query.
        Returns:
            - ret (:obj:`Any`): The queried attribute.

        .. note::
            If you want to get the attribute ``attr`` in ``armor[k]``, you should query "{k}_{attr}".
        """
        if len(self._armor) == 1:
            return getattr(self._armor['main'], key)
        else:
            name = 'main'
            for k in self._armor:
                if key.startswith(k):
                    name = k
                    key = key.split(k + '_')[1]
                    break
            return getattr(self._armor[name], key)
