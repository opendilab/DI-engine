from abc import ABC
import copy
from collections import OrderedDict
from typing import Any, Union, Optional

import torch

from .agent_plugin import register_plugin


class BaseAgent(ABC):

    def __init__(self, model: torch.nn.Module, pipeline_plugin_cfg: Optional[OrderedDict]) -> None:
        self._model = model
        register_plugin(self, pipeline_plugin_cfg)

    def forward(self, data: Any, param: Optional[dict] = None) -> Any:
        if param is not None:
            return self._model(data, **param)
        else:
            return self._model(data)

    def mode(self, train: bool) -> None:
        if train:
            self._model.train()
        else:
            self._model.eval()

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def state_dict(self) -> dict:
        return {'model': self._model.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self._model.load_state_dict(state_dict['model'])

    def reset(self) -> None:
        pass


model_plugin_cfg_set = set(['target'])


class AgentAggregator(object):

    def __init__(self, agent_type: type, model: torch.nn.Module, plugin_cfg: OrderedDict) -> None:
        assert issubclass(agent_type, BaseAgent)
        # TODO(nyz) different model and different pipeline_plugin_cfg
        model_plugin_cfg = {}
        for k in plugin_cfg:
            if k in model_plugin_cfg_set:
                model_plugin_cfg[k] = plugin_cfg.pop(k)
        self._agent = {}
        self._agent['main'] = agent_type(model, plugin_cfg)
        for k in model_plugin_cfg:
            # other agent default disenable grad
            plugin_cfg['grad'] = {'enable_grad': False}
            self._agent[k] = agent_type(copy.deepcopy(model), plugin_cfg)
            register_plugin(self._agent[k], OrderedDict({k: model_plugin_cfg[k]}))

    def __getattr__(self, key: str) -> Any:
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
