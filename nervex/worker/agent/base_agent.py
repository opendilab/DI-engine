from abc import ABC, abstractmethod
from typing import Any, Union, Optional
from collections import OrderedDict
import torch
from .agent_plugin import register_plugin


class BaseAgent(ABC):
    def __init__(self, model: torch.nn.Module, plugin_cfg: Union[OrderedDict, None]) -> None:
        self._model = model
        register_plugin(self, plugin_cfg)

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

    ##added
    def get_parameters(self):
        return self._model.parameters()

    def process_gradient(self):
        pass
