from abc import ABC, abstractmethod
from typing import Any, Tuple, Callable
import torch
from .agent_plugin import register_plugin


class BaseAgent(ABC):
    def __init__(self, model: torch.nn.Module, plugin_cfg) -> None:
        self._model = model
        register_plugin(self, plugin_cfg)

    def forward(self, data: Any, param: dict) -> Any:
        return self._model(data, param)

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
