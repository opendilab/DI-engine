"""
Copyright 2020 Sensetime X-lab. All Rights Reserved
"""
from abc import ABC, abstractmethod
from typing import Any


class BaseCompGraph(ABC):
    @abstractmethod
    def forward(self, data: Any) -> dict:
        raise NotImplementedError

    @abstractmethod
    def register_stats(self, variable_record: 'VariableRecord', tb_logger: 'TensorBoardLogger'):  # noqa
        """Input variable record and tensorboard logger. Return nothing."""
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def sync_gradients(self) -> None:
        raise NotImplementedError
