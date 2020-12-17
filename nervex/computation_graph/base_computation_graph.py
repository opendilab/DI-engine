"""
Copyright 2020 Sensetime X-lab. All Rights Reserved
"""
from abc import ABC, abstractmethod
from typing import Any


class BaseCompGraph(ABC):

    @abstractmethod
    def forward(self, data: Any, agent: Any) -> dict:
        raise NotImplementedError

    def register_stats(self, tb_logger: 'TensorBoardLogger'):  # noqa
        """Input variable record and tensorboard logger. Return nothing."""
        tb_logger.register_var('total_loss_avg')
        # subclass can override this method to extend its own statistics value.

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
