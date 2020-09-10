"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Alphastar implementation for supervised learning on linklink, including basic processes.
"""

import os.path as osp
import torch
import torch.nn as nn
import threading
import yaml
from easydict import EasyDict
from collections import OrderedDict

from nervex.utils import override, merge_dicts, pretty_print, read_config
from nervex.worker import BaseLearner
from nervex.computation_graph.sumo_dqn_computation_graph import SumoDqnGraph
from nervex.data.collate_fn import sumo_dqn_collect_fn

default_config = read_config(osp.join(osp.dirname(__file__), "sumo_dqn_learner_default_config.yaml"))


class SumoDqnLearner(BaseLearner):
    _name = "SumoDqnLearner"

    def __init__(self, cfg: dict):
        cfg = merge_dicts(default_config, cfg)
        super(SumoDqnLearner, self).__init__(cfg)

    @override(BaseLearner)
    def _setup_data_source(self):
        pass

    @override(BaseLearner)
    def _setup_computation_graph(self):
        self._computation_graph = SumoDqnGraph(self._cfg.learner)

    @override(BaseLearner)
    def _setup_optimizer(self):
        self._optimizer = torch.optim.Adam(
            self._computation_graph.agent.model.parameters(),
            lr=self._cfg.learner.learning_rate,
            weight_decay=self._cfg.learner.weight_decay
        )
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=[], gamma=1)

    @property
    @override(BaseLearner)
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    @override(BaseLearner)
    def lr_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self._lr_scheduler

    @property
    @override(BaseLearner)
    def computation_graph(self) -> 'BaseCompGraph':  # noqa
        return self._computation_graph
