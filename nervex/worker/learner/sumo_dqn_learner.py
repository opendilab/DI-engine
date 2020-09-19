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
from nervex.worker.agent.sumo_dqn_agent import SumoDqnLearnerAgent
from nervex.model.sumo_dqn.sumo_dqn_network import FCDQN
from nervex.envs.sumo.sumo_env import SumoWJ3Env
from nervex.computation_graph.sumo_dqn_computation_graph import SumoDqnGraph

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
    def _setup_agent(self):
        sumo_env = SumoWJ3Env(self._cfg.env)
        model = FCDQN(sumo_env.info().obs_space.shape, [v for k, v in sumo_env.info().act_space.shape.items()])
        if self._cfg.learner.use_cuda:
            model.cuda()
        self._agent = SumoDqnLearnerAgent(model, plugin_cfg={'is_double': self._cfg.learner.dqn.is_double})
        self._agent.mode(train=True)
        if self._agent.is_double:
            self._agent.target_mode(train=True)

    @override(BaseLearner)
    def _setup_computation_graph(self):
        self._computation_graph = SumoDqnGraph(self._cfg.learner)

    @override(BaseLearner)
    def _setup_optimizer(self):
        self._optimizer = torch.optim.Adam(
            self._agent.model.parameters(),
            lr=self._cfg.learner.learning_rate,
            weight_decay=self._cfg.learner.weight_decay
        )
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=[], gamma=1)
