"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Alphastar implementation for supervised learning on linklink, including basic processes.
"""

import os.path as osp

from nervex.data import default_collate
from nervex.model import FCDQN
from nervex.utils import override, merge_dicts, read_config, DistModule
from nervex.worker.learner import BaseLearner, register_learner
from app_zoo.sumo.envs.sumo_env import SumoWJ3Env
from app_zoo.sumo.computation_graph.sumo_dqn_computation_graph import SumoDqnGraph
from app_zoo.sumo.worker.agent.sumo_dqn_agent import SumoDqnLearnerAgent

default_config = read_config(osp.join(osp.dirname(__file__), "sumo_dqn_learner_default_config.yaml"))


class SumoDqnLearner(BaseLearner):
    _name = "SumoDqnLearner"

    def __init__(self, cfg: dict):
        cfg = merge_dicts(default_config, cfg)
        super(SumoDqnLearner, self).__init__(cfg)

    @override(BaseLearner)
    def _setup_data_source(self):
        self._collate_fn = default_collate
        batch_size = self._cfg.learner.batch_size

        def iterator():
            while True:
                data = self.get_data(batch_size)
                yield self._collate_fn(data)

        self._data_source = iterator()

    @override(BaseLearner)
    def _setup_agent(self):
        sumo_env = SumoWJ3Env(self._cfg.env)
        model = FCDQN(
            sumo_env.info().obs_space.shape, [v for k, v in sumo_env.info().act_space.shape.items()],
            dueling=self._cfg.learner.dqn.dueling
        )
        if self._cfg.learner.use_cuda:
            model.cuda()
        if self.use_distributed:
            model = DistModule(model)
        self._agent = SumoDqnLearnerAgent(model, plugin_cfg={'is_double': self._cfg.learner.dqn.is_double})
        self._agent.mode(train=True)
        if self._agent.is_double:
            self._agent.target_mode(train=True)

    @override(BaseLearner)
    def _setup_computation_graph(self):
        self._computation_graph = SumoDqnGraph(self._cfg.learner)


register_learner('sumo_dqn', SumoDqnLearner)
