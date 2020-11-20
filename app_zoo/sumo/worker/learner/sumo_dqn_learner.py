"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Alphastar implementation for supervised learning on linklink, including basic processes.
"""

import os.path as osp

from nervex.model import FCDQN
from nervex.utils import deep_merge_dicts
from nervex.utils import override, read_config, DistModule
from nervex.worker.learner import BaseLearner, register_learner
from nervex.worker.agent import create_dqn_learner_agent
from app_zoo.sumo.envs.sumo_env import SumoWJ3Env
from app_zoo.sumo.computation_graph.sumo_dqn_computation_graph import SumoDqnGraph

default_config = read_config(osp.join(osp.dirname(__file__), "sumo_dqn_learner_default_config.yaml"))


class SumoDqnLearner(BaseLearner):
    _name = "SumoDqnLearner"

    def __init__(self, cfg: dict):
        cfg = deep_merge_dicts(default_config, cfg)
        super(SumoDqnLearner, self).__init__(cfg)

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
        self._agent = create_dqn_learner_agent(model, self._cfg.learner.dqn.is_double)
        self._agent.mode(train=True)
        if self._agent.is_double:
            self._agent.target_mode(train=True)

    @override(BaseLearner)
    def _setup_computation_graph(self):
        self._computation_graph = SumoDqnGraph(self._cfg.learner)


register_learner('sumo_dqn', SumoDqnLearner)
