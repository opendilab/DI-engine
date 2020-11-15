import os.path as osp

from app_zoo.gfootball.computation_graph.gfootball_iql_computation_graph import GfootballIqlGraph
from nervex.data import default_collate
from app_zoo.gfootball.model.iql.iql_network import FootballIQL
from nervex.utils import deep_merge_dicts
from nervex.utils import override, read_config, DistModule
from app_zoo.gfootball.worker.agent.gfootball_agent import GfootballIqlLearnerAgent
from nervex.worker.learner import BaseLearner, register_learner
from nervex.torch_utils import CudaFetcher

default_learner_config = read_config(osp.join(osp.dirname(__file__), "gfootball_iql_learner_default_config.yaml"))


class GfootballIqlLearner(BaseLearner):
    _name = "GfootballIqlLearner"

    def __init__(self, cfg: dict):
        cfg = deep_merge_dicts(default_learner_config, cfg)
        super(GfootballIqlLearner, self).__init__(cfg)

    @override(BaseLearner)
    def _setup_agent(self):
        model = FootballIQL(cfg=self._cfg.model)
        if self._cfg.learner.use_cuda:
            model.cuda()
        if self.use_distributed:
            model = DistModule(model)
        self._agent = GfootballIqlLearnerAgent(model, plugin_cfg={'is_double': self._cfg.learner.dqn.is_double})
        self._agent.mode(train=True)
        if self._agent.is_double:
            self._agent.target_mode(train=True)

    @override(BaseLearner)
    def _setup_computation_graph(self):
        self._computation_graph = GfootballIqlGraph(self._cfg.learner)


register_learner('gfootball_dqn', GfootballIqlLearner)
