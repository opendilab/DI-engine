import os.path as osp

from nervex.model import FCQAC
from nervex.utils import deep_merge_dicts
from nervex.utils import override, read_config, DistModule
from nervex.worker.learner import BaseLearner, register_learner
from nervex.worker.agent import create_qac_learner_agent
from app_zoo.classic_control.pendulum.envs import PendulumEnv
from app_zoo.classic_control.pendulum.computation_graph.pendulum_ddpg_computation_graph import PendulumDdpgGraph

default_config = read_config(osp.join(osp.dirname(__file__), "pendulum_ddpg_learner_default_config.yaml"))


class PendulumDdpgLearner(BaseLearner):
    _name = "PendulumDdpgLearner"

    def __init__(self, cfg: dict):
        cfg = deep_merge_dicts(default_config, cfg)
        super(PendulumDdpgLearner, self).__init__(cfg)

    @override(BaseLearner)
    def _setup_agent(self):
        pendulum_env = PendulumEnv(self._cfg.env)
        env_info = pendulum_env.info()
        model = FCQAC(
            env_info.obs_space.shape, len(env_info.act_space.shape), env_info.act_space.value,
            use_twin_critic=True
        )
        if self._cfg.learner.use_cuda:
            model.cuda()
        if self.use_distributed:
            model = DistModule(model)
        self._agent = create_qac_learner_agent(model, self._cfg.learner.ddpg.is_double)
        self._agent.mode(train=True)
        if self._agent.is_double:
            self._agent.target_mode(train=True)

    @override(BaseLearner)
    def _setup_computation_graph(self):
        self._computation_graph = PendulumDdpgGraph(self._cfg.learner)


register_learner('pendulum_ddpg', PendulumDdpgLearner)
