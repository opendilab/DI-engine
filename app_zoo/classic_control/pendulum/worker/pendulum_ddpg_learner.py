import os.path as osp
import torch

from nervex.model import QAC
from nervex.utils import deep_merge_dicts
from nervex.worker.learner import BaseLearner, register_learner
from nervex.worker.agent import create_qac_learner_agent
from app_zoo.classic_control.pendulum.envs import PendulumEnv
from app_zoo.classic_control.pendulum.computation_graph.pendulum_ddpg_computation_graph import PendulumDdpgGraph


class PendulumDdpgLearner(BaseLearner):
    _name = "PendulumDdpgLearner"

    def __init__(self, cfg: dict):
        super(PendulumDdpgLearner, self).__init__(cfg)

    def _setup_agent(self):
        pendulum_env = PendulumEnv(self._cfg.env)
        env_info = pendulum_env.info()
        model = QAC(
            env_info.obs_space.shape,
            len(env_info.act_space.shape),
            env_info.act_space.value,
            use_twin_critic=self._cfg.learner.ddpg.use_twin_critic
        )
        if self._cfg.learner.use_cuda:
            model.cuda()
        self._agent = create_qac_learner_agent(model, self._cfg.learner.ddpg.use_noise, env_info.act_space.value)
        self._agent.mode(train=True)
        self._agent.target_mode(train=True)

    def _setup_computation_graph(self):
        self._computation_graph = PendulumDdpgGraph(self._cfg.learner)

    def _setup_optimizer(self) -> None:
        """
        Overview:
            Setup learner's optimizer and lr_scheduler.
            DDPG can set different learning rate for critic and actor network, so it overwrites base learner.
        """
        self._optimizer = torch.optim.Adam(
            [
                {
                    'params': self._agent.model._critic.parameters(),
                    'lr': self._cfg.learner.critic_learning_rate
                },
                {
                    'params': self._agent.model._actor.parameters(),
                    'lr': self._cfg.learner.actor_learning_rate
                },
            ],
            weight_decay=self._cfg.learner.weight_decay
        )
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=[], gamma=1)


register_learner('pendulum_ddpg', PendulumDdpgLearner)
