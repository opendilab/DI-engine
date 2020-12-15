import os.path as osp
import torch

from nervex.model import ATOCQAC
from nervex.utils import deep_merge_dicts
from nervex.worker.learner import BaseLearner, register_learner
from nervex.worker.agent import create_qac_learner_agent
from app_zoo.multiagent_particle.envs import ParticleEnv
from app_zoo.multiagent_particle.computation_graph.particle_atoc_computation_graph import ParticleAtocGraph


class ParticleAtocLearner(BaseLearner):
    _name = "ParticleAtocLearner"

    def __init__(self, cfg: dict):
        super(ParticleAtocLearner, self).__init__(cfg)

    def _setup_agent(self):
        particle_env = ParticleEnv(self._cfg.env)
        # get env infos
        env_info = particle_env.info()
        obs_sp = env_info.obs_space.get('agent0')
        act_sp = env_info.act_space.get('agent0')
        action_dim = act_sp.value['max'] + 1 - act_sp.value['min']
        obs_dim = obs_sp.shape[0]
        n_agent = particle_env.agent_num
        m_group = self._cfg.env.get("m_group")
        if not m_group:
            m_group = n_agent // 2
        t_initate = self._cfg.env.get("t_initate")
        thought_dim = self._cfg.env.get("thought_dim")
        if not thought_dim:
            thought_dim = ((action_dim + obs_dim) // 2) * 2
        model = ATOCQAC(
            obs_dim,
            action_dim,
            thought_dim,
            n_agent,
            m_group,
            t_initate,
        )
        if self._cfg.learner.use_cuda:
            model.cuda()
        self._agent = create_qac_learner_agent(model, self._cfg.learner.atoc.use_noise, env_info.act_space.value)
        self._agent.mode(train=True)
        self._agent.target_mode(train=True)

    def _setup_computation_graph(self):
        self._computation_graph = ParticleAtocGraph(self._cfg.learner)

    def _setup_optimizer(self) -> None:
        """
        Overview:
            Setup learner's optimizer and lr_scheduler.
            ATOC can set different learning rate for critic and actor network, so it overwrites base learner.

        .. note:: AtocNetwork's attention unit may need a different optimizer or not
        """
        # TODO consider whether we need a different optimizer for attention unit
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


register_learner('particle_atoc', ParticleAtocLearner)
