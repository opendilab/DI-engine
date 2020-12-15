from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch

from nervex.torch_utils import Adam
from nervex.rl_utils import dist_1step_td_data, dist_1step_td_error, epsilon_greedy
from nervex.model import NoiseDistributionFCDiscreteNet
from nervex.agent import Agent
from .base_policy import register_policy
from .dqn import DQNPolicy


class RainbowDQNPolicy(DQNPolicy):

    def _init_learn(self) -> None:
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._agent = Agent(self._model)
        algo_cfg = self._cfg.learn.algo
        self._gamma = algo_cfg.discount_factor
        self._v_max = self._cfg.model.v_max
        self._v_min = self._cfg.model.v_min
        self._n_atom = self._cfg.model.n_atom

        self._agent.add_model('target', update_type='assign', update_kwargs={'freq': algo_cfg.target_update_freq})
        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.add_plugin('target', 'grad', enable_grad=False)
        self._agent.mode(train=True)
        self._agent.target_mode(train=True)
        self._agent.reset()
        self._agent.target_reset()
        self._learn_setting_set = {}

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        # forward
        self._reset_noise(self._agent.model)
        self._reset_noise(self._agent.target_model)
        q_dist = self._agent.forward(data['obs'])['distribution']
        target_q_dist = self._agent.target_forward(data['next_obs'])['distribution']
        data = dist_1step_td_data(q_dist, target_q_dist, data['action'], data['reward'], data['done'], data['weight'])
        loss = dist_1step_td_error(data, self._gamma, self._v_min, self._v_max, self._n_atom)
        # update
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # after update
        self._agent.target_update(self._agent.state_dict()['model'])
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
        }

    def _init_collect(self) -> None:
        self._get_traj_length = self._cfg.collect.get_traj_length
        self._collect_agent = Agent(self._model)
        self._collect_agent.add_plugin('main', 'grad', enable_grad=False)
        self._collect_agent.add_plugin('main', 'argmax_sample')
        self._collect_agent.mode(train=True)
        self._collect_agent.reset()
        self._collect_setting_set = {}

    def _forward_collect(self, data: dict) -> dict:
        self._reset_noise(self._collect_agent.model)
        return self._collect_agent.forward(data)

    def _init_command(self) -> None:
        pass

    def _get_setting_collect(self, command_info: dict) -> dict:
        return {}

    def _create_model_from_cfg(self, cfg: dict) -> torch.nn.Module:
        return NoiseDistributionFCDiscreteNet(**cfg.model)

    def _reset_noise(self, model: torch.nn.Module):
        for m in model.modules():
            if hasattr(m, 'reset_noise'):
                m.reset_noise()


register_policy('rainbow_dqn', RainbowDQNPolicy)
