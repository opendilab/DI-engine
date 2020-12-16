from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple, deque
import torch

from nervex.torch_utils import Adam
from nervex.rl_utils import dist_nstep_td_data, dist_nstep_td_error, Adder
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
        self._nstep = algo_cfg.nstep
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
        reward = data['reward']
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        assert reward.shape == (self._cfg.learn.batch_size, self._nstep), reward.shape
        reward = reward.permute(1, 0).contiguous()
        self._reset_noise(self._agent.model)
        self._reset_noise(self._agent.target_model)
        q_dist = self._agent.forward(data['obs'])['distribution']
        target_q_dist = self._agent.target_forward(data['next_obs'])['distribution']
        data = dist_nstep_td_data(q_dist, target_q_dist, data['action'], reward, data['done'], data['weight'])
        loss = dist_nstep_td_error(data, self._gamma, self._v_min, self._v_max, self._n_atom, nstep=self._nstep)
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
        self._traj_len = self._cfg.collect.traj_len
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._use_cuda, self._unroll_len)
        self._collect_nstep = self._cfg.collect.algo.nstep
        self._collect_agent = Agent(self._model)
        self._collect_agent.add_plugin('main', 'grad', enable_grad=False)
        self._collect_agent.add_plugin('main', 'eps_greedy_sample')
        self._collect_agent.mode(train=True)
        self._collect_agent.reset()
        self._collect_setting_set = {'eps'}

    def _forward_collect(self, data: dict) -> dict:
        self._reset_noise(self._collect_agent.model)
        return self._collect_agent.forward(data, eps=self._eps)

    def _get_train_sample(self, traj_cache: deque, data_id: int) -> Union[None, List[Any]]:
        # adder is defined in _init_collect
        data = self._adder.get_traj(traj_cache, data_id, self._traj_len, return_num=self._nstep)
        data = self._adder.get_nstep_return(data, self._nstep, self._traj_len)
        return self._adder.get_train_sample(data)

    def _create_model_from_cfg(self, cfg: dict) -> torch.nn.Module:
        return NoiseDistributionFCDiscreteNet(**cfg.model)

    def _reset_noise(self, model: torch.nn.Module):
        for m in model.modules():
            if hasattr(m, 'reset_noise'):
                m.reset_noise()


register_policy('rainbow_dqn', RainbowDQNPolicy)
