from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch

from nervex.torch_utils import Adam
from nervex.rl_utils import dist_nstep_td_data, dist_nstep_td_error, Adder, iqn_nstep_td_data, iqn_nstep_td_error
from nervex.model import NoiseDistributionFCDiscreteNet, NoiseQuantileFCDiscreteNet
from nervex.agent import Agent
from .base_policy import register_policy
from .dqn import DQNPolicy


class RainbowDQNPolicy(DQNPolicy):
    r"""
    Overview:
        Rainbow DQN contain several improvements upon DQN, including:
            - target network
            - dueling architectur
            - prioritized experience replay
            - n_step return
            - noise net
            - distribution net

        Therefore, the RainbowDQNPolicy class inherit upon DQNPolicy class
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Init the learner agent of RainbowDQNPolicy

        Arguments:
            .. note::

                the _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): the learning rate fo the optimizer
            - gamma (:obj:`float`): the discount factor
            - nstep (:obj:`int`): the num of n step return
            - v_min (:obj:`float`): value distribution minimum value
            - v_max (:obj:`float`): value distribution maximum value
            - n_atom (:obj:`int`): the number of atom sample point
        """
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._agent = Agent(self._model)
        algo_cfg = self._cfg.learn.algo
        self._use_iqn = algo_cfg.get('use_iqn', False)
        self._gamma = algo_cfg.discount_factor
        self._nstep = algo_cfg.nstep
        if self._use_iqn:
            self._kappa = algo_cfg.kappa
            self._tau = algo_cfg.tau_num
            self._tau_prim = algo_cfg.tau_prim_num
            self._num_quantiles = algo_cfg.quantile_num

        else:
            self._v_max = self._cfg.model.v_max
            self._v_min = self._cfg.model.v_min
            self._n_atom = self._cfg.model.n_atom

        self._agent.add_model('target', update_type='assign', update_kwargs={'freq': algo_cfg.target_update_freq})
        self._agent.add_plugin('main', 'argmax_sample')
        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.add_plugin('target', 'grad', enable_grad=False)
        self._agent.mode(train=True)
        self._agent.target_mode(train=True)
        self._agent.reset()
        self._agent.target_reset()
        self._learn_setting_set = {}

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        """
        Overview:
            Forward and backward function of learn mode, acquire the data and calculate the loss and\
            optimize learner model

        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'next_obs', 'reward', 'action']

        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including cur_lr and total_loss
                - cur_lr (:obj:`float`): current learning rate
                - total_loss (:obj:`float`): the calculated loss
        """
        reward = data['reward']
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        assert reward.shape == (self._cfg.learn.batch_size, self._nstep), reward.shape
        reward = reward.permute(1, 0).contiguous()
        self._reset_noise(self._agent.model)
        self._reset_noise(self._agent.target_model)
        if self._use_iqn:
            ret = self._agent.forward(data['obs'], param={'num_quantiles': self._tau})
            q = ret['q']
            replay_quantiles = ret['quantiles']
            target_q = self._agent.target_forward(data['next_obs'], param={'num_quantiles': self._tau_prim})['q']
            self._reset_noise(self._agent.target_model)
            target_q_action = self._agent.forward(
                data['next_obs'], param={'num_quantiles': self._num_quantiles}
            )['action']
            data = iqn_nstep_td_data(
                q, target_q, data['action'], target_q_action, reward, data['done'], replay_quantiles, data['weight']
            )
            loss, td_error_per_sample = iqn_nstep_td_error(data, self._gamma, nstep=self._nstep, kappa=self._kappa)
        else:
            q_dist = self._agent.forward(data['obs'])['distribution']
            target_q_dist = self._agent.target_forward(data['next_obs'])['distribution']
            self._reset_noise(self._agent.target_model)
            target_q_action = self._agent.forward(data['next_obs'])['action']
            data = dist_nstep_td_data(
                q_dist, target_q_dist, data['action'], target_q_action, reward, data['done'], data['weight']
            )
            loss, td_error_per_sample = dist_nstep_td_error(
                data, self._gamma, self._v_min, self._v_max, self._n_atom, nstep=self._nstep
            )
        # update
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # after update
        self._agent.target_update(self._agent.state_dict()['model'])
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
        }

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init moethod. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect agent.

            .. note::
                the rainbow dqn enable the eps_greedy_sample, but might not need to use it, \
                    as the noise_net contain noise that can help exploration
        """
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

    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        r"""
        Overview:
            Reset the noise from noise net and collect output according to eps_greedy plugin

        Arguments:
            - data_id (:obj:`List` of :obj:`int`): Not used, set in arguments for consistency
            - data (:obj:`dict`): Dict type data, including at least ['obs'].

        Returns:
            - data (:obj:`dict`): The collected data
        """
        self._reset_noise(self._collect_agent.model)
        return self._collect_agent.forward(data, eps=self._eps)

    def _get_train_sample(self, traj: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data

        Arguments:
            - traj (:obj:`deque`): The trajactory's cache

        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        # adder is defined in _init_collect
        data = self._adder.get_traj(traj, self._traj_len, return_num=self._collect_nstep)
        data = self._adder.get_nstep_return_data(data, self._collect_nstep, self._traj_len)
        return self._adder.get_train_sample(data)

    def _create_model_from_cfg(self, cfg: dict, model_type: Optional[type] = None) -> torch.nn.Module:
        """
        Overview:
            Create a model according to input config. Defalut use NoiseDistributionFCDiscreteNet for 1 dim obs

        Arguments:
            - cfg (:obj:`dict`): Config, including the config contain model parameters
            - model_type (:obj:`type` or None): The type of the model to create, if this is not None, this\
                function will create an instance of the model_type.

        Returns:
            - model (:obj:`torch.nn.Module`): Generted model.
        """
        if model_type is None:
            if cfg.learn.algo.get('use_iqn'):
                return NoiseQuantileFCDiscreteNet(**cfg.model)
            else:
                return NoiseDistributionFCDiscreteNet(**cfg.model)
        else:
            return model_type(**cfg.model)

    def _reset_noise(self, model: torch.nn.Module):
        r"""
        Overview:
            Reset the noise of model

        Arguments:
            - model (:obj:`torch.nn.Module`): the model to reset, must contain reset_noise method
        """
        for m in model.modules():
            if hasattr(m, 'reset_noise'):
                m.reset_noise()


# regist rainbow_dqn policy in the policy maps
register_policy('rainbow_dqn', RainbowDQNPolicy)
