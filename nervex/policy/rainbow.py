from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
import copy

from nervex.torch_utils import Adam, to_device
from nervex.data import default_collate, default_decollate
from nervex.rl_utils import dist_nstep_td_data, dist_nstep_td_error, Adder, iqn_nstep_td_data, iqn_nstep_td_error
from nervex.model import NoiseDistributionFCDiscreteNet, NoiseQuantileFCDiscreteNet, model_wrap
from nervex.utils import POLICY_REGISTRY
from .dqn import DQNPolicy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('rainbow')
class RainbowDQNPolicy(DQNPolicy):
    r"""
    Overview:
        Rainbow DQN contain several improvements upon DQN, including:
            - target network
            - dueling architecture
            - prioritized experience replay
            - n_step return
            - noise net
            - distribution net

        Therefore, the RainbowDQNPolicy class inherit upon DQNPolicy class
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='rainbow',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=True,
        model=dict(
            # (float) Value of the smallest atom in the support set.
            # Default to -10.0.
            v_min=-10,
            # (float) Value of the smallest atom in the support set.
            # Default to 10.0.
            v_max=10,
            # (int) Number of atoms in the support set of the
            # value distribution. Default to 51.
            n_atom=51,
        ),
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            batch_size=32,
            learning_rate=0.001,
            # (float) L2 norm weight for network parameters.
            weight_decay=0.0,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # (int) N-step reward for target q_value estimation
            nstep=3,
            # (bool) Whether to use iqn for distribution loss
            iqn=False,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_step, n_episode] shoule be set
            n_sample=32,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # (int) Frequence of target network update.
            nstep=3,
        ),
        eval=dict(),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                # (float) End value for epsilon decay, in [0, 1].
                start=0.95,
                # (float) End value for epsilon decay, in [0, 1].
                end=0.05,
                # (int) Env steps of epsilon decay.
                decay=100000,
            ),
            replay_buffer=dict(
                # (int) Max size of replay buffer.
                replay_buffer_size=100000,
                # (int) Number of experiences in replay buffer when training starts. Defaults to 0.
                replay_buffer_start_size=2000,
                # (float) Prioritization exponent.
                alpha=0.6,
                # (float) Importance sample soft coefficient.
                # 0 means no correction, while 1 means full correction
                beta=0.4,
                # (int) Anneal step for beta: 0 means no annealing. Defaults to 0
                anneal_step=100000,
            )
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Init the learner model of RainbowDQNPolicy

        Arguments:
            - learning_rate (:obj:`float`): the learning rate fo the optimizer
            - gamma (:obj:`float`): the discount factor
            - nstep (:obj:`int`): the num of n step return
            - v_min (:obj:`float`): value distribution minimum value
            - v_max (:obj:`float`): value distribution maximum value
            - n_atom (:obj:`int`): the number of atom sample point
        """
        self._priority = self._cfg.priority
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._gamma = self._cfg.learn.discount_factor
        self._nstep = self._cfg.learn.nstep
        self._iqn = self._cfg.learn.iqn
        if self._iqn:
            self._huber_loss_threshold = self._cfg.learn.huber_loss_threshold
            self._quantile_thresholds_N = self._cfg.learn.quantile_thresholds_N
            self._quantile_thresholds_N_prime = self._cfg.learn.quantile_thresholds_N_prime
            self._quantile_thresholds_K = self._cfg.learn.quantile_thresholds_K
        else:
            self._v_max = self._cfg.model.v_max
            self._v_min = self._cfg.model.v_min
            self._n_atom = self._cfg.model.n_atom

        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.learn.target_update_freq}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._learn_model.reset()
        self._target_model.reset()

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
        data = default_preprocess_learn(
            data, use_priority=self._priority, ignore_done=self._cfg.learn.ignore_done, use_nstep=True
        )
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # Rainbow forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        # reset noise of noisenet for both main model and target model
        self._reset_noise(self._learn_model)
        self._reset_noise(self._target_model)
        if self._iqn:
            ret = self._learn_model.forward(data['obs'], num_quantiles=self._quantile_thresholds_N)
            q = ret['q']
            replay_quantiles = ret['quantiles']
            with torch.no_grad():
                target_q = self._target_model.forward(
                    data['next_obs'], num_quantiles=self._quantile_thresholds_N_prime
                )['q']
                self._reset_noise(self._learn_model)
                target_q_action = self._learn_model.forward(
                    data['next_obs'], num_quantiles=self._quantile_thresholds_K
                )['action']
            value_gamma = data.get('value_gamma', None)
            data = iqn_nstep_td_data(
                q, target_q, data['action'], target_q_action, data['reward'], data['done'], replay_quantiles,
                data['weight']
            )
            loss, td_error_per_sample = iqn_nstep_td_error(
                data, self._gamma, nstep=self._nstep, kappa=self._huber_loss_threshold, value_gamma=value_gamma
            )
        else:
            q_dist = self._learn_model.forward(data['obs'])['distribution']
            with torch.no_grad():
                target_q_dist = self._target_model.forward(data['next_obs'])['distribution']
                self._reset_noise(self._learn_model)
                target_q_action = self._learn_model.forward(data['next_obs'])['action']
            value_gamma = data.get('value_gamma', None)
            data = dist_nstep_td_data(
                q_dist, target_q_dist, data['action'], target_q_action, data['reward'], data['done'], data['weight']
            )
            loss, td_error_per_sample = dist_nstep_td_error(
                data, self._gamma, self._v_min, self._v_max, self._n_atom, nstep=self._nstep, value_gamma=value_gamma
            )
        # ====================
        # Rainbow update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
        }

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init moethod. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect model.

            .. note::
                the rainbow dqn enable the eps_greedy_sample, but might not need to use it, \
                    as the noise_net contain noise that can help exploration
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._cuda, self._unroll_len)
        self._nstep = self._cfg.collect.nstep
        self._gamma = self._cfg.collect.discount_factor
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, eps: float) -> dict:
        r"""
        Overview:
            Reset the noise from noise net and collect output according to eps_greedy plugin

        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].

        Returns:
            - data (:obj:`dict`): The collected data
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        self._reset_noise(self._collect_model)
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

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
        data = self._adder.get_nstep_return_data(traj, self._nstep, gamma=self._gamma)
        return self._adder.get_train_sample(data)

    def default_model(self) -> Tuple[str, List[str]]:
        if self._cfg.learn.iqn:
            return 'noise_quantile_fc', ['nervex.model.discrete_net.discrete_net']
        else:
            return 'noise_dist_fc', ['nervex.model.discrete_net.discrete_net']

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


@POLICY_REGISTRY.register('iqn')
class IQNPolicy(RainbowDQNPolicy):

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='iqn',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=True,
        model=dict(
            # (str) Type of beta function, chosen from ['uniform', 'CPW', 'CVaR','Pow']. Default to 'uniform'.
            beta_function_type='uniform',
        ),
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            batch_size=32,
            learning_rate=0.001,
            # (float) L2 norm weight for network parameters.
            weight_decay=0.0,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # (int) N-step reward for target q_value estimation
            nstep=3,
            # (bool) Whether to use iqn for distribution loss
            iqn=True,
            # (int) Number of quantile thresholds used in quantile regression. Default to 64.
            quantile_thresholds_N=64,
            # (int) Number of quantile thresholds used to sample from the return distribution at the next state.
            # Default to 32.
            quantile_thresholds_N_prime=32,
            # (int) Number of quantile thresholds used to compute greedy actions. Default to 32.
            quantile_thresholds_K=32,
            # (float) Threshold of Huber loss. In the IQN paper, this is denoted by kappa. Default to 1.0.
            huber_loss_threshold=1.0,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_step, n_episode] shoule be set
            n_sample=32,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # (int) Frequence of target network update.
            nstep=3,
        ),
        eval=dict(),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                # (float) End value for epsilon decay, in [0, 1].
                start=0.95,
                # (float) End value for epsilon decay, in [0, 1].
                end=0.05,
                # (int) Env steps of epsilon decay.
                decay=100000,
            ),
            replay_buffer=dict(
                # (int) Max size of replay buffer.
                replay_buffer_size=100000,
                # (int) Number of experiences in replay buffer when training starts. Defaults to 0.
                replay_buffer_start_size=2000,
                # (float) Prioritization exponent.
                alpha=0.6,
                # (float) Importance sample soft coefficient.
                # 0 means no correction, while 1 means full correction
                beta=0.4,
                # (int) Anneal step for beta: 0 means no annealing. Defaults to 0
                anneal_step=100000,
            )
        )
    )
