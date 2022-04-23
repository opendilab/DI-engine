from typing import List, Dict, Any, Tuple, Union
import copy
import torch

from ding.torch_utils import Adam, to_device
from ding.rl_utils import fqf_nstep_td_data, fqf_nstep_td_error, fqf_calculate_fraction_loss, evaluate_quantile_at_action, get_train_sample, get_nstep_return_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .dqn import DQNPolicy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('fqf')
class FQFPolicy(DQNPolicy):
    r"""
    Overview:
        Policy class of FQF algorithm.

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      fqf          | RL policy register name, refer to      | this arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | this arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4  ``priority``         bool     True           | Whether use priority(PER)              | priority sample,
                                                                                                 | update priority
        6  | ``other.eps``      float    0.05           | Start value for epsilon decay. It's
           | ``.start``                                 | small because rainbow use noisy net.
        7  | ``other.eps``      float    0.05           | End value for epsilon decay.
           | ``.end``
        8  | ``discount_``      float    0.97,          | Reward's future discount factor, aka.  | may be 1 when sparse
           | ``factor``                  [0.95, 0.999]  | gamma                                  | reward env
        9  ``nstep``            int      3,             | N-step reward discount sum for target
                                         [3, 5]         | q_value estimation
        10 | ``learn.update``   int      3              | How many updates(iterations) to train  | this args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        11 ``learn.kappa``      float    /              | Threshold of Huber loss
        == ==================== ======== ============== ======================================== =======================
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='fqf',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        # (float) Reward's future discount factor, aka. gamma.
        discount_factor=0.97,
        # (int) N-step reward for target q_value estimation
        nstep=1,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=3,
            batch_size=64,
            learning_rate_fraction=0.001,
            learning_rate_quantile=0.001,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (float) Threshold of Huber loss. In the FQF paper, this is denoted by kappa. Default to 1.0.
            kappa=1.0,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_step, n_episode] shoule be set
            # n_sample=8,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                # (int) Decay length(env step)
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=10000, )
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config, main and target models.
        """
        self._priority = self._cfg.priority
        # Optimizer
        self._fraction_loss_optimizer = Adam(self._model.head.quantiles_proposal.parameters(), lr=self._cfg.learn.learning_rate_fraction)
        self._quantile_loss_optimizer = Adam(
            list(self._model.head.Q.parameters()) + list(self._model.head.fqf_fc.parameters()) +
            list(self._model.encoder.parameters()),
            lr=self._cfg.learn.learning_rate_quantile
        )

        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep
        self._kappa = self._cfg.learn.kappa

        # use wrapper instead of plugin
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
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        data = default_preprocess_learn(
            data, use_priority=self._priority, ignore_done=self._cfg.learn.ignore_done, use_nstep=True
        )
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # Q-learning forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        # Current q value (main model)
        ret = self._learn_model.forward(data['obs'])
        q_value = ret['q']  # [batch, num_quantiles, 64(action_dim)]
        quantiles = ret['quantiles']  # [batch, num_quantiles+1]
        quantiles_hats = ret['quantiles_hats']  # [batch, num_quantiles]
        q_tau_i = ret['q_tau_i']

        # Target q value
        with torch.no_grad():
            # quantiles_hats = (quantiles[:, :-1] + quantiles[:, 1:]) / 2.     # [batch_size, num_quantiles]
            target_q_value = self._target_model.forward(data['next_obs'])['q']
            # Max q value action (main model)
            target_q_action = self._learn_model.forward(data['next_obs'])['action']

            q_s_a_hats = evaluate_quantile_at_action(q_value, data['action'])  # [batch_size, num_quantiles, 1]

        data_n = fqf_nstep_td_data(
            q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], quantiles_hats,
            data['weight']
        )
        value_gamma = data.get('value_gamma')

        fraction_loss = fqf_calculate_fraction_loss(q_tau_i.detach(), q_s_a_hats.detach(), quantiles, data['action'])

        quantile_loss, td_error_per_sample = fqf_nstep_td_error(
            data_n, self._gamma, nstep=self._nstep, kappa=self._kappa, value_gamma=value_gamma
        )

        # ====================
        # fraction_proposal network update
        # ====================
        self._fraction_loss_optimizer.zero_grad()
        fraction_loss.backward(retain_graph=True)
        if self._cfg.learn.multi_gpu:
            self.sync_gradients(self._learn_model)
        self._fraction_loss_optimizer.step()

        # ====================
        # Q-learning update
        # ====================
        self._quantile_loss_optimizer.zero_grad()
        quantile_loss.backward()
        if self._cfg.learn.multi_gpu:
            self.sync_gradients(self._learn_model)
        self._quantile_loss_optimizer.step()

        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr_fraction_loss': self._fraction_loss_optimizer.defaults['lr'],
            'cur_lr_quantile_loss': self._quantile_loss_optimizer.defaults['lr'],
            'fraction_loss': fraction_loss.item(),
            'quantile_loss': quantile_loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
            # Only discrete action satisfying len(data['action'])==1 can return this and draw histogram on tensorboard.
            # '[histogram]action_distribution': data['action'],
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_fraction_loss': self._fraction_loss_optimizer.state_dict(),
            'optimizer_quantile_loss': self._quantile_loss_optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        #self._optimizer.load_state_dict(state_dict['optimizer'])
        self._fraction_loss_optimizer.load_state_dict(state_dict['optimizer_fraction_loss'])
        self._quantile_loss_optimizer.load_state_dict(state_dict['optimizer_quantile_loss'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
            Enable the eps_greedy_sample
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.discount_factor  # necessary for parallel
        self._nstep = self._cfg.nstep  # necessary for parallel
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data
        Arguments:
            - data (:obj:`list`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'fqf', ['ding.model.template.q_learning']
