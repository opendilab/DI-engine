from typing import List, Dict, Any
import copy
import torch

from ding.torch_utils import Adam, to_device
from ding.rl_utils import m_q_1step_td_data, m_q_1step_td_error
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY

from .dqn import DQNPolicy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('mdqn')
class MDQNPolicy(DQNPolicy):
    """
    Overview:
        Policy class of Munchausen DQN algorithm, extended by auxiliary objectives.
        Paper link: https://arxiv.org/abs/2007.14430.
    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      mdqn           | RL policy register name, refer to      | This arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | This arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4  ``priority``         bool     False          | Whether use priority(PER)              | Priority sample,
                                                                                                 | update priority
        5  | ``priority_IS``    bool     False          | Whether use Importance Sampling Weight
           | ``_weight``                                | to correct biased update. If True,
                                                        | priority must be True.
        6  | ``discount_``      float    0.97,          | Reward's future discount factor, aka.  | May be 1 when sparse
           | ``factor``                  [0.95, 0.999]  | gamma                                  | reward env
        7  ``nstep``            int      1,             | N-step reward discount sum for target
                                         [3, 5]         | q_value estimation
        8  | ``learn.update``   int      1              | How many updates(iterations) to train  | This args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
           | ``_gpu``
        10 | ``learn.batch_``   int      32             | The number of samples of an iteration
           | ``size``
        11 | ``learn.learning`` float    0.001          | Gradient step length of an iteration.
           | ``_rate``
        12 | ``learn.target_``  int      2000           | Frequence of target network update.    | Hard(assign) update
           | ``update_freq``
        13 | ``learn.ignore_``  bool     False          | Whether ignore done for target value   | Enable it for some
           | ``done``                                   | calculation.                           | fake termination env
        14 ``collect.n_sample`` int      4              | The number of training samples of a    | It varies from
                                                        | call of collector.                     | different envs
        15 | ``collect.unroll`` int      1              | unroll length of an iteration          | In RNN, unroll_len>1
           | ``_len``
        16 | ``other.eps.type`` str      exp            | exploration rate decay type            | Support ['exp',
                                                                                                 | 'linear'].
        17 | ``other.eps.``     float    0.01           | start value of exploration rate        | [0,1]
           | ``start``
        18 | ``other.eps.``     float    0.001          | end value of exploration rate          | [0,1]
           | ``end``
        19 | ``other.eps.``     int      250000         | decay length of exploration            | greater than 0. set
           | ``decay``                                                                           | decay=250000 means
                                                                                                 | the exploration rate
                                                                                                 | decay from start
                                                                                                 | value to end value
                                                                                                 | during decay length.
        20 | ``entropy_tau``    float    0.003          | the ration of entropy in TD loss
        21 | ``alpha``          float    0.9            | the ration of Munchausen term to the
                                                        | TD loss
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='mdqn',
        # (bool) Whether to use cuda in policy.
        cuda=False,
        # (bool) Whether learning policy is the same as collecting data policy(on-policy).
        on_policy=False,
        # (bool) Whether to enable priority experience sample.
        priority=False,
        # (bool) Whether to use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (float) Discount factor(gamma) for returns.
        discount_factor=0.97,
        # (float) Entropy factor (tau) for Munchausen DQN.
        entropy_tau=0.03,
        # (float) Discount factor (alpha) for Munchausen term.
        m_alpha=0.9,
        # (int) The number of step for calculating target q_value.
        nstep=1,
        # learn_mode config
        learn=dict(
            # (int) How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=3,
            # (int) How many samples in a training batch
            batch_size=64,
            # (float) The step size of gradient descent
            learning_rate=0.001,
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (bool) Whether ignore done(usually for max step termination env).
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # Only one of [n_sample, n_episode] shoule be set.
            n_sample=4,
            # (int) Split episodes or trajectories into pieces with length `unroll_len`.
            unroll_len=1,
        ),
        eval=dict(),  # for compability
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                # (float) Epsilon start value.
                start=0.95,
                # (float) Epsilon end value.
                end=0.1,
                # (int) Decay length(env step).
                decay=10000,
            ),
            replay_buffer=dict(
                # (int) Maximum size of replay buffer. Usually, larger buffer size is better.
                replay_buffer_size=10000,
            ),
        ),
    )

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For MDQN, it contains \
            optimizer, algorithm-specific arguments such as entropy_tau, m_alpha and nstep, main and target model.
            This method will be called in ``__init__`` method if ``learn`` field is in ``enable_field``.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        .. note::
            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.

        .. note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        # Optimizer
        # set eps in order to consistent with the original paper implementation
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate, eps=0.0003125)

        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep
        self._entropy_tau = self._cfg.entropy_tau
        self._m_alpha = self._cfg.m_alpha

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        if 'target_update_freq' in self._cfg.learn:
            self._target_model = model_wrap(
                self._target_model,
                wrapper_name='target',
                update_type='assign',
                update_kwargs={'freq': self._cfg.learn.target_update_freq}
            )
        elif 'target_theta' in self._cfg.learn:
            self._target_model = model_wrap(
                self._target_model,
                wrapper_name='target',
                update_type='momentum',
                update_kwargs={'theta': self._cfg.learn.target_theta}
            )
        else:
            raise RuntimeError("DQN needs target network, please either indicate target_update_freq or target_theta")
        self._learn_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._learn_model.reset()
        self._target_model.reset()

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, action_gap, clip_frac, priority.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For MDQN, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
                ``reward``, ``next_obs``, ``done``. Sometimes, it also contains other keys such as ``weight`` \
                and ``value_gamma``.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The information dict that indicated training result, which will be \
                recorded in text log and tensorboard, values must be python scalar or a list of scalars. For the \
                detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for MDQNPolicy: ``ding.policy.tests.test_mdqn``.
        """
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=True
        )
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # Q-learning forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        # Current q value (main model)
        q_value = self._learn_model.forward(data['obs'])['logit']
        # Target q value
        with torch.no_grad():
            target_q_value_current = self._target_model.forward(data['obs'])['logit']
            target_q_value = self._target_model.forward(data['next_obs'])['logit']

        data_m = m_q_1step_td_data(
            q_value, target_q_value_current, target_q_value, data['action'], data['reward'].squeeze(0), data['done'],
            data['weight']
        )

        loss, td_error_per_sample, action_gap, clipfrac = m_q_1step_td_error(
            data_m, self._gamma, self._entropy_tau, self._m_alpha
        )
        # ====================
        # Q-learning update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        self._optimizer.step()

        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'q_value': q_value.mean().item(),
            'target_q_value': target_q_value.mean().item(),
            'priority': td_error_per_sample.abs().tolist(),
            'action_gap': action_gap.item(),
            'clip_frac': clipfrac.mean().item(),
        }

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return ['cur_lr', 'total_loss', 'q_value', 'action_gap', 'clip_frac']
