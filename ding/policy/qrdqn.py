from typing import List, Dict, Any, Tuple, Union
import copy
import torch

from ding.torch_utils import Adam, to_device
from ding.rl_utils import qrdqn_nstep_td_data, qrdqn_nstep_td_error, get_train_sample, get_nstep_return_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .dqn import DQNPolicy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('qrdqn')
class QRDQNPolicy(DQNPolicy):
    r"""
    Overview:
        Policy class of QRDQN algorithm. QRDQN (https://arxiv.org/pdf/1710.10044.pdf) is a distributional RL \
        algorithm, which is an extension of DQN. The main idea of QRDQN is to use quantile regression to \
        estimate the quantile of the distribution of the return value, and then use the quantile to calculate \
        the quantile loss.

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      qrdqn          | RL policy register name, refer to      | this arg is optional,
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
        type='qrdqn',
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

            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=3,
            batch_size=64,
            learning_rate=0.001,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_update_freq=100,
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

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.

        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.
        """
        return 'qrdqn', ['ding.model.template.q_learning']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For QRDQN, it mainly \
            contains optimizer, algorithm-specific arguments such as nstep and gamma. This method \
            also executes some special network initializations and prepares running mean/std monitor for value.
            This method will be called in ``__init__`` method if ``learn`` field is in ``enable_field``.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        .. note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
        """
        self._priority = self._cfg.priority
        # Optimizer
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)

        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep

        # use model_wrapper for specialized demands of different modes
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
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, current lr.

        Arguments:
            - data (:obj:`dict`): Input data used for policy forward, including the \
                collected training samples from replay buffer. For each element in dict, the key of the \
                dict is the name of data items and the value is the corresponding data. Usually, the value is \
                torch.Tensor or np.ndarray or there dict/list combinations. In the ``_forward_learn`` method, data \
                often need to first be stacked in the batch dimension by some utility functions such as \
                ``default_preprocess_learn``. \
                For QRDQN, each element in list is a dict containing at least the following keys: ``obs``, \
                ``action``, ``reward``, ``next_obs``. Sometimes, it also contains other keys such as ``weight``.

        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The output result dict of forward learn, \
                containing current lr, total_loss and priority. When discrete action satisfying \
                len(data['action'])==1, it also could contain ``action_distribution`` which is used \
                to draw histogram on tensorboard. For more information, please refer to the :class:`DQNPolicy`.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for QRDQNPolicy: ``ding.policy.tests.test_qrdqn``.
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
        q_value, tau = ret['q'], ret['tau']
        # Target q value
        with torch.no_grad():
            target_q_value = self._target_model.forward(data['next_obs'])['q']
            # Max q value action (main model)
            target_q_action = self._learn_model.forward(data['next_obs'])['action']

        data_n = qrdqn_nstep_td_data(
            q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], tau, data['weight']
        )
        value_gamma = data.get('value_gamma')
        loss, td_error_per_sample = qrdqn_nstep_td_error(
            data_n, self._gamma, nstep=self._nstep, value_gamma=value_gamma
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
            'priority': td_error_per_sample.abs().tolist(),
            # Only discrete action satisfying len(data['action'])==1 can return this and draw histogram on tensorboard.
            # '[histogram]action_distribution': data['action'],
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])
