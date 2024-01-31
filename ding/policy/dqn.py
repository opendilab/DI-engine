from typing import List, Dict, Any, Tuple
from collections import namedtuple
import copy
import torch

from ding.torch_utils import Adam, to_device, ContrastiveLoss
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error, get_nstep_return_data, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate

from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('dqn')
class DQNPolicy(Policy):
    """
    Overview:
        Policy class of DQN algorithm, extended by Double DQN/Dueling DQN/PER/multi-step TD.

    Config:
        == ===================== ======== ============== ======================================= =======================
        ID Symbol                Type     Default Value  Description                              Other(Shape)
        == ===================== ======== ============== ======================================= =======================
        1  ``type``              str      dqn            | RL policy register name, refer to     | This arg is optional,
                                                         | registry ``POLICY_REGISTRY``          | a placeholder
        2  ``cuda``              bool     False          | Whether to use cuda for network       | This arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``         bool     False          | Whether the RL algorithm is on-policy
                                                         | or off-policy
        4  ``priority``          bool     False          | Whether use priority(PER)             | Priority sample,
                                                                                                 | update priority
        5  | ``priority_IS``     bool     False          | Whether use Importance Sampling
           | ``_weight``                                 | Weight to correct biased update. If
                                                         | True, priority must be True.
        6  | ``discount_``       float    0.97,          | Reward's future discount factor, aka. | May be 1 when sparse
           | ``factor``                   [0.95, 0.999]  | gamma                                 | reward env
        7  ``nstep``             int      1,             | N-step reward discount sum for target
                                          [3, 5]         | q_value estimation
        8  | ``model.dueling``   bool     True           | dueling head architecture
        9  | ``model.encoder``   list     [32, 64,       | Sequence of ``hidden_size`` of        | default kernel_size
           | ``_hidden``         (int)    64, 128]       | subsequent conv layers and the        | is [8, 4, 3]
           | ``_size_list``                              | final dense layer.                    | default stride is
                                                                                                 | [4, 2 ,1]
        10 | ``model.dropout``   float    None           | Dropout rate for dropout layers.      | [0,1]
                                                                                                 | If set to ``None``
                                                                                                 | means no dropout
        11 | ``learn.update``    int      3              | How many updates(iterations) to train | This args can be vary
           | ``per_collect``                             | after collector's one collection.     | from envs. Bigger val
                                                         | Only valid in serial training         | means more off-policy
        12 | ``learn.batch_``    int      64             | The number of samples of an iteration
           | ``size``
        13 | ``learn.learning``  float    0.001          | Gradient step length of an iteration.
           | ``_rate``
        14 | ``learn.target_``   int      100            | Frequence of target network update.   | Hard(assign) update
           | ``update_freq``
        15 | ``learn.target_``   float    0.005          | Frequence of target network update.   | Soft(assign) update
           | ``theta``                                   | Only one of [target_update_freq,
           |                                             | target_theta] should be set
        16 | ``learn.ignore_``   bool     False          | Whether ignore done for target value  | Enable it for some
           | ``done``                                    | calculation.                          | fake termination env
        17 ``collect.n_sample``  int      [8, 128]       | The number of training samples of a   | It varies from
                                                         | call of collector.                    | different envs
        18 ``collect.n_episode`` int      8              | The number of training episodes of a  | only one of [n_sample
                                                         | call of collector                     | ,n_episode] should
                                                         |                                       | be set
        19 | ``collect.unroll``  int      1              | unroll length of an iteration         | In RNN, unroll_len>1
           | ``_len``
        20 | ``other.eps.type``  str      exp            | exploration rate decay type           | Support ['exp',
                                                                                                 | 'linear'].
        21 | ``other.eps.``      float    0.95           | start value of exploration rate       | [0,1]
           | ``start``
        22 | ``other.eps.``      float    0.1            | end value of exploration rate         | [0,1]
           | ``end``
        23 | ``other.eps.``      int      10000          | decay length of exploration           | greater than 0. set
           | ``decay``                                                                           | decay=10000 means
                                                                                                 | the exploration rate
                                                                                                 | decay from start
                                                                                                 | value to end value
                                                                                                 | during decay length.
        == ===================== ======== ============== ======================================= =======================
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='dqn',
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
        # (int) The number of steps for calculating target q_value.
        nstep=1,
        model=dict(
            # (list(int)) Sequence of ``hidden_size`` of subsequent conv layers and the final dense layer.
            encoder_hidden_size_list=[128, 128, 64],
        ),
        # learn_mode config
        learn=dict(
            # (int) How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=3,
            # (int) How many samples in a training batch.
            batch_size=64,
            # (float) The step size of gradient descent.
            learning_rate=0.001,
            # (int) Frequency of target network update.
            # Only one of [target_update_freq, target_theta] should be set.
            target_update_freq=100,
            # (float) Used for soft update of the target network.
            # aka. Interpolation factor in EMA update for target network.
            # Only one of [target_update_freq, target_theta] should be set.
            target_theta=0.005,
            # (bool) If set to True, the 'done' signals that indicate the end of an episode due to environment time
            # limits are disregarded. By default, this is set to False. This setting is particularly useful for tasks
            # that have a predetermined episode length, such as HalfCheetah and various other MuJoCo environments,
            # where the maximum length is capped at 1000 steps. When enabled, any 'done' signal triggered by reaching
            # the maximum episode steps will be overridden to 'False'. This ensures the accurate calculation of the
            # Temporal Difference (TD) error, using the formula `gamma * (1 - done) * next_v + reward`,
            # even when the episode surpasses the predefined step limit.
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # Only one of [n_sample, n_episode] should be set.
            n_sample=8,
            # (int) Split episodes or trajectories into pieces with length `unroll_len`.
            unroll_len=1,
        ),
        eval=dict(),  # for compatibility
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

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.

        .. note::
            The user can define and use customized network model but must obey the same interface definition indicated \
            by import_names path. For example about DQN, its registered name is ``dqn`` and the import_names is \
            ``ding.model.template.q_learning``.
        """
        return 'dqn', ['ding.model.template.q_learning']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For DQN, it mainly contains \
            optimizer, algorithm-specific arguments such as nstep and gamma, main and target model.
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
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)

        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep

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

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, q value, priority.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For DQN, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
                ``reward``, ``next_obs``, ``done``. Sometimes, it also contains other keys such as ``weight`` \
                and ``value_gamma``.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The information dict that indicated training result, which will be \
                recorded in text log and tensorboard, values must be python scalar or a list of scalars. For the \
                detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement your own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for DQNPolicy: ``ding.policy.tests.test_dqn``.
        """
        # Data preprocessing operations, such as stack data, cpu to cuda device
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=True
        )
        if self._cuda:
            data = to_device(data, self._device)
        # Q-learning forward
        self._learn_model.train()
        self._target_model.train()
        # Current q value (main model)
        q_value = self._learn_model.forward(data['obs'])['logit']
        # Target q value
        with torch.no_grad():
            target_q_value = self._target_model.forward(data['next_obs'])['logit']
            # Max q value action (main model), i.e. Double DQN
            target_q_action = self._learn_model.forward(data['next_obs'])['action']

        data_n = q_nstep_td_data(
            q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], data['weight']
        )
        value_gamma = data.get('value_gamma')
        loss, td_error_per_sample = q_nstep_td_error(data_n, self._gamma, nstep=self._nstep, value_gamma=value_gamma)

        # Update network parameters
        self._optimizer.zero_grad()
        loss.backward()
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        self._optimizer.step()

        # Postprocessing operations, such as updating target model, return logged values and priority.
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'q_value': q_value.mean().item(),
            'target_q_value': target_q_value.mean().item(),
            'priority': td_error_per_sample.abs().tolist(),
            # Only discrete action satisfying len(data['action'])==1 can return this and draw histogram on tensorboard.
            # '[histogram]action_distribution': data['action'],
        }

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return ['cur_lr', 'total_loss', 'q_value', 'target_q_value']

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model, target_model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For DQN, it contains the \
            collect_model to balance the exploration and exploitation with epsilon-greedy sample mechanism, and other \
            algorithm-specific arguments such as unroll_len and nstep.
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.

        .. tip::
            Some variables need to initialize independently in different modes, such as gamma and nstep in DQN. This \
            design is for the convenience of parallel execution of different policy modes.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.discount_factor  # necessary for parallel
        self._nstep = self._cfg.nstep  # necessary for parallel
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: Dict[int, Any], eps: float) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of collect mode (collecting training data by interacting with envs). Forward means \
            that the policy gets some necessary data (mainly observation) from the envs and then returns the output \
            data, such as the action to interact with the envs. Besides, this policy also needs ``eps`` argument for \
            exploration, i.e., classic epsilon-greedy exploration strategy.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
            - eps (:obj:`float`): The epsilon value for exploration.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action and \
                other necessary data for learn mode defined in ``self._process_transition`` method. The key of the \
                dict is the same as the input data, i.e. environment id.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for DQNPolicy: ``ding.policy.tests.test_dqn``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _get_train_sample(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory (transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. In DQN with nstep TD, a train sample is a processed transition. \
            This method is usually used in collectors to execute necessary \
            RL data preprocessing before training, which can help learner amortize relevant time consumption. \
            In addition, you can also implement this method as an identity function and do the data processing \
            in ``self._forward_learn`` method.
        Arguments:
            - transitions (:obj:`List[Dict[str, Any]`): The trajectory data (a list of transition), each element is \
                in the same format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each element is similar in format \
                to input transitions, but may contain more data for training, such as nstep reward and target obs.
        """
        transitions = get_nstep_return_data(transitions, self._nstep, gamma=self._gamma)
        return get_train_sample(transitions, self._unroll_len)

    def _process_transition(self, obs: torch.Tensor, policy_output: Dict[str, torch.Tensor],
                            timestep: namedtuple) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For DQN, it contains obs, next_obs, action, reward, done.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, such as stacked 2D image in Atari.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. For DQN, it contains the action and the logit (q_value) of the action.
            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method, \
                except all the elements have been transformed into tensor data. Usually, it contains the next obs, \
                reward, done, info, etc.
        Returns:
            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': policy_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For DQN, it contains the \
            eval model to greedily select action with argmax q_value mechanism.
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance by interacting with envs). Forward \
            means that the policy gets some necessary data (mainly observation) from the envs and then returns the \
            action to interact with the envs.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action. The \
                key of the dict is the same as the input data, i.e. environment id.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for DQNPolicy: ``ding.policy.tests.test_dqn``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def calculate_priority(self, data: Dict[int, Any], update_target_model: bool = False) -> Dict[str, Any]:
        """
        Overview:
            Calculate priority for replay buffer.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, a batch of data for training.
            - update_target_model (:obj:`bool`): Whether to update target model.
        Returns:
            - priority (:obj:`Dict[str, Any]`): Dict type priority data, values are python scalar or a list of scalars.
        ArgumentsKeys:
            - necessary: ``obs``, ``action``, ``reward``, ``next_obs``, ``done``
            - optional: ``value_gamma``
        ReturnsKeys:
            - necessary: ``priority``
        """

        if update_target_model:
            self._target_model.load_state_dict(self._learn_model.state_dict())

        data = default_preprocess_learn(
            data,
            use_priority=False,
            use_priority_IS_weight=False,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=True
        )
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # Q-learning forward
        # ====================
        self._learn_model.eval()
        self._target_model.eval()
        with torch.no_grad():
            # Current q value (main model)
            q_value = self._learn_model.forward(data['obs'])['logit']
            # Target q value
            target_q_value = self._target_model.forward(data['next_obs'])['logit']
            # Max q value action (main model), i.e. Double DQN
            target_q_action = self._learn_model.forward(data['next_obs'])['action']
            data_n = q_nstep_td_data(
                q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], data['weight']
            )
            value_gamma = data.get('value_gamma')
            loss, td_error_per_sample = q_nstep_td_error(
                data_n, self._gamma, nstep=self._nstep, value_gamma=value_gamma
            )
        return {'priority': td_error_per_sample.abs().tolist()}


@POLICY_REGISTRY.register('dqn_stdim')
class DQNSTDIMPolicy(DQNPolicy):
    """
    Overview:
        Policy class of DQN algorithm, extended by ST-DIM auxiliary objectives.
        ST-DIM paper link: https://arxiv.org/abs/1906.08226.
    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      dqn_stdim      | RL policy register name, refer to      | This arg is optional,
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
        8  | ``learn.update``   int      3              | How many updates(iterations) to train  | This args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
           | ``_gpu``
        10 | ``learn.batch_``   int      64             | The number of samples of an iteration
           | ``size``
        11 | ``learn.learning`` float    0.001          | Gradient step length of an iteration.
           | ``_rate``
        12 | ``learn.target_``  int      100            | Frequence of target network update.    | Hard(assign) update
           | ``update_freq``
        13 | ``learn.ignore_``  bool     False          | Whether ignore done for target value   | Enable it for some
           | ``done``                                   | calculation.                           | fake termination env
        14 ``collect.n_sample`` int      [8, 128]       | The number of training samples of a    | It varies from
                                                        | call of collector.                     | different envs
        15 | ``collect.unroll`` int      1              | unroll length of an iteration          | In RNN, unroll_len>1
           | ``_len``
        16 | ``other.eps.type`` str      exp            | exploration rate decay type            | Support ['exp',
                                                                                                 | 'linear'].
        17 | ``other.eps.``     float    0.95           | start value of exploration rate        | [0,1]
           | ``start``
        18 | ``other.eps.``     float    0.1            | end value of exploration rate          | [0,1]
           | ``end``
        19 | ``other.eps.``     int      10000          | decay length of exploration            | greater than 0. set
           | ``decay``                                                                           | decay=10000 means
                                                                                                 | the exploration rate
                                                                                                 | decay from start
                                                                                                 | value to end value
                                                                                                 | during decay length.
        20 | ``aux_loss``       float    0.001          | the ratio of the auxiliary loss to     | any real value,
           | ``_weight``                                | the TD loss                            | typically in
                                                                                                 | [-0.1, 0.1].
        == ==================== ======== ============== ======================================== =======================
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='dqn_stdim',
        # (bool) Whether to use cuda in policy.
        cuda=False,
        # (bool) Whether to learning policy is the same as collecting data policy (on-policy).
        on_policy=False,
        # (bool) Whether to enable priority experience sample.
        priority=False,
        # (bool) Whether to use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (float) Discount factor(gamma) for returns.
        discount_factor=0.97,
        # (int) The number of step for calculating target q_value.
        nstep=1,
        # (float) The weight of auxiliary loss to main loss.
        aux_loss_weight=0.001,
        # learn_mode config
        learn=dict(
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=3,
            # (int) How many samples in a training batch.
            batch_size=64,
            # (float) The step size of gradient descent.
            learning_rate=0.001,
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (bool) Whether ignore done(usually for max step termination env).
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # Only one of [n_sample, n_episode] shoule be set.
            # n_sample=8,
            # (int) Cut trajectories into pieces with length "unroll_len".
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
                # (int) Decay length (env step).
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
            Initialize the learn mode of policy, including related attributes and modules. For DQNSTDIM, it first \
            call super class's ``_init_learn`` method, then initialize extra auxiliary model, its optimizer, and the \
            loss weight. This method will be called in ``__init__`` method if ``learn`` field is in ``enable_field``.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        .. note::
            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.

        .. note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
        """
        super()._init_learn()
        x_size, y_size = self._get_encoding_size()
        self._aux_model = ContrastiveLoss(x_size, y_size, **self._cfg.aux_model)
        if self._cuda:
            self._aux_model.cuda()
        self._aux_optimizer = Adam(self._aux_model.parameters(), lr=self._cfg.learn.learning_rate)
        self._aux_loss_weight = self._cfg.aux_loss_weight

    def _get_encoding_size(self) -> Tuple[Tuple[int], Tuple[int]]:
        """
        Overview:
            Get the input encoding size of the ST-DIM axuiliary model.
        Returns:
            - info_dict (:obj:`Tuple[Tuple[int], Tuple[int]]`): The encoding size without the first (Batch) dimension.
        """
        obs = self._cfg.model.obs_shape
        if isinstance(obs, int):
            obs = [obs]
        test_data = {
            "obs": torch.randn(1, *obs),
            "next_obs": torch.randn(1, *obs),
        }
        if self._cuda:
            test_data = to_device(test_data, self._device)
        with torch.no_grad():
            x, y = self._model_encode(test_data)
        return x.size()[1:], y.size()[1:]

    def _model_encode(self, data: dict) -> Tuple[torch.Tensor]:
        """
        Overview:
            Get the encoding of the main model as input for the auxiliary model.
        Arguments:
            - data (:obj:`dict`): Dict type data, same as the _forward_learn input.
        Returns:
            - (:obj:`Tuple[torch.Tensor]`): the tuple of two tensors to apply contrastive embedding learning. \
                In ST-DIM algorithm, these two variables are the dqn encoding of `obs` and `next_obs` respectively.
        """
        assert hasattr(self._model, "encoder")
        x = self._model.encoder(data["obs"])
        y = self._model.encoder(data["next_obs"])
        return x, y

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, q value, priority, aux_loss.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For DQNSTDIM, each element in list is a dict containing at least the following keys: ``obs``, \
                ``action``, ``reward``, ``next_obs``, ``done``. Sometimes, it also contains other keys such as \
                ``weight`` and ``value_gamma``.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The information dict that indicated training result, which will be \
                recorded in text log and tensorboard, values must be python scalar or a list of scalars. For the \
                detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.
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

        # ======================
        # Auxiliary model update
        # ======================
        # RL network encoding
        # To train the auxiliary network, the gradients of x, y should be 0.
        with torch.no_grad():
            x_no_grad, y_no_grad = self._model_encode(data)
        # the forward function of the auxiliary network
        self._aux_model.train()
        aux_loss_learn = self._aux_model.forward(x_no_grad, y_no_grad)
        # the BP process of the auxiliary network
        self._aux_optimizer.zero_grad()
        aux_loss_learn.backward()
        if self._cfg.multi_gpu:
            self.sync_gradients(self._aux_model)
        self._aux_optimizer.step()

        # ====================
        # Q-learning forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        # Current q value (main model)
        q_value = self._learn_model.forward(data['obs'])['logit']
        # Target q value
        with torch.no_grad():
            target_q_value = self._target_model.forward(data['next_obs'])['logit']
            # Max q value action (main model)
            target_q_action = self._learn_model.forward(data['next_obs'])['action']

        data_n = q_nstep_td_data(
            q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], data['weight']
        )
        value_gamma = data.get('value_gamma')
        bellman_loss, td_error_per_sample = q_nstep_td_error(
            data_n, self._gamma, nstep=self._nstep, value_gamma=value_gamma
        )

        # ======================
        # Compute auxiliary loss
        # ======================
        x, y = self._model_encode(data)
        self._aux_model.eval()
        aux_loss_eval = self._aux_model.forward(x, y) * self._aux_loss_weight
        loss = aux_loss_eval + bellman_loss

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
            'bellman_loss': bellman_loss.item(),
            'aux_loss_learn': aux_loss_learn.item(),
            'aux_loss_eval': aux_loss_eval.item(),
            'total_loss': loss.item(),
            'q_value': q_value.mean().item(),
            'priority': td_error_per_sample.abs().tolist(),
            # Only discrete action satisfying len(data['action'])==1 can return this and draw histogram on tensorboard.
            # '[histogram]action_distribution': data['action'],
        }

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return ['cur_lr', 'bellman_loss', 'aux_loss_learn', 'aux_loss_eval', 'total_loss', 'q_value']

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'aux_optimizer': self._aux_optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])
        self._aux_optimizer.load_state_dict(state_dict['aux_optimizer'])
