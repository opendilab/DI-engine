from typing import List, Dict, Any, Tuple, Optional
from collections import namedtuple
import copy
import torch

from ding.torch_utils import RMSprop, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import timestep_collate, default_collate, default_decollate
from .base_policy import Policy


@POLICY_REGISTRY.register('qmix')
class QMIXPolicy(Policy):
    """
    Overview:
        Policy class of QMIX algorithm. QMIX is a multi-agent reinforcement learning algorithm, \
        you can view the paper in the following link https://arxiv.org/abs/1803.11485.
    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      qmix           | RL policy register name, refer to      | this arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     True           | Whether to use cuda for network        | this arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4. ``priority``         bool     False          | Whether use priority(PER)              | priority sample,
                                                                                                 | update priority
        5  | ``priority_``      bool     False          | Whether use Importance Sampling        | IS weight
           | ``IS_weight``                              | Weight to correct biased update.
        6  | ``learn.update_``  int      20             | How many updates(iterations) to train  | this args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        7  | ``learn.target_``   float    0.001         | Target network update momentum         | between[0,1]
           | ``update_theta``                           | parameter.
        8  | ``learn.discount`` float    0.99           | Reward's future discount factor, aka.  | may be 1 when sparse
           | ``_factor``                                | gamma                                  | reward env
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='qmix',
        # (bool) Whether to use cuda for network.
        cuda=True,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # learn_mode config
        learn=dict(
            # (int) How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=20,
            # (int) How many samples in a training batch.
            batch_size=32,
            # (float) The step size of gradient descent.
            learning_rate=0.0005,
            clip_value=100,
            # (float) Target network update momentum parameter, in [0, 1].
            target_update_theta=0.008,
            # (float) The discount factor for future rewards, in [0, 1].
            discount_factor=0.99,
            # (bool) Whether to use double DQN mechanism(target q for surpassing over estimation).
            double_q=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # In each collect phase, we collect a total of <n_sample> sequence samples, a sample with length unroll_len.
            # n_sample=32,
            # (int) Split trajectories into pieces with length ``unroll_len``, the length of timesteps
            # in each forward when training. In qmix, it is greater than 1 because there is RNN.
            unroll_len=10,
        ),
        eval=dict(),  # for compatibility
        other=dict(
            eps=dict(
                # (str) Type of epsilon decay.
                type='exp',
                # (float) Start value for epsilon decay, in [0, 1].
                start=1,
                # (float) Start value for epsilon decay, in [0, 1].
                end=0.05,
                # (int) Decay length(env step).
                decay=50000,
            ),
            replay_buffer=dict(
                # (int) Maximum size of replay buffer. Usually, larger buffer size is better.
                replay_buffer_size=5000,
            ),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For QMIX, ``ding.model.qmix.qmix``
        """
        return 'qmix', ['ding.model.template.qmix']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including some attributes and modules. For QMIX, it mainly contains \
            optimizer, algorithm-specific arguments such as gamma, main and target model. Because of the use of RNN, \
            all the models should be wrappered with ``hidden_state`` which needs to be initialized with proper size.
            This method will be called in ``__init__`` method if ``learn`` field is in ``enable_field``.

        .. tip::
            For multi-agent algorithm, we often need to use ``agent_num`` to initialize some necessary variables.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        .. note::
            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.

        .. note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
            - agent_num (:obj:`int`): Since this is a multi-agent algorithm, we need to input the agent num.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and not self._priority_IS_weight, "Priority is not implemented in QMIX"
        self._optimizer = RMSprop(
            params=self._model.parameters(),
            lr=self._cfg.learn.learning_rate,
            alpha=0.99,
            eps=0.00001,
            weight_decay=1e-5
        )
        self._gamma = self._cfg.learn.discount_factor

        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_update_theta}
        )
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
        )
        self._learn_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
        )
        self._learn_model.reset()
        self._target_model.reset()

    def _data_preprocess_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Preprocess the data to fit the required data format for learning
        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function
        Returns:
            - data (:obj:`Dict[str, Any]`): the processed data, from \
                [len=B, ele={dict_key: [len=T, ele=Tensor(any_dims)]}] -> {dict_key: Tensor([T, B, any_dims])}
        """
        # data preprocess
        data = timestep_collate(data)
        if self._cuda:
            data = to_device(data, self._device)
        data['weight'] = data.get('weight', None)
        data['done'] = data['done'].float()
        return data

    def _forward_learn(self, data: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data (trajectory for QMIX) from the replay buffer and then \
            returns the output result, including various training information such as loss, q value, grad_norm.
        Arguments:
            - data (:obj:`List[List[Dict[int, Any]]]`): The input data used for policy forward, including a batch of \
                training samples. For each dict element, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the time and \
                batch dimension by the utility functions ``self._data_preprocess_learn``. \
                For QMIX, each element in list is a trajectory with the length of ``unroll_len``, and the element in \
                trajectory list is a dict containing at least the following keys: ``obs``, ``action``, ``prev_state``, \
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
            For more detailed examples, please refer to our unittest for QMIXPolicy: ``ding.policy.tests.test_qmix``.
        """
        data = self._data_preprocess_learn(data)
        # ====================
        # Q-mix forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        # for hidden_state plugin, we need to reset the main model and target model
        self._learn_model.reset(state=data['prev_state'][0])
        self._target_model.reset(state=data['prev_state'][0])
        inputs = {'obs': data['obs'], 'action': data['action']}
        total_q = self._learn_model.forward(inputs, single_step=False)['total_q']

        if self._cfg.learn.double_q:
            next_inputs = {'obs': data['next_obs']}
            self._learn_model.reset(state=data['prev_state'][1])
            logit_detach = self._learn_model.forward(next_inputs, single_step=False)['logit'].clone().detach()
            next_inputs = {'obs': data['next_obs'], 'action': logit_detach.argmax(dim=-1)}
        else:
            next_inputs = {'obs': data['next_obs']}
        with torch.no_grad():
            target_total_q = self._target_model.forward(next_inputs, single_step=False)['total_q']

        with torch.no_grad():
            if data['done'] is not None:
                target_v = self._gamma * (1 - data['done']) * target_total_q + data['reward']
            else:
                target_v = self._gamma * target_total_q + data['reward']

        data = v_1step_td_data(total_q, target_total_q, data['reward'], data['done'], data['weight'])
        loss, td_error_per_sample = v_1step_td_error(data, self._gamma)
        # ====================
        # Q-mix update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._cfg.learn.clip_value)
        self._optimizer.step()
        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'total_q': total_q.mean().item() / self._cfg.model.agent_num,
            'target_reward_total_q': target_v.mean().item() / self._cfg.model.agent_num,
            'target_total_q': target_total_q.mean().item() / self._cfg.model.agent_num,
            'grad_norm': grad_norm,
        }

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        """
        Overview:
            Reset some stateful variables for learn mode when necessary, such as the hidden state of RNN or the \
            memory bank of some special algortihms. If ``data_id`` is None, it means to reset all the stateful \
            varaibles. Otherwise, it will reset the stateful variables according to the ``data_id``. For example, \
            different trajectories in ``data_id`` will have different hidden state in RNN.
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): The id of the data, which is used to reset the stateful variables \
                (i.e. RNN hidden_state in QMIX) specified by ``data_id``.
        """
        self._learn_model.reset(data_id=data_id)

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
            Initialize the collect mode of policy, including related attributes and modules. For QMIX, it contains the \
            collect_model to balance the exploration and exploitation with epsilon-greedy sample mechanism and \
            maintain the hidden state of rnn. Besides, there are some initialization operations about other \
            algorithm-specific arguments such as burnin_step, unroll_len and nstep.
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.collect.env_num,
            save_prev_state=True,
            init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
        )
        self._collect_model = model_wrap(self._collect_model, wrapper_name='eps_greedy_sample')
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
                other necessary data (prev_state) for learn mode defined in ``self._process_transition`` method. The \
                key of the dict is the same as the input data, i.e. environment id.

        .. note::
            RNN's hidden states are maintained in the policy, so we don't need pass them into data but to reset the \
            hidden states with ``_reset_collect`` method when episode ends. Besides, the previous hidden states are \
            necessary for training, so we need to return them in ``_process_transition`` method.
        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for QMIXPolicy: ``ding.policy.tests.test_qmix``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps, data_id=data_id)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        """
        Overview:
            Reset some stateful variables for eval mode when necessary, such as the hidden state of RNN or the \
            memory bank of some special algortihms. If ``data_id`` is None, it means to reset all the stateful \
            varaibles. Otherwise, it will reset the stateful variables according to the ``data_id``. For example, \
            different environments/episodes in evaluation in ``data_id`` will have different hidden state in RNN.
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): The id of the data, which is used to reset the stateful variables \
                (i.e., RNN hidden_state in QMIX) specified by ``data_id``.
        """
        self._collect_model.reset(data_id=data_id)

    def _process_transition(self, obs: torch.Tensor, policy_output: Dict[str, torch.Tensor],
                            timestep: namedtuple) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For QMIX, it contains obs, next_obs, action, prev_state, reward, done.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, usually including ``agent_obs`` \
                and ``global_obs`` in multi-agent environment like MPE and SMAC.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. For QMIX, it contains the action and the prev_state of RNN.
            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method, \
                except all the elements have been transformed into tensor data. Usually, it contains the next obs, \
                reward, done, info, etc.
        Returns:
            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'prev_state': policy_output['prev_state'],
            'action': policy_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory (transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. In QMIX, a train sample is processed transitions with unroll_len \
            length. This method is usually used in collectors to execute necessary \
            RL data preprocessing before training, which can help learner amortize revelant time consumption. \
            In addition, you can also implement this method as an identity function and do the data processing \
            in ``self._forward_learn`` method.
        Arguments:
            - transitions (:obj:`List[Dict[str, Any]`): The trajectory data (a list of transition), each element is \
                the same format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each sample is a fixed-length \
                trajectory, and each element in a sample is the similar format as input transitions.
        """
        return get_train_sample(transitions, self._unroll_len)

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For QMIX, it contains the \
            eval model to greedily select action with argmax q_value mechanism and main the hidden state.
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
        self._eval_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.eval.env_num,
            save_prev_state=True,
            init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
        )
        self._eval_model = model_wrap(self._eval_model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance by interacting with envs). Forward \
            means that the policy gets some necessary data (mainly observation) from the envs and then returns the \
            action to interact with the envs. ``_forward_eval`` often use argmax sample method to get actions that \
            q_value is the highest.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action. The \
                key of the dict is the same as the input data, i.e. environment id.

        .. note::
            RNN's hidden states are maintained in the policy, so we don't need pass them into data but to reset the \
            hidden states with ``_reset_eval`` method when the episode ends.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for QMIXPolicy: ``ding.policy.tests.test_qmix``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, data_id=data_id)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        """
        Overview:
            Reset some stateful variables for eval mode when necessary, such as the hidden state of RNN or the \
            memory bank of some special algortihms. If ``data_id`` is None, it means to reset all the stateful \
            varaibles. Otherwise, it will reset the stateful variables according to the ``data_id``. For example, \
            different environments/episodes in evaluation in ``data_id`` will have different hidden state in RNN.
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): The id of the data, which is used to reset the stateful variables \
                (i.e., RNN hidden_state in QMIX) specified by ``data_id``.
        """
        self._eval_model.reset(data_id=data_id)

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return ['cur_lr', 'total_loss', 'total_q', 'target_total_q', 'grad_norm', 'target_reward_total_q']
