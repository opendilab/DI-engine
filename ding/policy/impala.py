from collections import namedtuple
from typing import List, Dict, Any, Tuple

import torch
import treetensor.torch as ttorch

from ding.model import model_wrap
from ding.rl_utils import vtrace_data, vtrace_error_discrete_action, vtrace_error_continuous_action, get_train_sample
from ding.torch_utils import Adam, RMSprop, to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate, ttorch_collate
from ding.policy.base_policy import Policy


@POLICY_REGISTRY.register('impala')
class IMPALAPolicy(Policy):
    """
    Overview:
        Policy class of IMPALA algorithm. Paper link: https://arxiv.org/abs/1802.01561.

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      impala         | RL policy register name, refer to      | this arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | this arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4. ``priority``         bool     False          | Whether use priority(PER)              | priority sample,
                                                                                                 | update priority

        5  | ``priority_``      bool     False          | Whether use Importance Sampling Weight | If True, priority
           | ``IS_weight``                              |                                        | must be True
        6  ``unroll_len``       int      32             | trajectory length to calculate v-trace
                                                        | target
        7  | ``learn.update``   int      4              | How many updates(iterations) to train  | this args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='impala',
        # (bool) Whether to use cuda in policy.
        cuda=False,
        # (bool) Whether learning policy is the same as collecting data policy(on-policy).
        on_policy=False,
        # (bool) Whether to enable priority experience sample.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (str) Which kind of action space used in IMPALAPolicy, ['discrete', 'continuous'].
        action_space='discrete',
        # (int) the trajectory length to calculate v-trace target.
        unroll_len=32,
        # (bool) Whether to need policy data in process transition.
        transition_with_policy_data=True,
        # learn_mode config
        learn=dict(
            # (int) collect n_sample data, train model update_per_collect times.
            update_per_collect=4,
            # (int) the number of data for a train iteration.
            batch_size=16,
            # (float) The step size of gradient descent.
            learning_rate=0.0005,
            # (float) loss weight of the value network, the weight of policy network is set to 1.
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1.
            entropy_weight=0.0001,
            # (float) discount factor for future reward, defaults int [0, 1].
            discount_factor=0.99,
            # (float) additional discounting parameter.
            lambda_=0.95,
            # (float) clip ratio of importance weights.
            rho_clip_ratio=1.0,
            # (float) clip ratio of importance weights.
            c_clip_ratio=1.0,
            # (float) clip ratio of importance sampling.
            rho_pg_clip_ratio=1.0,
            # (str) The gradient clip operation type used in IMPALA, ['clip_norm', clip_value', 'clip_momentum_norm'].
            grad_clip_type=None,
            # (float) The gradient clip target value used in IMPALA.
            # If ``grad_clip_type`` is 'clip_norm', then the maximum of gradient will be normalized to this value.
            clip_value=0.5,
            # (str) Optimizer used to train the network, ['adam', 'rmsprop'].
            optim='adam',
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # Only one of [n_sample, n_episode] shoule be set.
            # n_sample=16,
        ),
        eval=dict(),  # for compatibility
        other=dict(
            replay_buffer=dict(
                # (int) Maximum size of replay buffer. Usually, larger buffer size is better.
                replay_buffer_size=1000,
                # (int) Maximum use times for a sample in buffer. If reaches this value, the sample will be removed.
                max_use=16,
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
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For example about IMPALA , its registered name is ``vac`` and the import_names is \
            ``ding.model.template.vac``.
        """
        return 'vac', ['ding.model.template.vac']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For IMPALA, it mainly \
            contains optimizer, algorithm-specific arguments such as loss weight and gamma, main (learn) model.
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
        assert self._cfg.action_space in ["continuous", "discrete"], self._cfg.action_space
        self._action_space = self._cfg.action_space
        # Optimizer
        optim_type = self._cfg.learn.optim
        if optim_type == 'rmsprop':
            self._optimizer = RMSprop(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        elif optim_type == 'adam':
            self._optimizer = Adam(
                self._model.parameters(),
                grad_clip_type=self._cfg.learn.grad_clip_type,
                clip_value=self._cfg.learn.clip_value,
                lr=self._cfg.learn.learning_rate
            )
        else:
            raise NotImplementedError("Now only support rmsprop and adam, but input is {}".format(optim_type))
        self._learn_model = model_wrap(self._model, wrapper_name='base')

        self._action_shape = self._cfg.model.action_shape
        self._unroll_len = self._cfg.unroll_len

        # Algorithm config
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._gamma = self._cfg.learn.discount_factor
        self._lambda = self._cfg.learn.lambda_
        self._rho_clip_ratio = self._cfg.learn.rho_clip_ratio
        self._c_clip_ratio = self._cfg.learn.c_clip_ratio
        self._rho_pg_clip_ratio = self._cfg.learn.rho_pg_clip_ratio

        # Main model
        self._learn_model.reset()

    def _data_preprocess_learn(self, data: List[Dict[str, Any]]):
        """
        Overview:
            Data preprocess function of learn mode.
            Convert list trajectory data to to trajectory data, which is a dict of tensors.
        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): List type data, a list of data for training. Each list element is a \
                dict, whose values are torch.Tensor or np.ndarray or dict/list combinations, keys include at least \
                'obs', 'next_obs', 'logit', 'action', 'reward', 'done'
        Returns:
            - data (:obj:`dict`): Dict type data. Values are torch.Tensor or np.ndarray or dict/list combinations. \
        ReturnsKeys:
            - necessary: 'logit', 'action', 'reward', 'done', 'weight', 'obs_plus_1'.
            - optional and not used in later computation: 'obs', 'next_obs'.'IS', 'collect_iter', 'replay_unique_id', \
                'replay_buffer_idx', 'priority', 'staleness', 'use'.
        ReturnsShapes:
            - obs_plus_1 (:obj:`torch.FloatTensor`): :math:`(T * B, obs_shape)`, where T is timestep, B is batch size \
                and obs_shape is the shape of single env observation
            - logit (:obj:`torch.FloatTensor`): :math:`(T, B, N)`, where N is action dim
            - action (:obj:`torch.LongTensor`): :math:`(T, B)`
            - reward (:obj:`torch.FloatTensor`): :math:`(T+1, B)`
            - done (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - weight (:obj:`torch.FloatTensor`): :math:`(T, B)`
        """
        elem = data[0]
        if isinstance(elem, dict):  # old pipeline
            data = default_collate(data)
        elif isinstance(elem, list):  # new task pipeline
            data = default_collate(default_collate(data))
        else:
            raise TypeError("not support element type ({}) in IMPALA".format(type(elem)))
        if self._cuda:
            data = to_device(data, self._device)
        if self._priority_IS_weight:
            assert self._priority, "Use IS Weight correction, but Priority is not used."
        if self._priority and self._priority_IS_weight:
            data['weight'] = data['IS']
        else:
            data['weight'] = data.get('weight', None)
        if isinstance(elem, dict):  # old pipeline
            for k in data:
                if isinstance(data[k], list):
                    data[k] = default_collate(data[k])
        data['obs_plus_1'] = torch.cat([data['obs'], data['next_obs'][-1:]], dim=0)  # shape (T+1)*B,env_obs_shape
        return data

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss and current learning rate.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For IMPALA, each element in list is a dict containing at least the following keys: ``obs``, \
                ``action``, ``logit``, ``reward``, ``next_obs``, ``done``. Sometimes, it also contains other keys such \
                as ``weight``.
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
            For more detailed examples, please refer to unittest for IMPALAPolicy: ``ding.policy.tests.test_impala``.
        """
        data = self._data_preprocess_learn(data)
        # ====================
        # IMPALA forward
        # ====================
        self._learn_model.train()
        output = self._learn_model.forward(
            data['obs_plus_1'].view((-1, ) + data['obs_plus_1'].shape[2:]), mode='compute_actor_critic'
        )
        target_logit, behaviour_logit, actions, values, rewards, weights = self._reshape_data(output, data)
        # Calculate vtrace error
        data = vtrace_data(target_logit, behaviour_logit, actions, values, rewards, weights)
        g, l, r, c, rg = self._gamma, self._lambda, self._rho_clip_ratio, self._c_clip_ratio, self._rho_pg_clip_ratio
        if self._action_space == 'continuous':
            vtrace_loss = vtrace_error_continuous_action(data, g, l, r, c, rg)
        elif self._action_space == 'discrete':
            vtrace_loss = vtrace_error_discrete_action(data, g, l, r, c, rg)

        wv, we = self._value_weight, self._entropy_weight
        total_loss = vtrace_loss.policy_loss + wv * vtrace_loss.value_loss - we * vtrace_loss.entropy_loss
        # ====================
        # IMPALA update
        # ====================
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': vtrace_loss.policy_loss.item(),
            'value_loss': vtrace_loss.value_loss.item(),
            'entropy_loss': vtrace_loss.entropy_loss.item(),
        }

    def _reshape_data(self, output: Dict[str, Any], data: Dict[str, Any]) -> Tuple:
        """
        Overview:
            Obtain weights for loss calculating, where should be 0 for done positions. Update values and rewards with \
            the weight.
        Arguments:
            - output (:obj:`Dict[int, Any]`): Dict type data, output of learn_model forward. \
                Values are torch.Tensor or np.ndarray or dict/list combinations,keys are value, logit.
            - data (:obj:`Dict[int, Any]`): Dict type data, input of policy._forward_learn Values are torch.Tensor or \
                np.ndarray or dict/list combinations. Keys includes at least ['logit', 'action', 'reward', 'done'].
        Returns:
            - data (:obj:`Tuple[Any]`): Tuple of target_logit, behaviour_logit, actions, values, rewards, weights.
        ReturnsShapes:
            - target_logit (:obj:`torch.FloatTensor`): :math:`((T+1), B, Obs_Shape)`, where T is timestep,\
                B is batch size and Obs_Shape is the shape of single env observation.
            - behaviour_logit (:obj:`torch.FloatTensor`): :math:`(T, B, N)`, where N is action dim.
            - actions (:obj:`torch.LongTensor`): :math:`(T, B)`
            - values (:obj:`torch.FloatTensor`): :math:`(T+1, B)`
            - rewards (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - weights (:obj:`torch.FloatTensor`): :math:`(T, B)`
        """
        if self._action_space == 'continuous':
            target_logit = {}
            target_logit['mu'] = output['logit']['mu'].reshape(self._unroll_len + 1, -1,
                                                               self._action_shape)[:-1
                                                                                   ]  # shape (T+1),B,env_action_shape
            target_logit['sigma'] = output['logit']['sigma'].reshape(self._unroll_len + 1, -1, self._action_shape
                                                                     )[:-1]  # shape (T+1),B,env_action_shape
        elif self._action_space == 'discrete':
            target_logit = output['logit'].reshape(self._unroll_len + 1, -1,
                                                   self._action_shape)[:-1]  # shape (T+1),B,env_action_shape
        behaviour_logit = data['logit']  # shape T,B
        actions = data['action']  # shape T,B for discrete # shape T,B,env_action_shape for continuous
        values = output['value'].reshape(self._unroll_len + 1, -1)  # shape T+1,B,env_action_shape
        rewards = data['reward']  # shape T,B
        weights_ = 1 - data['done'].float()  # shape T,B
        weights = torch.ones_like(rewards)  # shape T,B
        values[1:] = values[1:] * weights_
        weights[1:] = weights_[:-1]
        rewards = rewards * weights  # shape T,B
        return target_logit, behaviour_logit, actions, values, rewards, weights

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For IMPALA, it contains \
            the collect_model to balance the exploration and exploitation (e.g. the multinomial sample mechanism in \
            discrete action space), and other algorithm-specific arguments such as unroll_len.
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.
        """
        assert self._cfg.action_space in ["continuous", "discrete"]
        self._action_space = self._cfg.action_space
        if self._action_space == 'continuous':
            self._collect_model = model_wrap(self._model, wrapper_name='reparam_sample')
        elif self._action_space == 'discrete':
            self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample')

        self._collect_model.reset()

    def _forward_collect(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of collect mode (collecting training data by interacting with envs). Forward means \
            that the policy gets some necessary data (mainly observation) from the envs and then returns the output \
            data, such as the action to interact with the envs.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action and \
                other necessary data (action logit and value) for learn mode defined in ``self._process_transition`` \
                method. The key of the dict is the same as the input data, i.e. environment id.

        .. tip::
            If you want to add more tricks on this policy, like temperature factor in multinomial sample, you can pass \
            related data as extra keyword arguments of this method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to unittest for IMPALAPolicy: ``ding.policy.tests.test_impala``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        output = {i: d for i, d in zip(data_id, output)}
        return output

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory (transitions, a list of transition) data, process it into a list of sample that \
            can be used for training. In IMPALA, a train sample is processed transitions with unroll_len length.
        Arguments:
            - transitions (:obj:`List[Dict[str, Any]`): The trajectory data (a list of transition), each element is \
                the same format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each element is the similar format \
                as input transitions, but may contain more data for training.
        """
        return get_train_sample(data, self._unroll_len)

    def _process_transition(self, obs: torch.Tensor, policy_output: Dict[str, torch.Tensor],
                            timestep: namedtuple) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For IMPALA, it contains obs, next_obs, action, reward, done, logit.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, such as stacked 2D image in Atari.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. For IMPALA, it contains the action and the logit of the action.
            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method, \
                except all the elements have been transformed into tensor data. Usually, it contains the next obs, \
                reward, done, info, etc.
        Returns:
            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'logit': policy_output['logit'],
            'action': policy_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For IMPALA, it contains the \
            eval model to select optimial action (e.g. greedily select action with argmax mechanism in discrete action).
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
        assert self._cfg.action_space in ["continuous", "discrete"], self._cfg.action_space
        self._action_space = self._cfg.action_space
        if self._action_space == 'continuous':
            self._eval_model = model_wrap(self._model, wrapper_name='deterministic_sample')
        elif self._action_space == 'discrete':
            self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')

        self._eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance by interacting with envs). Forward \
            means that the policy gets some necessary data (mainly observation) from the envs and then returns the \
            action to interact with the envs. ``_forward_eval`` in IMPALA often uses deterministic sample to get \
            actions while ``_forward_collect`` usually uses stochastic sample method for balance exploration and \
            exploitation.
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
            For more detailed examples, please refer to unittest for IMPALAPolicy: ``ding.policy.tests.test_impala``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        output = {i: d for i, d in zip(data_id, output)}
        return output

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss', 'entropy_loss']
