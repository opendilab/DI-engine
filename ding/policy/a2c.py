from collections import namedtuple
from typing import List, Dict, Any, Tuple

import torch

from ding.model import model_wrap
from ding.rl_utils import a2c_data, a2c_error, get_gae_with_default_last_value, get_train_sample, \
    a2c_error_continuous
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY, split_data_generator
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('a2c')
class A2CPolicy(Policy):
    """
    Overview:
        Policy class of A2C (Advantage Actor-Critic) algorithm, proposed in https://arxiv.org/abs/1602.01783.
    """
    config = dict(
        # (str) Name of the registered RL policy (refer to the "register_policy" function).
        type='a2c',
        # (bool) Flag to enable CUDA for model computation.
        cuda=False,
        # (bool) Flag for using on-policy training (training policy is the same as the behavior policy).
        on_policy=True,
        # (bool) Flag for enabling priority experience replay. Must be False when priority_IS_weight is False.
        priority=False,
        # (bool) Flag for using Importance Sampling weights to correct updates. Requires `priority` to be True.
        priority_IS_weight=False,
        # (str) Type of action space used in the policy, with valid options ['discrete', 'continuous'].
        action_space='discrete',
        # learn_mode configuration
        learn=dict(
            # (int) Number of updates per data collection. A2C requires this to be set to 1.
            update_per_collect=1,
            # (int) Batch size for learning.
            batch_size=64,
            # (float) Learning rate for optimizer.
            learning_rate=0.001,
            # (Tuple[float, float]) Coefficients used for computing running averages of gradient and its square.
            betas=(0.9, 0.999),
            # (float) Term added to the denominator to improve numerical stability in optimizer.
            eps=1e-8,
            # (float) Maximum norm for gradients.
            grad_norm=0.5,
            # (float) Scaling factor for value network loss relative to policy network loss.
            value_weight=0.5,
            # (float) Weight of entropy regularization in the loss function.
            entropy_weight=0.01,
            # (bool) Flag to enable normalization of advantages.
            adv_norm=False,
            # (bool) If set to True, the 'done' signals that indicate the end of an episode due to environment time
            # limits are disregarded. By default, this is set to False. This setting is particularly useful for tasks
            # that have a predetermined episode length, such as HalfCheetah and various other MuJoCo environments,
            # where the maximum length is capped at 1000 steps. When enabled, any 'done' signal triggered by reaching
            # the maximum episode steps will be overridden to 'False'. This ensures the accurate calculation of the
            # Temporal Difference (TD) error, using the formula `gamma * (1 - done) * next_v + reward`,
            # even when the episode surpasses the predefined step limit.
            ignore_done=False,
        ),
        # collect_mode configuration
        collect=dict(
            # (int) The length of rollout for data collection.
            unroll_len=1,
            # (float) Discount factor for calculating future rewards, typically in the range [0, 1].
            discount_factor=0.9,
            # (float) Trade-off parameter for balancing TD-error and Monte Carlo error in GAE.
            gae_lambda=0.95,
        ),
        # eval_mode configuration (kept empty for compatibility purposes)
        eval=dict(),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Returns the default model configuration used by the A2C algorithm. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.

        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): \
                Tuple containing the registered model name and model's import_names.
        """
        return 'vac', ['ding.model.template.vac']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For A2C, it mainly \
            contains optimizer, algorithm-specific arguments such as value_weight, entropy_weight, adv_norm
            and grad_norm, and main model. \
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
        assert self._cfg.action_space in ["continuous", "discrete"]
        # Optimizer
        self._optimizer = Adam(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate,
            betas=self._cfg.learn.betas,
            eps=self._cfg.learn.eps
        )

        # Algorithm config
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._adv_norm = self._cfg.learn.adv_norm
        self._grad_norm = self._cfg.learn.grad_norm

        # Main and target models
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as policy_loss, value_loss, entropy_loss.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in the list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For A2C, each element in the list is a dict containing at least the following keys: \
                ['obs', 'action', 'adv', 'value', 'weight'].
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The information dict that indicated training result, which will be \
                recorded in text log and tensorboard, values must be python scalar or a list of scalars. For the \
                detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that is not supported, the main reason is that the corresponding model does not support \
             it. You can implement your own model rather than use the default model. For more information, please \
             raise an issue in GitHub repo, and we will continue to follow up.
        """
        # Data preprocessing operations, such as stack data, cpu to cuda device
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()

        for batch in split_data_generator(data, self._cfg.learn.batch_size, shuffle=True):
            # forward
            output = self._learn_model.forward(batch['obs'], mode='compute_actor_critic')

            adv = batch['adv']
            return_ = batch['value'] + adv
            if self._adv_norm:
                # norm adv in total train_batch
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            error_data = a2c_data(output['logit'], batch['action'], output['value'], adv, return_, batch['weight'])

            # Calculate A2C loss
            if self._action_space == 'continuous':
                a2c_loss = a2c_error_continuous(error_data)
            elif self._action_space == 'discrete':
                a2c_loss = a2c_error(error_data)

            wv, we = self._value_weight, self._entropy_weight
            total_loss = a2c_loss.policy_loss + wv * a2c_loss.value_loss - we * a2c_loss.entropy_loss

            # ====================
            # A2C-learning update
            # ====================
            self._optimizer.zero_grad()
            total_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self._learn_model.parameters()),
                max_norm=self._grad_norm,
            )
            self._optimizer.step()

        # =============
        # after update
        # =============
        # only record last updates information in logger
        return {
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': a2c_loss.policy_loss.item(),
            'value_loss': a2c_loss.value_loss.item(),
            'entropy_loss': a2c_loss.entropy_loss.item(),
            'adv_abs_max': adv.abs().max().item(),
            'grad_norm': grad_norm,
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
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
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For A2C, it contains the \
            collect_model to balance the exploration and exploitation with ``reparam_sample`` or \
            ``multinomial_sample`` mechanism, and other algorithm-specific arguments such as gamma and gae_lambda. \
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.
        """
        assert self._cfg.action_space in ["continuous", "discrete"]
        self._unroll_len = self._cfg.collect.unroll_len

        self._action_space = self._cfg.action_space
        if self._action_space == 'continuous':
            self._collect_model = model_wrap(self._model, wrapper_name='reparam_sample')
        elif self._action_space == 'discrete':
            self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample')
        self._collect_model.reset()
        # Algorithm
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda

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
                other necessary data for learn mode defined in ``self._process_transition`` method. The key of the \
                dict is the same as the input data, i.e. environment id.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor_critic')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, policy_output: Dict[str, torch.Tensor],
                            timestep: namedtuple) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For A2C, it contains obs, next_obs, action, value, reward, done.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, such as stacked 2D image in Atari.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. For A2C, it contains the action and the value of the state.
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
            'value': policy_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory (transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. In A2C, a train sample is a processed transition. \
            This method is usually used in collectors to execute necessary \
            RL data preprocessing before training, which can help the learner amortize relevant time consumption. \
            In addition, you can also implement this method as an identity function and do the data processing \
            in ``self._forward_learn`` method.
        Arguments:
            - transitions (:obj:`List[Dict[str, Any]`): The trajectory data (a list of transition), each element is \
                in the same format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each element is similar in format \
                to input transitions, but may contain more data for training, such as advantages.
        """
        transitions = get_gae_with_default_last_value(
            transitions,
            transitions[-1]['done'],
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            cuda=self._cuda,
        )
        return get_train_sample(transitions, self._unroll_len)

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For A2C, it contains the \
            eval model to greedily select action with ``argmax_sample`` mechanism (For discrete action space) and \
            ``deterministic_sample`` mechanism (For continuous action space). \
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
        assert self._cfg.action_space in ["continuous", "discrete"]
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
            action to interact with the envs.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action. The \
                key of the dict is the same as the input data, i.e., environment id.

        .. note::
            The input value can be ``torch.Tensor`` or dict/list combinations, current policy supports all of them. \
            For the data type that is not supported, the main reason is that the corresponding model does not \
            support it. You can implement your own model rather than use the default model. For more information, \
            please raise an issue in GitHub repo, and we will continue to follow up.
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
        return {i: d for i, d in zip(data_id, output)}

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss', 'entropy_loss', 'adv_abs_max', 'grad_norm']
