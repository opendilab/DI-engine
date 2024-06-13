from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
import copy
import numpy as np

from ding.torch_utils import Adam, to_device, to_dtype, unsqueeze, ContrastiveLoss
from ding.rl_utils import ppo_data, ppo_error, ppo_policy_error, ppo_policy_data, get_gae_with_default_last_value, \
    v_nstep_td_data, v_nstep_td_error, get_nstep_return_data, get_train_sample, gae, gae_data, ppo_error_continuous, \
    get_gae, ppo_policy_error_continuous
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY, split_data_generator, RunningMeanStd
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('ppo')
class PPOPolicy(Policy):
    """
    Overview:
        Policy class of on-policy version PPO algorithm. Paper link: https://arxiv.org/abs/1707.06347.
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='ppo',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy. (Note: in practice PPO can be off-policy used)
        on_policy=True,
        # (bool) Whether to use priority (priority sample, IS weight, update priority).
        priority=False,
        # (bool) Whether to use Importance Sampling Weight to correct biased update due to priority.
        # If True, priority must be True.
        priority_IS_weight=False,
        # (bool) Whether to recompurete advantages in each iteration of on-policy PPO.
        recompute_adv=True,
        # (str) Which kind of action space used in PPOPolicy, ['discrete', 'continuous', 'hybrid']
        action_space='discrete',
        # (bool) Whether to use nstep return to calculate value target, otherwise, use return = adv + value.
        nstep_return=False,
        # (bool) Whether to enable multi-agent training, i.e.: MAPPO.
        multi_agent=False,
        # (bool) Whether to need policy ``_forward_collect`` output data in process transition.
        transition_with_policy_data=True,
        # learn_mode config
        learn=dict(
            # (int) After collecting n_sample/n_episode data, how many epoches to train models.
            # Each epoch means the one entire passing of training data.
            epoch_per_collect=10,
            # (int) How many samples in a training batch.
            batch_size=64,
            # (float) The step size of gradient descent.
            learning_rate=3e-4,
            # (dict or None) The learning rate decay.
            # If not None, should contain key 'epoch_num' and 'min_lr_lambda'.
            # where 'epoch_num' is the total epoch num to decay the learning rate to min value,
            # 'min_lr_lambda' is the final decayed learning rate.
            lr_scheduler=None,
            # (float) The loss weight of value network, policy network weight is set to 1.
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1.
            entropy_weight=0.0,
            # (float) PPO clip ratio, defaults to 0.2.
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch.
            adv_norm=True,
            # (bool) Whether to use value norm with running mean and std in the whole training process.
            value_norm=True,
            # (bool) Whether to enable special network parameters initialization scheme in PPO, such as orthogonal init.
            ppo_param_init=True,
            # (str) The gradient clip operation type used in PPO, ['clip_norm', clip_value', 'clip_momentum_norm'].
            grad_clip_type='clip_norm',
            # (float) The gradient clip target value used in PPO.
            # If ``grad_clip_type`` is 'clip_norm', then the maximum of gradient will be normalized to this value.
            grad_clip_value=0.5,
            # (bool) Whether ignore done (usually for max step termination env).
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # Only one of [n_sample, n_episode] should be set.
            # n_sample=64,
            # (int) Split episodes or trajectories into pieces with length `unroll_len`.
            unroll_len=1,
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # (float) GAE lambda factor for the balance of bias and variance(1-step td and mc)
            gae_lambda=0.95,
        ),
        eval=dict(),  # for compability
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
            by import_names path. For example about PPO, its registered name is ``ppo`` and the import_names is \
            ``ding.model.template.vac``.

        .. note::
            Because now PPO supports both single-agent and multi-agent usages, so we can implement these functions \
            with the same policy and two different default models, which is controled by ``self._cfg.multi_agent``.
        """
        if self._cfg.multi_agent:
            return 'mavac', ['ding.model.template.mavac']
        else:
            return 'vac', ['ding.model.template.vac']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For PPO, it mainly contains \
            optimizer, algorithm-specific arguments such as loss weight, clip_ratio and recompute_adv. This method \
            also executes some special network initializations and prepares running mean/std monitor for value.
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
        assert not self._priority and not self._priority_IS_weight, "Priority is not implemented in PPO"

        assert self._cfg.action_space in ["continuous", "discrete", "hybrid"]
        self._action_space = self._cfg.action_space
        if self._cfg.learn.ppo_param_init:
            for n, m in self._model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            if self._action_space in ['continuous', 'hybrid']:
                # init log sigma
                if self._action_space == 'continuous':
                    if hasattr(self._model.actor_head, 'log_sigma_param'):
                        torch.nn.init.constant_(self._model.actor_head.log_sigma_param, -0.5)
                elif self._action_space == 'hybrid':  # actor_head[1]: ReparameterizationHead, for action_args
                    if hasattr(self._model.actor_head[1], 'log_sigma_param'):
                        torch.nn.init.constant_(self._model.actor_head[1].log_sigma_param, -0.5)

                for m in list(self._model.critic.modules()) + list(self._model.actor.modules()):
                    if isinstance(m, torch.nn.Linear):
                        # orthogonal initialization
                        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                        torch.nn.init.zeros_(m.bias)
                # do last policy layer scaling, this will make initial actions have (close to)
                # 0 mean and std, and will help boost performances,
                # see https://arxiv.org/abs/2006.05990, Fig.24 for details
                for m in self._model.actor.modules():
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.zeros_(m.bias)
                        m.weight.data.copy_(0.01 * m.weight.data)

        # Optimizer
        self._optimizer = Adam(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate,
            grad_clip_type=self._cfg.learn.grad_clip_type,
            clip_value=self._cfg.learn.grad_clip_value
        )

        # Define linear lr scheduler
        if self._cfg.learn.lr_scheduler is not None:
            epoch_num = self._cfg.learn.lr_scheduler['epoch_num']
            min_lr_lambda = self._cfg.learn.lr_scheduler['min_lr_lambda']

            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._optimizer,
                lr_lambda=lambda epoch: max(1.0 - epoch * (1.0 - min_lr_lambda) / epoch_num, min_lr_lambda)
            )

        self._learn_model = model_wrap(self._model, wrapper_name='base')

        # Algorithm config
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._clip_ratio = self._cfg.learn.clip_ratio
        self._adv_norm = self._cfg.learn.adv_norm
        self._value_norm = self._cfg.learn.value_norm
        if self._value_norm:
            self._running_mean_std = RunningMeanStd(epsilon=1e-4, device=self._device)
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda
        self._recompute_adv = self._cfg.recompute_adv
        # Main model
        self._learn_model.reset()

    def _forward_learn(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, clipfrac, approx_kl.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including the latest \
                collected training samples for on-policy algorithms like PPO. For each element in list, the key of the \
                dict is the name of data items and the value is the corresponding data. Usually, the value is \
                torch.Tensor or np.ndarray or there dict/list combinations. In the ``_forward_learn`` method, data \
                often need to first be stacked in the batch dimension by some utility functions such as \
                ``default_preprocess_learn``. \
                For PPO, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
                ``reward``, ``logit``, ``value``, ``done``. Sometimes, it also contains other keys such as ``weight``.
        Returns:
            - return_infos (:obj:`List[Dict[str, Any]]`): The information list that indicated training result, each \
                training iteration contains append a information dict into the final list. The list will be precessed \
                and recorded in text log and tensorboard. The value of the dict must be python scalar or a list of \
                scalars. For the detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. tip::
            The training procedure of PPO is two for loops. The outer loop trains all the collected training samples \
            with ``epoch_per_collect`` epochs. The inner loop splits all the data into different mini-batch with \
            the length of ``batch_size``.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for PPOPolicy: ``ding.policy.tests.test_ppo``.
        """
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        if self._cuda:
            data = to_device(data, self._device)
        data['obs'] = to_dtype(data['obs'], torch.float32)
        if 'next_obs' in data:
            data['next_obs'] = to_dtype(data['next_obs'], torch.float32)
        # ====================
        # PPO forward
        # ====================
        return_infos = []
        self._learn_model.train()

        for epoch in range(self._cfg.learn.epoch_per_collect):
            if self._recompute_adv:  # calculate new value using the new updated value network
                with torch.no_grad():
                    value = self._learn_model.forward(data['obs'], mode='compute_critic')['value']
                    next_value = self._learn_model.forward(data['next_obs'], mode='compute_critic')['value']
                    if self._value_norm:
                        value *= self._running_mean_std.std
                        next_value *= self._running_mean_std.std

                    traj_flag = data.get('traj_flag', None)  # traj_flag indicates termination of trajectory
                    compute_adv_data = gae_data(value, next_value, data['reward'], data['done'], traj_flag)
                    data['adv'] = gae(compute_adv_data, self._gamma, self._gae_lambda)

                    unnormalized_returns = value + data['adv']

                    if self._value_norm:
                        data['value'] = value / self._running_mean_std.std
                        data['return'] = unnormalized_returns / self._running_mean_std.std
                        self._running_mean_std.update(unnormalized_returns.cpu().numpy())
                    else:
                        data['value'] = value
                        data['return'] = unnormalized_returns

            else:  # don't recompute adv
                if self._value_norm:
                    unnormalized_return = data['adv'] + data['value'] * self._running_mean_std.std
                    data['return'] = unnormalized_return / self._running_mean_std.std
                    self._running_mean_std.update(unnormalized_return.cpu().numpy())
                else:
                    data['return'] = data['adv'] + data['value']

            for batch in split_data_generator(data, self._cfg.learn.batch_size, shuffle=True):
                output = self._learn_model.forward(batch['obs'], mode='compute_actor_critic')
                adv = batch['adv']
                if self._adv_norm:
                    # Normalize advantage in a train_batch
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Calculate ppo error
                if self._action_space == 'continuous':
                    ppo_batch = ppo_data(
                        output['logit'], batch['logit'], batch['action'], output['value'], batch['value'], adv,
                        batch['return'], batch['weight']
                    )
                    ppo_loss, ppo_info = ppo_error_continuous(ppo_batch, self._clip_ratio)
                elif self._action_space == 'discrete':
                    ppo_batch = ppo_data(
                        output['logit'], batch['logit'], batch['action'], output['value'], batch['value'], adv,
                        batch['return'], batch['weight']
                    )
                    ppo_loss, ppo_info = ppo_error(ppo_batch, self._clip_ratio)
                elif self._action_space == 'hybrid':
                    # discrete part (discrete policy loss and entropy loss)
                    ppo_discrete_batch = ppo_policy_data(
                        output['logit']['action_type'], batch['logit']['action_type'], batch['action']['action_type'],
                        adv, batch['weight']
                    )
                    ppo_discrete_loss, ppo_discrete_info = ppo_policy_error(ppo_discrete_batch, self._clip_ratio)
                    # continuous part (continuous policy loss and entropy loss, value loss)
                    ppo_continuous_batch = ppo_data(
                        output['logit']['action_args'], batch['logit']['action_args'], batch['action']['action_args'],
                        output['value'], batch['value'], adv, batch['return'], batch['weight']
                    )
                    ppo_continuous_loss, ppo_continuous_info = ppo_error_continuous(
                        ppo_continuous_batch, self._clip_ratio
                    )
                    # sum discrete and continuous loss
                    ppo_loss = type(ppo_continuous_loss)(
                        ppo_continuous_loss.policy_loss + ppo_discrete_loss.policy_loss, ppo_continuous_loss.value_loss,
                        ppo_continuous_loss.entropy_loss + ppo_discrete_loss.entropy_loss
                    )
                    ppo_info = type(ppo_continuous_info)(
                        max(ppo_continuous_info.approx_kl, ppo_discrete_info.approx_kl),
                        max(ppo_continuous_info.clipfrac, ppo_discrete_info.clipfrac)
                    )
                wv, we = self._value_weight, self._entropy_weight
                total_loss = ppo_loss.policy_loss + wv * ppo_loss.value_loss - we * ppo_loss.entropy_loss

                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                if self._cfg.learn.lr_scheduler is not None:
                    cur_lr = sum(self._lr_scheduler.get_last_lr()) / len(self._lr_scheduler.get_last_lr())
                else:
                    cur_lr = self._optimizer.defaults['lr']

                return_info = {
                    'cur_lr': cur_lr,
                    'total_loss': total_loss.item(),
                    'policy_loss': ppo_loss.policy_loss.item(),
                    'value_loss': ppo_loss.value_loss.item(),
                    'entropy_loss': ppo_loss.entropy_loss.item(),
                    'adv_max': adv.max().item(),
                    'adv_mean': adv.mean().item(),
                    'value_mean': output['value'].mean().item(),
                    'value_max': output['value'].max().item(),
                    'approx_kl': ppo_info.approx_kl,
                    'clipfrac': ppo_info.clipfrac,
                }
                if self._action_space == 'continuous':
                    return_info.update(
                        {
                            'act': batch['action'].float().mean().item(),
                            'mu_mean': output['logit']['mu'].mean().item(),
                            'sigma_mean': output['logit']['sigma'].mean().item(),
                        }
                    )
                return_infos.append(return_info)

        if self._cfg.learn.lr_scheduler is not None:
            self._lr_scheduler.step()

        return return_infos

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For PPO, it contains the \
            collect_model to balance the exploration and exploitation (e.g. the multinomial sample mechanism in \
            discrete action space), and other algorithm-specific arguments such as unroll_len and gae_lambda.
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.

        .. tip::
            Some variables need to initialize independently in different modes, such as gamma and gae_lambda in PPO. \
            This design is for the convenience of parallel execution of different policy modes.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        assert self._cfg.action_space in ["continuous", "discrete", "hybrid"], self._cfg.action_space
        self._action_space = self._cfg.action_space
        if self._action_space == 'continuous':
            self._collect_model = model_wrap(self._model, wrapper_name='reparam_sample')
        elif self._action_space == 'discrete':
            self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample')
        elif self._action_space == 'hybrid':
            self._collect_model = model_wrap(self._model, wrapper_name='hybrid_reparam_multinomial_sample')
        self._collect_model.reset()
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda
        self._recompute_adv = self._cfg.recompute_adv

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
            For more detailed examples, please refer to our unittest for PPOPolicy: ``ding.policy.tests.test_ppo``.
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

    def _process_transition(self, obs: torch.Tensor, policy_output: Dict[str, torch.Tensor],
                            timestep: namedtuple) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For PPO, it contains obs, next_obs, action, reward, done, logit, value.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, such as stacked 2D image in Atari.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. For PPO, it contains the state value, action and the logit of the action.
            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method, \
                except all the elements have been transformed into tensor data. Usually, it contains the next obs, \
                reward, done, info, etc.
        Returns:
            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.

        .. note::
            ``next_obs`` is used to calculate nstep return when necessary, so we place in into transition by default. \
            You can delete this field to save memory occupancy if you do not need nstep return.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': policy_output['action'],
            'logit': policy_output['logit'],
            'value': policy_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory (transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. In PPO, a train sample is a processed transition with new computed \
            ``traj_flag`` and ``adv`` field. This method is usually used in collectors to execute necessary \
            RL data preprocessing before training, which can help learner amortize revelant time consumption. \
            In addition, you can also implement this method as an identity function and do the data processing \
            in ``self._forward_learn`` method.
        Arguments:
            - transitions (:obj:`List[Dict[str, Any]`): The trajectory data (a list of transition), each element is \
                the same format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each element is the similar format \
                as input transitions, but may contain more data for training, such as GAE advantage.
        """
        data = transitions
        data = to_device(data, self._device)
        for transition in data:
            transition['traj_flag'] = copy.deepcopy(transition['done'])
        data[-1]['traj_flag'] = True

        if self._cfg.learn.ignore_done:
            data[-1]['done'] = False

        if data[-1]['done']:
            last_value = torch.zeros_like(data[-1]['value'])
        else:
            with torch.no_grad():
                last_value = self._collect_model.forward(
                    unsqueeze(data[-1]['next_obs'], 0), mode='compute_actor_critic'
                )['value']
            if len(last_value.shape) == 2:  # multi_agent case:
                last_value = last_value.squeeze(0)
        if self._value_norm:
            last_value *= self._running_mean_std.std
            for i in range(len(data)):
                data[i]['value'] *= self._running_mean_std.std
        data = get_gae(
            data,
            to_device(last_value, self._device),
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            cuda=False,
        )
        if self._value_norm:
            for i in range(len(data)):
                data[i]['value'] /= self._running_mean_std.std

        # remove next_obs for save memory when not recompute adv
        if not self._recompute_adv:
            for i in range(len(data)):
                data[i].pop('next_obs')
        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For PPO, it contains the \
            eval model to select optimial action (e.g. greedily select action with argmax mechanism in discrete action).
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
        assert self._cfg.action_space in ["continuous", "discrete", "hybrid"]
        self._action_space = self._cfg.action_space
        if self._action_space == 'continuous':
            self._eval_model = model_wrap(self._model, wrapper_name='deterministic_sample')
        elif self._action_space == 'discrete':
            self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        elif self._action_space == 'hybrid':
            self._eval_model = model_wrap(self._model, wrapper_name='hybrid_reparam_multinomial_sample')

        self._eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance by interacting with envs). Forward \
            means that the policy gets some necessary data (mainly observation) from the envs and then returns the \
            action to interact with the envs. ``_forward_eval`` in PPO often uses deterministic sample method to get \
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
            For more detailed examples, please refer to our unittest for PPOPolicy: ``ding.policy.tests.test_ppo``.
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
        variables = super()._monitor_vars_learn() + [
            'policy_loss',
            'value_loss',
            'entropy_loss',
            'adv_max',
            'adv_mean',
            'approx_kl',
            'clipfrac',
            'value_max',
            'value_mean',
        ]
        if self._action_space == 'continuous':
            variables += ['mu_mean', 'sigma_mean', 'sigma_grad', 'act']
        return variables


@POLICY_REGISTRY.register('ppo_pg')
class PPOPGPolicy(Policy):
    """
    Overview:
        Policy class of on policy version PPO algorithm (pure policy gradient without value network).
        Paper link: https://arxiv.org/abs/1707.06347.
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='ppo_pg',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy. (Note: in practice PPO can be off-policy used)
        on_policy=True,
        # (str) Which kind of action space used in PPOPolicy, ['discrete', 'continuous', 'hybrid']
        action_space='discrete',
        # (bool) Whether to enable multi-agent training, i.e.: MAPPO.
        multi_agent=False,
        # (bool) Whether to need policy data in process transition.
        transition_with_policy_data=True,
        # learn_mode config
        learn=dict(
            # (int) After collecting n_sample/n_episode data, how many epoches to train models.
            # Each epoch means the one entire passing of training data.
            epoch_per_collect=10,
            # (int) How many samples in a training batch.
            batch_size=64,
            # (float) The step size of gradient descent.
            learning_rate=3e-4,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1.
            entropy_weight=0.0,
            # (float) PPO clip ratio, defaults to 0.2.
            clip_ratio=0.2,
            # (bool) Whether to enable special network parameters initialization scheme in PPO, such as orthogonal init.
            ppo_param_init=True,
            # (str) The gradient clip operation type used in PPO, ['clip_norm', clip_value', 'clip_momentum_norm'].
            grad_clip_type='clip_norm',
            # (float) The gradient clip target value used in PPO.
            # If ``grad_clip_type`` is 'clip_norm', then the maximum of gradient will be normalized to this value.
            grad_clip_value=0.5,
            # (bool) Whether ignore done (usually for max step termination env).
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training episodes collected in one collection process. Only one of n_episode shoule be set.
            # n_episode=8,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
        ),
        eval=dict(),  # for compability
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.
        """
        return 'pg', ['ding.model.template.pg']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For PPOPG, it mainly \
            contains optimizer, algorithm-specific arguments such as loss weight and clip_ratio. This method \
            also executes some special network initializations.
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
        self._action_space = self._cfg.action_space
        if self._cfg.learn.ppo_param_init:
            for n, m in self._model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            if self._action_space == 'continuous':
                if hasattr(self._model.head, 'log_sigma_param'):
                    torch.nn.init.constant_(self._model.head.log_sigma_param, -0.5)
                for m in self._model.modules():
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.zeros_(m.bias)
                        m.weight.data.copy_(0.01 * m.weight.data)

        # Optimizer
        self._optimizer = Adam(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate,
            grad_clip_type=self._cfg.learn.grad_clip_type,
            clip_value=self._cfg.learn.grad_clip_value
        )

        self._learn_model = model_wrap(self._model, wrapper_name='base')

        # Algorithm config
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._clip_ratio = self._cfg.learn.clip_ratio
        self._gamma = self._cfg.collect.discount_factor
        # Main model
        self._learn_model.reset()

    def _forward_learn(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, clipfrac, approx_kl.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including the latest \
                collected training samples for on-policy algorithms like PPO. For each element in list, the key of the \
                dict is the name of data items and the value is the corresponding data. Usually, the value is \
                torch.Tensor or np.ndarray or there dict/list combinations. In the ``_forward_learn`` method, data \
                often need to first be stacked in the batch dimension by some utility functions such as \
                ``default_preprocess_learn``. \
                For PPOPG, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
                ``return``, ``logit``, ``done``. Sometimes, it also contains other keys such as ``weight``.
        Returns:
            - return_infos (:obj:`List[Dict[str, Any]]`): The information list that indicated training result, each \
                training iteration contains append a information dict into the final list. The list will be precessed \
                and recorded in text log and tensorboard. The value of the dict must be python scalar or a list of \
                scalars. For the detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. tip::
            The training procedure of PPOPG is two for loops. The outer loop trains all the collected training samples \
            with ``epoch_per_collect`` epochs. The inner loop splits all the data into different mini-batch with \
            the length of ``batch_size``.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.
        """

        data = default_preprocess_learn(data)
        if self._cuda:
            data = to_device(data, self._device)
        return_infos = []
        self._learn_model.train()

        for epoch in range(self._cfg.learn.epoch_per_collect):
            for batch in split_data_generator(data, self._cfg.learn.batch_size, shuffle=True):
                output = self._learn_model.forward(batch['obs'])

                ppo_batch = ppo_policy_data(
                    output['logit'], batch['logit'], batch['action'], batch['return'], batch['weight']
                )
                if self._action_space == 'continuous':
                    ppo_loss, ppo_info = ppo_policy_error_continuous(ppo_batch, self._clip_ratio)
                elif self._action_space == 'discrete':
                    ppo_loss, ppo_info = ppo_policy_error(ppo_batch, self._clip_ratio)
                total_loss = ppo_loss.policy_loss - self._entropy_weight * ppo_loss.entropy_loss

                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                return_info = {
                    'cur_lr': self._optimizer.defaults['lr'],
                    'total_loss': total_loss.item(),
                    'policy_loss': ppo_loss.policy_loss.item(),
                    'entropy_loss': ppo_loss.entropy_loss.item(),
                    'approx_kl': ppo_info.approx_kl,
                    'clipfrac': ppo_info.clipfrac,
                }
                if self._action_space == 'continuous':
                    return_info.update(
                        {
                            'act': batch['action'].float().mean().item(),
                            'mu_mean': output['logit']['mu'].mean().item(),
                            'sigma_mean': output['logit']['sigma'].mean().item(),
                        }
                    )
                return_infos.append(return_info)
        return return_infos

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For PPOPG, it contains \
            the collect_model to balance the exploration and exploitation (e.g. the multinomial sample mechanism in \
            discrete action space), and other algorithm-specific arguments such as unroll_len and gae_lambda.
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.

        .. tip::
            Some variables need to initialize independently in different modes, such as gamma and gae_lambda in PPO. \
            This design is for the convenience of parallel execution of different policy modes.
        """
        assert self._cfg.action_space in ["continuous", "discrete"], self._cfg.action_space
        self._action_space = self._cfg.action_space
        self._unroll_len = self._cfg.collect.unroll_len
        if self._action_space == 'continuous':
            self._collect_model = model_wrap(self._model, wrapper_name='reparam_sample')
        elif self._action_space == 'discrete':
            self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample')
        self._collect_model.reset()
        self._gamma = self._cfg.collect.discount_factor

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
                other necessary data (action logit) for learn mode defined in ``self._process_transition`` \
                method. The key of the dict is the same as the input data, i.e. environment id.

        .. tip::
            If you want to add more tricks on this policy, like temperature factor in multinomial sample, you can pass \
            related data as extra keyword arguments of this method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: torch.Tensor, policy_output: Dict[str, torch.Tensor],
                            timestep: namedtuple) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For PPOPG, it contains obs, action, reward, done, logit.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, such as stacked 2D image in Atari.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. For PPOPG, it contains the action and the logit of the action.
            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method, \
                except all the elements have been transformed into tensor data. Usually, it contains the next obs, \
                reward, done, info, etc.
        Returns:
            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.
        """
        transition = {
            'obs': obs,
            'action': policy_output['action'],
            'logit': policy_output['logit'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given entire episode data (a list of transition), process it into a list of sample that \
            can be used for training directly. In PPOPG, a train sample is a processed transition with new computed \
            ``return`` field. This method is usually used in collectors to execute necessary \
            RL data preprocessing before training, which can help learner amortize revelant time consumption. \
            In addition, you can also implement this method as an identity function and do the data processing \
            in ``self._forward_learn`` method.
        Arguments:
            - data (:obj:`List[Dict[str, Any]`): The episode data (a list of transition), each element is \
                the same format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each element is the similar format \
                as input transitions, but may contain more data for training, such as discounted episode return.
        """
        assert data[-1]['done'] is True, "PPO-PG needs a complete epsiode"

        if self._cfg.learn.ignore_done:
            raise NotImplementedError

        R = 0.
        for i in reversed(range(len(data))):
            R = self._gamma * R + data[i]['reward']
            data[i]['return'] = R

        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For PPOPG, it contains the \
            eval model to select optimial action (e.g. greedily select action with argmax mechanism in discrete action).
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
            action to interact with the envs. ``_forward_eval`` in PPO often uses deterministic sample method to get \
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
            For more detailed examples, please refer to our unittest for PPOPGPolicy: ``ding.policy.tests.test_ppo``.
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

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return super()._monitor_vars_learn() + [
            'policy_loss',
            'entropy_loss',
            'approx_kl',
            'clipfrac',
        ]


@POLICY_REGISTRY.register('ppo_offpolicy')
class PPOOffPolicy(Policy):
    """
    Overview:
        Policy class of off-policy version PPO algorithm. Paper link: https://arxiv.org/abs/1707.06347.
        This version is more suitable for large-scale distributed training.
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='ppo',
        # (bool) Whether to use cuda for network.
        cuda=False,
        on_policy=False,
        # (bool) Whether to use priority (priority sample, IS weight, update priority).
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (str) Which kind of action space used in PPOPolicy, ["continuous", "discrete", "hybrid"].
        action_space='discrete',
        # (bool) Whether to use nstep_return for value loss.
        nstep_return=False,
        # (int) The timestep of TD (temporal-difference) loss.
        nstep=3,
        # (bool) Whether to need policy data in process transition.
        transition_with_policy_data=True,
        # learn_mode config
        learn=dict(
            # (int) How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=5,
            # (int) How many samples in a training batch.
            batch_size=64,
            # (float) The step size of gradient descent.
            learning_rate=0.001,
            # (float) The loss weight of value network, policy network weight is set to 1.
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1.
            entropy_weight=0.01,
            # (float) PPO clip ratio, defaults to 0.2.
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch.
            adv_norm=False,
            # (bool) Whether to use value norm with running mean and std in the whole training process.
            value_norm=True,
            # (bool) Whether to enable special network parameters initialization scheme in PPO, such as orthogonal init.
            ppo_param_init=True,
            # (str) The gradient clip operation type used in PPO, ['clip_norm', clip_value', 'clip_momentum_norm'].
            grad_clip_type='clip_norm',
            # (float) The gradient clip target value used in PPO.
            # If ``grad_clip_type`` is 'clip_norm', then the maximum of gradient will be normalized to this value.
            grad_clip_value=0.5,
            # (bool) Whether ignore done (usually for max step termination env).
            ignore_done=False,
            # (float) The weight decay (L2 regularization) loss weight, defaults to 0.0.
            weight_decay=0.0,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # Only one of [n_sample, n_episode] shoule be set.
            # n_sample=64,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # (float) GAE lambda factor for the balance of bias and variance (1-step td and mc).
            gae_lambda=0.95,
        ),
        eval=dict(),  # for compability
        other=dict(
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
        """
        return 'vac', ['ding.model.template.vac']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For PPOOff, it mainly \
            contains optimizer, algorithm-specific arguments such as loss weight and clip_ratio. This method \
            also executes some special network initializations and prepares running mean/std monitor for value.
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
        assert not self._priority and not self._priority_IS_weight, "Priority is not implemented in PPOOff"

        assert self._cfg.action_space in ["continuous", "discrete", "hybrid"]
        self._action_space = self._cfg.action_space

        if self._cfg.learn.ppo_param_init:
            for n, m in self._model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            if self._action_space in ['continuous', 'hybrid']:
                # init log sigma
                if self._action_space == 'continuous':
                    if hasattr(self._model.actor_head, 'log_sigma_param'):
                        torch.nn.init.constant_(self._model.actor_head.log_sigma_param, -2.0)
                elif self._action_space == 'hybrid':  # actor_head[1]: ReparameterizationHead, for action_args
                    if hasattr(self._model.actor_head[1], 'log_sigma_param'):
                        torch.nn.init.constant_(self._model.actor_head[1].log_sigma_param, -0.5)

                for m in list(self._model.critic.modules()) + list(self._model.actor.modules()):
                    if isinstance(m, torch.nn.Linear):
                        # orthogonal initialization
                        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                        torch.nn.init.zeros_(m.bias)
                # do last policy layer scaling, this will make initial actions have (close to)
                # 0 mean and std, and will help boost performances,
                # see https://arxiv.org/abs/2006.05990, Fig.24 for details
                for m in self._model.actor.modules():
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.zeros_(m.bias)
                        m.weight.data.copy_(0.01 * m.weight.data)

        # Optimizer
        self._optimizer = Adam(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate,
            grad_clip_type=self._cfg.learn.grad_clip_type,
            clip_value=self._cfg.learn.grad_clip_value
        )

        self._learn_model = model_wrap(self._model, wrapper_name='base')

        # Algorithm config
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._clip_ratio = self._cfg.learn.clip_ratio
        self._adv_norm = self._cfg.learn.adv_norm
        self._value_norm = self._cfg.learn.value_norm
        if self._value_norm:
            self._running_mean_std = RunningMeanStd(epsilon=1e-4, device=self._device)
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda
        self._nstep = self._cfg.nstep
        self._nstep_return = self._cfg.nstep_return
        # Main model
        self._learn_model.reset()

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, clipfrac and approx_kl.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For PPOOff, each element in list is a dict containing at least the following keys: ``obs``, ``adv``, \
                ``action``, ``logit``, ``value``, ``done``. Sometimes, it also contains other keys such as ``weight`` \
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
        """
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=self._nstep_return)
        if self._cuda:
            data = to_device(data, self._device)
        data['obs'] = to_dtype(data['obs'], torch.float32)
        if 'next_obs' in data:
            data['next_obs'] = to_dtype(data['next_obs'], torch.float32)
        # ====================
        # PPO forward
        # ====================

        self._learn_model.train()

        with torch.no_grad():
            if self._value_norm:
                unnormalized_return = data['adv'] + data['value'] * self._running_mean_std.std
                data['return'] = unnormalized_return / self._running_mean_std.std
                self._running_mean_std.update(unnormalized_return.cpu().numpy())
            else:
                data['return'] = data['adv'] + data['value']

        # normal ppo
        if not self._nstep_return:
            output = self._learn_model.forward(data['obs'], mode='compute_actor_critic')
            adv = data['adv']

            if self._adv_norm:
                # Normalize advantage in a total train_batch
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # Calculate ppo loss
            if self._action_space == 'continuous':
                ppodata = ppo_data(
                    output['logit'], data['logit'], data['action'], output['value'], data['value'], adv, data['return'],
                    data['weight']
                )
                ppo_loss, ppo_info = ppo_error_continuous(ppodata, self._clip_ratio)
            elif self._action_space == 'discrete':
                ppodata = ppo_data(
                    output['logit'], data['logit'], data['action'], output['value'], data['value'], adv, data['return'],
                    data['weight']
                )
                ppo_loss, ppo_info = ppo_error(ppodata, self._clip_ratio)
            elif self._action_space == 'hybrid':
                # discrete part (discrete policy loss and entropy loss)
                ppo_discrete_batch = ppo_policy_data(
                    output['logit']['action_type'], data['logit']['action_type'], data['action']['action_type'], adv,
                    data['weight']
                )
                ppo_discrete_loss, ppo_discrete_info = ppo_policy_error(ppo_discrete_batch, self._clip_ratio)
                # continuous part (continuous policy loss and entropy loss, value loss)
                ppo_continuous_batch = ppo_data(
                    output['logit']['action_args'], data['logit']['action_args'], data['action']['action_args'],
                    output['value'], data['value'], adv, data['return'], data['weight']
                )
                ppo_continuous_loss, ppo_continuous_info = ppo_error_continuous(ppo_continuous_batch, self._clip_ratio)
                # sum discrete and continuous loss
                ppo_loss = type(ppo_continuous_loss)(
                    ppo_continuous_loss.policy_loss + ppo_discrete_loss.policy_loss, ppo_continuous_loss.value_loss,
                    ppo_continuous_loss.entropy_loss + ppo_discrete_loss.entropy_loss
                )
                ppo_info = type(ppo_continuous_info)(
                    max(ppo_continuous_info.approx_kl, ppo_discrete_info.approx_kl),
                    max(ppo_continuous_info.clipfrac, ppo_discrete_info.clipfrac)
                )

            wv, we = self._value_weight, self._entropy_weight
            total_loss = ppo_loss.policy_loss + wv * ppo_loss.value_loss - we * ppo_loss.entropy_loss

        else:
            output = self._learn_model.forward(data['obs'], mode='compute_actor')
            adv = data['adv']
            if self._adv_norm:
                # Normalize advantage in a total train_batch
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Calculate ppo loss
            if self._action_space == 'continuous':
                ppodata = ppo_policy_data(output['logit'], data['logit'], data['action'], adv, data['weight'])
                ppo_policy_loss, ppo_info = ppo_policy_error_continuous(ppodata, self._clip_ratio)
            elif self._action_space == 'discrete':
                ppodata = ppo_policy_data(output['logit'], data['logit'], data['action'], adv, data['weight'])
                ppo_policy_loss, ppo_info = ppo_policy_error(ppodata, self._clip_ratio)
            elif self._action_space == 'hybrid':
                # discrete part (discrete policy loss and entropy loss)
                ppo_discrete_data = ppo_policy_data(
                    output['logit']['action_type'], data['logit']['action_type'], data['action']['action_type'], adv,
                    data['weight']
                )
                ppo_discrete_loss, ppo_discrete_info = ppo_policy_error(ppo_discrete_data, self._clip_ratio)
                # continuous part (continuous policy loss and entropy loss, value loss)
                ppo_continuous_data = ppo_policy_data(
                    output['logit']['action_args'], data['logit']['action_args'], data['action']['action_args'], adv,
                    data['weight']
                )
                ppo_continuous_loss, ppo_continuous_info = ppo_policy_error_continuous(
                    ppo_continuous_data, self._clip_ratio
                )
                # sum discrete and continuous loss
                ppo_policy_loss = type(ppo_continuous_loss)(
                    ppo_continuous_loss.policy_loss + ppo_discrete_loss.policy_loss,
                    ppo_continuous_loss.entropy_loss + ppo_discrete_loss.entropy_loss
                )
                ppo_info = type(ppo_continuous_info)(
                    max(ppo_continuous_info.approx_kl, ppo_discrete_info.approx_kl),
                    max(ppo_continuous_info.clipfrac, ppo_discrete_info.clipfrac)
                )

            wv, we = self._value_weight, self._entropy_weight
            next_obs = data.get('next_obs')
            value_gamma = data.get('value_gamma')
            reward = data.get('reward')
            # current value
            value = self._learn_model.forward(data['obs'], mode='compute_critic')
            # target value
            next_data = {'obs': next_obs}
            target_value = self._learn_model.forward(next_data['obs'], mode='compute_critic')
            # TODO what should we do here to keep shape
            assert self._nstep > 1
            td_data = v_nstep_td_data(
                value['value'], target_value['value'], reward, data['done'], data['weight'], value_gamma
            )
            # calculate v_nstep_td critic_loss
            critic_loss, td_error_per_sample = v_nstep_td_error(td_data, self._gamma, self._nstep)
            ppo_loss_data = namedtuple('ppo_loss', ['policy_loss', 'value_loss', 'entropy_loss'])
            ppo_loss = ppo_loss_data(ppo_policy_loss.policy_loss, critic_loss, ppo_policy_loss.entropy_loss)
            total_loss = ppo_policy_loss.policy_loss + wv * critic_loss - we * ppo_policy_loss.entropy_loss

        # ====================
        # PPO update
        # ====================
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return_info = {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': ppo_loss.policy_loss.item(),
            'value': data['value'].mean().item(),
            'value_loss': ppo_loss.value_loss.item(),
            'entropy_loss': ppo_loss.entropy_loss.item(),
            'adv_abs_max': adv.abs().max().item(),
            'approx_kl': ppo_info.approx_kl,
            'clipfrac': ppo_info.clipfrac,
        }
        if self._action_space == 'continuous':
            return_info.update(
                {
                    'act': data['action'].float().mean().item(),
                    'mu_mean': output['logit']['mu'].mean().item(),
                    'sigma_mean': output['logit']['sigma'].mean().item(),
                }
            )
        return return_info

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For PPOOff, it contains \
            collect_model to balance the exploration and exploitation (e.g. the multinomial sample mechanism in \
            discrete action space), and other algorithm-specific arguments such as unroll_len and gae_lambda.
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.

        .. tip::
            Some variables need to initialize independently in different modes, such as gamma and gae_lambda in PPOOff.
            This design is for the convenience of parallel execution of different policy modes.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        assert self._cfg.action_space in ["continuous", "discrete", "hybrid"]
        self._action_space = self._cfg.action_space
        if self._action_space == 'continuous':
            self._collect_model = model_wrap(self._model, wrapper_name='reparam_sample')
        elif self._action_space == 'discrete':
            self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample')
        elif self._action_space == 'hybrid':
            self._collect_model = model_wrap(self._model, wrapper_name='hybrid_reparam_multinomial_sample')
        self._collect_model.reset()
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda
        self._nstep = self._cfg.nstep
        self._nstep_return = self._cfg.nstep_return
        self._value_norm = self._cfg.learn.value_norm
        if self._value_norm:
            self._running_mean_std = RunningMeanStd(epsilon=1e-4, device=self._device)

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
            For more detailed examples, please refer to our unittest for PPOOffPolicy: ``ding.policy.tests.test_ppo``.
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

    def _process_transition(self, obs: torch.Tensor, policy_output: Dict[str, torch.Tensor],
                            timestep: namedtuple) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For PPO, it contains obs, next_obs, action, reward, done, logit, value.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, such as stacked 2D image in Atari.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. For PPO, it contains the state value, action and the logit of the action.
            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method, \
                except all the elements have been transformed into tensor data. Usually, it contains the next obs, \
                reward, done, info, etc.
        Returns:
            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.

        .. note::
            ``next_obs`` is used to calculate nstep return when necessary, so we place in into transition by default. \
            You can delete this field to save memory occupancy if you do not need nstep return.
        """

        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'logit': policy_output['logit'],
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
            can be used for training directly. In PPO, a train sample is a processed transition with new computed \
            ``traj_flag`` and ``adv`` field. This method is usually used in collectors to execute necessary \
            RL data preprocessing before training, which can help learner amortize revelant time consumption. \
            In addition, you can also implement this method as an identity function and do the data processing \
            in ``self._forward_learn`` method.
        Arguments:
            - transitions (:obj:`List[Dict[str, Any]`): The trajectory data (a list of transition), each element is \
                the same format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each element is the similar format \
                as input transitions, but may contain more data for training, such as GAE advantage.
        """
        data = transitions
        data = to_device(data, self._device)
        for transition in data:
            transition['traj_flag'] = copy.deepcopy(transition['done'])
        data[-1]['traj_flag'] = True

        if self._cfg.learn.ignore_done:
            data[-1]['done'] = False

        if data[-1]['done']:
            last_value = torch.zeros_like(data[-1]['value'])
        else:
            with torch.no_grad():
                last_value = self._collect_model.forward(
                    unsqueeze(data[-1]['next_obs'], 0), mode='compute_actor_critic'
                )['value']
            if len(last_value.shape) == 2:  # multi_agent case:
                last_value = last_value.squeeze(0)
        if self._value_norm:
            last_value *= self._running_mean_std.std
            for i in range(len(data)):
                data[i]['value'] *= self._running_mean_std.std
        data = get_gae(
            data,
            to_device(last_value, self._device),
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            cuda=False,
        )
        if self._value_norm:
            for i in range(len(data)):
                data[i]['value'] /= self._running_mean_std.std

        if not self._nstep_return:
            return get_train_sample(data, self._unroll_len)
        else:
            return get_nstep_return_data(data, self._nstep)

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For PPOOff, it contains the \
            eval model to select optimial action (e.g. greedily select action with argmax mechanism in discrete action).
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
        assert self._cfg.action_space in ["continuous", "discrete", "hybrid"]
        self._action_space = self._cfg.action_space
        if self._action_space == 'continuous':
            self._eval_model = model_wrap(self._model, wrapper_name='deterministic_sample')
        elif self._action_space == 'discrete':
            self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        elif self._action_space == 'hybrid':
            self._eval_model = model_wrap(self._model, wrapper_name='hybrid_deterministic_argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance by interacting with envs). Forward \
            means that the policy gets some necessary data (mainly observation) from the envs and then returns the \
            action to interact with the envs. ``_forward_eval`` in PPO often uses deterministic sample method to get \
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
            For more detailed examples, please refer to our unittest for PPOOffPolicy: ``ding.policy.tests.test_ppo``.
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
        variables = super()._monitor_vars_learn() + [
            'policy_loss', 'value', 'value_loss', 'entropy_loss', 'adv_abs_max', 'approx_kl', 'clipfrac'
        ]
        if self._action_space == 'continuous':
            variables += ['mu_mean', 'sigma_mean', 'sigma_grad', 'act']
        return variables


@POLICY_REGISTRY.register('ppo_stdim')
class PPOSTDIMPolicy(PPOPolicy):
    """
    Overview:
        Policy class of on policy version PPO algorithm with ST-DIM auxiliary model.
        PPO paper link: https://arxiv.org/abs/1707.06347.
        ST-DIM paper link: https://arxiv.org/abs/1906.08226.
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='ppo_stdim',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy. (Note: in practice PPO can be off-policy used)
        on_policy=True,
        # (bool) Whether to use priority(priority sample, IS weight, update priority)
        priority=False,
        # (bool) Whether to use Importance Sampling Weight to correct biased update due to priority.
        # If True, priority must be True.
        priority_IS_weight=False,
        # (bool) Whether to recompurete advantages in each iteration of on-policy PPO
        recompute_adv=True,
        # (str) Which kind of action space used in PPOPolicy, ['discrete', 'continuous']
        action_space='discrete',
        # (bool) Whether to use nstep return to calculate value target, otherwise, use return = adv + value
        nstep_return=False,
        # (bool) Whether to enable multi-agent training, i.e.: MAPPO
        multi_agent=False,
        # (bool) Whether to need policy data in process transition
        transition_with_policy_data=True,
        # (float) The loss weight of the auxiliary model to the main loss.
        aux_loss_weight=0.001,
        aux_model=dict(
            # (int) the encoding size (of each head) to apply contrastive loss.
            encode_shape=64,
            # ([int, int]) the heads number of the obs encoding and next_obs encoding respectively.
            heads=[1, 1],
            # (str) the contrastive loss type.
            loss_type='infonce',
            # (float) a parameter to adjust the polarity between positive and negative samples.
            temperature=1.0,
        ),
        # learn_mode config
        learn=dict(
            # (int) After collecting n_sample/n_episode data, how many epoches to train models.
            # Each epoch means the one entire passing of training data.
            epoch_per_collect=10,
            # (int) How many samples in a training batch.
            batch_size=64,
            # (float) The step size of gradient descent.
            learning_rate=3e-4,
            # (float) The loss weight of value network, policy network weight is set to 1.
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1.
            entropy_weight=0.0,
            # (float) PPO clip ratio, defaults to 0.2.
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch.
            adv_norm=True,
            # (bool) Whether to use value norm with running mean and std in the whole training process.
            value_norm=True,
            # (bool) Whether to enable special network parameters initialization scheme in PPO, such as orthogonal init.
            ppo_param_init=True,
            # (str) The gradient clip operation type used in PPO, ['clip_norm', clip_value', 'clip_momentum_norm'].
            grad_clip_type='clip_norm',
            # (float) The gradient clip target value used in PPO.
            # If ``grad_clip_type`` is 'clip_norm', then the maximum of gradient will be normalized to this value.
            grad_clip_value=0.5,
            # (bool) Whether ignore done (usually for max step termination env).
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # Only one of [n_sample, n_episode] shoule be set.
            # n_sample=64,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # (float) GAE lambda factor for the balance of bias and variance (1-step td and mc).
            gae_lambda=0.95,
        ),
        eval=dict(),  # for compability
    )

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the auxiliary model, its optimizer, and the axuliary loss weight to the main loss.
        """
        super()._init_learn()
        x_size, y_size = self._get_encoding_size()
        self._aux_model = ContrastiveLoss(x_size, y_size, **self._cfg.aux_model)
        if self._cuda:
            self._aux_model.cuda()
        self._aux_optimizer = Adam(self._aux_model.parameters(), lr=self._cfg.learn.learning_rate)
        self._aux_loss_weight = self._cfg.aux_loss_weight

    def _get_encoding_size(self):
        """
        Overview:
            Get the input encoding size of the ST-DIM axuiliary model.
        Returns:
            - info_dict (:obj:`[Tuple, Tuple]`): The encoding size without the first (Batch) dimension.
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

    def _model_encode(self, data):
        """
        Overview:
            Get the encoding of the main model as input for the auxiliary model.
        Arguments:
            - data (:obj:`dict`): Dict type data, same as the _forward_learn input.
        Returns:
            - (:obj:`Tuple[Tensor]`): the tuple of two tensors to apply contrastive embedding learning.
                In ST-DIM algorithm, these two variables are the dqn encoding of `obs` and `next_obs`\
                respectively.
        """
        assert hasattr(self._model, "encoder")
        x = self._model.encoder(data["obs"])
        y = self._model.encoder(data["next_obs"])
        return x, y

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data
        Returns:
            - info_dict (:obj:`Dict[str, Any]`):
              Including current lr, total_loss, policy_loss, value_loss, entropy_loss, \
                        adv_abs_max, approx_kl, clipfrac
        """
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # PPO forward
        # ====================
        return_infos = []
        self._learn_model.train()

        for epoch in range(self._cfg.learn.epoch_per_collect):
            if self._recompute_adv:  # calculate new value using the new updated value network
                with torch.no_grad():
                    value = self._learn_model.forward(data['obs'], mode='compute_critic')['value']
                    next_value = self._learn_model.forward(data['next_obs'], mode='compute_critic')['value']
                    if self._value_norm:
                        value *= self._running_mean_std.std
                        next_value *= self._running_mean_std.std

                    traj_flag = data.get('traj_flag', None)  # traj_flag indicates termination of trajectory
                    compute_adv_data = gae_data(value, next_value, data['reward'], data['done'], traj_flag)
                    data['adv'] = gae(compute_adv_data, self._gamma, self._gae_lambda)

                    unnormalized_returns = value + data['adv']

                    if self._value_norm:
                        data['value'] = value / self._running_mean_std.std
                        data['return'] = unnormalized_returns / self._running_mean_std.std
                        self._running_mean_std.update(unnormalized_returns.cpu().numpy())
                    else:
                        data['value'] = value
                        data['return'] = unnormalized_returns

            else:  # don't recompute adv
                if self._value_norm:
                    unnormalized_return = data['adv'] + data['value'] * self._running_mean_std.std
                    data['return'] = unnormalized_return / self._running_mean_std.std
                    self._running_mean_std.update(unnormalized_return.cpu().numpy())
                else:
                    data['return'] = data['adv'] + data['value']

            for batch in split_data_generator(data, self._cfg.learn.batch_size, shuffle=True):
                # ======================
                # Auxiliary model update
                # ======================

                # RL network encoding
                # To train the auxiliary network, the gradients of x, y should be 0.
                with torch.no_grad():
                    x_no_grad, y_no_grad = self._model_encode(batch)
                # the forward function of the auxiliary network
                self._aux_model.train()
                aux_loss_learn = self._aux_model.forward(x_no_grad, y_no_grad)
                # the BP process of the auxiliary network
                self._aux_optimizer.zero_grad()
                aux_loss_learn.backward()
                if self._cfg.multi_gpu:
                    self.sync_gradients(self._aux_model)
                self._aux_optimizer.step()

                output = self._learn_model.forward(batch['obs'], mode='compute_actor_critic')
                adv = batch['adv']
                if self._adv_norm:
                    # Normalize advantage in a train_batch
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Calculate ppo loss
                if self._action_space == 'continuous':
                    ppo_batch = ppo_data(
                        output['logit'], batch['logit'], batch['action'], output['value'], batch['value'], adv,
                        batch['return'], batch['weight']
                    )
                    ppo_loss, ppo_info = ppo_error_continuous(ppo_batch, self._clip_ratio)
                elif self._action_space == 'discrete':
                    ppo_batch = ppo_data(
                        output['logit'], batch['logit'], batch['action'], output['value'], batch['value'], adv,
                        batch['return'], batch['weight']
                    )
                    ppo_loss, ppo_info = ppo_error(ppo_batch, self._clip_ratio)

                # ======================
                # Compute auxiliary loss
                # ======================

                # In total_loss BP, the gradients of x, y are required to update the encoding network.
                # The auxiliary network won't be updated since the self._optimizer does not contain
                # its weights.
                x, y = self._model_encode(data)
                self._aux_model.eval()
                aux_loss_eval = self._aux_model.forward(x, y) * self._aux_loss_weight

                wv, we = self._value_weight, self._entropy_weight
                total_loss = ppo_loss.policy_loss + wv * ppo_loss.value_loss - we * ppo_loss.entropy_loss\
                    + aux_loss_eval

                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                return_info = {
                    'cur_lr': self._optimizer.defaults['lr'],
                    'total_loss': total_loss.item(),
                    'aux_loss_learn': aux_loss_learn.item(),
                    'aux_loss_eval': aux_loss_eval.item(),
                    'policy_loss': ppo_loss.policy_loss.item(),
                    'value_loss': ppo_loss.value_loss.item(),
                    'entropy_loss': ppo_loss.entropy_loss.item(),
                    'adv_max': adv.max().item(),
                    'adv_mean': adv.mean().item(),
                    'value_mean': output['value'].mean().item(),
                    'value_max': output['value'].max().item(),
                    'approx_kl': ppo_info.approx_kl,
                    'clipfrac': ppo_info.clipfrac,
                }
                if self._action_space == 'continuous':
                    return_info.update(
                        {
                            'act': batch['action'].float().mean().item(),
                            'mu_mean': output['logit']['mu'].mean().item(),
                            'sigma_mean': output['logit']['sigma'].mean().item(),
                        }
                    )
                return_infos.append(return_info)
        return return_infos

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model, optimizer and aux_optimizer for \
            representation learning.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'aux_optimizer': self._aux_optimizer.state_dict(),
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
        self._aux_optimizer.load_state_dict(state_dict['aux_optimizer'])

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return super()._monitor_vars_learn() + ["aux_loss_learn", "aux_loss_eval"]
