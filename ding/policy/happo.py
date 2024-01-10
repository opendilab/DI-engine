from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
import copy
import numpy as np
from torch.distributions import Independent, Normal

from ding.torch_utils import Adam, to_device, to_dtype, unsqueeze, ContrastiveLoss
from ding.rl_utils import happo_data, happo_error, happo_policy_error, happo_policy_data, \
    v_nstep_td_data, v_nstep_td_error, get_train_sample, gae, gae_data, happo_error_continuous, \
    get_gae
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY, split_data_generator, RunningMeanStd
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('happo')
class HAPPOPolicy(Policy):
    """
    Overview:
        Policy class of on policy version HAPPO algorithm. Paper link: https://arxiv.org/abs/2109.11251.
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='happo',
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
        # (str) Which kind of action space used in PPOPolicy, ['discrete', 'continuous', 'hybrid']
        action_space='discrete',
        # (bool) Whether to use nstep return to calculate value target, otherwise, use return = adv + value
        nstep_return=False,
        # (bool) Whether to enable multi-agent training, i.e.: MAPPO
        multi_agent=False,
        # (bool) Whether to need policy data in process transition
        transition_with_policy_data=True,
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) The loss weight of value network, policy network weight is set to 1
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.0,
            # (float) PPO clip ratio, defaults to 0.2
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=True,
            value_norm=True,
            ppo_param_init=True,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
            ignore_done=False,
        ),
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            # n_sample=64,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # (float) GAE lambda factor for the balance of bias and variance(1-step td and mc)
            gae_lambda=0.95,
        ),
        eval=dict(),
    )

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For HAPPO, it mainly \
            contains optimizer, algorithm-specific arguments such as loss weight, clip_ratio and recompute_adv. This \
            method also executes some special network initializations and prepares running mean/std monitor for value.
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

        assert self._cfg.action_space in ["continuous", "discrete"]
        self._action_space = self._cfg.action_space
        if self._cfg.learn.ppo_param_init:
            for n, m in self._model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            if self._action_space in ['continuous']:
                # init log sigma
                for agent_id in range(self._cfg.agent_num):
                    # if hasattr(self._model.agent_models[agent_id].actor_head, 'log_sigma_param'):
                    #     torch.nn.init.constant_(self._model.agent_models[agent_id].actor_head.log_sigma_param, 1)
                    # The above initialization step has been changed to reparameterizationHead.
                    for m in list(self._model.agent_models[agent_id].critic.modules()) + \
                    list(self._model.agent_models[agent_id].actor.modules()):
                        if isinstance(m, torch.nn.Linear):
                            # orthogonal initialization
                            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                            torch.nn.init.zeros_(m.bias)
                    # do last policy layer scaling, this will make initial actions have (close to)
                    # 0 mean and std, and will help boost performances,
                    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
                    for m in self._model.agent_models[agent_id].actor.modules():
                        if isinstance(m, torch.nn.Linear):
                            torch.nn.init.zeros_(m.bias)
                            m.weight.data.copy_(0.01 * m.weight.data)

        # Add the actor/critic parameters of each HAVACAgent in HAVAC to the parameter list of actor/critic_optimizer
        actor_params = []
        critic_params = []
        for agent_idx in range(self._model.agent_num):
            actor_params.append({'params': self._model.agent_models[agent_idx].actor.parameters()})
            critic_params.append({'params': self._model.agent_models[agent_idx].critic.parameters()})

        self._actor_optimizer = Adam(
            actor_params,
            lr=self._cfg.learn.learning_rate,
            grad_clip_type=self._cfg.learn.grad_clip_type,
            clip_value=self._cfg.learn.grad_clip_value,
            # eps = 1e-5,
        )

        self._critic_optimizer = Adam(
            critic_params,
            lr=self._cfg.learn.critic_learning_rate,
            grad_clip_type=self._cfg.learn.grad_clip_type,
            clip_value=self._cfg.learn.grad_clip_value,
            # eps = 1e-5,
        )

        self._learn_model = model_wrap(self._model, wrapper_name='base')
        # self._learn_model = model_wrap(
        #     self._model,
        #     wrapper_name='hidden_state',
        #     state_num=self._cfg.learn.batch_size,
        #     init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
        # )

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

    def prepocess_data_agent(self, data: Dict[str, Any]):
        """
        Overview:
            Preprocess data for agent dim. This function is used in learn mode. \
            It will be called recursively to process nested dict data. \
            It will transpose the data with shape (B, agent_num, ...) to (agent_num, B, ...). \
        Arguments:
            - data (:obj:`dict`): Dict type data, where each element is the data of an agent of dict type.
        Returns:
            - ret (:obj:`dict`): Dict type data, where each element is the data of an agent of dict type.
        """
        ret = {}
        for key, value in data.items():
            if isinstance(value, dict):
                ret[key] = self.prepocess_data_agent(value)
            elif isinstance(value, torch.Tensor) and len(value.shape) > 1:
                ret[key] = value.transpose(0, 1)
            else:
                ret[key] = value
        return ret

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): List type data, where each element is the data of an agent of dict type.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`):
              Including current lr, total_loss, policy_loss, value_loss, entropy_loss, \
                        adv_abs_max, approx_kl, clipfrac
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, clipfrac, approx_kl.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including the latest \
                collected training samples for on-policy algorithms like HAPPO. For each element in list, the key of \
                dict is the name of data items and the value is the corresponding data. Usually, the value is \
                torch.Tensor or np.ndarray or there dict/list combinations. In the ``_forward_learn`` method, data \
                often need to first be stacked in the batch dimension by some utility functions such as \
                ``default_preprocess_learn``. \
                For HAPPO, each element in list is a dict containing at least the following keys: ``obs``, \
                ``action``, ``reward``, ``logit``, ``value``, ``done``. Sometimes, it also contains other keys \
                such as ``weight``.
        Returns:
            - return_infos (:obj:`List[Dict[str, Any]]`): The information list that indicated training result, each \
                training iteration contains append a information dict into the final list. The list will be precessed \
                and recorded in text log and tensorboard. The value of the dict must be python scalar or a list of \
                scalars. For the detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. tip::
            The training procedure of HAPPO is three for loops. The outermost loop trains each agent separately. \
            The middle loop trains all the collected training samples with ``epoch_per_collect`` epochs. The inner \
            loop splits all the data into different mini-batch with the length of ``batch_size``.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for HAPPOPolicy: ``ding.policy.tests.test_happo``.
        """
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        all_data_len = data['obs']['agent_state'].shape[0]
        # fator is the ratio of the old and new strategies of the first m-1 agents, initialized to 1.
        # Each transition has its own factor. ref: http://arxiv.org/abs/2109.11251
        factor = torch.ones(all_data_len, 1)  # (B, 1)
        if self._cuda:
            data = to_device(data, self._device)
            factor = to_device(factor, self._device)
        # process agent dim
        data = self.prepocess_data_agent(data)
        # ====================
        # PPO forward
        # ====================
        return_infos = []
        self._learn_model.train()

        for agent_id in range(self._cfg.agent_num):
            agent_data = {}
            for key, value in data.items():
                if value is not None:
                    if type(value) is dict:
                        agent_data[key] = {k: v[agent_id] for k, v in value.items()}  # not feasible for rnn
                    elif len(value.shape) > 1:
                        agent_data[key] = data[key][agent_id]
                    else:
                        agent_data[key] = data[key]
                else:
                    agent_data[key] = data[key]

            # update factor
            agent_data['factor'] = factor
            # calculate old_logits of all data in buffer for later factor
            inputs = {
                'obs': agent_data['obs'],
                # 'actor_prev_state': agent_data['actor_prev_state'],
                # 'critic_prev_state': agent_data['critic_prev_state'],
            }
            old_logits = self._learn_model.forward(agent_id, inputs, mode='compute_actor')['logit']

            for epoch in range(self._cfg.learn.epoch_per_collect):
                if self._recompute_adv:  # calculate new value using the new updated value network
                    with torch.no_grad():
                        inputs['obs'] = agent_data['obs']
                        # value = self._learn_model.forward(agent_id, agent_data['obs'], mode='compute_critic')['value']
                        value = self._learn_model.forward(agent_id, inputs, mode='compute_critic')['value']
                        inputs['obs'] = agent_data['next_obs']
                        next_value = self._learn_model.forward(agent_id, inputs, mode='compute_critic')['value']
                        if self._value_norm:
                            value *= self._running_mean_std.std
                            next_value *= self._running_mean_std.std

                        traj_flag = agent_data.get('traj_flag', None)  # traj_flag indicates termination of trajectory
                        compute_adv_data = gae_data(
                            value, next_value, agent_data['reward'], agent_data['done'], traj_flag
                        )
                        agent_data['adv'] = gae(compute_adv_data, self._gamma, self._gae_lambda)

                        unnormalized_returns = value + agent_data['adv']

                        if self._value_norm:
                            agent_data['value'] = value / self._running_mean_std.std
                            agent_data['return'] = unnormalized_returns / self._running_mean_std.std
                            self._running_mean_std.update(unnormalized_returns.cpu().numpy())
                        else:
                            agent_data['value'] = value
                            agent_data['return'] = unnormalized_returns

                else:  # don't recompute adv
                    if self._value_norm:
                        unnormalized_return = agent_data['adv'] + agent_data['value'] * self._running_mean_std.std
                        agent_data['return'] = unnormalized_return / self._running_mean_std.std
                        self._running_mean_std.update(unnormalized_return.cpu().numpy())
                    else:
                        agent_data['return'] = agent_data['adv'] + agent_data['value']

                for batch in split_data_generator(agent_data, self._cfg.learn.batch_size, shuffle=True):
                    inputs = {
                        'obs': batch['obs'],
                        # 'actor_prev_state': batch['actor_prev_state'],
                        # 'critic_prev_state': batch['critic_prev_state'],
                    }
                    output = self._learn_model.forward(agent_id, inputs, mode='compute_actor_critic')
                    adv = batch['adv']
                    if self._adv_norm:
                        # Normalize advantage in a train_batch
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Calculate happo error
                    if self._action_space == 'continuous':
                        happo_batch = happo_data(
                            output['logit'], batch['logit'], batch['action'], output['value'], batch['value'], adv,
                            batch['return'], batch['weight'], batch['factor']
                        )
                        happo_loss, happo_info = happo_error_continuous(happo_batch, self._clip_ratio)
                    elif self._action_space == 'discrete':
                        happo_batch = happo_data(
                            output['logit'], batch['logit'], batch['action'], output['value'], batch['value'], adv,
                            batch['return'], batch['weight'], batch['factor']
                        )
                        happo_loss, happo_info = happo_error(happo_batch, self._clip_ratio)
                    wv, we = self._value_weight, self._entropy_weight
                    total_loss = happo_loss.policy_loss + wv * happo_loss.value_loss - we * happo_loss.entropy_loss

                    # actor update
                    # critic update
                    self._actor_optimizer.zero_grad()
                    self._critic_optimizer.zero_grad()
                    total_loss.backward()
                    self._actor_optimizer.step()
                    self._critic_optimizer.step()

                    return_info = {
                        'agent{}_cur_lr'.format(agent_id): self._actor_optimizer.defaults['lr'],
                        'agent{}_total_loss'.format(agent_id): total_loss.item(),
                        'agent{}_policy_loss'.format(agent_id): happo_loss.policy_loss.item(),
                        'agent{}_value_loss'.format(agent_id): happo_loss.value_loss.item(),
                        'agent{}_entropy_loss'.format(agent_id): happo_loss.entropy_loss.item(),
                        'agent{}_adv_max'.format(agent_id): adv.max().item(),
                        'agent{}_adv_mean'.format(agent_id): adv.mean().item(),
                        'agent{}_value_mean'.format(agent_id): output['value'].mean().item(),
                        'agent{}_value_max'.format(agent_id): output['value'].max().item(),
                        'agent{}_approx_kl'.format(agent_id): happo_info.approx_kl,
                        'agent{}_clipfrac'.format(agent_id): happo_info.clipfrac,
                    }
                    if self._action_space == 'continuous':
                        return_info.update(
                            {
                                'agent{}_act'.format(agent_id): batch['action'].float().mean().item(),
                                'agent{}_mu_mean'.format(agent_id): output['logit']['mu'].mean().item(),
                                'agent{}_sigma_mean'.format(agent_id): output['logit']['sigma'].mean().item(),
                            }
                        )
                    return_infos.append(return_info)
            # calculate the factor
            inputs = {
                'obs': agent_data['obs'],
                # 'actor_prev_state': agent_data['actor_prev_state'],
            }
            new_logits = self._learn_model.forward(agent_id, inputs, mode='compute_actor')['logit']
            if self._cfg.action_space == 'discrete':
                dist_new = torch.distributions.categorical.Categorical(logits=new_logits)
                dist_old = torch.distributions.categorical.Categorical(logits=old_logits)
            elif self._cfg.action_space == 'continuous':
                dist_new = Normal(new_logits['mu'], new_logits['sigma'])
                dist_old = Normal(old_logits['mu'], old_logits['sigma'])
            logp_new = dist_new.log_prob(agent_data['action'])
            logp_old = dist_old.log_prob(agent_data['action'])
            if len(logp_new.shape) > 1:
                # for logp with shape(B, action_shape), we need to calculate the product of all action dimensions.
                factor = factor * torch.prod(
                    torch.exp(logp_new - logp_old), dim=-1
                ).reshape(all_data_len, 1).detach()  # attention the shape
            else:
                # for logp with shape(B, ), directly calculate factor
                factor = factor * torch.exp(logp_new - logp_old).reshape(all_data_len, 1).detach()
        return return_infos

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode optimizer and model.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn mode. It contains the \
                state_dict of current policy network and optimizer.
        """
        return {
            'model': self._learn_model.state_dict(),
            'actor_optimizer': self._actor_optimizer.state_dict(),
            'critic_optimizer': self._critic_optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict of learn mode optimizer and model.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn mode. It contains the state_dict \
                of current policy network and optimizer.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self._critic_optimizer.load_state_dict(state_dict['critic_optimizer'])

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For HAPPO, it contains \
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
        self._unroll_len = self._cfg.collect.unroll_len
        assert self._cfg.action_space in ["continuous", "discrete"]
        self._action_space = self._cfg.action_space
        if self._action_space == 'continuous':
            self._collect_model = model_wrap(self._model, wrapper_name='reparam_sample')
        elif self._action_space == 'discrete':
            self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample')
        self._collect_model.reset()
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda
        self._recompute_adv = self._cfg.recompute_adv

    def _forward_collect(self, data: Dict[int, Any]) -> dict:
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
            For more detailed examples, please refer to our unittest for HAPPOPolicy: ``ding.policy.tests.test_happo``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {k: v.transpose(0, 1) for k, v in data.items()}  # not feasible for rnn
        self._collect_model.eval()
        with torch.no_grad():
            outputs = []
            for agent_id in range(self._cfg.agent_num):
                # output = self._collect_model.forward(agent_id, data, mode='compute_actor_critic')
                single_agent_obs = {k: v[agent_id] for k, v in data.items()}
                input = {
                    'obs': single_agent_obs,
                }
                output = self._collect_model.forward(agent_id, input, mode='compute_actor_critic')
                outputs.append(output)
            # transfer data from (M, B, N)->(B, M, N)
            result = {}
            for key in outputs[0].keys():
                if isinstance(outputs[0][key], dict):
                    subkeys = outputs[0][key].keys()
                    stacked_subvalues = {}
                    for subkey in subkeys:
                        stacked_subvalues[subkey] = \
                            torch.stack([output[key][subkey] for output in outputs], dim=0).transpose(0, 1)
                    result[key] = stacked_subvalues
                else:
                    # If Value is tensor, stack it directly
                    if isinstance(outputs[0][key], torch.Tensor):
                        result[key] = torch.stack([output[key] for output in outputs], dim=0).transpose(0, 1)
                    else:
                        # If it is not tensor, assume that it is a non-stackable data type \
                        # (such as int, float, etc.), and directly retain the original value
                        result[key] = [output[key] for output in outputs]
        output = result
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For HAPPO, it contains obs, next_obs, action, reward, done, logit, value.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep.
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
            'action': model_output['action'],
            'logit': model_output['logit'],
            'value': model_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        """
        Overview:
            For a given trajectory (transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. In HAPPO, a train sample is a processed transition with new computed \
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
                last_values = []
                for agent_id in range(self._cfg.agent_num):
                    inputs = {'obs': {k: unsqueeze(v[agent_id], 0) for k, v in data[-1]['next_obs'].items()}}
                    last_value = self._collect_model.forward(agent_id, inputs, mode='compute_actor_critic')['value']
                    last_values.append(last_value)
                last_value = torch.cat(last_values)
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
        assert self._cfg.action_space in ["continuous", "discrete"]
        self._action_space = self._cfg.action_space
        if self._action_space == 'continuous':
            self._eval_model = model_wrap(self._model, wrapper_name='deterministic_sample')
        elif self._action_space == 'discrete':
            self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance by interacting with envs). Forward \
            means that the policy gets some necessary data (mainly observation) from the envs and then returns the \
            action to interact with the envs. ``_forward_eval`` in HAPPO often uses deterministic sample method to \
            get actions while ``_forward_collect`` usually uses stochastic sample method for balance exploration and \
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
            For more detailed examples, please refer to our unittest for HAPPOPolicy: ``ding.policy.tests.test_happo``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        # transfer data from (B, M, N)->(M, B, N)
        data = {k: v.transpose(0, 1) for k, v in data.items()}  # not feasible for rnn
        self._eval_model.eval()
        with torch.no_grad():
            outputs = []
            for agent_id in range(self._cfg.agent_num):
                single_agent_obs = {k: v[agent_id] for k, v in data.items()}
                input = {
                    'obs': single_agent_obs,
                }
                output = self._eval_model.forward(agent_id, input, mode='compute_actor')
                outputs.append(output)
        output = self.revert_agent_data(outputs)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For example about HAPPO, its registered name is ``happo`` and the import_names is \
            ``ding.model.template.havac``.
        """
        return 'havac', ['ding.model.template.havac']

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
        prefixes = [f'agent{i}_' for i in range(self._cfg.agent_num)]
        variables = [prefix + var for prefix in prefixes for var in variables]
        return variables

    def revert_agent_data(self, data: list):
        """
        Overview:
            Revert the data of each agent to the original data format.
        Arguments:
            - data (:obj:`list`): List type data, where each element is the data of an agent of dict type.
        Returns:
            - ret (:obj:`dict`): Dict type data, where each element is the data of an agent of dict type.
        """
        ret = {}
        # Traverse all keys of the first output
        for key in data[0].keys():
            if isinstance(data[0][key], torch.Tensor):
                # If the value corresponding to the current key is tensor, stack N tensors
                stacked_tensor = torch.stack([output[key] for output in data], dim=0)
                ret[key] = stacked_tensor.transpose(0, 1)
            elif isinstance(data[0][key], dict):
                # If the value corresponding to the current key is a dictionary, recursively \
                # call the function to process the contents inside the dictionary.
                ret[key] = self.revert_agent_data([output[key] for output in data])
        return ret
