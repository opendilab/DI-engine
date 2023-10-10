from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
import copy
import numpy as np

from ding.torch_utils import Adam, to_device, to_dtype, unsqueeze, ContrastiveLoss
from ding.rl_utils import happo_data, happo_error, happo_policy_error, happo_policy_data, get_gae_with_default_last_value, \
    v_nstep_td_data, v_nstep_td_error, get_nstep_return_data, get_train_sample, gae, gae_data, happo_error_continuous, \
    get_gae
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY, split_data_generator, RunningMeanStd
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('happo')
class HAPPOPolicy(Policy):
    r"""
    Overview:
        Policy class of on policy version HAPPO algorithm.
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
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config and the main model.
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
        """
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # PPO forward
        # ====================
        return_infos = []
        self._learn_model.train()
        factor = torch.ones(*data[0]['obs'].shape[:-1], 1)     # (L, M, 1)

        for agent_id in self.agent_num:
            agent_data = data[agent_id]
            # update factor
            agent_data['factor'] = factor
            # calculate old_logits of all data in buffer for later factor
            inputs = {
                'obs': agent_data['obs'],
                'actor_prev_state': agent_data['actor_prev_state'],
            }
            old_logits = self._learn_model.forward(agent_id, inputs, mode='compute_actor')

            for epoch in range(self._cfg.learn.epoch_per_collect):
                if self._recompute_adv:  # calculate new value using the new updated value network
                    with torch.no_grad():
                        value = self._learn_model.forward(agent_data['obs'], mode='compute_critic')['value']
                        next_value = self._learn_model.forward(agent_data['next_obs'], mode='compute_critic')['value']
                        if self._value_norm:
                            value *= self._running_mean_std.std
                            next_value *= self._running_mean_std.std

                        traj_flag = agent_data.get('traj_flag', None)  # traj_flag indicates termination of trajectory
                        compute_adv_data = gae_data(value, next_value, agent_data['reward'], agent_data['done'], traj_flag)
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
                        'actor_prev_state': batch['actor_prev_state'],
                        'critic_prev_state': batch['critic_prev_state'],
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
                            batch['return'], batch['weight']
                        )
                        happo_loss, happo_info = happo_error_continuous(happo_batch, self._clip_ratio)
                    elif self._action_space == 'discrete':
                        happo_batch = happo_data(
                            output['logit'], batch['logit'], batch['action'], output['value'], batch['value'], adv,
                            batch['return'], batch['weight']
                        )
                        happo_loss, happo_info = happo_error(happo_batch, self._clip_ratio)
                    wv, we = self._value_weight, self._entropy_weight
                    total_loss = happo_loss.policy_loss + wv * happo_loss.value_loss - we * happo_loss.entropy_loss

                    self._optimizer.zero_grad()
                    total_loss.backward()
                    self._optimizer.step()

                    # calculate the factor
                    inputs = {
                        'obs': agent_data['obs'],
                        'actor_prev_state': agent_data['actor_prev_state'],
                    }
                    new_logits = self._learn_model.forward(agent_id, inputs, mode='compute_actor')
                    factor = factor*torch.prod(torch.exp(new_logits-old_logits),dim=-1) # attention the shape

                    return_info = {
                        'cur_lr': self._optimizer.defaults['lr'],
                        'total_loss': total_loss.item(),
                        'policy_loss': happo_loss.policy_loss.item(),
                        'value_loss': happo_loss.value_loss.item(),
                        'entropy_loss': happo_loss.entropy_loss.item(),
                        'adv_max': adv.max().item(),
                        'adv_mean': adv.mean().item(),
                        'value_mean': output['value'].mean().item(),
                        'value_max': output['value'].max().item(),
                        'approx_kl': happo_info.approx_kl,
                        'clipfrac': happo_info.clipfrac,
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

    def default_model(self) -> Tuple[str, List[str]]:
        return 'havac', ['ding.model.template.mavac']

    def _monitor_vars_learn(self) -> List[str]:
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