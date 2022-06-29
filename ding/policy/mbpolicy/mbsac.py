from typing import Dict, Any, List
from functools import partial
import copy

import torch
from torch import Tensor
from torch import nn
from torch.distributions import Normal, Independent, TransformedDistribution, TanhTransform
from easydict import EasyDict

from ding.torch_utils import to_device, fold_batch, unfold_batch, unsqueeze_repeat
from ding.utils import POLICY_REGISTRY, deep_merge_dicts
from ding.policy import SACPolicy
from ding.rl_utils import generalized_lambda_returns
from ding.policy.common_utils import default_preprocess_learn

from .utils import q_evaluation


@POLICY_REGISTRY.register('mbsac')
class MBSACPolicy(SACPolicy):
    r"""
       Overview:
           Model based SAC with value expansion (arXiv: 1803.00101)
           and value gradient (arXiv: 1510.09142) w.r.t lambda-return.

           https://arxiv.org/pdf/1803.00101.pdf
           https://arxiv.org/pdf/1510.09142.pdf

       Config:
           == ====================   ========    =============  ==================================
           ID Symbol                 Type        Default Value  Description
           == ====================   ========    =============  ==================================
           1  ``learn._lambda``      float       0.8            | Lambda for TD-lambda return.
           2  ``learn.grad_clip`     float       100.0          | Max norm of gradients.
           3  ``learn.sample_state`` bool        True           | Whether to sample states or tra-
                                                                |   nsitions from environment buffer.
           == ====================   ========    =============  ==================================

        .. note::
            For other configs, please refer to ding.policy.sac.SACPolicy.
       """

    config = dict(
        learn=dict(
            # (float) Lambda for TD-lambda return.
            lambda_=0.8,
            # (float) Max norm of gradients.
            grad_clip=100,
            # (bool) Whether to sample states or transitions from environment buffer.
            sample_state=True,
        )
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = copy.deepcopy(cls.config)
        cfg = EasyDict(deep_merge_dicts(super().config, cfg))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def _init_learn(self) -> None:
        super()._init_learn()
        self._target_model.requires_grad_(False)

        self._lambda = self._cfg.learn.lambda_
        self._grad_clip = self._cfg.learn.grad_clip
        self._sample_state = self._cfg.learn.sample_state
        self._auto_alpha = self._cfg.learn.auto_alpha
        # TODO: auto alpha
        assert not self._auto_alpha, "NotImplemented"

        # TODO: TanhTransform leads to NaN
        def actor_fn(obs: Tensor):
            # (mu, sigma) = self._learn_model.forward(
            #     obs, mode='compute_actor')['logit']
            # # enforce action bounds
            # dist = TransformedDistribution(
            #     Independent(Normal(mu, sigma), 1), [TanhTransform()])
            # action = dist.rsample()
            # log_prob = dist.log_prob(action)
            # return action, -self._alpha.detach() * log_prob
            (mu, sigma) = self._learn_model.forward(obs, mode='compute_actor')['logit']
            dist = Independent(Normal(mu, sigma), 1)
            pred = dist.rsample()
            action = torch.tanh(pred)

            log_prob = dist.log_prob(
                pred
            ) + 2 * (pred + torch.nn.functional.softplus(-2. * pred) - torch.log(torch.tensor(2.))).sum(-1)
            return action, -self._alpha.detach() * log_prob

        self._actor_fn = actor_fn

        def critic_fn(obss: Tensor, actions: Tensor, model: nn.Module):
            eval_data = {'obs': obss, 'action': actions}
            q_values = model.forward(eval_data, mode='compute_critic')['q_value']
            return q_values

        self._critic_fn = critic_fn

    def _forward_learn(self, data: dict, world_model, envstep) -> Dict[str, Any]:
        # preprocess data
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if self._cuda:
            data = to_device(data, self._device)

        if len(data['action'].shape) == 1:
            data['action'] = data['action'].unsqueeze(1)

        self._learn_model.train()
        self._target_model.train()

        # TODO: use treetensor
        # rollout length is determined by world_model.rollout_length_scheduler
        if self._sample_state:
            # data['reward'], ... are not used
            obss, actions, rewards, aug_rewards, dones = \
                world_model.rollout(data['obs'], self._actor_fn, envstep)
        else:
            obss, actions, rewards, aug_rewards, dones = \
                world_model.rollout(data['next_obs'], self._actor_fn, envstep)
            obss = torch.concat([data['obs'].unsqueeze(0), obss])
            actions = torch.concat([data['action'].unsqueeze(0), actions])
            rewards = torch.concat([data['reward'].unsqueeze(0), rewards])
            aug_rewards = torch.concat([torch.zeros_like(data['reward']).unsqueeze(0), aug_rewards])
            dones = torch.concat([data['done'].unsqueeze(0), dones])

        dones = torch.concat([torch.zeros_like(data['done']).unsqueeze(0), dones])

        # (T+1, B)
        target_q_values = q_evaluation(obss, actions, partial(self._critic_fn, model=self._target_model))
        if self._twin_critic:
            target_q_values = torch.min(target_q_values[0], target_q_values[1]) + aug_rewards
        else:
            target_q_values = target_q_values + aug_rewards

        # (T, B)
        lambda_return = generalized_lambda_returns(target_q_values, rewards, self._gamma, self._lambda, dones[1:])

        # (T, B)
        # If S_t terminates, we should not consider loss from t+1,...
        weight = (1 - dones[:-1].detach()).cumprod(dim=0)

        # (T+1, B)
        q_values = q_evaluation(obss.detach(), actions.detach(), partial(self._critic_fn, model=self._learn_model))
        if self._twin_critic:
            critic_loss = 0.5 * torch.square(q_values[0][:-1] - lambda_return.detach()) \
                        + 0.5 * torch.square(q_values[1][:-1] - lambda_return.detach())
        else:
            critic_loss = 0.5 * torch.square(q_values[:-1] - lambda_return.detach())

        # value expansion loss
        critic_loss = (critic_loss * weight).mean()

        # value gradient loss
        policy_loss = -(lambda_return * weight).mean()

        # alpha_loss  = None

        loss_dict = {
            'critic_loss': critic_loss,
            'policy_loss': policy_loss,
            # 'alpha_loss':  alpha_loss.detach(),
        }

        norm_dict = self._update(loss_dict)

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        self._target_model.update(self._learn_model.state_dict())

        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'alpha': self._alpha.item(),
            'target_q_value': target_q_values.detach().mean().item(),
            **norm_dict,
            **loss_dict,
        }

    def _update(self, loss_dict):
        # update critic
        self._optimizer_q.zero_grad()
        loss_dict['critic_loss'].backward()
        critic_norm = nn.utils.clip_grad_norm_(self._model.critic.parameters(), self._grad_clip)
        self._optimizer_q.step()
        # update policy
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        policy_norm = nn.utils.clip_grad_norm_(self._model.actor.parameters(), self._grad_clip)
        self._optimizer_policy.step()
        # update temperature
        # self._alpha_optim.zero_grad()
        # loss_dict['alpha_loss'].backward()
        # self._alpha_optim.step()
        return {'policy_norm': policy_norm, 'critic_norm': critic_norm}

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        alpha_loss = ['alpha_loss'] if self._auto_alpha else []
        return [
            'policy_loss',
            'critic_loss',
            'policy_norm',
            'critic_norm',
            'cur_lr_q',
            'cur_lr_p',
            'alpha',
            'target_q_value',
        ] + alpha_loss


@POLICY_REGISTRY.register('stevesac')
class STEVESACPolicy(SACPolicy):
    r"""
       Overview:
           Model based SAC with stochastic value expansion (arXiv 1807.01675).\
           This implementation also uses value gradient w.r.t the same STEVE target.

           https://arxiv.org/pdf/1807.01675.pdf

       Config:
           == ====================    ========    =============  =====================================
           ID Symbol                  Type        Default Value  Description
           == ====================    ========    =============  =====================================
           1  ``learn.grad_clip`      float       100.0          | Max norm of gradients.
           2  ``learn.ensemble_size`` int         1              | The number of ensemble world models.
           == ====================    ========    =============  =====================================

        .. note::
            For other configs, please refer to ding.policy.sac.SACPolicy.
       """

    config = dict(
        learn=dict(
            # (float) Max norm of gradients.
            grad_clip=100,
            # (int) The number of ensemble world models.
            ensemble_size=1,
        )
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = copy.deepcopy(cls.config)
        cfg = EasyDict(deep_merge_dicts(super().config, cfg))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def _init_learn(self) -> None:
        super()._init_learn()
        self._target_model.requires_grad_(False)

        self._grad_clip = self._cfg.learn.grad_clip
        self._ensemble_size = self._cfg.learn.ensemble_size
        self._auto_alpha = self._cfg.learn.auto_alpha
        # TODO: auto alpha
        assert not self._auto_alpha, "NotImplemented"

        def actor_fn(obs: Tensor):
            obs, dim = fold_batch(obs, 1)
            (mu, sigma) = self._learn_model.forward(obs, mode='compute_actor')['logit']
            dist = Independent(Normal(mu, sigma), 1)
            pred = dist.rsample()
            action = torch.tanh(pred)

            log_prob = dist.log_prob(
                pred
            ) + 2 * (pred + torch.nn.functional.softplus(-2. * pred) - torch.log(torch.tensor(2.))).sum(-1)
            aug_reward = -self._alpha.detach() * log_prob

            return unfold_batch(action, dim), unfold_batch(aug_reward, dim)

        self._actor_fn = actor_fn

        def critic_fn(obss: Tensor, actions: Tensor, model: nn.Module):
            eval_data = {'obs': obss, 'action': actions}
            q_values = model.forward(eval_data, mode='compute_critic')['q_value']
            return q_values

        self._critic_fn = critic_fn

    def _forward_learn(self, data: dict, world_model, envstep) -> Dict[str, Any]:
        # preprocess data
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if self._cuda:
            data = to_device(data, self._device)

        if len(data['action'].shape) == 1:
            data['action'] = data['action'].unsqueeze(1)

        # [B, D] -> [E, B, D]
        data['next_obs'] = unsqueeze_repeat(data['next_obs'], self._ensemble_size)
        data['reward'] = unsqueeze_repeat(data['reward'], self._ensemble_size)
        data['done'] = unsqueeze_repeat(data['done'], self._ensemble_size)

        self._learn_model.train()
        self._target_model.train()

        obss, actions, rewards, aug_rewards, dones = \
            world_model.rollout(data['next_obs'], self._actor_fn, envstep, keep_ensemble=True)
        rewards = torch.concat([data['reward'].unsqueeze(0), rewards])
        dones = torch.concat([data['done'].unsqueeze(0), dones])

        # (T, E, B)
        target_q_values = q_evaluation(obss, actions, partial(self._critic_fn, model=self._target_model))
        if self._twin_critic:
            target_q_values = torch.min(target_q_values[0], target_q_values[1]) + aug_rewards
        else:
            target_q_values = target_q_values + aug_rewards

        # (T+1, E, B)
        discounts = ((1 - dones) * self._gamma).cumprod(dim=0)
        discounts = torch.concat([torch.ones_like(discounts)[:1], discounts])
        # (T, E, B)
        cum_rewards = (rewards * discounts[:-1]).cumsum(dim=0)
        discounted_q_values = target_q_values * discounts[1:]
        steve_return = cum_rewards + discounted_q_values
        # (T, B)
        steve_return_mean = steve_return.mean(1)
        with torch.no_grad():
            steve_return_inv_var = 1 / (1e-8 + steve_return.var(1, unbiased=False))
            steve_return_weight = steve_return_inv_var / (1e-8 + steve_return_inv_var.sum(dim=0))
        # (B, )
        steve_return = (steve_return_mean * steve_return_weight).sum(0)

        eval_data = {'obs': data['obs'], 'action': data['action']}
        q_values = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']
        if self._twin_critic:
            critic_loss = 0.5 * torch.square(q_values[0] - steve_return.detach()) \
                        + 0.5 * torch.square(q_values[1] - steve_return.detach())
        else:
            critic_loss = 0.5 * torch.square(q_values - steve_return.detach())

        critic_loss = critic_loss.mean()

        policy_loss = -steve_return.mean()

        # alpha_loss  = None

        loss_dict = {
            'critic_loss': critic_loss,
            'policy_loss': policy_loss,
            # 'alpha_loss':  alpha_loss.detach(),
        }

        norm_dict = self._update(loss_dict)

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        self._target_model.update(self._learn_model.state_dict())

        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'alpha': self._alpha.item(),
            'target_q_value': target_q_values.detach().mean().item(),
            **norm_dict,
            **loss_dict,
        }

    def _update(self, loss_dict):
        # update critic
        self._optimizer_q.zero_grad()
        loss_dict['critic_loss'].backward()
        critic_norm = nn.utils.clip_grad_norm_(self._model.critic.parameters(), self._grad_clip)
        self._optimizer_q.step()
        # update policy
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        policy_norm = nn.utils.clip_grad_norm_(self._model.actor.parameters(), self._grad_clip)
        self._optimizer_policy.step()
        # update temperature
        # self._alpha_optim.zero_grad()
        # loss_dict['alpha_loss'].backward()
        # self._alpha_optim.step()
        return {'policy_norm': policy_norm, 'critic_norm': critic_norm}

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        alpha_loss = ['alpha_loss'] if self._auto_alpha else []
        return [
            'policy_loss',
            'critic_loss',
            'policy_norm',
            'critic_norm',
            'cur_lr_q',
            'cur_lr_p',
            'alpha',
            'target_q_value',
        ] + alpha_loss
