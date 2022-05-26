from typing import Dict, Any, List
from functools import partial
import copy

import torch
from torch import Tensor
from torch import nn
from torch.distributions import Normal, Independent, TransformedDistribution, TanhTransform
from easydict import EasyDict

from ding.torch_utils import to_device
from ding.utils import POLICY_REGISTRY, deep_merge_dicts
from ding.policy import SACPolicy
from ding.rl_utils import generalized_lambda_returns
from ding.policy.common_utils import default_preprocess_learn

from .utils import q_evaluation


@POLICY_REGISTRY.register('mbsac')
class MBSACPolicy(SACPolicy):
    r"""
       Overview:
           Model based SAC.

       Config:
            TODO
       """

    config = dict(
        learn=dict(
            # (float) Lambda for TD return.
            _lambda=0.8,
            # (float) Gradient clip norm.
            grad_clip=100,
            # (bool) Whether to sample transitions or only states from environment buffer.
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

        self._lambda = self._cfg.learn._lambda
        self._grad_clip = self._cfg.learn.grad_clip
        self._sample_state = self._cfg.learn.sample_state

        # TODO: complete auto alpha
        self._auto_alpha = False

        # TODO: use TanhTransform()
        def actor_fn(obs: Tensor):
            # (mu, sigma) = self._learn_model.forward(
            #     obs, mode='compute_actor')['logit']
            # # enforce action bounds
            # dist = TransformedDistribution(
            #     Independent(Normal(mu, sigma), 1), [TanhTransform()])
            # action = dist.rsample()
            # log_prob = dist.log_prob(action)
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
            obss = torch.concat([data['obs'][None, :], obss])
            actions = torch.concat([data['action'][None, :], actions])
            rewards = torch.concat([data['reward'][None, :], rewards])
            aug_rewards = torch.concat([torch.zeros_like(data['reward'])[None, :], aug_rewards])
            dones = torch.concat([data['done'][None, :], dones])

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
        # note: torch.tensor(0).log().exp() = tensor(0.)
        # example: tensor([0., 0., 1., 0.]).log().cumsum(dim=0).exp() = tensor([1., 1., 0., 0.])
        weight = (1 - dones[:-1].detach()).log().cumsum(dim=0).exp()

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

        self._update(loss_dict)

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        self._target_model.update(self._learn_model.state_dict())

        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            # 'priority': td_error_per_sample.abs().tolist(),
            # 'td_error': td_error_per_sample.detach().mean().item(),
            # 'alpha': self._alpha.item(),
            'target_q_value': target_q_values.detach().mean().item(),
            **loss_dict
        }

    def _update(self, loss_dict):
        # update critic
        self._optimizer_q.zero_grad()
        loss_dict['critic_loss'].backward()
        nn.utils.clip_grad_norm_(self._model.critic.parameters(), self._grad_clip)
        self._optimizer_q.step()
        # update policy
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        nn.utils.clip_grad_norm_(self._model.actor.parameters(), self._grad_clip)
        self._optimizer_policy.step()
        # update temperature
        # self._alpha_optim.zero_grad()
        # loss_dict['alpha_loss'].backward()
        # self._alpha_optim.step()
