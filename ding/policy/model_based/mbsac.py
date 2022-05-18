from typing import List, Dict, Any, Tuple, Union
from functools import partial
import copy

import torch
from torch.distributions import Normal, Independent, TransformedDistribution, TanhTransform
from easydict import EasyDict

from ding.torch_utils import to_device
from ding.utils import POLICY_REGISTRY, deep_merge_dicts
from ding.policy.sac import SACPolicy
from ding.rl_utils import generalized_lambda_returns
from ding.policy.common_utils import default_preprocess_learn

from .utils import flatten_batch, unflatten_batch, q_evaluation, rollout

@POLICY_REGISTRY.register('mbsac')
class MBSACPolicy(SACPolicy):
    r"""
       Overview:
           Policy class of SAC algorithm with model-based features including value expansion and value gradient.

       Config:
            TODO
       """

    config = dict(
        learn=dict(
            # (int) Model-based value expansion horizon H (arXiv 1803.00101). 
            # (int) Model-based value gradient horizon H (arXiv 1510.09142). 
            horizon=0,
            # 
            _lambda=0.8,
            # (bool) Whether to sample states or transitions from environment buffer. 
            sample_state=True,


            # note:
            # horizon=0, sample_state=False -> vanilla sac
            # horizon!=0, _lambda=1  -> value gradient / value expansion
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

        # self._ve_horizon = self._cfg.learn.ve_horizon
        # self._vg_horizon = self._cfg.learn.vg_horizon
        self._horizon = self._cfg.learn.horizon
        self._lambda = self._cfg.learn._lambda
        self._sample_state = self._cfg.learn._sample_state

        # TODO: set _env and assert exists
        # assert (self._value_expansion_horizon > 0 or self._value_gradient_horizon > 0) \
        #         and hasattr(self, '_env_model') and hasattr(self, '_model_env'), "_env_model missing"

        # helper functions to provide unified API
        def forward_fn(obs):
            (mu, sigma) = self._learn_model.forward(
                obs, mode='compute_actor')['logit']
            dist = TransformedDistribution(
                Independent(Normal(mu, sigma), 1), [TanhTransform()])
            action = dist.rsample()
            log_prob = dist.log_prob(action)
            return action, -self._alpha.detach()*log_prob
        self._forward_fn = forward_fn

        def eval_fn(obss, actions, model):
            eval_data = {'obs': obss, 'action': actions}
            q_values = model.forward(eval_data, mode='compute_critic')['q_value']
            return q_values
        self._eval_fn = eval_fn


    def _forward_learn(self, data: dict) -> Dict[str, Any]:
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

        self._learn_model.train()
        self._target_model.train()
            
        if self._sample_state:
            obss, actions, rewards, aug_rewards, dones = \
                rollout(data['obs'], self.forward_fn, self._ve_horizon)
        else:
            obss, actions, rewards, aug_rewards, dones = \
                rollout(data['next_obs'], self.forward_fn, self._ve_horizon)
            obss        = torch.concat([obss,    data['obs'][None,:]])
            actions     = torch.concat([actions, data['action'][None,:]])
            rewards     = torch.concat([rewards, data['reward'][None,:]])
            aug_rewards = torch.concat([aug_rewards, torch.zeros_like(data['reward'])[None,:]])
            dones       = torch.concat([dones,   data['done'][None,:]])


        target_q_values = q_evaluation(obss, actions, 
                                       partial(self._eval_fn, model=self._target_model))
        if self._twin_critic:
            target_q_values = torch.min(target_q_values[0],
                                        target_q_values[1]) + aug_rewards
        else:
            target_q_values = target_q_values + aug_rewards

        lambda_return = generalized_lambda_returns(target_q_values, 
                                                   rewards, 
                                                   self._gamma, 
                                                   self._lambda, 
                                                   dones)

        weight = (1 - dones.detach()).log().cumsum(dim=0).exp()

        # critic_loss
        q_values = q_evaluation(obss.detach(), actions.detach(), 
                                partial(self._eval_fn, model=self._learn_model))
        if self._twin_critic:
            critic_loss = 0.5 * torch.square(q_values[0], lambda_return.detach()) \
                        + 0.5 * torch.square(q_values[1], lambda_return.detach())
        else: 
            critic_loss = 0.5 * torch.square(q_values, lambda_return.detach())
        critic_loss = (critic_loss * weight).mean()

        # policy_loss
        policy_loss = -(lambda_return * weight).mean()

        # alpha_loss

        loss_dict = {
            'critic_loss': critic_loss.detach(),
            'policy_loss': policy_loss.detach(),
            'alpha_loss':  alpha_loss.detach(),
        }

        self.update(loss_dict)

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        self._target_model.update(self._learn_model.state_dict())

        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'priority': td_error_per_sample.abs().tolist(),
            'td_error': td_error_per_sample.detach().mean().item(),
            'alpha': self._alpha.item(),
            'target_q_value': target_q_value.detach().mean().item(),
            **loss_dict
        }
    
    def _alpha_loss(self, aug_rewards):
        log_prob = -aug_rewards/self._alpha
        if self._auto_alpha:
            if self._log_space:
                log_prob = log_prob + self._target_entropy
                alpha_loss = -(self._log_alpha * log_prob.detach()).mean()

                self._alpha_optim.zero_grad()
                loss_dict['alpha_loss'].backward()
                self._alpha_optim.step()
                self._alpha = self._log_alpha.detach().exp()
            else:
                log_prob = log_prob + self._target_entropy
                loss_dict['alpha_loss'] = -(self._alpha * log_prob.detach()).mean()

                self._alpha_optim.zero_grad()
                loss_dict['alpha_loss'].backward()
                self._alpha_optim.step()
                self._alpha = max(0, self._alpha)

    
    def _update(self, loss_dict):
        # update critic
        self._optimizer_q.zero_grad()
        loss_dict['critic_loss'].backward()
        self._optimizer_q.step()
        # update policy
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()
        # update temperature
        self._alpha_optim.zero_grad()
        loss_dict['alpha_loss'].backward()
        self._alpha_optim.step()

