from typing import List, Dict, Any, Tuple, Union
import copy

import torch
from torch.distributions import Normal, Independent
from easydict import EasyDict

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error
from ding.utils import POLICY_REGISTRY, deep_merge_dicts
from ding.policy.sac import SACPolicy
from ding.policy.common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('mbsac')
class MBSACPolicy(SACPolicy):
    r"""
       Overview:
           Policy class of SAC algorithm with model-based features including value expansion and value gradient.

       Config:
           == ===================================  ========    =============  ================================= =======================
           ID Symbol                               Type        Default Value  Description                       Other(Shape)
           == ===================================  ========    =============  ================================= =======================
           1  ``value_expansion_horizon``          int         0              TODO
           2  ``value_expansion_norm``             bool        True
           3  ``value_expansion_type``             str         'mve'
           4  ``value_expansion_grad_clip_value``  float       .0
           5  ``value_gradient_horizon``           int         0
           6  ``value_gradient_norm``              bool        True 
           7  ``value_gradient_grad_clip_value``   float       .0
           == ===================================  ========    =============  ================================= =======================
       """

    config = dict(
        learn=dict(
            # (int) Model-based value expansion horizon H (arXiv 1803.00101). 
            value_expansion_horizon=0,
            # (int) Whether to use value expansion norm 1/(H + 1).
            value_expansion_norm=True,
            # (str) The style of value expansion to use.
            # Support MVE with td-k trick (arXiv 1803.00101).
            # STEVE style value expansion (arXiv 1807.01675) will be supported in the future.
            value_expansion_type='mve', # 'steve' or 'mve'
            # (float) Gradient clips norm for value expansion.
            # Gradient clip is deactivate when value_expansion_grad_clip_value=0. 
            value_expansion_grad_clip_value=0,

            value_expansion_weight_decay=0,

            # (int) Model-based value gradient horizon H (arXiv 1510.09142). 
            value_gradient_horizon=0,
            # (int) Whether to use value gradient norm 1/(H + 1).
            value_gradient_norm=True,
            # (float) Gradient clips norm for value gradient.
            # Gradient clip is deactivate when value_gradient_grad_clip_value=0.
            value_gradient_grad_clip_value=0,

            value_gradient_weight_decay=0,
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

        self._value_expansion_horizon = self._cfg.learn.value_expansion_horizon
        self._value_expansion_type = self._cfg.learn.value_expansion_type
        self._value_expansion_norm = self._cfg.learn.value_expansion_norm
        self._value_expansion_grad_clip_value = self._cfg.learn.value_expansion_grad_clip_value
        self._value_expansion_weight_decay = self._cfg.learn.value_expansion_weight_decay
        # TODO: implement steve style value expansion
        self._value_expansion_type = 'mve'

        self._value_gradient_horizon = self._cfg.learn.value_gradient_horizon
        self._value_gradient_norm = self._cfg.learn.value_gradient_norm
        self._value_gradient_grad_clip_value = self._cfg.learn.value_gradient_grad_clip_value
        self._value_gradient_weight_decay = self._cfg.learn.value_gradient_weight_decay

        self._history_vars = dict()
        self._history_loss = dict()

        self._optimizer_q = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_q,
            weight_decay=self._value_expansion_weight_decay,
            grad_clip_type='clip_value' if self._value_expansion_grad_clip_value else None,
            clip_value=self._value_expansion_grad_clip_value
        )
        self._optimizer_policy = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_policy,
            weight_decay=self._value_gradient_weight_decay,
            grad_clip_type='clip_value' if self._value_gradient_grad_clip_value else None,
            clip_value=self._value_gradient_grad_clip_value
        )

        # assert (self._value_expansion_horizon > 0 or self._value_gradient_horizon > 0) \
        #         and hasattr(self, '_env_model') and hasattr(self, '_model_env'), "_env_model missing"

    
    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        mode = None
        if 'mode' in data:
            mode = data['mode']
            data = data['data']

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

        if mode == 'policy' or not mode:
            self._update_policy(data['obs'])
        if mode == 'value' or not mode:
            self._update_value(data)

        if self._auto_alpha:
            self._update_temperature(data['obs'])


        self._history_vars['total_loss'] = sum(self._history_loss.values())
        self._history_vars.update(self._history_loss)

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        if mode == 'value' or not mode:
            self._target_model.update(self._learn_model.state_dict())
        return self._history_vars

    
    def _forward_helper(self, obs):


        if not torch.isfinite(obs).all():
            raise RuntimeError(f'obs contains infinite elements: \n{obs}')
        for name, param in self._model.actor.named_parameters():
            if not torch.isfinite(param.data).all():
                raise RuntimeError(f'{name} contains infinite elements: \n{param}')
        (mu, sigma) = self._learn_model.forward(obs, mode='compute_actor')['logit']
        dist = Independent(Normal(mu, sigma), 1)
        pred = dist.rsample()
        action = torch.tanh(pred)
        # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
        log_prob = dist.log_prob(pred) + 2 * torch.log(torch.cosh(pred)).sum(-1)

        return action, log_prob


    def _update_value(self, data):

        value_loss = 0

        obs      = data['obs']
        action   = data['action']
        next_obs = data['next_obs']
        reward   = data['reward']
        done     = data['done']

        # TODO: steve
        if self._value_expansion_type == 'mve':

            obs_list       = [obs, next_obs]
            action_list    = [action]
            reward_list    = [reward]
            done_list      = [done]
            done_mask_list = [torch.zeros_like(next_obs.sum(-1)).bool(), done.bool()]

            with torch.no_grad():
                # td-k trick

                for _ in range(self._value_expansion_horizon):
                    action, log_prob = self._forward_helper(next_obs)
                    reward, next_obs  = self._env_model.batch_predict(next_obs, action)
                    reward = reward - self._alpha * log_prob
                    done = self._model_env.termination_fn(next_obs)
                    done_mask = done_mask_list[-1] | done
                    obs_list.append(next_obs)
                    action_list.append(action)
                    reward_list.append(reward)
                    done_list.append(done)
                    done_mask_list.append(done_mask)

                action, log_prob = self._forward_helper(next_obs)
                eval_data = {'obs': next_obs, 'action': action}
                target_q_value = self._target_model.forward(eval_data, mode='compute_critic')['q_value']
                # the value of a policy according to the maximum entropy objective
                if self._twin_critic:
                    # find min one as target q value
                    target_q_value = torch.min(target_q_value[0],
                                               target_q_value[1]) - self._alpha * log_prob
                else:
                    target_q_value = target_q_value - self._alpha * log_prob

            for obs, action, reward, done, done_mask in reversed(
                    list(zip(obs_list[:-1], action_list, reward_list, done_list, done_mask_list[:-1]))):
                
                eval_data = {'obs': obs, 'action': action}
                q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']

                if self._twin_critic:
                    q_data0 = v_1step_td_data(q_value[0], target_q_value, reward, done.int(), 1-done_mask.int())
                    loss, td_error_per_sample0 = v_1step_td_error(q_data0, self._gamma)
                    value_loss += loss
                    q_data1 = v_1step_td_data(q_value[1], target_q_value, reward, done.int(), 1-done_mask.int())
                    loss, td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
                    value_loss += loss
                    td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2
                else:
                    q_data = v_1step_td_data(q_value, target_q_value, reward, done.int(), 1-done_mask.int())
                    loss, td_error_per_sample = v_1step_td_error(q_data, self._gamma)
                    value_loss += loss 

                target_q_value = reward + (1 - done.int()) * self._gamma * target_q_value

            if self._value_expansion_norm:
                value_loss = value_loss / (self._value_expansion_horizon + 1) 

            self._history_loss['value_loss'] = value_loss
            self._history_vars.update({
                'cur_lr_q': self._optimizer_q.defaults['lr'],
                # 'priority': td_error_per_sample.abs().tolist(),
                'td_error': td_error_per_sample.detach().mean().item(),
                've_rollout_termination_ratio': done_mask_list[-2].sum() / done_mask_list[-2].numel(),
            })

            self._optimizer_q.zero_grad()
            self._history_loss['value_loss'].backward()
            for name, param in self._model.critic.named_parameters():
                if not torch.isfinite(param.grad).all():
                    raise RuntimeError(f'{name} gradient contains infinite elements: \n{param}')
            self._optimizer_q.step()
            for name, param in self._model.critic.named_parameters():
                if not torch.isfinite(param.data).all():
                    raise RuntimeError(f'{name} data contains infinite elements after step: \n{name}')
                

    def _update_policy(self, obs):
        policy_loss = 0

        done_mask = torch.zeros_like(obs.sum(-1)).bool()
        for i in range(self._value_gradient_horizon):
            action, log_prob = self._forward_helper(obs)
            reward, obs  = self._env_model.batch_predict(obs, action)
            policy_loss += (self._gamma ** i) * (
                (1 - done_mask.int()) * (self._alpha * log_prob  - reward)).mean()
            done = self._model_env.termination_fn(obs)
            done_mask = done_mask | done

        # calculate the q value for the final state
        action, log_prob = self._forward_helper(obs)
        eval_data = {'obs': obs, 'action': action}
        new_q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']
        if self._twin_critic:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])
        # TODO: if self._value_expansion_horizon > 0 and self._value_expansion_type = 'steve'
        #   new_q_value = new_q_value.mean(0)
        policy_loss += (self._gamma ** self._value_gradient_horizon) * (
            (1 - done_mask.int()) * (self._alpha * log_prob - new_q_value)).mean()

        if self._value_gradient_norm:
            policy_loss = policy_loss / (self._value_gradient_horizon + 1)

        self._history_loss['policy_loss'] = policy_loss
        self._history_vars.update({
            'vg_rollout_termination_ratio': done_mask.sum() / done_mask.numel(),
            'cur_lr_p': self._optimizer_policy.defaults['lr']
        })

        # update policy network
        self._optimizer_policy.zero_grad()
        with torch.autograd.detect_anomaly():
            self._history_loss['policy_loss'].backward()
            # for name, param in self._model.actor.named_parameters():
            #     if not torch.isfinite(param.grad).all():
            #         raise RuntimeError(f'{name} gradient contains infinite elements: \n{param}')
            self._optimizer_policy.step()
            # for name, param in self._model.actor.named_parameters():
            #     if not torch.isfinite(param.data).all():
            #         raise RuntimeError(f'{name} data contains infinite elements: \n{param}')


    def _update_temperature(self, obs):
        action, log_prob = self._forward_helper(obs)

        if self._log_space:
            log_prob = log_prob + self._target_entropy
            self._history_loss['alpha_loss'] = -(self._log_alpha * log_prob.detach()).mean()

            self._alpha_optim.zero_grad()
            self._history_loss['alpha_loss'].backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        else:
            log_prob = log_prob + self._target_entropy
            self._history_loss['alpha_loss'] = -(self._alpha * log_prob.detach()).mean()

            self._alpha_optim.zero_grad()
            self._history_loss['alpha_loss'].backward()
            self._alpha_optim.step()
            self._alpha = max(0, self._alpha)

        self._history_vars.update({'alpha': self._alpha.item()})


    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """

        return [
            'alpha_loss',
            'policy_loss',
            'value_loss',
            'cur_lr_q',
            'cur_lr_p',
            'alpha',
            'td_error',
            've_rollout_termination_ratio',
            'vg_rollout_termination_ratio',
        ]

