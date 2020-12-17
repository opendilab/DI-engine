import sys
import os

from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch

from nervex.torch_utils import Adam
from nervex.rl_utils import value_data, soft_q_data, soft_q_error, value_error, Adder
from nervex.model import SAC
from nervex.agent import Agent
from nervex.policy.base_policy import Policy, register_policy
from nervex.policy.common_policy import CommonPolicy


class SACPolicy(CommonPolicy):

    def _init_learn(self) -> None:
        self._optimizer_q = Adam(
            self._model.q_net.parameters(),
            lr=self._cfg.learn.learning_rate_q,
            weight_decay=self._cfg.learn.weight_decay
        )
        self._optimizer_value = Adam(
            self._model.value_net.parameters(),
            lr=self._cfg.learn.learning_rate_value,
            weight_decay=self._cfg.learn.weight_decay
        )
        self._optimizer_policy = Adam(
            self._model.policy_net.parameters(),
            lr=self._cfg.learn.learning_rate_policy,
            weight_decay=self._cfg.learn.weight_decay
        )
        self._agent = Agent(self._model)
        algo_cfg = self._cfg.learn.algo
        self._algo_cfg_learn = algo_cfg
        self._gamma = algo_cfg.discount_factor
        self._reparameterization = algo_cfg.reparameterization
        self._policy_std_reg_weight = algo_cfg.policy_std_reg_weight
        self._policy_mean_reg_weight = algo_cfg.policy_mean_reg_weight

        self._use_twin_q = algo_cfg.use_twin_q

        self._agent.add_model('target', update_type='momentum', update_kwargs={'theta': algo_cfg.target_theta})
        if algo_cfg.use_noise:
            self._agent.add_plugin(
                'target',
                'action_noise',
                noise_type='gauss',
                noise_kwargs={
                    'mu': 0.0,
                    'sigma': algo_cfg.noise_sigma
                },
                noise_range=algo_cfg.noise_range,
            )
        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.add_plugin('target', 'grad', enable_grad=False)
        self._agent.mode(train=True)
        self._agent.target_mode(train=True)
        self._agent.reset()
        self._agent.target_reset()
        self._learn_setting_set = {}
        self._forward_learn_cnt = 0

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        loss_dict = {}

        obs = data.get('obs')
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        action = data.get('action')
        done = data.get('done')

        # evaluate
        eval_data = self._agent.forward(data, param={'mode': 'evaluate'})

        mean = eval_data["mean"]
        log_std = eval_data["log_std"]
        new_action = eval_data["action"]
        log_prob = eval_data["log_prob"]

        # predict q_value and v_value
        q_value = self._agent.forward(data, param={'mode': 'compute_q'})['q_value']
        v_value = self._agent.forward(data, param={'mode': 'compute_value'})['v_value']

        # compute q loss
        next_data = {'obs': next_obs}
        target_v_value = self._agent.target_forward(next_data, param={'mode': 'compute_value'})['v_value']
        q_data = soft_q_data(target_v_value, reward, done, q_value)
        q_loss = soft_q_error(q_data, self._gamma)
        loss_dict['q_loss'] = q_loss

        # compute value loss
        eval_data['obs'] = obs
        new_q_value = self._agent.forward(eval_data, param={'mode': 'compute_q'})['q_value']
        next_v_value = new_q_value - log_prob
        v_data = value_data(v_value, next_v_value)
        value_loss = value_error(v_data)
        loss_dict['value_loss'] = value_loss

        # compute policy loss
        if not self._reparameterization:
            target_log_policy = new_q_value - v_value
            policy_loss = (log_prob * (log_prob - target_log_policy).detach()).mean()
        else:
            policy_loss = (log_prob - new_q_value.detach()).mean()

        std_reg_loss = self._policy_std_reg_weight * (log_std ** 2).mean()
        mean_reg_loss = self._policy_mean_reg_weight * (mean ** 2).mean()

        policy_loss += std_reg_loss + mean_reg_loss
        loss_dict['policy_loss'] = policy_loss
        loss_dict['total_loss'] = sum(loss_dict.values())

        # update
        self._optimizer_q.zero_grad()
        q_loss.backward()
        self._optimizer_q.step()

        self._optimizer_value.zero_grad()
        value_loss.backward()
        self._optimizer_value.step()

        self._optimizer_policy.zero_grad()
        policy_loss.backward()
        self._optimizer_policy.step()

        # target update
        self._forward_learn_cnt += 1
        self._agent.target_update(self._agent.state_dict()['model'])
        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_v': self._optimizer_value.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            **loss_dict
        }

    def _init_collect(self) -> None:
        self._traj_len = self._cfg.collect.traj_len
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._use_cuda, self._unroll_len)
        self._collect_agent = Agent(self._model)
        algo_cfg = self._cfg.collect.algo
        self._collect_agent.add_plugin(
            'main',
            'action_noise',
            noise_type='gauss',
            noise_kwargs={
                'mu': 0.0,
                'sigma': algo_cfg.noise_sigma
            },
            noise_range=None,
        )
        self._collect_agent.add_plugin('main', 'grad', enable_grad=False)
        self._collect_agent.mode(train=False)
        self._collect_agent.reset()
        self._collect_setting_set = {}

    def _forward_collect(self, data: dict) -> dict:
        output = self._collect_agent.forward(data, param={'mode': 'compute_action'})
        return output

    def _process_transition(self, obs: Any, agent_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': agent_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _init_eval(self) -> None:
        self._eval_agent = Agent(self._model)
        self._eval_agent.add_plugin('main', 'grad', enable_grad=False)
        self._eval_agent.mode(train=False)
        self._eval_agent.reset()
        self._eval_setting_set = {}

    def _forward_eval(self, data: dict) -> dict:
        output = self._eval_agent.forward(data, param={'mode': 'compute_action'})
        return output

    def _init_command(self) -> None:
        pass

    def _create_model_from_cfg(self, cfg: dict) -> torch.nn.Module:
        return SAC(**cfg.model)


register_policy('sac', SACPolicy)
