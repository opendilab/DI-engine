from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
import sys
import os
import torch
import torch.nn.functional as F

from nervex.torch_utils import Adam
from nervex.rl_utils import v_1step_td_data, v_1step_td_error, Adder
from nervex.agent import Agent
from nervex.policy.base_policy import Policy, register_policy
from nervex.policy.common_policy import CommonPolicy


class SACPolicy(CommonPolicy):
    r"""
    Overview:
        Policy class of SAC algorithm.
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init q, value and policy's optimizers, algorithm config, main and target agents.
        """
        # Optimizers
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

        # Algorithm config
        algo_cfg = self._cfg.learn.algo
        self._algo_cfg_learn = algo_cfg
        self._gamma = algo_cfg.discount_factor
        self._alpha = algo_cfg.alpha
        self._reparameterization = algo_cfg.reparameterization
        self._policy_std_reg_weight = algo_cfg.policy_std_reg_weight
        self._policy_mean_reg_weight = algo_cfg.policy_mean_reg_weight
        self._use_twin_q = algo_cfg.use_twin_q

        # Main and target agents
        self._agent = Agent(self._model)
        self._agent.add_model('target', update_type='momentum', update_kwargs={'theta': algo_cfg.target_theta})
        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.add_plugin('target', 'grad', enable_grad=False)
        self._agent.mode(train=True)
        self._agent.target_mode(train=True)
        self._agent.reset()
        self._agent.target_reset()
        self._learn_setting_set = {}
        self._forward_learn_cnt = 0

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        loss_dict = {}

        obs = data.get('obs')
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        action = data.get('action')
        done = data.get('done')

        # evaluate to get action distribution
        eval_data = self._agent.forward(data, param={'mode': 'evaluate'})
        mean = eval_data["mean"]
        log_std = eval_data["log_std"]
        new_action = eval_data["action"]
        log_prob = eval_data["log_prob"]

        # predict q value and v value
        q_value = self._agent.forward(data, param={'mode': 'compute_q'})['q_value']
        v_value = self._agent.forward(data, param={'mode': 'compute_value'})['v_value']

        # =================
        # q network
        # =================
        # compute q loss
        next_data = {'obs': next_obs}
        target_v_value = self._agent.target_forward(next_data, param={'mode': 'compute_value'})['v_value']
        if self._use_twin_q:
            q_data0 = v_1step_td_data(q_value[0], target_v_value, reward, done, data['weight'])
            loss_dict['q_loss'], td_error_per_sample0 = v_1step_td_error(q_data0, self._gamma)
            q_data1 = v_1step_td_data(q_value[1], target_v_value, reward, done, data['weight'])
            loss_dict['q_twin_loss'], td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
            td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2
        else:
            q_data = v_1step_td_data(q_value, target_v_value, reward, done, data['weight'])
            loss_dict['q_loss'], td_error_per_sample = v_1step_td_error(q_data, self._gamma)

        # update q network
        self._optimizer_q.zero_grad()
        loss_dict['q_loss'].backward()
        if self._use_twin_q:
            loss_dict['q_twin_loss'].backward()
        self._optimizer_q.step()

        # =================
        # value network
        # =================
        # compute value loss
        eval_data['obs'] = obs
        new_q_value = self._agent.forward(eval_data, param={'mode': 'compute_q'})['q_value']
        if self._use_twin_q:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])
        # new_q_value: (bs, ), log_prob: (bs, act_dim) -> next_v_value: (bs, )
        next_v_value = (new_q_value.unsqueeze(-1) - self._alpha * log_prob).mean(dim=-1)
        loss_dict['value_loss'] = F.mse_loss(v_value, next_v_value.detach())

        # update value network
        self._optimizer_value.zero_grad()
        loss_dict['value_loss'].backward()
        self._optimizer_value.step()

        # =================
        # policy network
        # =================
        # compute policy loss
        if not self._reparameterization:
            target_log_policy = new_q_value - v_value
            policy_loss = (log_prob * (log_prob - target_log_policy.unsqueeze(-1))).mean()
        else:
            policy_loss = (self._alpha * log_prob - new_q_value.unsqueeze(-1)).mean()

        std_reg_loss = self._policy_std_reg_weight * (log_std ** 2).mean()
        mean_reg_loss = self._policy_mean_reg_weight * (mean ** 2).mean()

        policy_loss += std_reg_loss + mean_reg_loss
        loss_dict['policy_loss'] = policy_loss
        loss_dict['total_loss'] = sum(loss_dict.values())

        # update policy network
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        self._agent.target_update(self._agent.state_dict()['model'])
        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_v': self._optimizer_value.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'priority': td_error_per_sample.abs().tolist(),
            **loss_dict
        }

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect agent.
            Use action noise for exploration.
        """
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

    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data_id (:obj:`List[int]`): Not used in this policy, set in arguments for consistency.
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        output = self._collect_agent.forward(data, param={'mode': 'compute_action'})
        return output

    def _process_transition(self, obs: Any, agent_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - agent_output (:obj:`dict`): Output of collect agent, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step, i.e. next_obs).
        Return:
            - transition (:obj:`Dict[str, Any]`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': agent_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval agent. Unlike learn and collect agent, eval agent does not need noise.
        """
        self._eval_agent = Agent(self._model)
        self._eval_agent.add_plugin('main', 'grad', enable_grad=False)
        self._eval_agent.mode(train=False)
        self._eval_agent.reset()
        self._eval_setting_set = {}

    def _forward_eval(self, data_id: List[int], data: dict) -> dict:
        r"""
        Overview:
            Forward function for eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data_id (:obj:`List[int]`): Not used in this policy.
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        output = self._eval_agent.forward(data, param={'mode': 'compute_action', 'deterministic_eval': True})
        return output

    def _init_command(self) -> None:
        r"""
        Overview:
            Command mode init method. Called by ``self.__init__``.
        """
        pass

    def default_model(self) -> Tuple[str, List[str]]:
        return 'sac', ['nervex.model.sac']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        q_twin = ['q_twin_loss'] if self._use_twin_q else []
        return super()._monitor_vars_learn() + [
            'policy_loss', 'value_loss', 'q_loss', 'cur_lr_q', 'cur_lr_v', 'cur_lr_p'
        ] + q_twin


register_policy('sac', SACPolicy)
