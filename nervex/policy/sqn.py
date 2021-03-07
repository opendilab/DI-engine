from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
import sys
import os
import math
import numpy as np
import torch
import torch.nn.functional as F

from nervex.torch_utils import Adam, one_hot
from nervex.rl_utils import Adder, epsilon_greedy
from nervex.armor import Armor
from nervex.model import FCDiscreteNet
from nervex.policy.base_policy import Policy, register_policy
from nervex.policy.common_policy import CommonPolicy


class SQNModel(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(SQNModel, self).__init__()
        self.q0 = FCDiscreteNet(*args, **kwargs)
        self.q1 = FCDiscreteNet(*args, **kwargs)

    def forward(self, data: dict) -> dict:
        output0 = self.q0(data)
        output1 = self.q1(data)
        return {
            'q_value': [output0['logit'], output1['logit']],
            'logit': output0['logit'],
        }


class SQNPolicy(CommonPolicy):
    r"""
    Overview:
        Policy class of SQN algorithm (arxiv: 1912.10891).
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init q, value and policy's optimizers, algorithm config, main and target armors.
        """
        # Optimizers
        self._optimizer_q = Adam(
            self._model.parameters(), lr=self._cfg.learn.learning_rate_q, weight_decay=self._cfg.learn.weight_decay
        )

        # Algorithm config
        algo_cfg = self._cfg.learn.algo
        self._algo_cfg_learn = algo_cfg
        self._gamma = algo_cfg.discount_factor
        self._action_dim = np.prod(self._cfg.model.action_dim)
        self._target_entropy = -self._action_dim
        self._action_one_hot = one_hot(torch.arange(self._action_dim).long(),
                                       self._action_dim).unsqueeze(1).to(self._device)  # N, 1, N

        self._log_alpha = torch.FloatTensor([math.log(algo_cfg.alpha)]).to(self._device).requires_grad_(True)
        self._optimizer_alpha = torch.optim.Adam([self._log_alpha], lr=self._cfg.learn.learning_rate_alpha)

        # Main and target armors
        self._armor = Armor(self._model)
        self._armor.add_model('target', update_type='momentum', update_kwargs={'theta': algo_cfg.target_theta})
        self._armor.mode(train=True)
        self._armor.target_mode(train=True)
        self._armor.reset()
        self._armor.target_reset()
        self._learn_setting_set = {}
        self._forward_learn_cnt = 0

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        """
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs', 'done',\
                'weight']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Learn info, including current lr and loss.
        """
        obs = data.get('obs')
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        action = data.get('action')
        done = data.get('done')

        q_value = self._armor.forward({'obs': obs})['q_value']
        alpha = torch.exp(self._log_alpha.detach().clone())
        with torch.no_grad():
            next_q_value = self._armor.forward({'obs': next_obs})['q_value']
            target_q_value = self._armor.target_forward({'obs': next_obs})['q_value']
            target_v_value = []
            for i in range(2):
                # (N, 1, N) x (1, B, N) -> sum(dim=-1)
                q_a_n = (self._action_one_hot * next_q_value[i].unsqueeze(0)).sum(dim=-1)  # N, B
                pi = torch.softmax(q_a_n / alpha, dim=-1)  # N, B
                target_v_value_i = (pi * q_a_n).sum(dim=0) - alpha * (pi * torch.log(pi)).sum(dim=0)  # B,
                target_v_value.append(target_v_value_i)
            target_v_value = torch.min(*target_v_value)
        target_v_value = reward + (1 - done) * self._gamma * target_v_value
        batch_range = torch.arange(action.shape[0])
        q0_loss = F.mse_loss(q_value[0][batch_range, action], target_v_value)
        q1_loss = F.mse_loss(q_value[1][batch_range, action], target_v_value)

        self._optimizer_q.zero_grad()
        total_q_loss = q0_loss + q1_loss
        total_q_loss.backward()
        self._optimizer_q.step()

        log_pi = torch.log(torch.softmax(q_value[0] * one_hot(action, self._action_dim), dim=-1))
        alpha_loss = -(torch.exp(self._log_alpha) * (self._target_entropy + log_pi.detach())).mean()
        self._optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self._optimizer_alpha.step()

        self._armor.target_update(self._armor.state_dict()['model'])

        self._forward_learn_cnt += 1
        # target update
        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_alpha': self._optimizer_alpha.defaults['lr'],
            'q0_loss': q0_loss.item(),
            'q1_loss': q1_loss.item(),
            'alpha_loss': alpha_loss.item()
        }

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect armor.
            Use action noise for exploration.
        """
        self._traj_len = self._cfg.collect.traj_len
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._use_cuda, self._unroll_len)
        self._collect_armor = Armor(self._model)
        self._collect_armor.add_plugin('main', 'eps_greedy_sample')
        self._collect_armor.mode(train=False)
        self._collect_armor.reset()
        self._collect_setting_set = {'eps'}

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
        with torch.no_grad():
            output = self._collect_armor.forward(data, eps=self._eps)
        return output

    def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - armor_output (:obj:`dict`): Output of collect armor, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step, i.e. next_obs).
        Return:
            - transition (:obj:`Dict[str, Any]`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': armor_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval armor, which use argmax for selecting action
        """
        self._eval_armor = Armor(self._model)
        self._eval_armor.add_plugin('main', 'argmax_sample')
        self._eval_armor.mode(train=False)
        self._eval_armor.reset()
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
        with torch.no_grad():
            output = self._eval_armor.forward(data)
        return output

    def _init_command(self) -> None:
        r"""
        Overview:
            Command mode init method. Called by ``self.__init__``.
        """
        eps_cfg = self._cfg.command.eps
        self.epsilon_greedy = epsilon_greedy(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def _get_setting_collect(self, command_info: dict) -> dict:
        r"""
        Overview:
            Collect mode setting information including eps
        Arguments:
            - command_info (:obj:`dict`): Dict type, including at least ['learner_step']
        Returns:
           - collect_setting (:obj:`dict`): Including eps in collect mode.
        """
        learner_step = command_info['learner_step']
        return {'eps': self.epsilon_greedy(learner_step)}

    def _create_model(self, cfg: dict, model: Optional[Union[type, torch.nn.Module]] = None) -> torch.nn.Module:
        assert model is None
        return SQNModel(**cfg.model)

    def default_model(self) -> None:
        # placeholder
        pass

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        return super()._monitor_vars_learn() + ['cur_lr_q', 'cur_lr_alpha', 'q0_loss', 'q1_loss', 'alpha_loss']


register_policy('sqn', SQNPolicy)
