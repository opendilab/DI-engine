from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
import sys
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import pdb

from nervex.torch_utils import Adam
from nervex.rl_utils import Adder, epsilon_greedy
from nervex.armor import Armor
from nervex.model import FCDiscreteNet
from nervex.policy.base_policy import Policy, register_policy
from nervex.policy.common_policy import CommonPolicy
from torch.distributions.categorical import Categorical


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
        self._target_entropy = algo_cfg.get('target_entropy', self._cfg.model.action_dim / 10)

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
        # Q-function
        q_value = self._armor.forward({'obs': obs})['q_value']
        q0 = q_value[0]
        q1 = q_value[1]
        batch_range = torch.arange(action.shape[0])
        q0_a = q0[batch_range, action]
        q1_a = q1[batch_range, action]
        # Target
        with torch.no_grad():
            target_q_value = self._armor.target_forward({'obs': next_obs})['q_value']
            q0_targ = target_q_value[0]
            q1_targ = target_q_value[1]
            q_targ = torch.min(q0_targ, q1_targ)
            # discrete policy
            alpha = torch.exp(self._log_alpha.clone())
            # TODO use q_targ or q0 for pi
            pi = torch.softmax(q_targ / alpha, dim=-1)  # N, B
            log_pi = torch.log(pi)
            # v = \sum_a \pi(a | s) (Q(s, a) - \alpha \log(\pi(a|s)))
            target_v_value = (pi * (q_targ - alpha * log_pi)).sum(axis=-1)
            # q = r + \gamma v
            q_backup = reward + (1 - done) * self._gamma * target_v_value

        # update Q
        q0_loss = F.mse_loss(q0_a, q_backup)
        q1_loss = F.mse_loss(q1_a, q_backup)

        self._optimizer_q.zero_grad()
        total_q_loss = q0_loss + q1_loss
        total_q_loss.backward()
        self._optimizer_q.step()

        debug_q = self._armor.forward({'obs': obs})['q_value']
        if torch.isnan(debug_q[0]).sum().item() > 0:
            pdb.set_trace()

        # update alpha
        # TODO: use main_network or target_network
        entropy = (-pi * log_pi).sum(axis=-1)
        expect_entropy = (pi * self._target_entropy).sum(axis=-1)
        alpha_loss = self._log_alpha * (entropy - expect_entropy).mean()

        self._optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self._optimizer_alpha.step()

        # target update
        self._armor.target_update(self._armor.state_dict()['model'])
        self._forward_learn_cnt += 1

        # some useful info
        return {
            # 'cur_lr_q': self._optimizer_q.defaults['lr'],
            # 'cur_lr_alpha': self._optimizer_alpha.defaults['lr'],
            'q0_loss': q0_loss.item(),
            'q1_loss': q1_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'entropy': entropy.mean().item(),
            'alpha': math.exp(self._log_alpha.item()),
            'q0_value': q0_a.mean().item(),
            'q1_value': q1_a.mean().item(),
        }

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect armor.
            Use action noise for exploration.
        """
        self._traj_len = self._cfg.collect.traj_len
        if self._traj_len == "inf":
            self._traj_len = float("inf")
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
        # start with random action for better exploration
        output = self._collect_armor.forward(data, eps=self._eps)
        if self._forward_learn_cnt > self._cfg.command.eps.decay:
            try:
                logits = output['logit'] / math.exp(self._log_alpha.item())
                prob = torch.softmax(logits - logits.max(axis=-1, keepdim=True).values, dim=-1)
                pi_action = torch.multinomial(prob, 1)
                output['action'] = pi_action
            except RuntimeError:
                pdb.set_trace()
                # print("================data===================\n" * 8)
                # print(data)
                # print("================output===================\n" * 8)
                # print(output)
                # print("================logits===================\n" * 8)
                # print(logits)
                # print("=================prob==================\n" * 8)
                # print(prob)
                # print("=================END==============\n" * 8)
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
        return ['q0_loss', 'q1_loss', 'alpha_loss', 'alpha', 'entropy', 'q0_value', 'q1_value']


register_policy('sqn', SQNPolicy)
