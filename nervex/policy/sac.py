from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

from nervex.torch_utils import Adam, to_device
from nervex.data import default_collate, default_decollate
from nervex.rl_utils import v_1step_td_data, v_1step_td_error, Adder
from nervex.armor import Armor
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('sac')
class SACPolicy(Policy):
    r"""
    Overview:
        Policy class of SAC algorithm.
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init q, value and policy's optimizers, algorithm config, main and target armors.
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
        # Init auto alpha
        self._is_auto_alpha = algo_cfg.get('is_auto_alpha', None)
        if self._is_auto_alpha:
            self._target_entropy = -np.prod(self._cfg.model.action_dim)
            self._log_alpha = torch.log(torch.tensor([algo_cfg.alpha]))
            self._log_alpha = self._log_alpha.to(device='cuda' if self._use_cuda else 'cpu').requires_grad_()
            self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=self._cfg.learn.learning_rate_alpha)
            self._is_auto_alpha = True
            assert self._log_alpha.shape == torch.Size([1]) and self._log_alpha.requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = algo_cfg.alpha
            self._is_auto_alpha = False
        self._reparameterization = algo_cfg.reparameterization
        self._policy_std_reg_weight = algo_cfg.policy_std_reg_weight
        self._policy_mean_reg_weight = algo_cfg.policy_mean_reg_weight
        self._use_twin_q = algo_cfg.use_twin_q

        # Main and target armors
        self._armor = Armor(self._model)
        self._armor.add_model('target', update_type='momentum', update_kwargs={'theta': algo_cfg.target_theta})
        self._armor.reset()
        self._armor.target_reset()
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
        data = default_preprocess_learn(
            data,
            use_priority=self._cfg.get('use_priority', False),
            ignore_done=self._cfg.learn.get('ignore_done', False),
            use_nstep=False
        )
        if self._use_cuda:
            data = to_device(data, self._device)

        self._armor.model.train()
        self._armor.target_model.train()
        obs = data.get('obs')
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        action = data.get('action')
        done = data.get('done')

        # evaluate to get action distribution
        eval_data = self._armor.forward(data['obs'], param={'mode': 'evaluate'})
        mean = eval_data["mean"]
        log_std = eval_data["log_std"]
        log_prob = eval_data["log_prob"]

        # predict q value and v value
        q_value = self._armor.forward(data, param={'mode': 'compute_q'})['q_value']
        v_value = self._armor.forward(data['obs'], param={'mode': 'compute_value'})['v_value']

        # =================
        # q network
        # =================
        # compute q loss
        with torch.no_grad():
            next_v_value = self._armor.target_forward(next_obs, param={'mode': 'compute_value'})['v_value']
        if self._use_twin_q:
            q_data0 = v_1step_td_data(q_value[0], next_v_value, reward, done, data['weight'])
            loss_dict['q_loss'], td_error_per_sample0 = v_1step_td_error(q_data0, self._gamma)
            q_data1 = v_1step_td_data(q_value[1], next_v_value, reward, done, data['weight'])
            loss_dict['q_twin_loss'], td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
            td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2
        else:
            q_data = v_1step_td_data(q_value, next_v_value, reward, done, data['weight'])
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
        new_q_value = self._armor.forward(eval_data, param={'mode': 'compute_q'})['q_value']
        if self._use_twin_q:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])
        # new_q_value: (bs, ), log_prob: (bs, act_dim) -> target_v_value: (bs, )
        target_v_value = (new_q_value.unsqueeze(-1) - self._alpha * log_prob).mean(dim=-1)
        loss_dict['value_loss'] = F.mse_loss(v_value, target_v_value.detach())

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

        # update policy network
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()

        #  compute alpha loss
        if self._is_auto_alpha:
            log_prob = log_prob.detach() + self._target_entropy
            loss_dict['alpha_loss'] = -(self._log_alpha * log_prob).mean()

            self._alpha_optim.zero_grad()
            loss_dict['alpha_loss'].backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        loss_dict['total_loss'] = sum(loss_dict.values())

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        self._armor.target_update(self._armor.state_dict()['model'])
        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_v': self._optimizer_value.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'priority': td_error_per_sample.abs().tolist(),
            **loss_dict
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        ret = {
            'model': self._model.state_dict(),
            'optimizer_q': self._optimizer_q.state_dict(),
            'optimizer_value': self._optimizer_value.state_dict(),
            'optimizer_policy': self._optimizer_policy.state_dict(),
        }
        if self._is_auto_alpha:
            ret.update({'optimizer_alpha': self._alpha_optim.state_dict()})
        return ret

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._model.load_state_dict(state_dict['model'])
        self._optimizer_q.load_state_dict(state_dict['optimizer_q'])
        self._optimizer_value.load_state_dict(state_dict['optimizer_value'])
        self._optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        if self._is_auto_alpha:
            self._alpha_optim.load_state_dict(state_dict['optimizer_alpha'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect armor.
            Use action noise for exploration.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._use_cuda, self._unroll_len)
        self._collect_armor = Armor(self._model)
        algo_cfg = self._cfg.collect.algo
        self._collect_armor.add_plugin(
            'main',
            'action_noise',
            noise_type='gauss',
            noise_kwargs={
                'mu': 0.0,
                'sigma': algo_cfg.noise_sigma
            },
            noise_range=None,
        )
        self._collect_armor.reset()

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._use_cuda:
            data = to_device(data, self._device)
        self._collect_armor.model.eval()
        with torch.no_grad():
            output = self._collect_armor.forward(data, param={'mode': 'compute_action'})
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

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

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        return self._adder.get_train_sample(data)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval armor. Unlike learn and collect armor, eval armor does not need noise.
        """
        self._eval_armor = Armor(self._model)
        self._eval_armor.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function for eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._use_cuda:
            data = to_device(data, self._device)
        self._eval_armor.model.eval()
        with torch.no_grad():
            output = self._eval_armor.forward(data, param={'mode': 'compute_action', 'deterministic_eval': True})
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

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
        if self._is_auto_alpha:
            return super()._monitor_vars_learn() + [
                'alpha_loss', 'policy_loss', 'value_loss', 'q_loss', 'cur_lr_q', 'cur_lr_v', 'cur_lr_p'
            ] + q_twin
        else:
            return super()._monitor_vars_learn() + [
                'policy_loss', 'value_loss', 'q_loss', 'cur_lr_q', 'cur_lr_v', 'cur_lr_p'
            ] + q_twin
