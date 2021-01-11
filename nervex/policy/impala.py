from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
import torch.nn.functional as F

from nervex.data import default_collate, default_decollate
from nervex.torch_utils import to_device
from nervex.torch_utils import Adam
from nervex.rl_utils import Adder, vtrace_data, vtrace_error
from nervex.model import FCValueAC, ConvValueAC
from nervex.agent import Agent
from .base_policy import Policy, register_policy
from .common_policy import CommonPolicy


class IMPALAPolicy(CommonPolicy):

    def _init_learn(self) -> None:
        grad_clip_type = self._cfg.learn.get("grad_clip_type", None)
        clip_value = self._cfg.learn.get("clip_value", None)
        optim_type = self._cfg.learn.get("optim", "adam")
        if optim_type == 'rmsprop':
            self._optimizer = torch.optim.RMSprop(
                self._model.parameters(),
                lr=self._cfg.learn.learning_rate
            )
        elif optim_type == 'adam':
            self._optimizer = Adam(
                self._model.parameters(),
                grad_clip_type=grad_clip_type,
                clip_value=clip_value,
                lr=self._cfg.learn.learning_rate
            )
        else:
            raise NotImplementedError
        self._agent = Agent(self._model)

        self._action_dim = self._cfg.model.action_dim

        algo_cfg = self._cfg.learn.algo
        self._value_weight = algo_cfg.value_weight
        self._entropy_weight = algo_cfg.entropy_weight
        self._gamma = algo_cfg.discount_factor
        self._lambda = algo_cfg.lambda_
        self._rho_clip_ratio = algo_cfg.rho_clip_ratio
        self._c_clip_ratio = algo_cfg.c_clip_ratio
        self._rho_pg_clip_ratio = algo_cfg.rho_pg_clip_ratio

        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.mode(train=True)
        self._agent.reset()
        self._learn_setting_set = {}

    def _data_preprocess_learn(self, data: List[Dict[str, Any]]) -> dict:
        # data preprocess
        data = default_collate(data)
        if self._use_cuda:
            data = to_device(data, 'cuda')
        data['done'] = torch.cat(data['done'], dim=0).reshape(self._unroll_len, -1).float()
        use_priority = self._cfg.get('use_priority', False)
        if use_priority:
            data['weight'] = data['IS']
        else:
            data['weight'] = data.get('weight', None)
        data['obs_plus_1'] = torch.cat((data['obs'] + data['next_obs'][-1:]), dim=0)
        data['logit'] = torch.cat(data['logit'], dim=0).reshape(self._unroll_len, -1, self._action_dim)
        data['action'] = torch.cat(data['action'], dim=0).reshape(self._unroll_len, -1)
        data['reward'] = torch.cat(data['reward'], dim=0).reshape(self._unroll_len, -1)
        data['weight'] = torch.cat(data['weight'], dim=0).reshape(self._unroll_len, -1) if data['weight'] else None
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        # forward
        output = self._agent.forward(data['obs_plus_1'], param={'mode': 'compute_action_value'})
        target_logit, behaviour_logit, actions, values, rewards, weights = self._reshape_data(output, data)
        # calculate vtrace error
        data = vtrace_data(target_logit, behaviour_logit, actions, values, rewards, weights)
        g, l, r, c, rg = self._gamma, self._lambda, self._rho_clip_ratio, self._c_clip_ratio, self._rho_pg_clip_ratio
        vtrace_loss = vtrace_error(data, g, l, r, c, rg)
        wv, we = self._value_weight, self._entropy_weight
        total_loss = vtrace_loss.policy_loss + wv * vtrace_loss.value_loss - we * vtrace_loss.entropy_loss
        # update
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': vtrace_loss.policy_loss.item(),
            'value_loss': vtrace_loss.value_loss.item(),
            'entropy_loss': vtrace_loss.entropy_loss.item(),
        }

    def _reshape_data(self, output: dict, data: dict) -> tuple:
        target_logit = output['logit'].reshape(self._unroll_len + 1, -1, self._action_dim)[:-1]
        values = output['value'].reshape(self._unroll_len + 1, -1)
        behaviour_logit = data['logit']
        actions = data['action']
        rewards = data['reward']
        weights_ = 1 - data['done']
        weights = torch.ones_like(rewards)
        values[1:] = values[1:] * weights_
        weights[1:] = weights_[:-1]
        rewards = rewards * weights
        return target_logit, behaviour_logit, actions, values, rewards, weights

    def _init_collect(self) -> None:
        self._traj_len = self._cfg.collect.traj_len
        self._unroll_len = self._cfg.collect.unroll_len
        if self._traj_len == 'inf':
            self._traj_len = float('inf')
        # v_trace need v_t+1
        assert self._traj_len > 1, "IMPALA traj len should be greater than 1"
        self._collect_agent = Agent(self._model)
        self._collect_agent.add_plugin('main', 'multinomial_sample')
        self._collect_agent.add_plugin('main', 'grad', enable_grad=False)
        self._collect_agent.mode(train=False)
        self._collect_agent.reset()
        self._collect_setting_set = {}
        self._adder = Adder(self._use_cuda, self._unroll_len)

    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        return self._collect_agent.forward(data, param={'mode': 'compute_action_value'})

    def _process_transition(self, obs: Any, agent_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'logit': agent_output['logit'],
            'action': agent_output['action'],
            'value': agent_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _init_eval(self) -> None:
        self._eval_agent = Agent(self._model)
        self._eval_agent.add_plugin('main', 'argmax_sample')
        self._eval_agent.add_plugin('main', 'grad', enable_grad=False)
        self._eval_agent.mode(train=False)
        self._eval_agent.reset()
        self._eval_setting_set = {}

    def _forward_eval(self, data_id: List[int], data: dict) -> dict:
        return self._eval_agent.forward(data, param={'mode': 'compute_action'})

    def _init_command(self) -> None:
        pass

    def _create_model_from_cfg(self, cfg: dict, model_type: Optional[type] = None) -> torch.nn.Module:
        if model_type is None:
            if isinstance(cfg.model.obs_dim, List):
                return ConvValueAC(**cfg.model)
            else:
                return FCValueAC(**cfg.model)
        else:
            return model_type(**cfg.model)

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss', 'entropy_loss']


register_policy('impala', IMPALAPolicy)
