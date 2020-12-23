from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
import numpy as np
from easydict import EasyDict
import copy

from nervex.torch_utils import Adam
from nervex.rl_utils import ppo_data, ppo_error, epsilon_greedy
from nervex.model import FCValueAC
from nervex.agent import Agent
from .base_policy import Policy, register_policy
from .common_policy import CommonPolicy


class PPOPolicy(CommonPolicy):

    def _init_learn(self) -> None:
        self._optimizer = Adam(self._model.parameters(),
                               lr=self._cfg.learn.learning_rate)
        algo_cfg = self._cfg.learn.algo
        self._value_weight = algo_cfg.value_weight
        self._entropy_weight = algo_cfg.entropy_weight
        self._clip_ratio = algo_cfg.clip_ratio
        self._model.train()
        self._learn_setting_set = {}
        self._continous = self._cfg.policy.continous

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        # forward
        output = self._model(data['obs'], mode="compute_action_value")
        adv = data['adv']
        # norm adv in total train_batch
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # return = value + adv
        return_ = data['value'] + adv
        # calculate ppo error
        data = ppo_data(
            output['logit'], data['logit'], data['action'], output['value'], data['value'], adv, return_, data['weight']
        )
        ppo_loss, ppo_info = ppo_error(data, self._clip_ratio)
        wv, we = self._value_weight, self._entropy_weight
        total_loss = ppo_loss.policy_loss + wv * \
            ppo_loss.value_loss - we * ppo_loss.entropy_loss
        # update
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': ppo_loss.policy_loss.item(),
            'value_loss': ppo_loss.value_loss.item(),
            'entropy_loss': ppo_loss.entropy_loss.item(),
            'adv_abs_max': adv.abs().max().item(),
            'approx_kl': ppo_info.approx_kl,
            'clipfrac': ppo_info.clipfrac,
        }

    def _init_collect(self) -> None:
        self._traj_len = self._cfg.collect.traj_len
        self._unroll_len = self._cfg.collect.unroll_len
        assert(self._unroll_len == 1)
        if self._traj_len == 'inf':
            self._traj_len = float('inf')
        self._collect_setting_set = {'eps'}
        # self._adder = Adder(self._use_cuda, self._unroll_len)
        algo_cfg = self._cfg.collect.algo
        self._gamma = algo_cfg.discount_factor
        self._gae_lambda = algo_cfg.gae_lambda

    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        with torch.no_grad():
            ret = self._model(data['obs'], mode="compute_action_value")
            logit, value = ret['logit'], ret['value']
        if isinstance(logit, torch.Tensor):
            logit, value = [logit], [value]
        action = []
        for i, l in enumerate(logit):
            if np.random.random() > self._eps:
                action.append(l.argmax(dim=-1))
            else:
                action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
        if len(action) == 1:
            action, logit, value = action[0], logit[0], value[0]
        output = {'action': action, 'logit': logit, 'value': value}
        return output

    def _process_transition(self, obs: Any, output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': obs,
            'logit': output['logit'],
            'action': output['action'],
            'value': output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _init_eval(self) -> None:
        self._eval_setting_set = {}

    def _forward_eval(self, data_id: List[int], data: dict) -> dict:
        with torch.no_grad():
            ret = self._model(data['obs'], mode="compute_action_value")
            logit, value = ret['logit'], ret['value']
        if isinstance(logit, torch.Tensor):
            logit, value = [logit], [value]
        action = []
        for i, l in enumerate(logit):
            action.append(l.argmax(dim=-1))
        if len(action) == 1:
            action, logit, value = action[0], logit[0], value[0]
        output = {'action': action, 'logit': logit, 'value': value}
        return output

    def _create_model_from_cfg(self, cfg: dict) -> torch.nn.Module:
        return FCValueAC(**cfg.model)

    def _init_command(self) -> None:
        eps_cfg = self._cfg.command.eps
        self.epsilon_greedy = epsilon_greedy(
            eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def _get_setting_collect(self, command_info: dict) -> dict:
        learner_step = command_info['learner_step']
        return {'eps': self.epsilon_greedy(learner_step)}

    def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
        data = self._get_traj(traj_cache, self._traj_len, return_num=1)
        if self._traj_len == float('inf'):
            assert data[-1]['done'], "episode must be terminated by done=True"
        data = self._gae(
            data, gamma=self._gamma, gae_lambda=self._gae_lambda
        )
        return data

    def _get_traj(self, data: deque, traj_len: int, return_num: int = 0) -> list:
        num = min(traj_len, len(data))  # traj_len can be inf
        traj = [data.popleft() for _ in range(num)]
        for i in range(min(return_num, len(data))):
            data.appendleft(copy.deepcopy(traj[-(i + 1)]))
        return traj

    def _gae(self, data: List[Dict[str, Any]], gamma: float = 0.99,
             gae_lambda: float = 0.97) -> List[Dict[str, Any]]:
        if data[-1]['done']:
            last_value = torch.zeros(1)
        else:
            last_value = data[-1]['value']
            data = data[:-1]
        value = torch.stack([d['value'] for d in data] + [last_value])
        reward = torch.stack([d['reward'] for d in data])
        delta = reward + gamma * value[1:] - value[:-1]
        factor = gamma * gae_lambda
        adv = torch.zeros_like(reward)
        gae_item = 0.
        for t in reversed(range(reward.shape[0])):
            gae_item = delta[t] + factor * gae_item
            adv[t] += gae_item
        for i in range(len(data)):
            data[i]['adv'] = adv[i]
        return data

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        self._model.train()

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        self._model.eval()


register_policy('ppo_vanilla', PPOPolicy)
