from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch

from nervex.torch_utils import Adam
from nervex.rl_utils import a2c_data, a2c_error, Adder
from nervex.model import FCValueAC
from nervex.agent import Agent
from nervex.worker import TransitionBuffer
from .base_policy import Policy, register_policy
from .common_policy import CommonPolicy


class A2CPolicy(CommonPolicy):

    def _init_learn(self) -> None:
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._agent = Agent(self._model)
        algo_cfg = self._cfg.learn.algo
        self._value_weight = algo_cfg.value_weight
        self._entropy_weight = algo_cfg.entropy_weight

        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.mode(train=True)
        self._agent.reset()
        self._learn_setting_set = {}

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        # forward
        output = self._agent.forward(data['obs'], param={'mode': 'compute_action_value'})
        adv = data['adv']
        # norm adv in total train_batch
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # return = value + adv
        return_ = data['value'] + adv
        # calculate a2c error
        data = a2c_data(output['logit'], data['action'], output['value'], adv, return_, data['weight'])
        a2c_loss = a2c_error(data)
        total_loss = a2c_loss.policy_loss + self._value_weight * a2c_loss.value_loss - self._entropy_weight * a2c_loss.entropy_loss
        # update
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': a2c_loss.policy_loss.item(),
            'value_loss': a2c_loss.value_loss.item(),
            'entropy_loss': a2c_loss.entropy_loss.item(),
            'adv_abs_max': adv.abs().max().item(),
        }

    def _init_collect(self) -> None:
        self._get_traj_length = self._cfg.collect.get_traj_length
        if self._get_traj_length == 'inf':
            self._get_traj_length = float('inf')
        self._collect_agent = Agent(self._model)
        self._collect_agent.add_plugin('main', 'multinomial_sample')
        self._collect_agent.add_plugin('main', 'grad', enable_grad=False)
        self._collect_agent.mode(train=False)
        self._collect_agent.reset()
        self._collect_setting_set = {}
        self._adder = Adder(self._use_cuda)
        algo_cfg = self._cfg.collect.algo
        self._gamma = algo_cfg.discount_factor
        self._gae_lambda = algo_cfg.gae_lambda

    def _forward_collect(self, data: dict) -> dict:
        return self._collect_agent.forward(data, param={'mode': 'compute_action_value'})

    def _process_transition(self, obs: Any, agent_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': agent_output['action'],
            'value': agent_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_trajectory(self, transitions: TransitionBuffer, data_id: int, done: bool) -> Union[None, List[Any]]:
        if not done and len(transitions[data_id]) < self._get_traj_length + 1:
            return None
        else:
            ret = list(reversed([transitions[data_id].pop() for _ in range(len(transitions[data_id]))]))
            # get adv
            if self._get_traj_length == float('inf'):
                assert ret[-1]['done'], "episode must be terminated by done=True"
            if done:
                ret = self._adder.get_gae(
                    ret, last_value=torch.zeros(1), gamma=self._gamma, gae_lambda=self._gae_lambda
                )
            else:
                ret = self._adder.get_gae(
                    ret[:-1], last_value=ret[-1]['value'], gamma=self._gamma, gae_lambda=self._gae_lambda
                )
                transitions.append(data_id, ret[-1])
            return ret

    def _init_eval(self) -> None:
        self._eval_agent = Agent(self._model)
        self._eval_agent.add_plugin('main', 'argmax_sample')
        self._eval_agent.add_plugin('main', 'grad', enable_grad=False)
        self._eval_agent.mode(train=False)
        self._eval_agent.reset()
        self._eval_setting_set = {}

    def _forward_eval(self, data: dict) -> dict:
        return self._eval_agent.forward(data, param={'mode': 'compute_action'})

    def _init_command(self) -> None:
        pass

    def _create_model_from_cfg(self, cfg: dict) -> torch.nn.Module:
        return FCValueAC(**cfg.model)


register_policy('a2c', A2CPolicy)
