from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch

from nervex.torch_utils import Adam
from nervex.rl_utils import q_1step_td_data, q_1step_td_error, epsilon_greedy
from nervex.agent import Agent
from .base_policy import Policy
from .common_policy import CommonPolicy


class DQNPolicy(CommonPolicy):

    def _init_learn(self) -> None:
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._agent = Agent(self._model)
        algo_cfg = self._cfg.learn.algo
        self._gamma = algo_cfg.discount_factor

        self._agent.add_model('target', update_type='assign', update_kwargs={'freq': algo_cfg.target_update_freq})
        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.add_plugin('target', 'grad', enable_grad=False)
        self._agent.mode(train=True)
        self._agent.target_mode(train=True)
        self._agent.reset()
        self._agent.target_reset()

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        # forward
        q_value = self._agent.forward(data['obs'])['logit']
        target_q_value = self._agent.target_forward(data['next_obs'])['logit']
        data = q_1step_td_data(q_value, target_q_value, data['action'], data['reward'], data['done'], data['weight'])
        loss, info = q_1step_td_error(data, self._gamma)
        # update
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # after update
        self._agent.target_update(self._agent.state_dict())
        return {
            'total_loss': loss.item(),
            'td_error_per_data': info.td_error_per_data,
        }

    def _init_collect(self) -> None:
        self._get_traj_length = self._cfg.collect.get_traj_length
        self._eps = self._cfg.collect.get('eps', 0.05)
        self._collect_agent = Agent(self._model)
        self._collect_agent.add_plugin('main', 'eps_greedy_sample')
        self._collect_agent.add_plugin('main', 'grad', enable_grad=False)
        self._collect_agent.reset()

    def _forward_collect(self, data: dict) -> dict:
        self._collect_agent.mode(train=False)
        output = self._collect_agent.forward(data, eps=self._eps)
        self._collect_agent.mode(train=True)
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
        self._eval_agent.add_plugin('main', 'argmax_sample')
        self._eval_agent.add_plugin('main', 'grad', enable_grad=False)
        self._eval_agent.reset()

    def _forward_eval(self, data: dict) -> dict:
        self._eval_agent.mode(train=False)
        output = self._eval_agent.forward(data)
        self._eval_agent.mode(train=True)
        return output

    def _init_control(self) -> None:
        eps_cfg = self._cfg.control.eps
        self.epsilon_greedy = epsilon_greedy(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def _get_setting_collect(self) -> dict:
        return {'eps': self.epsilon_greedy(self.learn_step)}

    def _create_model_from_cfg(self, cfg: dict) -> torch.nn.Module:
        raise NotImplementedError
