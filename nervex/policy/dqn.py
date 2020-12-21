from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
import torch
from easydict import EasyDict

from nervex.torch_utils import Adam
from nervex.rl_utils import q_1step_td_data, q_1step_td_error, epsilon_greedy, Adder
from nervex.model import FCDiscreteNet, ConvDiscreteNet
from nervex.agent import Agent
from .base_policy import Policy, register_policy
from .common_policy import CommonPolicy


class DQNPolicy(CommonPolicy):

    def _init_learn(self) -> None:
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._agent = Agent(self._model)
        algo_cfg = self._cfg.learn.algo
        self._gamma = algo_cfg.discount_factor

        self._agent.add_model('target', update_type='assign', update_kwargs={'freq': algo_cfg.target_update_freq})
        self._agent.add_plugin('main', 'argmax_sample')
        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.add_plugin('target', 'grad', enable_grad=False)
        self._agent.mode(train=True)
        self._agent.target_mode(train=True)
        self._agent.reset()
        self._agent.target_reset()
        self._learn_setting_set = {}

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        # forward
        q_value = self._agent.forward(data['obs'])['logit']
        target_q_value = self._agent.target_forward(data['next_obs'])['logit']
        target_q_action = self._agent.forward(data['next_obs'])['action']
        data = q_1step_td_data(
            q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], data['weight']
        )
        loss = q_1step_td_error(data, self._gamma)
        # update
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # after update
        self._agent.target_update(self._agent.state_dict()['model'])
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
        }

    def _init_collect(self) -> None:
        self._traj_len = self._cfg.collect.traj_len
        if self._traj_len == "inf":
            self._traj_len == float("inf")
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._use_cuda, self._unroll_len)
        self._collect_agent = Agent(self._model)
        self._collect_agent.add_plugin('main', 'eps_greedy_sample')
        self._collect_agent.add_plugin('main', 'grad', enable_grad=False)
        self._collect_agent.mode(train=False)
        self._collect_agent.reset()
        self._collect_setting_set = {'eps'}

    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        return self._collect_agent.forward(data, eps=self._eps)

    def _process_transition(self, obs: Any, agent_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': agent_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _init_eval(self) -> None:
        self._eval_agent = Agent(self._model)
        self._eval_agent.add_plugin('main', 'argmax_sample')
        self._eval_agent.add_plugin('main', 'grad', enable_grad=False)
        self._eval_agent.mode(train=False)
        self._eval_agent.reset()
        self._eval_setting_set = {}

    def _forward_eval(self, data_id: List[int], data: dict) -> dict:
        return self._eval_agent.forward(data)

    def _init_command(self) -> None:
        eps_cfg = self._cfg.command.eps
        self.epsilon_greedy = epsilon_greedy(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def _get_setting_collect(self, command_info: dict) -> dict:
        learner_step = command_info['learner_step']
        return {'eps': self.epsilon_greedy(learner_step)}

    def _create_model_from_cfg(self, cfg: dict, model_type: Optional[type] = None) -> torch.nn.Module:
        if model_type is None:
            return FCDiscreteNet(**cfg.model)
        else:
            return model_type(**cfg.model)


register_policy('dqn', DQNPolicy)
