from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
from easydict import EasyDict

from nervex.torch_utils import Adam, to_device
from nervex.rl_utils import v_1step_td_data, v_1step_td_error, epsilon_greedy, Adder
from nervex.model import CollaQ
from nervex.agent import Agent
from nervex.data import timestep_collate
from .base_policy import Policy, register_policy
from .common_policy import CommonPolicy


class CollaQPolicy(CommonPolicy):

    def _init_learn(self) -> None:
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._agent = Agent(self._model)
        algo_cfg = self._cfg.learn.algo
        self._gamma = algo_cfg.discount_factor
        self._alpha = algo_cfg.get("collaQ_loss_factor", 1.0)

        self._agent.add_model('target', update_type='momentum', update_kwargs={'theta': algo_cfg.target_update_theta})
        self._agent.add_plugin(
            'main',
            'hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [[None for _ in range(self._cfg.learn.agent_num)] for _ in range(3)]
        )
        self._agent.add_plugin(
            'target',
            'hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [[None for _ in range(self._cfg.learn.agent_num)] for _ in range(3)]
        )
        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.add_plugin('target', 'grad', enable_grad=False)
        self._agent.mode(train=True)
        self._agent.target_mode(train=True)
        self._agent.reset()
        self._agent.target_reset()
        self._learn_setting_set = {}

    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        # data preprocess
        data = timestep_collate(data)
        if self._use_cuda:
            data = to_device(data, 'cuda')
        data['weight'] = data.get('weight', None)
        data['done'] = data['done'].float()
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        # forward
        self._agent.reset(state=data['prev_state'][0])
        self._agent.target_reset(state=data['prev_state'][0])
        inputs = {'obs': data['obs'], 'action': data['action']}
        ret = self._agent.forward(inputs, param={'single_step': False})
        total_q = ret['total_q']
        agent_colla_alone_q = ret['agent_colla_alone_q'].sum(-1).sum(-1)
        total_q = self._agent.forward(inputs, param={'single_step': False})['total_q']
        next_inputs = {'obs': data['next_obs']}
        target_total_q = self._agent.target_forward(next_inputs, param={'single_step': False})['total_q']
        #td_loss
        td_data = v_1step_td_data(total_q, target_total_q, data['reward'], data['done'], data['weight'])
        td_loss = v_1step_td_error(td_data, self._gamma)
        #collaQ loss
        colla_loss = (agent_colla_alone_q ** 2).mean()

        loss = colla_loss * self._alpha + td_loss
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
            self._traj_len = float("inf")
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._use_cuda, self._unroll_len)
        self._collect_agent = Agent(self._model)
        self._collect_agent.add_plugin(
            'main',
            'hidden_state',
            state_num=self._cfg.collect.env_num,
            save_prev_state=True,
            init_fn=lambda: [[None for _ in range(self._cfg.learn.agent_num)] for _ in range(3)]
        )
        self._collect_agent.add_plugin('main', 'eps_greedy_sample')
        self._collect_agent.add_plugin('main', 'grad', enable_grad=False)
        self._collect_agent.mode(train=False)
        self._collect_agent.reset()
        self._collect_setting_set = {'eps'}

    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        return self._collect_agent.forward(data, eps=self._eps, data_id=data_id)

    def _process_transition(self, obs: Any, agent_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'prev_state': agent_output['prev_state'],
            'action': agent_output['action'],
            'agent_colla_alone_q': agent_output['agent_colla_alone_q'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _init_eval(self) -> None:
        self._eval_agent = Agent(self._model)
        self._eval_agent.add_plugin(
            'main',
            'hidden_state',
            state_num=self._cfg.eval.env_num,
            save_prev_state=True,
            init_fn=lambda: [[None for _ in range(self._cfg.learn.agent_num)] for _ in range(3)]
        )
        self._eval_agent.add_plugin('main', 'argmax_sample')
        self._eval_agent.add_plugin('main', 'grad', enable_grad=False)
        self._eval_agent.mode(train=False)
        self._eval_agent.reset()
        self._eval_setting_set = {}

    def _forward_eval(self, data_id: List[int], data: dict) -> dict:
        return self._eval_agent.forward(data, data_id=data_id)

    def _init_command(self) -> None:
        eps_cfg = self._cfg.command.eps
        self.epsilon_greedy = epsilon_greedy(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def _get_setting_collect(self, command_info: dict) -> dict:
        learner_step = command_info['learner_step']
        return {'eps': self.epsilon_greedy(learner_step)}

    def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
        data = self._adder.get_traj(traj_cache, self._traj_len, return_num=0)
        return self._adder.get_train_sample(data)

    def _create_model_from_cfg(self, cfg: dict, model_type: Optional[type] = None) -> torch.nn.Module:
        if model_type is None:
            return CollaQ(**cfg.model)
        else:
            return model_type(**cfg.model)


register_policy('collaQ', CollaQPolicy)
