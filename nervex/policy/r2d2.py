from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple, deque
import torch
from easydict import EasyDict

from nervex.torch_utils import Adam, to_device
from nervex.rl_utils import q_nstep_td_data, q_nstep_td_error, q_nstep_td_error_with_rescale, epsilon_greedy, Adder
from nervex.model import FCRDiscreteNet
from nervex.agent import Agent
from nervex.data import timestep_collate
from .base_policy import Policy, register_policy
from .common_policy import CommonPolicy


class R2D2Policy(CommonPolicy):

    def _init_learn(self) -> None:
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._agent = Agent(self._model)
        algo_cfg = self._cfg.learn.algo
        self._gamma = algo_cfg.discount_factor
        self._nstep = algo_cfg.nstep
        self._use_value_rescale = algo_cfg.use_value_rescale
        self._burnin_step = algo_cfg.burnin_step

        self._agent.add_model('target', update_type='assign', update_kwargs={'freq': algo_cfg.target_update_freq})
        self._agent.add_plugin('main', 'hidden_state', state_num=self._cfg.learn.batch_size)
        self._agent.add_plugin('target', 'hidden_state', state_num=self._cfg.learn.batch_size)
        self._agent.add_plugin('main', 'argmax_sample')
        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.add_plugin('target', 'grad', enable_grad=False)
        self._agent.mode(train=True)
        self._agent.target_mode(train=True)
        self._learn_setting_set = {}

    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        # data preprocess
        data = timestep_collate(data)
        if self._use_cuda:
            data = to_device(data, 'cuda')
        assert len(data['obs']) == 2 * self._nstep + self._burnin_step, data['obs'].shape
        bs = self._burnin_step
        data['weight'] = data.get('weight', [None for _ in range(self._nstep)])
        ignore_done = self._cfg.learn.get('ignore_done', False)
        if ignore_done:
            data['done'] = [None for _ in range(self._nstep)]
        else:
            data['done'] = data['done'][bs:bs + self._nstep].float()
        data['action'] = data['action'][bs:bs + self._nstep]
        data['reward'] = data['reward'][bs:]
        data['burnin_obs'] = data['obs'][:bs]
        data['main_obs'] = data['obs'][bs:bs + self._nstep]
        data['target_obs'] = data['obs'][bs + self._nstep:]
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        # forward
        self._agent.reset(data_id=None, state=data['prev_state'][0])
        self._agent.target_reset(data_id=None, state=data['prev_state'][0])
        if len(data['burnin_obs']) != 0:
            with torch.no_grad():
                inputs = {'obs': data['burnin_obs'], 'enable_fast_timestep': True}
                _ = self._agent.forward(inputs)
                _ = self._agent.target_forward(inputs)
        inputs = {'obs': data['main_obs'], 'enable_fast_timestep': True}
        q_value = self._agent.forward(inputs)['logit']
        next_inputs = {'obs': data['target_obs'], 'enable_fast_timestep': True}
        target_q_value = self._agent.target_forward(next_inputs)['logit']
        target_q_action = self._agent.forward(next_inputs)['action']

        action, reward, done, weight = data['action'], data['reward'], data['done'], data['weight']
        reward = reward.permute(0, 2, 1).contiguous()  # T, B, nstep -> T, nstep, B
        loss = []
        for t in range(self._nstep):
            td_data = q_nstep_td_data(q_value[t], target_q_value[t], action[t], target_q_action[t], reward[t], done[t], weight[t])
            if self._use_value_rescale:
                loss.append(q_nstep_td_error_with_rescale(td_data, self._gamma, self._nstep))
            else:
                loss.append(q_nstep_td_error(td_data, self._gamma, self._nstep))
        loss = sum(loss) / (len(loss) + 1e-8)
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
        self._collect_nstep = self._cfg.collect.algo.nstep
        self._collect_burnin_step = self._cfg.collect.algo.burnin_step
        self._traj_len = self._cfg.collect.traj_len
        self._unroll_len = self._cfg.collect.unroll_len
        assert self._traj_len >= self._unroll_len
        assert self._unroll_len == self._collect_burnin_step + 2 * self._collect_nstep
        self._adder = Adder(self._use_cuda, self._unroll_len)
        self._collect_agent = Agent(self._model)
        self._collect_agent.add_plugin(
            'main', 'hidden_state', state_num=self._cfg.collect.env_num, save_prev_state=True
        )
        self._collect_agent.add_plugin('main', 'eps_greedy_sample')
        self._collect_agent.add_plugin('main', 'grad', enable_grad=False)
        self._collect_agent.mode(train=False)
        self._collect_agent.reset()
        self._collect_setting_set = {'eps'}

    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        return self._collect_agent.forward(data, data_id=data_id, eps=self._eps)

    def _process_transition(self, obs: Any, agent_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': obs,
            'action': agent_output['action'],
            'prev_state': agent_output['prev_state'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
        data = self._adder.get_traj(traj_cache, self._traj_len, return_num=self._collect_burnin_step)
        data = self._adder.get_nstep_return_data(data, self._nstep, self._traj_len)
        return self._adder.get_train_sample(data)

    def _init_eval(self) -> None:
        self._eval_agent = Agent(self._model)
        self._eval_agent.add_plugin('main', 'hidden_state', state_num=self._cfg.eval.env_num)
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

    def _create_model_from_cfg(self, cfg: dict) -> torch.nn.Module:
        return FCRDiscreteNet(**cfg.model)


register_policy('r2d2', R2D2Policy)
