from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
from easydict import EasyDict

from nervex.torch_utils import Adam, to_device
from nervex.rl_utils import q_nstep_td_data, q_nstep_td_error, q_nstep_td_error_with_rescale, epsilon_greedy, Adder
from nervex.agent import Agent
from nervex.data import timestep_collate
from .base_policy import Policy, register_policy
from .common_policy import CommonPolicy


class R2D2Policy(CommonPolicy):
    r"""
    Overview:
        Policy class of R2D2, from paper `Recurrent Experience Replay in Distributed Reinforcement Learning` .

        R2D2 proposed that several tricks should be used to improve upon DRQN,
        namely some recurrent experience replay trick such as burn-in.
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Init the learner agent of R2D2Policy

        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
            - gamma (:obj:`float`): The discount factor
            - nstep (:obj:`int`): The num of n step return
            - use_value_rescale (:obj:`bool`): Whether to use value rescaled loss in algorithm
            - burnin_step (:obj:`int`): The num of step of burnin
        """
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

    def _data_preprocess_learn(self, data: List[Dict[str, Any]]) -> dict:
        r"""
        Overview:
            Preprocess the data to fit the required data format for learning

        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function

        Returns:
            - data (:obj:`Dict[str, Any]`): the processed data, including at least \
                ['main_obs', 'target_obs', 'burnin_obs', 'action', 'reward', 'done', 'weight']
        """
        # data preprocess
        data = timestep_collate(data)
        if self._use_cuda:
            data = to_device(data, 'cuda')
        assert len(data['obs']) == 2 * self._nstep + self._burnin_step, data['obs'].shape  # todo: why 2*a+b
        bs = self._burnin_step
        data['weight'] = data.get('weight', [None for _ in range(self._nstep)])
        ignore_done = self._cfg.learn.get('ignore_done', False)
        if ignore_done:
            data['done'] = [None for _ in range(self._nstep)]
        else:
            data['done'] = data['done'][bs:bs + self._nstep].float()
        data['action'] = data['action'][bs:bs + self._nstep]
        data['reward'] = data['reward'][bs:]
        # split obs into three parts ['burnn_obs'(0~bs), 'main_obs'(bs~bs+nstep), 'target_obs'(bs+nstep~bss+2nstep)]
        data['burnin_obs'] = data['obs'][:bs]
        data['main_obs'] = data['obs'][bs:bs + self._nstep]
        data['target_obs'] = data['obs'][bs + self._nstep:]
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
            Acquire the data, calculate the loss and optimize learner model.

        Arguments:
            - data (:obj:`dict`): Dict type data, including at least \
                ['main_obs', 'target_obs', 'burnin_obs', 'action', 'reward', 'done', 'weight']

        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including cur_lr and total_loss
                - cur_lr (:obj:`float`): Current learning rate
                - total_loss (:obj:`float`): The calculated loss
        """
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
        # T, B, nstep -> T, nstep, B
        reward = reward.permute(0, 2, 1).contiguous()
        loss = []
        td_error = []
        for t in range(self._nstep):
            td_data = q_nstep_td_data(
                q_value[t], target_q_value[t], action[t], target_q_action[t], reward[t], done[t], weight[t]
            )
            if self._use_value_rescale:
                l, e = q_nstep_td_error_with_rescale(td_data, self._gamma, self._nstep)
                loss.append(l)
                td_error.append(e.abs())
            else:
                l, e = q_nstep_td_error(td_data, self._gamma, self._nstep)
                loss.append(l)
                td_error.append(e.abs())
        loss = sum(loss) / (len(loss) + 1e-8)
        td_error_per_sample = sum(td_error) / (len(td_error) + 1e-8)
        # update
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # after update
        self._agent.target_update(self._agent.state_dict()['model'])
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
        }

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init moethod. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect agent.
        """
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
        r"""
        Overview:
            Collect output according to eps_greedy plugin

        Arguments:
            - data_id (:obj:`List` of :obj:`int`): Not used, set in arguments for consistency
            - data (:obj:`dict`): Dict type data, including at least ['obs'].

        Returns:
            - data (:obj:`dict`): The collected data
        """
        return self._collect_agent.forward(data, data_id=data_id, eps=self._eps)

    def _process_transition(self, obs: Any, agent_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - agent_output (:obj:`dict`): Output of collect agent, including at least ['action', 'prev_state']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'action': agent_output['action'],
            'prev_state': agent_output['prev_state'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data

        Arguments:
            - traj_cache (:obj:`deque`): The trajectory's cache

        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        data = self._adder.get_traj(traj_cache, self._traj_len, return_num=self._collect_burnin_step)
        data = self._adder.get_nstep_return_data(data, self._collect_nstep, self._traj_len)
        return self._adder.get_train_sample(data)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval agent with argmax strategy.
        """
        self._eval_agent = Agent(self._model)
        self._eval_agent.add_plugin('main', 'hidden_state', state_num=self._cfg.eval.env_num)
        self._eval_agent.add_plugin('main', 'argmax_sample')
        self._eval_agent.add_plugin('main', 'grad', enable_grad=False)
        self._eval_agent.mode(train=False)
        self._eval_agent.reset()
        self._eval_setting_set = {}

    def _forward_eval(self, data_id: List[int], data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode, similar to ``self._forward_collect``.

        Arguments:
            - data_id (:obj:`List[int]`): Not used in this policy.
            - data (:obj:`dict`): Dict type data, including at least ['obs'].

        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        return self._eval_agent.forward(data, data_id=data_id)

    def _init_command(self) -> None:
        r"""
        Overview:
            Command mode init method. Called by ``self.__init__``.
            Init the self.epsilon_greedy according to eps config
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

    def default_model(self) -> Tuple[str, List[str]]:
        return 'fcr_discrete_net', ['nervex.model.discrete_net.discrete_net']


# regist r2d2 policy in the policy maps
register_policy('r2d2', R2D2Policy)
