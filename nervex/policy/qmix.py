from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
from easydict import EasyDict

from nervex.torch_utils import Adam, to_device
from nervex.rl_utils import v_1step_td_data, v_1step_td_error, epsilon_greedy, Adder
from nervex.model import QMix
from nervex.agent import Armor
from nervex.data import timestep_collate
from .base_policy import Policy, register_policy
from .common_policy import CommonPolicy


class QMIXPolicy(CommonPolicy):
    r"""
    Overview:
        Policy class of QMIX algorithm. QMIX is a multiagent reinforcement learning algorithm, \
            you can view the paper in the following link <https://arxiv.org/abs/1803.11485>_
    """

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the learner agent of QMIXPolicy
        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
            - gamma (:obj:`float`): The discount factor
            - agent_num (:obj:`int`): Since this is a multi-agent algorithm, \
                we need to input the agent num
            - batch_size (:obj:`int`): Need batch size info to init hidden_state plugins
        """
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._agent = Armor(self._model)
        algo_cfg = self._cfg.learn.algo
        self._gamma = algo_cfg.discount_factor

        self._agent.add_model('target', update_type='momentum', update_kwargs={'theta': algo_cfg.target_update_theta})
        self._agent.add_plugin(
            'main',
            'hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._agent.add_plugin(
            'target',
            'hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.add_plugin('target', 'grad', enable_grad=False)
        self._agent.mode(train=True)
        self._agent.target_mode(train=True)
        self._agent.reset()
        self._agent.target_reset()
        self._learn_setting_set = {}

    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        r"""
        Overview:
            Preprocess the data to fit the required data format for learning

        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function

        Returns:
            - data (:obj:`Dict[str, Any]`): the processed data, from \
                [len=B, ele={dict_key: [len=T, ele=Tensor(any_dims)]}] -> {dict_key: Tensor([T, B, any_dims])}
        """
        # data preprocess
        data = timestep_collate(data)
        if self._use_cuda:
            data = to_device(data, 'cuda')
        data['weight'] = data.get('weight', None)
        data['done'] = data['done'].float()
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least \
                ['obs', 'next_obs', 'action', 'reward', 'next_obs', 'prev_state', 'done']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        # ====================
        # Q-mix forward
        # ====================
        # for hidden_state plugin, we need to reset the main agent and target agent
        self._agent.reset(state=data['prev_state'][0])
        self._agent.target_reset(state=data['prev_state'][0])
        inputs = {'obs': data['obs'], 'action': data['action']}
        total_q = self._agent.forward(inputs, param={'single_step': False})['total_q']
        next_inputs = {'obs': data['next_obs']}
        target_total_q = self._agent.target_forward(next_inputs, param={'single_step': False})['total_q']

        data = v_1step_td_data(total_q, target_total_q, data['reward'], data['done'], data['weight'])
        loss, td_error_per_sample = v_1step_td_error(data, self._gamma)
        # ====================
        # Q-mix update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # =============
        # after update
        # =============
        self._agent.target_update(self._agent.state_dict()['model'])
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
        }

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect agent.
            Enable the eps_greedy_sample and the hidden_state plugin.
        """
        self._traj_len = self._cfg.collect.traj_len
        if self._traj_len == "inf":
            self._traj_len = float("inf")
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._use_cuda, self._unroll_len)
        self._collect_agent = Armor(self._model)
        self._collect_agent.add_plugin(
            'main',
            'hidden_state',
            state_num=self._cfg.collect.env_num,
            save_prev_state=True,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._collect_agent.add_plugin('main', 'eps_greedy_sample')
        self._collect_agent.add_plugin('main', 'grad', enable_grad=False)
        self._collect_agent.mode(train=False)
        self._collect_agent.reset()
        self._collect_setting_set = {'eps'}

    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        r"""
        Overview:
            Forward function for collect mode with eps_greedy
        Arguments:
            - data_id (:obj:`List` of :obj:`int`): Not used, set in arguments for consistency
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - data (:obj:`dict`): The collected data
        """
        return self._collect_agent.forward(data, eps=self._eps, data_id=data_id)

    def _process_transition(self, obs: Any, agent_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - agent_output (:obj:`dict`): Output of collect agent, including at least ['action', 'prev_state']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'prev_state': agent_output['prev_state'],
            'action': agent_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval agent with argmax strategy and the hidden_state plugin.
        """
        self._eval_agent = Armor(self._model)
        self._eval_agent.add_plugin(
            'main',
            'hidden_state',
            state_num=self._cfg.eval.env_num,
            save_prev_state=True,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._eval_agent.add_plugin('main', 'argmax_sample')
        self._eval_agent.add_plugin('main', 'grad', enable_grad=False)
        self._eval_agent.mode(train=False)
        self._eval_agent.reset()
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
        return self._eval_agent.forward(data, data_id=data_id)

    def _init_command(self) -> None:
        r"""
        Overview:
            Command mode init method. Called by ``self.__init__``.
            Set the eps_greedy rule according to the config for command
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

    def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory from the adder.
        Arguments:
            - traj (:obj:`deque`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        data = self._adder.get_traj(traj_cache, self._traj_len, return_num=0)
        return self._adder.get_train_sample(data)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'qmix', ['nervex.model.qmix.qmix']


register_policy('qmix', QMIXPolicy)
