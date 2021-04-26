from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
import copy
from easydict import EasyDict

from nervex.torch_utils import Adam, to_device
from nervex.rl_utils import v_1step_td_data, v_1step_td_error, epsilon_greedy, Adder
from nervex.model import CollaQ
from nervex.armor import model_wrap
from nervex.data import timestep_collate, default_collate, default_decollate
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy


@POLICY_REGISTRY.register('collaq')
class CollaQPolicy(Policy):
    r"""
    Overview:
        Policy class of CollaQ algorithm. CollaQ is a multi-agent reinforcement learning algorithm
    """

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the learner armor of CollaQPolicy
        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
            - gamma (:obj:`float`): The discount factor
            - alpha (:obj:`float`): The collaQ loss factor, the weight for calculating MARL loss
            - agent_num (:obj:`int`): Since this is a multi-agent algorithm, we need to input the agent num.
            - batch_size (:obj:`int`): Need batch size info to init hidden_state plugins
        """
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        algo_cfg = self._cfg.learn.algo
        self._gamma = algo_cfg.discount_factor
        self._alpha = algo_cfg.get("collaQ_loss_factor", 1.0)

        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': algo_cfg.target_update_theta}
        )
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [[None for _ in range(self._cfg.learn.agent_num)] for _ in range(3)]
        )
        self._learn_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [[None for _ in range(self._cfg.learn.agent_num)] for _ in range(3)]
        )
        self._learn_model.reset()
        self._target_model.reset()

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
            data = to_device(data, self._device)
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
        data = self._data_preprocess_learn(data)
        # ====================
        # CollaQ forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        # for hidden_state plugin, we need to reset the main armor and target armor
        self._learn_model.reset(state=data['prev_state'][0])
        self._target_model.reset(state=data['prev_state'][0])
        inputs = {'obs': data['obs'], 'action': data['action']}
        ret = self._learn_model.forward(inputs, single_step=False)
        total_q = ret['total_q']
        agent_colla_alone_q = ret['agent_colla_alone_q'].sum(-1).sum(-1)
        total_q = self._learn_model.forward(inputs, single_step=False)['total_q']
        next_inputs = {'obs': data['next_obs']}
        with torch.no_grad():
            target_total_q = self._target_model.forward(next_inputs, single_step=False)['total_q']

        # td_loss calculation
        td_data = v_1step_td_data(total_q, target_total_q, data['reward'], data['done'], data['weight'])
        td_loss, _ = v_1step_td_error(td_data, self._gamma)
        # collaQ loss calculation
        colla_loss = (agent_colla_alone_q ** 2).mean()
        # combine loss with factor
        loss = colla_loss * self._alpha + td_loss
        # ====================
        # CollaQ update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
        }

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        self._learn_model.reset(data_id=data_id)

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect armor.
            Enable the eps_greedy_sample and the hidden_state plugin.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._use_cuda, self._unroll_len)
        self._collect_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.collect.env_num,
            save_prev_state=True,
            init_fn=lambda: [[None for _ in range(self._cfg.learn.agent_num)] for _ in range(3)]
        )
        self._collect_model = model_wrap(self._collect_model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, eps: float) -> dict:
        r"""
        Overview:
            Forward function for collect mode with eps_greedy
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - data (:obj:`dict`): The collected data
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._use_cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps, data_id=data_id)
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        self._collect_model.reset(data_id=data_id)

    def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - armor_output (:obj:`dict`): Output of collect armor, including at least \
                ['action', 'prev_state', 'agent_colla_alone_q']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'prev_state': armor_output['prev_state'],
            'action': armor_output['action'],
            'agent_colla_alone_q': armor_output['agent_colla_alone_q'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval armor with argmax strategy and the hidden_state plugin.
        """
        self._eval_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.eval.env_num,
            save_prev_state=True,
            init_fn=lambda: [[None for _ in range(self._cfg.learn.agent_num)] for _ in range(3)]
        )
        self._eval_model = model_wrap(self._eval_model, wrapper_name='argmax_sample')
        self._eval_model.reset()

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
        data = {'obs': data}
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, data_id=data_id)
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        self._eval_model.reset(data_id=data_id)

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory from the adder
        Arguments:
            - data (:obj:`deque`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        return self._adder.get_train_sample(data)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'collaq', ['nervex.model.qmix.qmix']
