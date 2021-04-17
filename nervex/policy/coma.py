from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
from easydict import EasyDict

from nervex.torch_utils import Adam, to_device
from nervex.rl_utils import coma_data, coma_error, epsilon_greedy, Adder
from nervex.model import ComaNetwork
from nervex.armor import Armor
from nervex.data import timestep_collate
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy
from .common_policy import CommonPolicy


@POLICY_REGISTRY.register('coma')
class COMAPolicy(CommonPolicy):

    def _init_learn(self) -> None:
        """
        Overview:
            Init the learner armor of COMAPolicy

        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
            - gamma (:obj:`float`): The discount factor
            - lambda (:obj:`float`): The lambda factor, determining the mix of bootstrapping\
                vs further accumulation of multistep returns at each timestep,
            - value_wight(:obj:`float`): The weight of value loss in total loss
            - entropy_weight(:obj:`float`): The weight of entropy loss in total loss
            - agent_num (:obj:`int`): Since this is a multi-agent algorithm, we need to input the agent num.
            - batch_size (:obj:`int`): Need batch size info to init hidden_state plugins
        """
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._armor = Armor(self._model)
        algo_cfg = self._cfg.learn.algo
        self._gamma = algo_cfg.discount_factor
        self._lambda = algo_cfg.td_lambda
        self._value_weight = algo_cfg.value_weight
        self._entropy_weight = algo_cfg.entropy_weight

        self._armor.add_model('target', update_type='momentum', update_kwargs={'theta': algo_cfg.target_update_theta})
        self._armor.add_plugin(
            'main',
            'hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._armor.add_plugin(
            'target',
            'hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._armor.mode(train=True)
        self._armor.target_mode(train=True)
        self._armor.reset()
        self._armor.target_reset()

    def _data_preprocess_learn(self, data: List[Any]) -> Tuple[dict, dict]:
        r"""
        Overview:
            Preprocess the data to fit the required data format for learning

        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function, the Dict
                in data should contain keys including at least ['obs', 'action', 'reward']

        Returns:
            - data (:obj:`Dict[str, Any]`): the processed data, including at least \
                ['obs', 'action', 'reward', 'done', 'weight']
            - data_info (:obj:`dict`): the data info, such as replay_buffer_idx, replay_unique_id
        """
        data_info = {
            'replay_buffer_idx': [d.get('replay_buffer_idx', None) for d in data],
            'replay_unique_id': [d.get('replay_unique_id', None) for d in data],
        }
        # data preprocess
        data = timestep_collate(data)
        assert set(data.keys()) > set(['obs', 'action', 'reward'])
        if self._use_cuda:
            data = to_device(data, self._device)
        data['weight'] = data.get('weight', None)
        data['done'] = data['done'].float()
        return data, data_info

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode, acquire the data and calculate the loss and\
            optimize learner model

        Arguments:
            - data (:obj:`dict`): Dict type data, including at least \
                ['obs', 'action', 'reward', 'done', 'weight']

        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including cur_lr and total_loss
                - cur_lr (:obj:`float`): Current learning rate
                - total_loss (:obj:`float`): The calculated loss
                - policy_loss (:obj:`float`): The policy(actor) loss of coma
                - value_loss (:obj:`float`): The value(critic) loss of coma
                - entropy_loss (:obj:`float`): The entropy loss
        """
        # forward
        self._armor.reset(state=data['prev_state'][0])
        self._armor.target_reset(state=data['prev_state'][0])
        q_value = self._armor.forward(data, param={'mode': 'compute_q_value'})['q_value']
        with torch.no_grad():
            target_q_value = self._armor.target_forward(data, param={'mode': 'compute_q_value'})['q_value']
        logit = self._armor.forward(data, param={'mode': 'compute_action'})['logit']

        data = coma_data(logit, data['action'], q_value, target_q_value, data['reward'], data['weight'])
        coma_loss = coma_error(data, self._gamma, self._lambda)
        total_loss = coma_loss.policy_loss + self._value_weight * coma_loss.q_value_loss - self._entropy_weight * \
            coma_loss.entropy_loss

        # update
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        # after update
        self._armor.target_update(self._armor.state_dict()['model'])
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': coma_loss.policy_loss.item(),
            'value_loss': coma_loss.q_value_loss.item(),
            'entropy_loss': coma_loss.entropy_loss.item(),
        }

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init moethod. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect armor.
            Armor has eps_greedy_sample plugin and hidden state plugin
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._use_cuda, self._unroll_len)
        self._collect_armor = Armor(self._model)
        self._collect_armor.add_plugin(
            'main',
            'hidden_state',
            state_num=self._cfg.collect.env_num,
            save_prev_state=True,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._collect_armor.add_plugin('main', 'eps_greedy_sample')
        self._collect_armor.mode(train=False)
        self._collect_armor.reset()

    def _forward_collect(self, data_id: List[int], data: dict, eps: float) -> dict:
        r"""
        Overview:
            Collect output according to eps_greedy plugin

        Arguments:
            - data_id (:obj:`List` of :obj:`int`): Not used, set in arguments for consistency
            - data (:obj:`dict`): Dict type data, including at least ['obs'].

        Returns:
            - data (:obj:`dict`): The collected data
        """
        with torch.no_grad():
            output = self._collect_armor.forward(data, eps=eps, data_id=data_id, param={'mode': 'compute_action'})
        return output

    def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - armor_output (:obj:`dict`): Output of collect armor, including at least ['action', 'prev_state']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'prev_state': armor_output['prev_state'],
            'action': armor_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval armor with argmax strategy and hidden_state plugin.
        """
        self._eval_armor = Armor(self._model)
        self._eval_armor.add_plugin(
            'main',
            'hidden_state',
            state_num=self._cfg.eval.env_num,
            save_prev_state=True,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._eval_armor.add_plugin('main', 'argmax_sample')
        self._eval_armor.mode(train=False)
        self._eval_armor.reset()

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
        with torch.no_grad():
            output = self._eval_armor.forward(data, data_id=data_id, param={'mode': 'compute_action'})
        return output

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory, then sample from trajectory

        Arguments:
            - data (:obj:`deque`): The trajectory's cache

        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        return self._adder.get_train_sample(data)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'coma', ['nervex.model.coma.coma']

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss', 'entropy_loss']
