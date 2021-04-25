from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
import copy
import torch.nn.functional as F

from nervex.torch_utils import Adam, RMSprop, to_device
from nervex.data import default_collate, default_decollate
from nervex.rl_utils import Adder, vtrace_data, vtrace_error
from nervex.model import FCValueAC, ConvValueAC
from nervex.armor import model_wrap
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy


@POLICY_REGISTRY.register('impala')
class IMPALAPolicy(Policy):
    r"""
    Overview:
        Policy class of IMPALA algorithm.
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config and the main armor.
        """
        # Optimizer
        grad_clip_type = self._cfg.learn.get("grad_clip_type", None)
        clip_value = self._cfg.learn.get("clip_value", None)
        optim_type = self._cfg.learn.get("optim", "adam")
        if optim_type == 'rmsprop':
            self._optimizer = RMSprop(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        elif optim_type == 'adam':
            self._optimizer = Adam(
                self._model.parameters(),
                grad_clip_type=grad_clip_type,
                clip_value=clip_value,
                lr=self._cfg.learn.learning_rate
            )
        else:
            raise NotImplementedError
        self._model = model_wrap(self._model, wrapper_name='base')

        self._action_dim = self._cfg.model.action_dim
        self._unroll_len = self._cfg.learn.unroll_len

        # Algorithm config
        algo_cfg = self._cfg.learn.algo
        self._value_weight = algo_cfg.value_weight
        self._entropy_weight = algo_cfg.entropy_weight
        self._gamma = algo_cfg.discount_factor
        self._lambda = algo_cfg.lambda_
        self._rho_clip_ratio = algo_cfg.rho_clip_ratio
        self._c_clip_ratio = algo_cfg.c_clip_ratio
        self._rho_pg_clip_ratio = algo_cfg.rho_pg_clip_ratio

        # Main armor
        self._model.reset()

    def _data_preprocess_learn(self, data: List[Dict[str, Any]]) -> dict:
        r"""
        Overview:
            Data preprocess function of learn mode.
            Convert list trajectory data as the tensor trajectory data
        Arguments:
            - data (:obj:`dict`): Dict type data
        Returns:
            - data (:obj:`dict`)
        """
        data = default_collate(data)
        if self._use_cuda:
            data = to_device(data, self._device)
        data['done'] = torch.cat(data['done'], dim=0).reshape(self._unroll_len, -1).float()
        use_priority = self._cfg.get('use_priority', False)
        if use_priority:
            data['weight'] = data['IS']
        else:
            data['weight'] = data.get('weight', None)
        data['obs_plus_1'] = torch.cat((data['obs'] + data['next_obs'][-1:]), dim=0)
        data['logit'] = torch.cat(data['logit'], dim=0).reshape(self._unroll_len, -1, self._action_dim)
        data['action'] = torch.cat(data['action'], dim=0).reshape(self._unroll_len, -1)
        data['reward'] = torch.cat(data['reward'], dim=0).reshape(self._unroll_len, -1)
        data['weight'] = torch.cat(data['weight'], dim=0).reshape(self._unroll_len, -1) if data['weight'] else None
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data
        Returns:
            - info_dict (:obj:`Dict[str, Any]`):
              Including current lr, total_loss, policy_loss, value_loss and entropy_loss
        """
        data = self._data_preprocess_learn(data)
        # ====================
        # IMPALA forward
        # ====================
        self._model.train()
        output = self._model.forward(data['obs_plus_1'], mode='compute_action_value')
        target_logit, behaviour_logit, actions, values, rewards, weights = self._reshape_data(output, data)
        # Calculate vtrace error
        data = vtrace_data(target_logit, behaviour_logit, actions, values, rewards, weights)
        g, l, r, c, rg = self._gamma, self._lambda, self._rho_clip_ratio, self._c_clip_ratio, self._rho_pg_clip_ratio
        vtrace_loss = vtrace_error(data, g, l, r, c, rg)
        wv, we = self._value_weight, self._entropy_weight
        total_loss = vtrace_loss.policy_loss + wv * vtrace_loss.value_loss - we * vtrace_loss.entropy_loss
        # ====================
        # IMPALA update
        # ====================
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': vtrace_loss.policy_loss.item(),
            'value_loss': vtrace_loss.value_loss.item(),
            'entropy_loss': vtrace_loss.entropy_loss.item(),
        }

    def _reshape_data(self, output: dict, data: dict) -> tuple:
        r"""
        Overview:
            Obtain weights for loss calculating, where should be 0 for done positions
            Update values and rewards with the weight
        """
        target_logit = output['logit'].reshape(self._unroll_len + 1, -1, self._action_dim)[:-1]
        values = output['value'].reshape(self._unroll_len + 1, -1)
        behaviour_logit = data['logit']
        actions = data['action']
        rewards = data['reward']
        weights_ = 1 - data['done']
        weights = torch.ones_like(rewards)
        values[1:] = values[1:] * weights_
        weights[1:] = weights_[:-1]
        rewards = rewards * weights
        return target_logit, behaviour_logit, actions, values, rewards, weights

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._model.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect armor.
        """
        self._collect_unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample')
        self._collect_model.reset()
        self._adder = Adder(self._use_cuda, self._collect_unroll_len)

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function for collect mode
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - data (:obj:`dict`): The collected data
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._use_cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_action_value')
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> dict:
        """
        Overview:
               Generate dict type transition data from inputs.
        Arguments:
                - obs (:obj:`Any`): Env observation
                - armor_output (:obj:`dict`): Output of collect armor, including at least ['action']
                - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\
                       (here 'obs' indicates obs after env step).
        Returns:
               - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'logit': armor_output['logit'],
            'action': armor_output['action'],
            'value': armor_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        return self._adder.get_train_sample(data)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval armor with argmax strategy.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function for eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data_id (:obj:`List[int]`): Not used in this policy.
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._use_cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, mode='compute_action')
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        return 'fc_vac', ['nervex.model.actor_critic.value_ac']

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss', 'entropy_loss']
