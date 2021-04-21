from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch

from nervex.torch_utils import Adam, to_device
from nervex.rl_utils import ppo_data, ppo_error, Adder
from nervex.data import default_collate, default_decollate
from nervex.model import FCValueAC, ConvValueAC
from nervex.armor import Armor
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('ppo')
class PPOPolicy(Policy):
    r"""
    Overview:
        Policy class of PPO algorithm.
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config and the main armor.
        """
        # Optimizer
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._armor = Armor(self._model)

        # Algorithm config
        algo_cfg = self._cfg.learn.algo
        self._value_weight = algo_cfg.value_weight
        self._entropy_weight = algo_cfg.entropy_weight
        self._clip_ratio = algo_cfg.clip_ratio
        self._use_adv_norm = algo_cfg.get('use_adv_norm', False)

        # Main armor
        self._armor.reset()

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data
        Returns:
            - info_dict (:obj:`Dict[str, Any]`):
              Including current lr, total_loss, policy_loss, value_loss, entropy_loss, \
                        adv_abs_max, approx_kl, clipfrac
        """
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.get('ignore_done', False), use_nstep=False)
        if self._use_cuda:
            data = to_device(data, self._device)
        # ====================
        # PPO forward
        # ====================
        self._armor.model.train()
        output = self._armor.forward(data['obs'], param={'mode': 'compute_action_value'})
        adv = data['adv']
        if self._use_adv_norm:
            # Normalize advantage in a total train_batch
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return_ = data['value'] + adv
        # Calculate ppo error
        data = ppo_data(
            output['logit'], data['logit'], data['action'], output['value'], data['value'], adv, return_, data['weight']
        )
        ppo_loss, ppo_info = ppo_error(data, self._clip_ratio)
        wv, we = self._value_weight, self._entropy_weight
        total_loss = ppo_loss.policy_loss + wv * ppo_loss.value_loss - we * ppo_loss.entropy_loss
        # ====================
        # PPO update
        # ====================
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': ppo_loss.policy_loss.item(),
            'value_loss': ppo_loss.value_loss.item(),
            'entropy_loss': ppo_loss.entropy_loss.item(),
            'adv_abs_max': adv.abs().max().item(),
            'approx_kl': ppo_info.approx_kl,
            'clipfrac': ppo_info.clipfrac,
        }

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
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_armor = Armor(self._model)
        self._collect_armor.add_plugin('main', 'multinomial_sample')
        self._collect_armor.reset()
        self._adder = Adder(self._use_cuda, self._unroll_len)
        algo_cfg = self._cfg.collect.algo
        self._gamma = algo_cfg.discount_factor
        self._gae_lambda = algo_cfg.gae_lambda

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
        self._collect_armor.model.eval()
        with torch.no_grad():
            output = self._collect_armor.forward(data, param={'mode': 'compute_action_value'})
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
            'logit': armor_output['logit'],
            'action': armor_output['action'],
            'value': armor_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and calculate GAE, return one data to cache for next time calculation
        Arguments:
            - data (:obj:`deque`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        # adder is defined in _init_collect
        data = self._adder.get_gae_with_default_last_value(
            data, data[-1]['done'], gamma=self._gamma, gae_lambda=self._gae_lambda
        )
        return self._adder.get_train_sample(data)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval armor with argmax strategy.
        """
        self._eval_armor = Armor(self._model)
        self._eval_armor.add_plugin('main', 'argmax_sample')
        self._eval_armor.reset()

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
        self._eval_armor.model.eval()
        with torch.no_grad():
            output = self._eval_armor.forward(data, param={'mode': 'compute_action'})
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        return 'fc_vac', ['nervex.model.actor_critic.value_ac']

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + [
            'policy_loss', 'value_loss', 'entropy_loss', 'adv_abs_max', 'approx_kl', 'clipfrac'
        ]
