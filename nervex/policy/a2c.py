from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch

from nervex.rl_utils import a2c_data, a2c_error, Adder, nstep_return_data, nstep_return
from nervex.torch_utils import Adam
from nervex.model import FCValueAC, ConvValueAC
from nervex.armor import Armor
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy
from .common_policy import CommonPolicy


@POLICY_REGISTRY.register('a2c')
class A2CPolicy(CommonPolicy):
    r"""
    Overview:
        Policy class of A2C algorithm.
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config, main and target armors.
        """
        # Optimizer
        self._optimizer = Adam(
            self._model.parameters(), lr=self._cfg.learn.learning_rate, weight_decay=self._cfg.learn.weight_decay
        )

        # Algorithm config
        algo_cfg = self._cfg.learn.algo
        self._value_weight = algo_cfg.value_weight
        self._entropy_weight = algo_cfg.entropy_weight
        self._learn_use_nstep_return = algo_cfg.get('use_nstep_return', False)
        self._learn_gamma = algo_cfg.get('discount_factor', 0.99)
        self._learn_nstep = algo_cfg.get('nstep', 1)
        self._use_adv_norm = algo_cfg.get('use_adv_norm', False)

        # Main and target armors
        self._armor = Armor(self._model)
        self._armor.mode(train=True)
        self._armor.reset()

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs','adv']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        # forward
        output = self._armor.forward(data['obs'], param={'mode': 'compute_action_value'})

        adv = data['adv']
        if self._use_adv_norm:
            # norm adv in total train_batch
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        with torch.no_grad():
            if self._learn_use_nstep_return:
                # use nstep return
                next_value = self._armor.forward(data['next_obs'], param={'mode': 'compute_action_value'})['value']
                reward = data['reward'].permute(1, 0).contiguous()
                nstep_data = nstep_return_data(reward, next_value, data['done'])
                return_ = nstep_return(nstep_data, self._learn_gamma, self._learn_nstep).detach()
            else:
                # Return = value + adv
                return_ = data['value'] + adv
        data = a2c_data(output['logit'], data['action'], output['value'], adv, return_, data['weight'])

        # Calculate A2C loss
        a2c_loss = a2c_error(data)
        wv, we = self._value_weight, self._entropy_weight
        total_loss = a2c_loss.policy_loss + wv * a2c_loss.value_loss - we * a2c_loss.entropy_loss

        # ====================
        # A2C-learning update
        # ====================

        self._optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self._model.parameters()),
            max_norm=0.5,
        )
        self._optimizer.step()

        # =============
        # after update
        # =============
        return {
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': a2c_loss.policy_loss.item(),
            'value_loss': a2c_loss.value_loss.item(),
            'entropy_loss': a2c_loss.entropy_loss.item(),
            'adv_abs_max': adv.abs().max().item(),
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
        self._collect_armor.mode(train=False)
        self._collect_armor.reset()
        self._adder = Adder(self._use_cuda, self._unroll_len)
        algo_cfg = self._cfg.collect.algo
        self._gamma = algo_cfg.discount_factor
        self._gae_lambda = algo_cfg.gae_lambda
        self._collect_use_nstep_return = algo_cfg.get('use_nstep_return', False)
        self._collect_nstep = algo_cfg.get('nstep', 1)

    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        r"""
        Overview:
            Forward function for collect mode
        Arguments:
            - data_id (:obj:`List` of :obj:`int`): Not used, set in arguments for consistency
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - data (:obj:`dict`): The collected data
        """
        with torch.no_grad():
            output = self._collect_armor.forward(data, param={'mode': 'compute_action_value'})
        return output

    def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - armor_output (:obj:`dict`): Output of collect armor, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """

        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': armor_output['action'],
            'value': armor_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data
        Arguments:
            - data (:obj:`deque`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        # adder is defined in _init_collect
        data = self._adder.get_gae_with_default_last_value(
            data, data[-1]['done'], gamma=self._gamma, gae_lambda=self._gae_lambda
        )
        if self._collect_use_nstep_return:
            data = self._adder.get_nstep_return_data(data, self._collect_nstep)
        return self._adder.get_train_sample(data)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval armor with argmax strategy.
        """

        self._eval_armor = Armor(self._model)
        self._eval_armor.add_plugin('main', 'argmax_sample')
        self._eval_armor.mode(train=False)
        self._eval_armor.reset()

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
        with torch.no_grad():
            output = self._eval_armor.forward(data, param={'mode': 'compute_action'})
        return output

    def default_model(self) -> Tuple[str, List[str]]:
        return 'fc_vac', ['nervex.model.actor_critic.value_ac']

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss', 'entropy_loss', 'adv_abs_max']
