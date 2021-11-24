from ding.utils.default_helper import deep_merge_dicts
from .base_policy import Policy
from ding.model import model_wrap
from ding.utils.data import default_collate, default_decollate
from ding.rl_utils import get_train_sample
from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
from .command_mode_policy_instance import DummyCommandModePolicy
import torch
from ding.torch_utils import Adam, to_device
from .common_utils import default_preprocess_learn
from ding.rl_utils.sil import sil_data, sil_error


def create_sil(policy: Policy, cfg):
    sil_policy = SILCommand(policy, cfg)
    return sil_policy


class SIL(Policy):
    r"""
    Overview:
        Policy class of SIL algorithm.
    """
    sil_config = dict(
        value_weight=0.5,
        learning_rate=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    def __init__(self, policy: Policy, cfg):
        self.base_policy = policy
        self._model = policy._model
        cfg.policy.other.sil = deep_merge_dicts(self.sil_config, cfg.policy.other.sil)
        super(SIL, self).__init__(cfg.policy, model=policy._model, enable_field=policy._enable_field)

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config, main and target models.
        """
        self.base_policy._init_learn()
        # Optimizer
        self._optimizer = Adam(
            self._model.parameters(),
            lr=self._cfg.other.sil.learning_rate,
            betas=self._cfg.other.sil.betas,
            eps=self._cfg.other.sil.eps
        )
        self._priority = self.base_policy._priority
        self._vw = self._cfg.other.sil.value_weight
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()

    # def _data_preprocess_learn(self, data):

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs','adv']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        info1 = self.base_policy._forward_learn(data['base_policy'])
        data = data['sil']
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        output = self._learn_model.forward(data['obs'], mode='compute_actor_critic')
        data = sil_data(output['logit'], data['action'], output['value'], data['total_reward'])
        policy_loss, value_loss = sil_error(data)
        total_loss = policy_loss + self._vw * value_loss
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        info2 = {
            'sil_total_loss': total_loss.item(),
            'sil_policy_loss': policy_loss.item(),
            'sil_value_loss': value_loss.item(),
        }
        info1.update(info2)
        return info1

    def _state_dict_learn(self) -> Dict[str, Any]:
        base_data = self.base_policy._state_dict_learn()
        base_data.update(
            {
                'sil_model': self._learn_model.state_dict(),
                'sil_optimizer': self._optimizer.state_dict(),
            }
        )
        return base_data

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self.base_policy._load_state_dict_learn(state_dict)
        self._learn_model.load_state_dict(state_dict['sil_model'])
        self._optimizer.load_state_dict(state_dict['sil_optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
        """
        self.base_policy._init_collect()
        # Algorithm
        self._gamma = self.base_policy._gamma
        self._unroll_len = self.base_policy._unroll_len

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function for collect mode
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - data (:obj:`dict`): The collected data
        """
        return self.base_policy._forward_collect(data)

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        return self.base_policy._process_transition(obs, model_output, timestep)

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data
        Arguments:
            - data (:obj:`list`): The trajectory's buffer list
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        total_reward = 0
        postprocess_data = []
        # calcuate monte-carlo return
        for d in reversed(data):
            total_reward = d['reward'] + self._gamma * total_reward
            d['total_reward'] = total_reward
            postprocess_data.append(d)
        postprocess_data.reverse()
        postprocess_data = self.base_policy._get_train_sample(postprocess_data)
        return postprocess_data

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy.
        """
        self.base_policy._init_eval()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function for eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        return self.base_policy._forward_eval(data)

    def default_model(self) -> Tuple[str, List[str]]:
        return self.base_policy.default_model()

    def _monitor_vars_learn(self) -> List[str]:
        return self.base_policy._monitor_vars_learn() + [
            'sil_total_loss',
            'sil_policy_loss',
            'sil_value_loss',
        ]


class SILCommand(SIL, DummyCommandModePolicy):
    pass
