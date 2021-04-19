from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
from easydict import EasyDict
from copy import deepcopy
import numpy as np

from nervex.torch_utils import Adam
from nervex.rl_utils import q_1step_td_data, q_1step_td_error, epsilon_greedy
from nervex.model import FCDiscreteNet
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy
from .common_policy import CommonPolicy


@POLICY_REGISTRY.register('dqn_vanilla')
class DQNVanillaPolicy(CommonPolicy):

    def _init_learn(self) -> None:
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        algo_cfg = self._cfg.learn.algo
        self._gamma = algo_cfg.discount_factor
        self._target_model = deepcopy(self._model)
        self._model.train()
        self._target_model.train()

        self._update_count = 0
        self._target_update_freq = algo_cfg.target_update_freq

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        # forward
        with torch.enable_grad():
            ret = self._model(data['obs'])
            q_value = self._model(data['obs'])['logit']
        with torch.no_grad():
            target_q_value = self._target_model(data['next_obs'])['logit']
            target_q_action = self._model(data['next_obs'])['logit'].argmax(dim=-1)
        data = q_1step_td_data(
            q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], data['weight']
        )
        loss = q_1step_td_error(data, self._gamma)
        # update
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        state = self._model.state_dict()
        target_state = self._target_model.state_dict()
        # after update
        if (self._update_count + 1) % self._target_update_freq == 0:
            self._target_model.load_state_dict(self._model.state_dict(), strict=True)
        self._update_count += 1

        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
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
        self._unroll_len = self._cfg.collect.unroll_len

    def _forward_collect(self, data_id: List[int], data: dict, eps: float) -> dict:
        with torch.no_grad():
            logit = self._model(data['obs'])['logit']
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        action = []
        for i, l in enumerate(logit):
            if np.random.random() > eps:
                action.append(l.argmax(dim=-1))
            else:
                action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output = {'action': action}
        return output

    def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': armor_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _init_eval(self) -> None:
        pass

    def _forward_eval(self, data_id: List[int], data: dict) -> dict:
        with torch.no_grad():
            logit = self._model(data['obs'])['logit']
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        action = []
        for i, l in enumerate(logit):
            action.append(l.argmax(dim=-1))
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output = {'action': action}
        return output

    def default_model(self) -> Tuple[str, List[str]]:
        return 'fc_discrete_net', ['nervex.model.discrete_net.discrete_net']

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        return data

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        self._model.train()

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        self._model.eval()

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        self._model.eval()
