from collections import namedtuple, deque
from typing import Optional, List, Dict, Any, Tuple, Union
import torch
from easydict import EasyDict
import time

from nervex.model import create_model
from nervex.utils import import_module, allreduce, broadcast, get_rank, POLICY_REGISTRY
from nervex.policy import Policy, CommonPolicy
from nervex.rl_utils import Adder

from nervex.worker.collector.tests.speed_test.utils import random_change


def policy_sleep(duration: float) -> None:
    time.sleep(duration)


class FakePolicy(CommonPolicy):

    def __init__(
            self,
            cfg: dict,
            model: Optional[Union[type, torch.nn.Module]] = None,
            enable_field: Optional[List[str]] = None
    ) -> None:
        self._use_cuda = cfg.use_cuda and torch.cuda.is_available()
        self._init_collect()
        self._forward_time = cfg.get('forward_time', 0.)

    def _init_learn(self) -> None:
        pass

    def _init_collect(self) -> None:
        self._adder = Adder(self._use_cuda, 1)

    def _init_eval(self) -> None:
        pass

    def _init_command(self) -> None:
        pass

    def default_model(self) -> Tuple[str, List[str]]:
        pass

    def _create_model(self, cfg: dict, model: Optional[Union[type, torch.nn.Module]] = None) -> torch.nn.Module:
        pass

    def _forward_eval(self, data_id: List[int], data: dict) -> dict:
        pass

    def _forward_learn(self, data_id: List[int], data: dict) -> dict:
        pass

    # *************************************** collect function ************************************
    # def _data_preprocess_collect(self, data: Dict[int, Any]) -> Tuple[List[int], dict]:
    #     raise NotImplementedError

    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        policy_sleep(random_change(self._forward_time))
        return {'action': torch.ones(data['obs'].shape[0], 2)}
        # pass

    def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': armor_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    # def _data_postprocess_collect(self, data_id: List[int], data: dict) -> Dict[int, dict]:
    #     raise NotImplementedError

    # def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
    #     return self._adder.get_train_sample(data)

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        pass

    # --- tiasnhou ---
    # def __init__(self, dict_state=False, need_state=True):
    #     super().__init__()
    #     self.dict_state = dict_state
    #     self.need_state = need_state

    # def forward(self, batch, state=None):
    #     if self.need_state:
    #         if state is None:
    #             state = np.zeros((len(batch.obs), 2))
    #         else:
    #             state += 1
    #     if self.dict_state:
    #         return Batch(act=np.ones(len(batch.obs['index'])), state=state)
    #     return Batch(act=np.ones(len(batch.obs)), state=state)

    # def learn(self):
    #     pass
