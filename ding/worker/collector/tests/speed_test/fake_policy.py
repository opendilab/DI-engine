from collections import namedtuple, deque
from typing import Optional, List, Dict, Any, Tuple, Union
import torch
from easydict import EasyDict
import time

from ding.model import create_model
from ding.utils import import_module, allreduce, broadcast, get_rank, POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from ding.policy import Policy
from ding.rl_utils import get_train_sample

from ding.worker.collector.tests.speed_test.utils import random_change


class FakePolicy(Policy):
    config = dict(
        cuda=False,
        on_policy=False,
        forward_time=0.002,
        learn=dict(),
        collect=dict(
            n_sample=80,
            unroll_len=1,
            collector=dict(collect_print_freq=1000000),
        ),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=10000, ), ),
    )

    def __init__(
            self,
            cfg: dict,
            model: Optional[Union[type, torch.nn.Module]] = None,
            enable_field: Optional[List[str]] = None
    ) -> None:
        self._cfg = cfg
        self._cuda = cfg.cuda and torch.cuda.is_available()
        self._init_collect()
        self._forward_time = cfg.forward_time
        self._on_policy = cfg.on_policy
        self.policy_sum = 0
        self.policy_times = 0

    def policy_sleep(self, duration):
        time.sleep(duration)
        self.policy_sum += duration
        self.policy_times += 1

    def _init_learn(self) -> None:
        pass

    def _init_collect(self) -> None:
        self._unroll_len = 1

    def _init_eval(self) -> None:
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

    def _forward_collect(self, data: dict, **kwargs) -> dict:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        self.policy_sleep(random_change(self._forward_time))
        output = {'action': torch.ones(data.shape[0], 1).long()}
        output = default_decollate(output)
        output = {i: d for i, d in zip(data_id, output)}
        return output

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        return get_train_sample(data, self._unroll_len)

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        pass
