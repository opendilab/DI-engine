from typing import Any, List, Union, Optional
import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY

@ENV_REGISTRY.register('mountain_car')
class MountainCar(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._env = gym.make('MountainCar-v0')
        self._init_flag = False
        pass

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        pass

    def reset(self) -> np.ndarray:
        pass

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        pass