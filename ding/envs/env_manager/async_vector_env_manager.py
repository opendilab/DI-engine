from ding.torch_utils.data_helper import to_tensor
from .base_env_manager import BaseEnvManager, EnvState
from typing import Any, Union, List, Tuple, Dict, Callable, Optional
from easydict import EasyDict
from collections import namedtuple
from .async_vector_env import AsyncVectorEnv
import gym
from gym import spaces
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.torch_utils import to_ndarray, to_list
import numpy as np
from ding.utils import PropagatingThread, LockContextType, LockContext, ENV_MANAGER_REGISTRY


@ENV_MANAGER_REGISTRY.register('gym_async')
class AsyncVectorEnvManager(BaseEnvManager):
    """
    Overview:
        Create an AsyncVectorEnvManager to manage multiple environments.
        Each Environment is run by a respective subprocess.
    Interfaces:
        seed, ready_obs, step, reset
    """

    def __init__(self, env_fn: List[Callable], cfg: EasyDict = ...) -> None:
        super().__init__(env_fn, cfg=cfg)
        self.env_fns = env_fn
        self.env_manager = AsyncVectorEnv(
            env_fns=self.env_fns,
            # observation_space=observation_space,
            # action_space=action_space,
            shared_memory=False,
        )

    def reset(self, reset_param: Optional[Dict] = None) -> None:
        self.reset_observations = self.env_manager.reset()

    def step(self, actions: Dict[int, Any]) -> Dict[int, namedtuple]:
        actions = [v.item() for k, v in actions.items()]
        timestep = self.env_manager.step(actions)
        timestep_collate = {}
        for i in range(len(actions)):
            timestep_collate[i] = BaseEnvTimestep(timestep[0][i], timestep[1][i], timestep[2][i], timestep[3][i])
        return timestep_collate

    @property
    def ready_obs(self) -> Dict[int, Any]:
        return {i: self.reset_observations[i] for i in range(len(self.reset_observations))}

    def seed(self, seed: Union[Dict[int, int], List[int], int], dynamic_seed: bool = None) -> None:
        self.env_manager.seed(seed)