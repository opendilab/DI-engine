from typing import Any, Union, List, Tuple, Dict, Callable, Optional
import logging
import numpy as np
from easydict import EasyDict
from collections import namedtuple
from gym.vector.async_vector_env import AsyncVectorEnv

from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.torch_utils import to_ndarray, to_list
from ding.utils import PropagatingThread, LockContextType, LockContext, ENV_MANAGER_REGISTRY
from .base_env_manager import BaseEnvManager


@ENV_MANAGER_REGISTRY.register('gym_vector')
class GymVectorEnvManager(BaseEnvManager):
    """
    Overview:
        Create an GymVectorEnvManager to manage multiple environments.
        Each Environment is run by a respective subprocess.
    Interfaces:
        seed, ready_obs, step, reset
    """
    config = dict(shared_memory=False, )

    def __init__(self, env_fn: List[Callable], cfg: EasyDict) -> None:
        """
        .. note::
            ``env_fn`` must create gym-type environment instance, which may different DI-engine environment.
        """
        self._cfg = cfg
        self._env_fn = env_fn
        self._env_num = len(self._env_fn)
        self._closed = True
        self._env_replay_path = None

        self.env_manager = AsyncVectorEnv(
            env_fns=self._env_fn,
            # observation_space=observation_space,
            # action_space=action_space,
            shared_memory=cfg.shared_memory,
        )
        self.final_eval_reward = [0. for _ in range(self._env_num)]

    def reset(self, reset_param: Optional[Dict] = None) -> None:
        assert reset_param is None
        self.reset_observations = self.env_manager.reset()
        self.final_eval_reward = [0. for _ in range(self._env_num)]

    def step(self, actions: Dict[int, Any]) -> Dict[int, namedtuple]:
        elem = list(actions.values())[0]
        if isinstance(elem, np.ndarray) and elem.shape == (1, ):
            actions = [v.item() for k, v in actions.items()]
        timestep = self.env_manager.step(actions)
        timestep_collate_result = {}
        for i in range(len(actions)):
            timestep_collate_result[i] = BaseEnvTimestep(timestep[0][i], timestep[1][i], timestep[2][i], timestep[3][i])
            self.final_eval_reward[i] += timestep_collate_result[i].reward
            if timestep_collate_result[i].done:
                timestep_collate_result[i].info['final_eval_reward'] = self.final_eval_reward[i]
                self.final_eval_reward[i] = 0
        return timestep_collate_result

    @property
    def ready_obs(self) -> Dict[int, Any]:
        return {i: self.reset_observations[i] for i in range(len(self.reset_observations))}

    def seed(self, seed: Union[Dict[int, int], List[int], int], dynamic_seed: bool = None) -> None:
        # TODO dynamic_seed
        self.env_manager.seed(seed)
        logging.warning("gym env doesn't support dynamic_seed in different episode")
