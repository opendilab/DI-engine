from typing import Any, Union, List, Tuple, Dict, Callable, Optional
from ditk import logging
import numpy as np
from easydict import EasyDict
from collections import namedtuple
import gym
from gym.vector.async_vector_env import AsyncVectorEnv

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list
from ding.utils import PropagatingThread, LockContextType, LockContext, ENV_MANAGER_REGISTRY
from .base_env_manager import BaseEnvManager
from .base_env_manager import EnvState


@ENV_MANAGER_REGISTRY.register('gym_vector')
class GymVectorEnvManager(BaseEnvManager):
    """
    Overview:
        Create an GymVectorEnvManager to manage multiple environments.
        Each Environment is run by a respective subprocess.
    Interfaces:
        seed, ready_obs, step, reset, close
    """
    config = dict(shared_memory=False, episode_num=float("inf"))

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
        # env_ref is used to acquire some common attributes of env, like obs_shape and act_shape
        self._env_ref = self._env_fn[0]()
        self._env_states = {i: EnvState.VOID for i in range(self._env_num)}

        self._episode_num = self._cfg.episode_num
        self._env_episode_count = {i: 0 for i in range(self.env_num)}

        self._env_manager = AsyncVectorEnv(
            env_fns=self._env_fn,
            # observation_space=observation_space,
            # action_space=action_space,
            shared_memory=cfg.shared_memory,
        )
        self._env_states = {i: EnvState.INIT for i in range(self._env_num)}
        self._final_eval_reward = [0. for _ in range(self._env_num)]

    def reset(self, reset_param: Optional[Dict] = None) -> None:
        assert reset_param is None
        self._closed = False
        for env_id in range(self.env_num):
            self._env_states[env_id] = EnvState.RESET
        self._ready_obs = self._env_manager.reset()
        for env_id in range(self.env_num):
            self._env_states[env_id] = EnvState.RUN
        self._final_eval_reward = [0. for _ in range(self._env_num)]

    def step(self, actions: Dict[int, Any]) -> Dict[int, namedtuple]:
        assert isinstance(actions, Dict), type(actions)

        env_ids_given = list(actions.keys())
        for env_id in range(self.env_num):
            if env_id not in actions.keys():
                actions[env_id] = self._env_ref.random_action()
        """actions should be sorted by keys, since the original implementation
        of the step method in gym accepts list-type actions"""
        actions = dict(sorted(actions.items()))

        actions = list(actions.values())
        elem = actions[0]
        if not isinstance(elem, np.ndarray):
            raise Exception('DI-engine only accept np.ndarray-type action!')
        if elem.shape == (1, ):
            actions = [v.item() for v in actions]

        timestep = self._env_manager.step(actions)
        timestep_collate_result = {}
        for i in range(self.env_num):
            if i in env_ids_given:
                # Fix the compatability of API for both gym>=0.24.0 and gym<0.24.0
                # https://github.com/openai/gym/pull/2773
                if gym.version.VERSION >= '0.24.0':
                    timestepinfo = {}
                    for k, v in timestep[3].items():
                        timestepinfo[k] = v[i]
                    timestep_collate_result[i] = BaseEnvTimestep(
                        timestep[0][i], timestep[1][i], timestep[2][i], timestepinfo
                    )
                else:
                    timestep_collate_result[i] = BaseEnvTimestep(
                        timestep[0][i], timestep[1][i], timestep[2][i], timestep[3][i]
                    )
                self._final_eval_reward[i] += timestep_collate_result[i].reward
                if timestep_collate_result[i].done:
                    timestep_collate_result[i].info['final_eval_reward'] = self._final_eval_reward[i]
                    self._final_eval_reward[i] = 0
                    self._env_episode_count[i] += 1
                    if self._env_episode_count[i] >= self._episode_num:
                        self._env_states[i] = EnvState.DONE
                    else:
                        self._env_states[i] = EnvState.RESET
                        if all([self._env_states[i] == EnvState.RESET for i in range(self.env_num)]):
                            self.reset()
                else:
                    self._ready_obs[i] = timestep_collate_result[i].obs

        return timestep_collate_result

    @property
    def ready_obs(self) -> Dict[int, Any]:
        return {
            i: self._ready_obs[i]
            for i in range(len(self._ready_obs)) if self._env_episode_count[i] < self._episode_num
        }

    def seed(self, seed: Union[Dict[int, int], List[int], int], dynamic_seed: bool = None) -> None:
        self._env_manager.seed(seed)
        # TODO dynamic_seed
        logging.warning("gym env doesn't support dynamic_seed in different episode")

    def close(self) -> None:
        """
        Overview:
            Release the environment resources
            Since not calling super.__init__, no need to release BaseEnvManager's resources
        """
        if self._closed:
            return
        self._closed = True
        self._env_ref.close()
        self._env_manager.close()
        self._env_manager.close_extras(terminate=True)
