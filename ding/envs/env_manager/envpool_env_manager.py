import gym
from easydict import EasyDict
from copy import deepcopy
import numpy as np
from collections import namedtuple
from typing import Any, Union, List, Tuple, Dict, Callable, Optional
import logging
try:
    import envpool
except ImportError:
    import sys
    logging.warning("Please install envpool first, use 'pip install envpool'")
    envpool = None

from ding.envs import BaseEnvTimestep
from ding.utils import ENV_MANAGER_REGISTRY, deep_merge_dicts


@ENV_MANAGER_REGISTRY.register('env_pool')
class PoolEnvManager:
    '''
    Overview:
        Envpool now supports Atari, Classic Control, Toy Text, ViZDoom.
        Here we list some commonly used env_ids as follows.
        For more examples, you can refer to <https://envpool.readthedocs.io/en/latest/api/atari.html>.

        - Atari: "Pong-v5", "SpaceInvaders-v5", "Qbert-v5"
        - Classic Control: "CartPole-v0", "CartPole-v1", "Pendulum-v0"
    '''

    @classmethod
    def default_config(cls) -> EasyDict:
        return EasyDict(deepcopy(cls.config))

    config = dict(
        type='envpool',
        # Sync  mode: batch_size == env_num
        # Async mode: batch_size <  env_num
        env_num=8,
        batch_size=8,
        # Unlike other env managers, envpool's seed should be specified in config.
        seed=0,
    )

    def __init__(self, cfg: EasyDict) -> None:
        self._cfg = cfg
        self._env_num = cfg.env_num
        self._batch_size = cfg.batch_size
        self._seed = cfg.seed
        self._ready_obs = {}
        self._closed = True

    def launch(self) -> None:
        assert self._closed, "Please first close the env manager"
        self._envs = envpool.make(
            self._cfg.env_id, env_type="gym", num_envs=self._env_num, batch_size=self._batch_size, seed=self._seed
        )
        self._closed = False
        self.reset()

    def reset(self) -> None:
        self._envs.async_reset()
        obs, _, _, info = self._envs.recv()
        env_id = info['env_id']
        print(env_id)
        obs = obs.astype(np.float32)
        self._ready_obs = {i: o for i, o in zip(env_id, obs)}

    def step(self, action) -> Dict[int, namedtuple]:
        env_id = np.array(list(action.keys()))
        action = np.array(list(action.values()))
        if len(action.shape) == 2:
            action = action.squeeze(1)
        self._envs.send(action, env_id)

        obs, rew, done, info = self._envs.recv()
        obs = obs.astype(np.float32)
        rew = rew.astype(np.float32)
        env_id = info['env_id']
        timesteps = {}
        self._ready_obs = {}
        for i in range(len(env_id)):
            d = bool(done[i])
            r = rew[i:i + 1]
            timesteps[env_id[i]] = BaseEnvTimestep(obs[i], r, d, info={'env_id': i, 'final_eval_reward': 0.})
            self._ready_obs[env_id[i]] = obs[i]
        return timesteps

    def close(self) -> None:
        if self._closed:
            return
        # Envpool has no `close` API
        self._closed = True

    def seed(self, seed, dynamic_seed=False) -> None:
        # Envpool's seed is set in `envpool.make`. This method is preserved for compatibility.
        logging.warning("envpool doesn't support dynamic_seed in different episode")

    @property
    def env_num(self) -> int:
        return self._env_num

    @property
    def ready_obs(self) -> Dict[int, Any]:
        return self._ready_obs
