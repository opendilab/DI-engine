import gym
from easydict import EasyDict
from copy import deepcopy
import numpy as np
from collections import namedtuple
from typing import Any, Union, List, Tuple, Dict, Callable, Optional
from ditk import logging
try:
    import envpool
except ImportError:
    import sys
    logging.warning("Please install envpool first, use 'pip install envpool'")
    envpool = None

from ding.envs import BaseEnvTimestep
from ding.utils import ENV_MANAGER_REGISTRY, deep_merge_dicts
from ding.torch_utils import to_ndarray


@ENV_MANAGER_REGISTRY.register('env_pool')
class PoolEnvManager:
    '''
    Overview:
        Envpool now supports Atari, Classic Control, Toy Text, ViZDoom.
        Here we list some commonly used env_ids as follows.
        For more examples, you can refer to <https://envpool.readthedocs.io/en/latest/api/atari.html>.

        - Atari: "Pong-v5", "SpaceInvaders-v5", "Qbert-v5"
        - Classic Control: "CartPole-v0", "CartPole-v1", "Pendulum-v1"
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
    )

    def __init__(self, cfg: EasyDict) -> None:
        self._cfg = cfg
        self._env_num = cfg.env_num
        self._batch_size = cfg.batch_size
        self._ready_obs = {}
        self._closed = True
        self._seed = None

    def launch(self) -> None:
        assert self._closed, "Please first close the env manager"
        if self._seed is None:
            seed = 0
        else:
            seed = self._seed
        self._envs = envpool.make(
            task_id=self._cfg.env_id,
            env_type="gym",
            num_envs=self._env_num,
            batch_size=self._batch_size,
            seed=seed,
            episodic_life=self._cfg.episodic_life,
            reward_clip=self._cfg.reward_clip,
            stack_num=self._cfg.stack_num,
            gray_scale=self._cfg.gray_scale,
            frame_skip=self._cfg.frame_skip
        )
        self._closed = False
        self.reset()

    def reset(self) -> None:
        self._ready_obs = {}
        self._envs.async_reset()
        while True:
            obs, _, _, info = self._envs.recv()
            env_id = info['env_id']
            obs = obs.astype(np.float32)
            self._ready_obs = deep_merge_dicts({i: o for i, o in zip(env_id, obs)}, self._ready_obs)
            if len(self._ready_obs) == self._env_num:
                break
        self._final_eval_reward = [0. for _ in range(self._env_num)]

    def step(self, action: dict) -> Dict[int, namedtuple]:
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
            r = to_ndarray([rew[i]])
            self._final_eval_reward[env_id[i]] += r
            timesteps[env_id[i]] = BaseEnvTimestep(obs[i], r, d, info={'env_id': i})
            if d:
                timesteps[env_id[i]].info['final_eval_reward'] = self._final_eval_reward[env_id[i]]
                self._final_eval_reward[env_id[i]] = 0.
            self._ready_obs[env_id[i]] = obs[i]
        return timesteps

    def close(self) -> None:
        if self._closed:
            return
        # Envpool has no `close` API
        self._closed = True

    def seed(self, seed: int, dynamic_seed=False) -> None:
        # The i-th environment seed in Envpool will be set with i+seed, so we don't do extra transformation here
        self._seed = seed
        logging.warning("envpool doesn't support dynamic_seed in different episode")

    @property
    def env_num(self) -> int:
        return self._env_num

    @property
    def ready_obs(self) -> Dict[int, Any]:
        return self._ready_obs
