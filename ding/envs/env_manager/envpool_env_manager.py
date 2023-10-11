import gym
from easydict import EasyDict
from copy import deepcopy
import numpy as np
import torch
import treetensor.torch as ttorch
import treetensor.numpy as tnp
from collections import namedtuple
import enum
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


class EnvState(enum.IntEnum):
    VOID = 0
    INIT = 1
    RUN = 2
    RESET = 3
    DONE = 4
    ERROR = 5
    NEED_RESET = 6


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

        kwargs = {}
        if "episodic_life" in self._cfg:
            kwargs["episodic_life"] = self._cfg.episodic_life
        if "reward_clip" in self._cfg:
            kwargs["reward_clip"] = self._cfg.reward_clip
        if "stack_num" in self._cfg:
            kwargs["stack_num"] = self._cfg.stack_num
        if "gray_scale" in self._cfg:
            kwargs["gray_scale"] = self._cfg.gray_scale
        if "frame_skip" in self._cfg:
            kwargs["frame_skip"] = self._cfg.frame_skip

        self._envs = envpool.make(
            task_id=self._cfg.env_id,
            env_type="gym",
            num_envs=self._env_num,
            batch_size=self._batch_size,
            seed=seed,
            **kwargs
        )
        self._action_space = self._envs.action_space
        self._observation_space = self._envs.observation_space
        self._closed = False
        self.reset()

    def reset(self) -> None:
        self._ready_obs = {}
        self._envs.async_reset()
        while True:
            obs, _, _, info = self._envs.recv()
            env_id = info['env_id']
            obs = obs.astype(np.float32)
            obs /= 255.0
            self._ready_obs = deep_merge_dicts({i: o for i, o in zip(env_id, obs)}, self._ready_obs)
            if len(self._ready_obs) == self._env_num:
                break
        self._eval_episode_return = [0. for _ in range(self._env_num)]

    def step(self, action: dict) -> Dict[int, namedtuple]:
        env_id = np.array(list(action.keys()))
        action = np.array(list(action.values()))
        if len(action.shape) == 2:
            action = action.squeeze(1)
        self._envs.send(action, env_id)

        obs, rew, done, info = self._envs.recv()
        obs = obs.astype(np.float32)
        obs /= 255.0
        rew = rew.astype(np.float32)
        env_id = info['env_id']
        timesteps = {}
        self._ready_obs = {}
        for i in range(len(env_id)):
            d = bool(done[i])
            r = to_ndarray([rew[i]])
            self._eval_episode_return[env_id[i]] += r
            timesteps[env_id[i]] = BaseEnvTimestep(obs[i], r, d, info={'env_id': i})
            if d:
                timesteps[env_id[i]].info['eval_episode_return'] = self._eval_episode_return[env_id[i]]
                self._eval_episode_return[env_id[i]] = 0.
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

    @property
    def observation_space(self) -> 'gym.spaces.Space':  # noqa
        try:
            return self._observation_space
        except AttributeError:
            self.launch()
            self.close()
            return self._observation_space

    @property
    def action_space(self) -> 'gym.spaces.Space':  # noqa
        try:
            return self._action_space
        except AttributeError:
            self.launch()
            self.close()
            return self._action_space


@ENV_MANAGER_REGISTRY.register('env_pool_v2')
class PoolEnvManagerV2:
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
        super().__init__()
        self._cfg = cfg
        self._env_num = cfg.env_num
        self._batch_size = cfg.batch_size
        self._ready_obs = {}
        self._closed = True
        self._seed = None
        self._test = False

    def launch(self) -> None:
        assert self._closed, "Please first close the env manager"
        if self._seed is None:
            seed = 0
        else:
            seed = self._seed

        kwargs = {}
        if "episodic_life" in self._cfg:
            kwargs["episodic_life"] = self._cfg.episodic_life
        if "reward_clip" in self._cfg:
            kwargs["reward_clip"] = self._cfg.reward_clip
        if "stack_num" in self._cfg:
            kwargs["stack_num"] = self._cfg.stack_num
        if "gray_scale" in self._cfg:
            kwargs["gray_scale"] = self._cfg.gray_scale
        if "frame_skip" in self._cfg:
            kwargs["frame_skip"] = self._cfg.frame_skip
        if "test" in self._cfg:
            self._test = self._cfg.test

        self._envs = envpool.make(
            task_id=self._cfg.env_id,
            env_type="gym",
            num_envs=self._env_num,
            batch_size=self._batch_size,
            seed=seed,
            **kwargs
        )
        self._action_space = self._envs.action_space
        self._observation_space = self._envs.observation_space
        self._closed = False
        self.reset()

    def reset(self) -> None:
        self._ready_obs = {}
        self._envs.async_reset()
        while True:
            obs, _, _, info = self._envs.recv()
            env_id = info['env_id']
            obs = obs.astype(np.float32)
            obs /= 255.0
            self._ready_obs = deep_merge_dicts({i: o for i, o in zip(env_id, obs)}, self._ready_obs)
            if len(self._ready_obs) == self._env_num:
                break
        self._eval_episode_return = [0. for _ in range(self._env_num)]

    def step(self, action: Union[List, np.ndarray]) -> Dict[int, namedtuple]:
        env_id = np.array(list(self._ready_obs.keys()))
        action = np.array(action)
        if len(action.shape) == 2:
            action = action.squeeze(1)
        self._envs.send(action, env_id)

        obs, rew, done, info = self._envs.recv()
        if self._test:
            assert all(info['env_id'] == env_id)
        obs = obs.astype(np.float32)
        obs /= 255.0
        rew = rew.astype(np.float32)
        env_id = info['env_id']
        timesteps = {}
        new_data = []
        self._ready_obs = {}
        for i in range(len(env_id)):
            d = bool(done[i])
            r = to_ndarray([rew[i]])
            self._eval_episode_return[env_id[i]] += r
            info_dict = {'env_id': i}
            timesteps[env_id[i]] = BaseEnvTimestep(obs[i], r, d, info=info_dict)
            if d:
                info_dict['eval_episode_return'] = self._eval_episode_return[env_id[i]]
                timesteps[env_id[i]].info['eval_episode_return'] = info_dict['eval_episode_return']
                self._eval_episode_return[env_id[i]] = 0.
            new_data.append(tnp.array({'obs': obs[i], 'reward': r, 'done': d, 'info': info_dict, 'env_id': env_id[i]}))
            self._ready_obs[env_id[i]] = obs[i]
        return new_data

    def step_v2(self, action: Dict[int, np.ndarray]) -> Dict[int, namedtuple]:
        # env_id = np.array(list(action.keys()))
        # action = np.array(action)
        # if len(action.shape) == 2:
        #     action = action.squeeze(1)
        self._envs.send(action)

        obs, rew, done, info = self._envs.recv()
        if self._test:
            assert all(info['env_id'] == env_id)
        obs = obs.astype(np.float32)
        obs /= 255.0
        rew = rew.astype(np.float32)
        env_id = info['env_id']
        timesteps = {}
        new_data = []
        self._ready_obs = {}
        for i in range(len(env_id)):
            d = bool(done[i])
            r = to_ndarray([rew[i]])
            self._eval_episode_return[env_id[i]] += r
            info_dict = {'env_id': i}
            timesteps[env_id[i]] = BaseEnvTimestep(obs[i], r, d, info=info_dict)
            if d:
                info_dict['eval_episode_return'] = self._eval_episode_return[env_id[i]]
                timesteps[env_id[i]].info['eval_episode_return'] = info_dict['eval_episode_return']
                self._eval_episode_return[env_id[i]] = 0.
            new_data.append(tnp.array({'obs': obs[i], 'reward': r, 'done': d, 'info': info_dict, 'env_id': env_id[i]}))
            self._ready_obs[env_id[i]] = obs[i]
        return new_data

    def close(self) -> None:
        if self._closed:
            return
        # Envpool has no `close` API
        self._closed = True

    @property
    def closed(self) -> None:
        return self._closed

    def seed(self, seed: int, dynamic_seed=False) -> None:
        # The i-th environment seed in Envpool will be set with i+seed, so we don't do extra transformation here
        self._seed = seed
        logging.warning("envpool doesn't support dynamic_seed in different episode")

    @property
    def env_num(self) -> int:
        return self._env_num

    @property
    def ready_obs(self) -> tnp.array:
        if isinstance(self._ready_obs, dict):
            obs = [tnp.array(o) for k, o in self._ready_obs.items()]
            return tnp.stack(obs)
        else:
            raise NotImplementedError

    @property
    def ready_obs_v2(self) -> tnp.array:
        if self._ready_obs is not None:
            return self._ready_obs
        else:
            raise ValueError

    @property
    def observation_space(self) -> 'gym.spaces.Space':  # noqa
        try:
            return self._observation_space
        except AttributeError:
            self.launch()
            self.close()
            self._ready_obs = {}
            return self._observation_space

    @property
    def action_space(self) -> 'gym.spaces.Space':  # noqa
        try:
            return self._action_space
        except AttributeError:
            self.launch()
            self.close()
            self._ready_obs = {}
            return self._action_space


@ENV_MANAGER_REGISTRY.register('env_pool_v3')
class PoolEnvManagerV3:
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
        super().__init__()
        self._cfg = cfg
        self._env_num = cfg.env_num
        self._batch_size = cfg.batch_size
        self._ready_obs = {}
        self._closed = True
        self._seed = None
        self._test = False

    def launch(self) -> None:
        assert self._closed, "Please first close the env manager"
        if self._seed is None:
            seed = 0
        else:
            seed = self._seed

        kwargs = {}
        if "episodic_life" in self._cfg:
            kwargs["episodic_life"] = self._cfg.episodic_life
        if "reward_clip" in self._cfg:
            kwargs["reward_clip"] = self._cfg.reward_clip
        if "stack_num" in self._cfg:
            kwargs["stack_num"] = self._cfg.stack_num
        if "gray_scale" in self._cfg:
            kwargs["gray_scale"] = self._cfg.gray_scale
        if "frame_skip" in self._cfg:
            kwargs["frame_skip"] = self._cfg.frame_skip
        if "test" in self._cfg:
            self._test = self._cfg.test

        self._envs = envpool.make(
            task_id=self._cfg.env_id,
            env_type="gym",
            num_envs=self._env_num,
            batch_size=self._batch_size,
            seed=seed,
            **kwargs
        )
        self._action_space = self._envs.action_space
        self._observation_space = self._envs.observation_space
        self._closed = False
        self.reset()

    def reset(self) -> None:
        self._ready_obs = {}
        self._ready_obs_send = {}
        self._ready_action_send = {}
        self._envs.async_reset()
        while True:
            obs, _, _, info = self._envs.recv()
            env_id = info['env_id']
            obs = obs.astype(np.float32)
            obs /= 255.0
            self._ready_obs = deep_merge_dicts({i: o for i, o in zip(env_id, obs)}, self._ready_obs)
            if len(self._ready_obs) == self._env_num:
                break
        self._eval_episode_return = [0. for _ in range(self._env_num)]

    def step(self, action: Union[List, np.ndarray]) -> Dict[int, namedtuple]:
        env_id = np.array(list(self._ready_obs.keys()))
        action = np.array(action)
        if len(action.shape) == 2:
            action = action.squeeze(1)
        self._envs.send(action, env_id)

        obs, rew, done, info = self._envs.recv()
        if self._test:
            assert all(info['env_id'] == env_id)
        obs = obs.astype(np.float32)
        obs /= 255.0
        rew = rew.astype(np.float32)
        env_id = info['env_id']
        timesteps = {}
        new_data = []
        self._ready_obs = {}
        for i in range(len(env_id)):
            d = bool(done[i])
            r = to_ndarray([rew[i]])
            self._eval_episode_return[env_id[i]] += r
            info_dict = {'env_id': i}
            timesteps[env_id[i]] = BaseEnvTimestep(obs[i], r, d, info=info_dict)
            if d:
                info_dict['eval_episode_return'] = self._eval_episode_return[env_id[i]]
                timesteps[env_id[i]].info['eval_episode_return'] = info_dict['eval_episode_return']
                self._eval_episode_return[env_id[i]] = 0.
            new_data.append(tnp.array({'obs': obs[i], 'reward': r, 'done': d, 'info': info_dict, 'env_id': env_id[i]}))
            self._ready_obs[env_id[i]] = obs[i]
        return new_data

    def collect_data(self, num, policy=None, policy_forward_kwargs=None):
        if self.closed:
            self.launch()

        new_data = []

        while len(new_data) < num:

            obs_to_send = self._ready_obs
            env_id_to_send = list(obs_to_send.keys())
            num_to_send = len(obs_to_send.keys())
            if num_to_send > 0:
                if policy:
                    action_to_send = policy.forward(obs_to_send, **policy_forward_kwargs)
                else:
                    #random policy
                    action_to_send = {i: {"action": np.array([self._action_space.sample()])} for i in env_id_to_send}
                self._ready_obs_send.update(obs_to_send)
                self._ready_action_send.update(action_to_send)
                action = np.array([action_to_send[i]['action'] for i in env_id_to_send])
                if action.ndim == 2 and action.shape[1] == 1:
                    action = action.squeeze(1)
                env_id = np.array(env_id_to_send)
                self._envs.send(action, env_id)



            next_obs, rew, done, info = self._envs.recv()
            next_obs = next_obs.astype(np.float32)
            next_obs /= 255.0
            rew = rew.astype(np.float32)
            env_id = info['env_id']

            self._ready_obs = {}
            for i in range(len(env_id)):
                new_data.append(
                    ttorch.tensor(
                        {
                            'obs': self._ready_obs_send[env_id[i]],
                            'action': self._ready_action_send[env_id[i]]['action'],
                            'next_obs': next_obs[i],
                            'reward': np.array([rew[i]]),
                            'done': done[i]
                        }
                    )
                )
                self._ready_obs[env_id[i]] = next_obs[i]

        return new_data

    def collect_data_nstep(self, num, n_step=1, policy=None, policy_forward_kwargs=None):
        if self.closed:
            self.launch()

        new_data = []
        

        while len(new_data) < num:

            obs_to_send = self._ready_obs
            env_id_to_send = list(obs_to_send.keys())
            num_to_send = len(obs_to_send.keys())
            if num_to_send > 0:
                if policy:
                    action_to_send = policy.forward(obs_to_send, **policy_forward_kwargs)
                else:
                    #random policy
                    action_to_send = {i: {"action": np.array([self._action_space.sample()])} for i in env_id_to_send}
                self._ready_obs_send.update(obs_to_send)
                self._ready_action_send.update(action_to_send)
                action = np.array([action_to_send[i]['action'] for i in env_id_to_send])
                if action.ndim == 2 and action.shape[1] == 1:
                    action = action.squeeze(1)
                env_id = np.array(env_id_to_send)
                self._envs.send(action, env_id)

            next_obs, rew, done, info = self._envs.recv()
            next_obs = next_obs.astype(np.float32)
            next_obs /= 255.0
            rew = rew.astype(np.float32)
            env_id = info['env_id']

            self._ready_obs = {}
            for i in range(len(env_id)):
                new_data.append(
                    ttorch.tensor(
                        {
                            'obs': self._ready_obs_send[env_id[i]],
                            'action': self._ready_action_send[env_id[i]]['action'],
                            'next_obs': next_obs[i],
                            'reward': np.array([rew[i]]),
                            'done': done[i]
                        }
                    )
                )
                self._ready_obs[env_id[i]] = next_obs[i]

        return new_data

    def close(self) -> None:
        if self._closed:
            return
        # Envpool has no `close` API
        self._closed = True

    @property
    def closed(self) -> None:
        return self._closed

    def seed(self, seed: int, dynamic_seed=False) -> None:
        # The i-th environment seed in Envpool will be set with i+seed, so we don't do extra transformation here
        self._seed = seed
        logging.warning("envpool doesn't support dynamic_seed in different episode")

    @property
    def env_num(self) -> int:
        return self._env_num

    @property
    def ready_obs(self) -> tnp.array:
        if isinstance(self._ready_obs, dict):
            obs = [tnp.array(o) for k, o in self._ready_obs.items()]
            return tnp.stack(obs)
        else:
            raise NotImplementedError

    @property
    def ready_obs_v2(self) -> tnp.array:
        if self._ready_obs is not None:
            return self._ready_obs
        else:
            raise ValueError

    @property
    def observation_space(self) -> 'gym.spaces.Space':  # noqa
        try:
            return self._observation_space
        except AttributeError:
            self.launch()
            self.close()
            self._ready_obs = {}
            return self._observation_space

    @property
    def action_space(self) -> 'gym.spaces.Space':  # noqa
        try:
            return self._action_space
        except AttributeError:
            self.launch()
            self.close()
            self._ready_obs = {}
            return self._action_space


@ENV_MANAGER_REGISTRY.register('env_pool_v4')
class PoolEnvManagerV4:
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
        env_num=8,
        batch_size=8,
    )

    def __init__(self, cfg: EasyDict) -> None:
        super().__init__()
        self._cfg = cfg
        self._env_num = cfg.env_num
        self._batch_size = cfg.batch_size

        self._closed = True
        self._seed = None
        self._test = False

    def launch(self) -> None:
        assert self._closed, "Please first close the env manager"
        if self._seed is None:
            seed = 0
        else:
            seed = self._seed

        kwargs = {}
        if "episodic_life" in self._cfg:
            kwargs["episodic_life"] = self._cfg.episodic_life
        if "reward_clip" in self._cfg:
            kwargs["reward_clip"] = self._cfg.reward_clip
        if "stack_num" in self._cfg:
            kwargs["stack_num"] = self._cfg.stack_num
        if "gray_scale" in self._cfg:
            kwargs["gray_scale"] = self._cfg.gray_scale
        if "frame_skip" in self._cfg:
            kwargs["frame_skip"] = self._cfg.frame_skip
        if "test" in self._cfg:
            self._test = self._cfg.test

        self._envs = envpool.make(
            task_id=self._cfg.env_id,
            env_type="gym",
            num_envs=self._env_num,
            batch_size=self._batch_size,
            seed=seed,
            **kwargs
        )
        self._action_space = self._envs.action_space
        self._observation_space = self._envs.observation_space
        self._closed = False
        return self.reset()

    def reset(self) -> None:
        self._envs.async_reset()
        ready_obs={}
        while True:
            obs, _, _, info = self._envs.recv()
            env_id = info['env_id']
            obs = obs.astype(np.float32)
            obs /= 255.0
            ready_obs = deep_merge_dicts({i: o for i, o in zip(env_id, obs)}, ready_obs)
            if len(ready_obs) == self._env_num:
                break
        self._eval_episode_return = [0. for _ in range(self._env_num)]

        return ready_obs

    def send_action(self, action, env_id) -> Dict[int, namedtuple]:
        self._envs.send(action, env_id)

    def receive_data(self):
        next_obs, rew, done, info = self._envs.recv()
        next_obs = next_obs.astype(np.float32)
        next_obs /= 255.0
        rew = rew.astype(np.float32)
        
        return next_obs, rew, done, info
    
    def close(self) -> None:
        if self._closed:
            return
        # Envpool has no `close` API
        self._closed = True

    @property
    def closed(self) -> None:
        return self._closed

    def seed(self, seed: int, dynamic_seed=False) -> None:
        # The i-th environment seed in Envpool will be set with i+seed, so we don't do extra transformation here
        self._seed = seed
        logging.warning("envpool doesn't support dynamic_seed in different episode")

    @property
    def env_num(self) -> int:
        return self._env_num

    @property
    def observation_space(self) -> 'gym.spaces.Space':  # noqa
        try:
            return self._observation_space
        except AttributeError:
            self.launch()
            self.close()
            self._ready_obs = {}
            return self._observation_space

    @property
    def action_space(self) -> 'gym.spaces.Space':  # noqa
        try:
            return self._action_space
        except AttributeError:
            self.launch()
            self.close()
            self._ready_obs = {}
            return self._action_space
