from typing import Any, List, Dict, Union, Optional
import time
import gym
import gym_hybrid
import copy
import numpy as np
from easydict import EasyDict
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common import EnvElementInfo, affine_transform
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('gym_hybrid')
class GymHybridEnv(BaseEnv):
    default_env_id = ['Sliding-v0', 'Moving-v0']

    def __init__(self, cfg: EasyDict) -> None:
        self._cfg = cfg
        self._env_id = cfg.env_id
        assert self._env_id in self.default_env_id
        self._act_scale = cfg.act_scale
        self._init_flag = False
        self._replay_path = None

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make(self._env_id)
            if self._replay_path is not None:
                self._env = gym.wrappers.Monitor(
                    self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
                )
                self._env.metadata["render.modes"] = ["human", "rgb_array"]
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
            )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        self._final_eval_reward = 0
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Dict) -> BaseEnvTimestep:
        if self._act_scale:
            # acceleration_value.
            action['action_args'][0] = affine_transform(action['action_args'][0], min_val=0, max_val=1)
            # rotation_value. Following line can be omitted, because in the affine_transform function,
            # we have already done the clip(-1,1) operation
            action['action_args'][1] = affine_transform(action['action_args'][1], min_val=-1, max_val=1)
            action = [action['action_type'], action['action_args']]
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        obs = to_ndarray(obs)
        if isinstance(obs, list):  # corner case
            for i in range(len(obs)):
                if len(obs[i].shape) == 0:
                    obs[i] = np.array([obs[i]])
            obs = np.concatenate(obs)
        assert isinstance(obs, np.ndarray) and obs.shape == (10, )
        obs = obs.astype(np.float32)

        rew = to_ndarray([rew])  # wrapped to be transfered to a numpy array with shape (1,)
        if isinstance(rew, list):
            rew = rew[0]
        assert isinstance(rew, np.ndarray) and rew.shape == (1, )
        info['action_args_mask'] = np.array([[1, 0], [0, 1], [0, 0]])
        return BaseEnvTimestep(obs, rew, done, info)

    def random_action(self) -> Dict:
        # action_type: 0, 1, 2
        # action_args:
        #   - acceleration_value: [0, 1]
        #   - rotation_value: [-1, 1]
        raw_action = self._action_space.sample()
        return {'action_type': raw_action[0], 'action_args': raw_action[1]}

    def __repr__(self) -> str:
        return "DI-engine gym hybrid Env"

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space
