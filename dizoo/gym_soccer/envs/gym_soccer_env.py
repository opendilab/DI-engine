from typing import Any, List, Union, Optional
import numpy as np
import gym
import gym_soccer
import torch
from ding.utils import ENV_REGISTRY
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElementInfo
from ding.torch_utils import to_tensor, to_ndarray, to_list
from gym.utils import seeding


@ENV_REGISTRY.register('gym_soccer')
class GymSoccerEnv(BaseEnv):
    default_env_id = ['Soccer-v0', 'SoccerEmptyGoal-v0', 'SoccerAgainstKeeper-v0']

    def __init__(self, cfg: dict = {}) -> None:
        self._cfg = cfg
        self._env_id = cfg.env_id
        assert self._env_id in self.default_env_id
        self._init_flag = False
        self._replay_path = None

    def reset(self) -> torch.Tensor:
        if not self._init_flag:
            self._env = gym.make(self._env_id)
            self._init_flag = True
            if self._replay_path is not None:
                self._env = gym.wrappers.Monitor(
                    self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
                )
                self._env.metadata["render.modes"] = ["human", "rgb_array"]

        self._final_eval_reward = 0
        obs = self._env._reset()
        obs = to_ndarray(obs).astype(np.float32)
        return obs

    def step(self, action: List) -> BaseEnvTimestep:
        obs, rew, done, info = self._env._step(action)
        self._final_eval_reward += rew
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        obs = to_ndarray(obs).astype(np.float32)
        # reward wrapped to be transfered to a Tensor with shape (1,)
        rew = to_ndarray([rew])
        # '1' indicates the discrete action is associated with the continuous parameters
        info['action_args_mask'] = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])
        return BaseEnvTimestep(obs, rew, done, info)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        self._init_flag = False

    def get_random_action(self):
        # action_type: 0, 1, 2
        # action_args:
        #   - power: [0, 100]
        #   - direction: [-180, 180]
        return self._env.action_space.sample()

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T(
                (6, ),
                {
                    # [min, max)
                    'min': -1,
                    'max': 2,
                    'dtype': np.float32,
                },
            ),
            act_space=T(
                (3, ),
                {
                    # [min, max)
                    'min': 0,
                    'max': 3,
                    'dtype': int,
                },
            ),
            rew_space=T(
                (1, ),
                {
                    # [min, max)
                    'min': 0,
                    'max': 2.0,
                    'dtype': int,
                },
            ),
            use_wrappers=None,
        )

    def __repr__(self) -> str:
        return "DI-engine gym soccer Env"

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
