import gym
import copy
import numpy as np
from typing import List
from easydict import EasyDict
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_ndarray
from ding.envs import BaseEnv, BaseEnvTimestep
from .sokoban_wrappers import wrap_sokoban


@ENV_REGISTRY.register('sokoban')
class SokobanEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._env_id = cfg.env_id
        self._init_flag = False
        self._save_replay = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = self._make_env(only_info=False)
            self._init_flag = True

            if self._save_replay:
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix='rl-video-{}'.format(id(self))
                )

            self._env.observation_space.dtype = np.float32  # To unify the format of envs in DI-engine
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
            )

            if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
                np_seed = 100 * np.random.randint(1, 1000)
                self._env.seed(self._seed + np_seed)
            elif hasattr(self, '_seed'):
                self._env.seed(self._seed)
            obs = self._env.reset()
            obs = to_ndarray(obs).astype('float32')
            self._eval_episode_return = 0.
            return obs

    def step(self, action: np.array):
        action = to_ndarray(action)
        obs, rew, done, info = self._env.step(int(action))
        self._eval_episode_return += rew
        obs = to_ndarray(obs).astype('float32')
        rew = to_ndarray([rew])  # wrapped to be transfered to a array with shape (1,)
        if done:
            info['eval_episode_return'] = self._eval_episode_return
        return BaseEnvTimestep(obs, rew, done, info)

    def _make_env(self, only_info=False):
        return wrap_sokoban(
            self._env_id,
            norm_obs=self._cfg.get('norm_obs', EasyDict(use_norm=False, )),
            norm_reward=self._cfg.get('norm_reward', EasyDict(use_norm=False, )),
            only_info=only_info
        )

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def enable_save_replay(self, replay_path) -> None:
        if replay_path is None:
            replay_path = './video'
        self._save_replay = True
        self._replay_path = replay_path

    def __repr__(self) -> str:
        return "DI-engine Sokoban Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_cfg = copy.deepcopy(cfg)
        collector_env_num = collector_cfg.pop('collector_env_num', 1)
        return [collector_cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_env_num = evaluator_cfg.pop('evaluator_env_num', 1)
        evaluator_cfg.norm_reward = EasyDict(use_norm=False, )
        return [evaluator_cfg for _ in range(evaluator_env_num)]

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space
