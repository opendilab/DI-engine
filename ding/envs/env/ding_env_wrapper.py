from typing import List, Optional, Union, Dict
from easydict import EasyDict
import gym
import gymnasium
import copy
import numpy as np
import treetensor.numpy as tnp

from ding.envs.common.common_function import affine_transform
from ding.envs.env_wrappers import create_env_wrapper
from ding.torch_utils import to_ndarray
from ding.utils import CloudPickleWrapper
from .base_env import BaseEnv, BaseEnvTimestep
from .default_wrapper import get_default_wrappers


class DingEnvWrapper(BaseEnv):

    def __init__(self, env: gym.Env = None, cfg: dict = None, seed_api: bool = True, caller: str = 'collector') -> None:
        """
        You can pass in either an env instance, or a config to create an env instance:
            - An env instance: Parameter `env` must not be `None`, but should be the instance.
                               Do not support subprocess env manager; Thus usually used in simple env.
            - A config to create an env instance: Parameter `cfg` dict must contain `env_id`.
        """
        self._env = None
        self._raw_env = env
        self._cfg = cfg
        self._seed_api = seed_api  # some env may disable `env.seed` api
        self._caller = caller
        if self._cfg is None:
            self._cfg = dict()
        self._cfg = EasyDict(self._cfg)
        if 'act_scale' not in self._cfg:
            self._cfg.act_scale = False
        if 'rew_clip' not in self._cfg:
            self._cfg.rew_clip = False
        if 'env_wrapper' not in self._cfg:
            self._cfg.env_wrapper = 'default'
        if 'env_id' not in self._cfg:
            self._cfg.env_id = None
        if env is not None:
            self._env = env
            self._wrap_env(caller)
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._action_space.seed(0)  # default seed
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
            )
            self._init_flag = True
        else:
            assert 'env_id' in self._cfg
            self._init_flag = False
            self._observation_space = None
            self._action_space = None
            self._reward_space = None
        # Only if user specifies the replay_path, will the video be saved. So its inital value is None.
        self._replay_path = None

    # override
    def reset(self) -> None:
        if not self._init_flag:
            self._env = gym.make(self._cfg.env_id)
            self._wrap_env(self._caller)
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
            )
            self._init_flag = True
        if self._replay_path is not None:
            self._env = gym.wrappers.RecordVideo(
                self._env,
                video_folder=self._replay_path,
                episode_trigger=lambda episode_id: True,
                name_prefix='rl-video-{}'.format(id(self))
            )
            self._replay_path = None
        if isinstance(self._env, gym.Env):
            if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
                np_seed = 100 * np.random.randint(1, 1000)
                if self._seed_api:
                    self._env.seed(self._seed + np_seed)
                self._action_space.seed(self._seed + np_seed)
            elif hasattr(self, '_seed'):
                if self._seed_api:
                    self._env.seed(self._seed)
                self._action_space.seed(self._seed)
            obs = self._env.reset()
        elif isinstance(self._env, gymnasium.Env):
            if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
                np_seed = 100 * np.random.randint(1, 1000)
                self._action_space.seed(self._seed + np_seed)
                obs = self._env.reset(seed=self._seed + np_seed)
            elif hasattr(self, '_seed'):
                self._action_space.seed(self._seed)
                obs = self._env.reset(seed=self._seed)
            else:
                obs = self._env.reset()
        else:
            raise RuntimeError("not support env type: {}".format(type(self._env)))
        if self.observation_space.dtype == np.float32:
            obs = to_ndarray(obs, dtype=np.float32)
        else:
            obs = to_ndarray(obs)
        return obs

    # override
    def close(self) -> None:
        try:
            self._env.close()
            del self._env
        except:  # noqa
            pass

    # override
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    # override
    def step(self, action: Union[np.int64, np.ndarray]) -> BaseEnvTimestep:
        action = self._judge_action_type(action)
        if self._cfg.act_scale:
            action = affine_transform(action, min_val=self._env.action_space.low, max_val=self._env.action_space.high)
        obs, rew, done, info = self._env.step(action)
        if self._cfg.rew_clip:
            rew = max(-10, rew)
        rew = np.float32(rew)
        if self.observation_space.dtype == np.float32:
            obs = to_ndarray(obs, dtype=np.float32)
        else:
            obs = to_ndarray(obs)
        rew = to_ndarray([rew], np.float32)
        return BaseEnvTimestep(obs, rew, done, info)

    def _judge_action_type(self, action: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
        if isinstance(action, int):
            return action
        elif isinstance(action, np.int64):
            return int(action)
        elif isinstance(action, np.ndarray):
            if action.shape == ():
                action = action.item()
            elif action.shape == (1, ) and action.dtype == np.int64:
                action = action.item()
            return action
        elif isinstance(action, dict):
            for k, v in action.items():
                action[k] = self._judge_action_type(v)
            return action
        elif isinstance(action, tnp.ndarray):
            return self._judge_action_type(action.json())
        else:
            raise TypeError(
                '`action` should be either int/np.ndarray or dict of int/np.ndarray, but get {}: {}'.format(
                    type(action), action
                )
            )

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        if isinstance(random_action, np.ndarray):
            pass
        elif isinstance(random_action, int):
            random_action = to_ndarray([random_action], dtype=np.int64)
        elif isinstance(random_action, dict):
            random_action = to_ndarray(random_action)
        else:
            raise TypeError(
                '`random_action` should be either int/np.ndarray or dict of int/np.ndarray, but get {}: {}'.format(
                    type(random_action), random_action
                )
            )
        return random_action

    def _wrap_env(self, caller: str = 'collector') -> None:
        # wrapper_cfgs: Union[str, List]
        wrapper_cfgs = self._cfg.env_wrapper
        if isinstance(wrapper_cfgs, str):
            wrapper_cfgs = get_default_wrappers(wrapper_cfgs, self._cfg.env_id, caller)
        # self._wrapper_cfgs: List[Union[Callable, Dict]]
        self._wrapper_cfgs = wrapper_cfgs
        for wrapper in self._wrapper_cfgs:
            # wrapper: Union[Callable, Dict]
            if isinstance(wrapper, Dict):
                self._env = create_env_wrapper(self._env, wrapper)
            else:  # Callable, such as lambda anonymous function
                self._env = wrapper(self._env)

    def __repr__(self) -> str:
        return "DI-engine Env({}), generated by DingEnvWrapper".format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        actor_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(actor_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    @property
    def observation_space(self) -> gym.spaces.Space:
        if self._observation_space.dtype == np.float64:
            self._observation_space.dtype = np.float32
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def clone(self, caller: str = 'collector') -> BaseEnv:
        try:
            spec = copy.deepcopy(self._raw_env.spec)
            raw_env = CloudPickleWrapper(self._raw_env)
            raw_env = copy.deepcopy(raw_env).data
            raw_env.__setattr__('spec', spec)
        except Exception:
            raw_env = self._raw_env
        return DingEnvWrapper(raw_env, self._cfg, self._seed_api, caller)
