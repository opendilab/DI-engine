from typing import List, Optional
import gym
import copy
import numpy as np

from ding.envs.common.env_element import EnvElementInfo
from ding.envs.env_wrappers import create_env_wrapper, get_env_wrapper_cls
from ding.torch_utils import to_ndarray
from ding.utils import ENV_WRAPPER_REGISTRY, import_module
from .base_env import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from .default_wrapper import get_default_wrappers


class DingEnvWrapper(BaseEnv):

    def __init__(self, env: gym.Env = None, cfg: dict = None) -> None:
        # If env is None, assert cfg contains import_names + type, which are used to instantiate env instances.
        # If env is passed in, lazy_init is disabled (use init_flag to annotate)
        self._cfg = cfg
        if self._cfg is None:
            self._cfg = dict()
        if env is not None:
            self._init_flag = True
            self._env = env
            self.observation_space = self._env.observation_space
            self.action_space = self._env.action_space
            self.reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32)
        else:
            # TODO: need this assert?
            # assert 'type' in cfg and 'import_names' in cfg, 'Lazy init, but cfg does not contain necessary keys: {}'.format(
            #     cfg)
            self._init_flag = False
            self.observation_space = None
            self.action_space = None
            self.reward_space = None
        # Only if user specifies the replay_path, will the video be saved. So its inital value is None.
        self._replay_path = None

    # override
    def reset(self) -> None:
        if not self._init_flag:
            self._env = self._make_env()
            self._init_flag = True
            self.observation_space = self._env.observation_space
            self.action_space = self._env.action_space
            self.reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32)
            # self._update_space_shape()
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        if self._replay_path is not None:
            self._env = gym.wrappers.Monitor(
                self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
            )
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        self._action_type = self._cfg.get('action_type', 'scalar')
        return obs

    # override
    def close(self) -> None:
        self._env.close()

    # override
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    # override
    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        # TODO: hybird env's dict action
        # assert isinstance(action, np.ndarray), type(action)
        # if action.shape == (1, ) and self._action_type == 'scalar':
        #     action = action.squeeze()
        obs, rew, done, info = self._env.step(action)
        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)
        return BaseEnvTimestep(obs, rew, done, info)

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        # if not isinstance(random_action, np.ndarray):
        #     random_action = to_ndarray([self.action_space.sample()]).astype(np.float32)
        return random_action

    # def info(self) -> BaseEnvInfo:
    #     obs_space = self._env.observation_space
    #     act_space = self._env.action_space
    #     return BaseEnvInfo(
    #         agent_num=1,
    #         obs_space=EnvElementInfo(
    #             shape=obs_space.shape,
    #             value={
    #                 'min': obs_space.low,
    #                 'max': obs_space.high,
    #                 'dtype': np.float32
    #             },
    #         ),
    #         act_space=EnvElementInfo(
    #             shape=(act_space.n, ),
    #             value={
    #                 'min': 0,
    #                 'max': act_space.n,
    #                 'dtype': np.float32
    #             },
    #         ),
    #         rew_space=EnvElementInfo(
    #             shape=1,
    #             value={
    #                 'min': -1,
    #                 'max': 1,
    #                 'dtype': np.float32
    #             },
    #         ),
    #         use_wrappers=None
    #     )

    def _make_env(self) -> gym.Env:
        env = gym.make(self._cfg.env_id)
        wrapper_cfgs = self._cfg.get('env_wrapper', [])
        if isinstance(wrapper_cfgs, str):
            wrapper_cfgs = get_default_wrappers(self._cfg.env_id, wrapper_cfgs)
        self._wrapper_cfgs = wrapper_cfgs
        for wrapper_cfg in self._wrapper_cfgs:
            if wrapper_cfg.get('disable', False):
                continue
            env = create_env_wrapper(env, wrapper_cfg)
        return env

    def _update_space_shape(self) -> None:
        spaces = (self.observation_space.shape, self.action_space.shape, self.reward_space.shape)
        # wrapper_cfgs = self._cfg.get('env_wrapper', [])
        # if isinstance(wrapper_cfgs, str):
        #     wrapper_cfgs = default_wrappers[wrapper_cfgs]
        for wrapper_cfg in self._wrapper_cfgs:
            if wrapper_cfg.get('disable', False):
                continue
            wrapper_cls = get_env_wrapper_cls(wrapper_cfg)
            spaces = wrapper_cls.new_shape(*spaces, **wrapper_cfg.get('kwargs', {}))
        self.observation_space._shape, self.action_space._shape, self.reward_space._shape = spaces

    def __repr__(self) -> str:
        return "DI-engine Env({})".format(self._cfg.env_id)

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
