from typing import Any, Union, List
import copy
import torch
import numpy as np

from nervex.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.envs.common.common_function import affine_transform
from nervex.torch_utils import to_tensor, to_ndarray, to_list
from .mujoco_wrappers import wrap_mujoco
from nervex.utils import ENV_REGISTRY

MUJOCO_INFO_DICT = {
    'Ant-v3': BaseEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(
            shape=(111, ),
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf"),
                'dtype': np.float32
            },
        ),
        act_space=EnvElementInfo(
            shape=(8, ),
            value={
                'min': -1.0,
                'max': 1.0,
                'dtype': np.float32
            },
        ),
        rew_space=EnvElementInfo(
            shape=1,
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf")
            },
        ),
        use_wrappers=None,
    ),
    'Hopper-v2': BaseEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(
            shape=(11, ),
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf"),
                'dtype': np.float32
            },
        ),
        act_space=EnvElementInfo(
            shape=(3, ),
            value={
                'min': -1.0,
                'max': 1.0,
                'dtype': np.float32
            },
        ),
        rew_space=EnvElementInfo(
            shape=1,
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf")
            },
        ),
        use_wrappers=None,
    ),
    'Walker2d-v2': BaseEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(
            shape=(17, ),
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf"),
                'dtype': np.float32
            },
        ),
        act_space=EnvElementInfo(
            shape=(6, ),
            value={
                'min': -1.0,
                'max': 1.0,
                'dtype': np.float32
            },
        ),
        rew_space=EnvElementInfo(
            shape=1,
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf")
            },
        ),
        use_wrappers=None,
    ),
    'HalfCheetah-v3': BaseEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(
            shape=(17, ),
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf"),
                'dtype': np.float32
            },
        ),
        act_space=EnvElementInfo(
            shape=(6, ),
            value={
                'min': -1.0,
                'max': 1.0,
                'dtype': np.float32
            },
        ),
        rew_space=EnvElementInfo(
            shape=1,
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf")
            },
        ),
        use_wrappers=None,
    ),
    'Hopper-v3': BaseEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(
            shape=(11, ),
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf"),
                'dtype': np.float32
            },
        ),
        act_space=EnvElementInfo(
            shape=(3, ),
            value={
                'min': -1.0,
                'max': 1.0,
                'dtype': np.float32
            },
        ),
        rew_space=EnvElementInfo(
            shape=1,
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf")
            },
        ),
        use_wrappers=None,
    ),
    'InvertedPendulum-v2': BaseEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(
            shape=(4, ),
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf"),
                'dtype': np.float32
            },
        ),
        act_space=EnvElementInfo(
            shape=(1, ),
            value={
                'min': -1.0,
                'max': 1.0,
                'dtype': np.float32
            },
        ),
        rew_space=EnvElementInfo(
            shape=1,
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf")
            },
        ),
        use_wrappers=None,
    ),
    'InvertedDoublePendulum-v2': BaseEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(
            shape=(11, ),
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf"),
                'dtype': np.float32
            },
        ),
        act_space=EnvElementInfo(
            shape=(1, ),
            value={
                'min': -1.0,
                'max': 1.0,
                'dtype': np.float32
            },
        ),
        rew_space=EnvElementInfo(
            shape=1,
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf")
            },
        ),
        use_wrappers=None,
    ),
    'Reacher-v2': BaseEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(
            shape=(11, ),
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf"),
                'dtype': np.float32
            },
        ),
        act_space=EnvElementInfo(
            shape=(2, ),
            value={
                'min': -1.0,
                'max': 1.0,
                'dtype': np.float32
            },
        ),
        rew_space=EnvElementInfo(
            shape=1,
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf")
            },
        ),
        use_wrappers=None,
    ),
    'Walker2d-v3': BaseEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(
            shape=(17, ),
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf"),
                'dtype': np.float32
            },
        ),
        act_space=EnvElementInfo(
            shape=(6, ),
            value={
                'min': -1.0,
                'max': 1.0,
                'dtype': np.float32
            },
        ),
        rew_space=EnvElementInfo(
            shape=1,
            value={
                'min': np.float64("-inf"),
                'max': np.float64("inf")
            },
        ),
        use_wrappers=None,
    ),
}


@ENV_REGISTRY.register('mujoco')
class MujocoEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._use_act_scale = cfg.use_act_scale
        self._init_flag = False

    def reset(self) -> torch.FloatTensor:
        if not self._init_flag:
            self._env = self._make_env(only_info=False)
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        obs = self._env.reset()
        obs = to_ndarray(obs)
        self._final_eval_reward = 0.
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Union[torch.Tensor, np.ndarray, list]) -> BaseEnvTimestep:
        action = to_ndarray(action)
        if self._use_act_scale:
            action_range = self.info().act_space.value
            action = affine_transform(action, min_val=action_range['min'], max_val=action_range['max'])
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        obs = to_ndarray(obs)
        rew = to_ndarray([rew])  # wrapped to be transfered to a Tensor with shape (1,)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        if self._cfg.env_id in MUJOCO_INFO_DICT:
            info = copy.deepcopy(MUJOCO_INFO_DICT[self._cfg.env_id])
            info.use_wrappers = self._make_env(only_info=True)
            return info
        else:
            raise NotImplementedError('{} not found in MUJOCO_INFO_DICT [{}]'\
                .format(self._cfg.env_id, MUJOCO_INFO_DICT.keys()))

    def _make_env(self, only_info=False):
        return wrap_mujoco(
            self._cfg.env_id,
            norm_obs=self._cfg.get('norm_obs', None),
            norm_reward=self._cfg.get('norm_reward', None),
            only_info=only_info
        )

    def __repr__(self) -> str:
        return "nerveX Mujoco Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_cfg = copy.deepcopy(cfg)
        collector_env_num = collector_cfg.pop('collector_env_num', 1)
        return [collector_cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_env_num = evaluator_cfg.pop('evaluator_env_num', 1)
        evaluator_cfg.norm_reward.use_norm = False
        return [evaluator_cfg for _ in range(evaluator_env_num)]
