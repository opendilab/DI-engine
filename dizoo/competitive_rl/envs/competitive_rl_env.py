from typing import Any, Union, List
import copy
import numpy as np
import gym
import competitive_rl

from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo, update_shape
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray, to_list
from .competitive_rl_env_wrapper import BuiltinOpponentWrapper, wrap_env
from ding.utils import ENV_REGISTRY

competitive_rl.register_competitive_envs()
"""
The observation spaces:
cPong-v0: Box(210, 160, 3)
cPongDouble-v0: Tuple(Box(210, 160, 3), Box(210, 160, 3))
cCarRacing-v0: Box(96, 96, 1)
cCarRacingDouble-v0: Box(96, 96, 1)

The action spaces:
cPong-v0: Discrete(3)
cPongDouble-v0: Tuple(Discrete(3), Discrete(3))
cCarRacing-v0: Box(2,)
cCarRacingDouble-v0: Dict(0:Box(2,), 1:Box(2,))

cPongTournament-v0
"""

COMPETITIVERL_INFO_DICT = {
    'cPongDouble-v0': BaseEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(
            shape=(210, 160, 3),
            # shape=(4, 84, 84),
            value={
                'min': 0,
                'max': 255,
                'dtype': np.float32
            },
        ),
        act_space=EnvElementInfo(
            shape=(1, ),  # different with https://github.com/cuhkrlcourse/competitive-rl#usage
            value={
                'min': 0,
                'max': 3,
                'dtype': np.float32
            },
        ),
        rew_space=EnvElementInfo(
            shape=(1, ),
            value={
                'min': np.float32("-inf"),
                'max': np.float32("inf"),
                'dtype': np.float32
            },
        ),
        use_wrappers=None,
    ),
}


@ENV_REGISTRY.register('competitive_rl')
class CompetitiveRlEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._env_id = self._cfg.env_id

        # opponent_type is used to control builtin opponent agent, which is useful in evaluator.
        is_evaluator = self._cfg.get("is_evaluator", False)
        opponent_type = None
        if is_evaluator:
            opponent_type = self._cfg.get("opponent_type", None)
        self._builtin_wrap = self._env_id == "cPongDouble-v0" and is_evaluator and opponent_type == "builtin"
        self._opponent = self._cfg.get('eval_opponent', 'RULE_BASED')

        self._init_flag = False

    def reset(self) -> np.ndarray:
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
        obs = self.process_obs(obs)  # process

        if self._builtin_wrap:
            self._final_eval_reward = np.array([0.])
        else:
            self._final_eval_reward = np.array([0., 0.])
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:
        action = to_ndarray(action)
        action = self.process_action(action)  # process

        obs, rew, done, info = self._env.step(action)

        if not isinstance(rew, tuple):
            rew = [rew]
        rew = np.array(rew)
        self._final_eval_reward += rew

        obs = to_ndarray(obs)
        obs = self.process_obs(obs)  # process

        if done:
            info['final_eval_reward'] = self._final_eval_reward

        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        if self._env_id in COMPETITIVERL_INFO_DICT:
            info = copy.deepcopy(COMPETITIVERL_INFO_DICT[self._env_id])
            info.use_wrappers = self._make_env(only_info=True)
            obs_shape, act_shape, rew_shape = update_shape(
                info.obs_space.shape, info.act_space.shape, info.rew_space.shape, info.use_wrappers.split('\n')
            )
            info.obs_space.shape = obs_shape
            info.act_space.shape = act_shape
            info.rew_space.shape = rew_shape
            if not self._builtin_wrap:
                info.obs_space.shape = (2, ) + info.obs_space.shape
                info.act_space.shape = (2, )
                info.rew_space.shape = (2, )
            return info
        else:
            raise NotImplementedError('{} not found in COMPETITIVERL_INFO_DICT [{}]'\
                .format(self._env_id, COMPETITIVERL_INFO_DICT.keys()))

    def _make_env(self, only_info=False):
        return wrap_env(self._env_id, self._builtin_wrap, self._opponent, only_info=only_info)

    def __repr__(self) -> str:
        return "DI-engine Competitve RL Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_cfg = copy.deepcopy(cfg)
        collector_env_num = collector_cfg.pop('collector_env_num', 1)
        collector_cfg.is_evaluator = False
        return [collector_cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_env_num = evaluator_cfg.pop('evaluator_env_num', 1)
        evaluator_cfg.is_evaluator = True
        return [evaluator_cfg for _ in range(evaluator_env_num)]

    def process_action(self, action: np.ndarray) -> Union[tuple, dict, np.ndarray]:
        # If in double agent env, transfrom action passed in from outside to tuple or dict type.
        if self._env_id == "cPongDouble-v0" and not self._builtin_wrap:
            return (action[0].squeeze(), action[1].squeeze())
        elif self._env_id == "cCarRacingDouble-v0":
            return {0: action[0].squeeze(), 1: action[1].squeeze()}
        else:
            return action.squeeze()

    def process_obs(self, obs: Union[tuple, np.ndarray]) -> Union[tuple, np.ndarray]:
        # Copy observation for car racing double agent env, in case to be in alignment with pong double agent env.
        if self._env_id == "cCarRacingDouble-v0":
            obs = np.stack([obs, copy.deepcopy(obs)])
        return obs
