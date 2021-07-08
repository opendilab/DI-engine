from typing import List
import copy
import numpy as np

from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.utils import ENV_REGISTRY
from .atari_env import AtariEnv, ATARIENV_INFO_DICT


@ENV_REGISTRY.register('atari_multi_discrete')
class AtariMultiDiscreteEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._multi_env_num = cfg['multi_env_num']
        self._env = [AtariEnv(cfg) for _ in range(self._multi_env_num)]
        self._env_done = {i: False for i in range(self._multi_env_num)}
        self._done_obs = {i: None for i in range(self._multi_env_num)}
        self._final_eval_reward = 0.
        self._cfg = cfg

    def reset(self) -> np.ndarray:
        obs = []
        for e in self._env:
            obs.append(e.reset())
        self._env_done = {i: False for i in range(self._multi_env_num)}
        self._done_obs = {i: None for i in range(self._multi_env_num)}
        self._final_eval_reward = 0.
        return np.concatenate(obs, axis=0)

    def close(self) -> None:
        for e in self._env:
            e.close()

    def seed(self, seed: int) -> None:
        for i, e in enumerate(self._env):
            e.seed(seed + i)

    def step(self, action: list) -> BaseEnvTimestep:
        timestep = []
        for i, (a, e) in enumerate(zip(action, self._env)):
            if not self._env_done[i]:
                timestep.append(e.step(a))
        reward = sum([t.reward for t in timestep])
        done = all([t.done for t in timestep])

        obs = []
        j = 0
        for i in range(self._multi_env_num):
            if self._env_done[i]:
                obs.append(self._done_obs[i])
            else:
                if timestep[j].done:
                    # print('done', i, timestep[j].info['final_eval_reward'])
                    self._final_eval_reward += timestep[j].info['final_eval_reward']
                    self._env_done[i] = True
                    self._done_obs[i] = copy.deepcopy(timestep[j].obs)
                obs.append(timestep[j].obs)
                j += 1
        obs = np.concatenate(obs, axis=0)
        info = {}
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, reward, done, info)

    def info(self) -> BaseEnvInfo:
        info = self._env[0].info()
        T = EnvElementInfo
        if self._cfg.env_id in ATARIENV_INFO_DICT:
            obs_shape = list(ATARIENV_INFO_DICT[self._cfg.env_id].obs_space.shape)
            n = ATARIENV_INFO_DICT[self._cfg.env_id].act_space.shape[0]
        else:
            raise NotImplementedError('{} not found in ATARIENV_INFO_DICT [{}]'\
                .format(self._cfg.env_id, ATARIENV_INFO_DICT.keys()))
        obs_shape[0] = obs_shape[0] * self._multi_env_num
        obs_space = T(obs_shape, {'dtype': np.float32}, None, None)
        act_shape = tuple([n for _ in range(self._multi_env_num)])
        act_space = T(act_shape, {'dtype': np.float32}, None, None)
        rew_space = T(1, {'min': -self._multi_env_num, 'max': self._multi_env_num, 'dtype': np.float32}, None, None)
        return BaseEnvInfo(
            agent_num=self._multi_env_num,
            obs_space=obs_space,
            act_space=act_space,
            rew_space=rew_space,
        )

    def __repr__(self) -> str:
        return "DI-engine Atari Multi Discrete Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num', 1)
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num', 1)
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]
