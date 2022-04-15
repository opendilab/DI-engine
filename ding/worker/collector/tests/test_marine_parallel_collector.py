from typing import Any, Union, List
import copy
import torch
import numpy as np
import pytest
import os
import gym

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ding.entry import parallel_pipeline
from .fake_cpong_dqn_config import fake_cpong_dqn_config, fake_cpong_dqn_create_config, fake_cpong_dqn_system_config


@ENV_REGISTRY.register('fake_competitive_rl')
class FakeCompetitiveRlEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._is_evaluator = cfg.is_evaluator
        self.num_agents = 2
        self.observation_space = gym.spaces.Box(low=0, high=256, shape=(2, 4, 84, 84), dtype=np.int64)
        self.action_space = gym.spaces.Box(low=0, high=3, shape=(1, ), dtype=np.float32)
        self.reward_space = gym.spaces.Box(
            low=np.float32("-inf"), high=np.float32("inf"), shape=(1, ), dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        self._step_times = 0
        obs_shape = (4, 84, 84)
        if not self._is_evaluator:
            obs_shape = (2, ) + obs_shape
        obs = np.random.randint(0, 256, obs_shape).astype(np.float32)
        return obs

    def close(self) -> None:
        pass

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        pass

    def step(self, action: Union[torch.Tensor, np.ndarray, list]) -> BaseEnvTimestep:
        obs_shape = (4, 84, 84)
        if not self._is_evaluator:
            obs_shape = (2, ) + obs_shape
        obs = np.random.randint(0, 256, obs_shape).astype(np.float32)
        rew = np.array([1.]) if self._is_evaluator else np.array([1., -1.])
        done = False if self._step_times < 20 else True
        info = {}
        if done:
            info['final_eval_reward'] = np.array([21.]) if self._is_evaluator else np.array([5., -5.])
        self._step_times += 1
        return BaseEnvTimestep(obs, rew, done, info)

    def __repr__(self) -> str:
        return "Fake Competitve RL Env({})".format(self._cfg.env_id)

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


@pytest.mark.unittest
def test_1v1_collector():
    parallel_pipeline([fake_cpong_dqn_config, fake_cpong_dqn_create_config, fake_cpong_dqn_system_config], 0)
    os.popen("rm -rf data log policy ckpt* total_config.py")
