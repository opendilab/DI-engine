import gym
import torch
import numpy as np
from gym import spaces
from typing import Any
from nervex.envs import BaseEnv
from nervex.envs.common.env_element import EnvElement
from nervex.torch_utils import to_tensor, tensor_to_list
from app_zoo.multiagent_particle.envs.make_env import make_env
from app_zoo.multiagent_particle.envs.multiagent.multi_discrete import MultiDiscrete


class ParicleEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._env_name = cfg.get("env_name", "simple")
        self._env = make_env(self._env_name)
        # TODO(zwh) discrete action definition optimization, not all the input action is discrete
        self._env.discrete_action_input = True
        self.agent_num = self._env.n

    def reset(self) -> torch.Tensor:
        if hasattr(self, '_seed'):
            # Note: the real env instance only has a empty seed method, only pass
            self._env.seed(self._seed)
        obs_n = self._env.reset()
        obs_n = to_tensor(obs_n, torch.float)
        return obs_n

    def close(self) -> None:
        # Note: the real env instance only has a empty close method, only pass
        self._env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed

    def _process_action(self, action: list):
        return tensor_to_list(action)

    def step(self, action: list) -> BaseEnv.timestep:
        action = self._process_action(action)
        obs_n, rew_n, done_n, info_n = self._env.step(action)
        obs_n = [to_tensor(obs, torch.float) for obs in obs_n]
        rew_n = [to_tensor(rew, torch.float) for rew in rew_n]
        return BaseEnv.timestep(obs_n, rew_n, done_n, info_n)

    def info(self) -> BaseEnv.info_template:
        T = EnvElement.info_template
        act_space = {}
        obs_space = {}
        rew_space = {}
        for i in range(self._env.n):
            obs_space['agent' + str(i)] = T(
                self._env.observation_space[i].shape, {
                    'min': -np.inf,
                    'max': +np.inf,
                    'dtype': float
                }, None, None
            )
            rew_space['agent' + str(i)] = T((1, ), {'min': -np.inf, 'max': +np.inf, 'dtype': float}, None, None)
            # print("action_space is ", self._env.action_space)
            act = self._env.action_space[i]
            if isinstance(act, MultiDiscrete):
                act_space['agent' + str(i)] = T(
                    (act.shape, ), {
                        'min': [int(l) for l in list(act.low)],
                        'max': [int(h) for h in list(act.high)]
                    }, None, None
                )
            elif isinstance(act, spaces.Tuple):
                #are not used in our environment yet
                act_space['agent' + str(i)] = T((len(act.spaces), ), {'space': act.spaces}, None, None)
            elif isinstance(act, spaces.Discrete):
                act_space['agent' + str(i)] = T((1, ), {'min': 0, 'max': act.n - 1, 'dtype': int}, None, None)
            elif isinstance(act, spaces.Box):
                act_space['agent' +
                          str(i)] = T(act.shape, {
                              'min': act.low,
                              'max': act.high,
                              'dtype': act.dtype
                          }, None, None)
        return BaseEnv.info_template(
            agent_num=self.agent_num,
            obs_space=obs_space,
            act_space=act_space,
            rew_space=rew_space,
        )

    def __repr__(self) -> str:
        return "nervex wrapped Multiagent particle Env({})".format(self._cfg.env_id)
