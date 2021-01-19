from typing import Any, List
import gym
import torch
import numpy as np

from collections import OrderedDict,namedtuple
from typing import Any, Dict, Optional, Union,List

from nervex.envs import BaseEnv, register_env, BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.torch_utils import to_tensor, to_ndarray, to_list


goaltimestep = namedtuple('GoalEnvTimestep', ['obs', 'goal', 'reward', 'done', 'info'])


class BitFlipEnv(BaseEnv):


    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._n_bits = cfg.n_bits
        self._state = torch.zeros(size = (self._n_bits,),dtype=torch.float32,requires_grad=False)
        self._goal = torch.ones(size = (self._n_bits,),dtype=torch.float32,requires_grad=False)
        self._curr_step = 0
        self._maxsize = self._n_bits
        self._final_eval_reward = 0

    def reset(self) -> torch.Tensor:
        self._curr_step = 0
        self._final_eval_reward = 0
        self._state = torch.randint(high=2, size =(self._n_bits,),dtype=torch.float32,requires_grad=False)
        self._goal = torch.randint(high=2, size =(self._n_bits,),dtype=torch.float32,requires_grad=False)

        while torch.all(self._state == self._goal):
            self._goal = torch.randint(high=2, size =(self._n_bits,),dtype=torch.float32,requires_grad=False)

        obs = torch.cat([self._state,self._goal],dim = 0)

        return obs

    def close(self) -> None:
        pass

    def check_success(self, state, goal):
        return torch.all(self._state == self._goal).item()

    def seed(self, seed: int) -> None:
        self._seed = seed
        torch.manual_seed(self._seed)

    def step(self, action: torch.Tensor):
        self._curr_step += 1
        if action.shape == (1, ):
            action = action.squeeze()  # 0-dim tensor
        # action = action.numpy()

        self._state[action] = 1 - self._state[action]
        if self.check_success(self._state, self._goal):
            rew = torch.FloatTensor([1])
            done = True
        else:
            rew = torch.FloatTensor([0])
            done = False
        self._final_eval_reward += rew
        if self._curr_step >= self._maxsize:
            done = True
        info = {}
        if done:
            info['final_eval_reward'] = self._final_eval_reward.item()

        obs = torch.cat([self._state,self._goal],dim = 0)

        return goaltimestep(obs,self._goal, rew, done, info)

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T(
                (4, ), {
                    'min': [-4.8, float("-inf"), -0.42, float("-inf")],
                    'max': [4.8, float("inf"), 0.42, float("inf")],
                    'dtype': float,
                }, None, None
            ),
            # [min, max)
            act_space=T((self._n_bits, ), {
                'min': 0,
                'max': self._n_bits
            }, None, None),
            rew_space=T((1, ), {
                'min': 0.0,
                'max': 1.0
            }, None, None),
        )

    def __repr__(self) -> str:
        return "nerveX BitFlip Env({})".format('bitflip')


register_env('bitflip', BitFlipEnv)
