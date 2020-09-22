import copy
import os
from collections import namedtuple
import sys
from typing import List, Any, Union
from torchvision import transforms

from nervex.envs.env.base_env import BaseEnv
from nervex.envs.gym.pong.action.pong_action_runner import PongRawActionRunner
from nervex.envs.gym.pong.reward.pong_reward_runner import PongRewardRunner
from nervex.envs.gym.pong.obs.pong_obs_runner import PongObsRunner
import torch
import numpy as np
import gym


def transform(height, width):
    return transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Grayscale(),
         transforms.Resize((height, width)),
         transforms.ToTensor()]
    )


class PongEnv(BaseEnv):
    r"""
    Notes:
        To see information about atari_env: https://blog.csdn.net/qq_27008079/article/details/100126060
    """
    timestep = namedtuple('PongTimestep', ['obs', 'reward', 'done', 'rest_lives', 'info'])

    info_template = namedtuple('PongEnvInfo', ['obs_space', 'act_space', 'rew_space', 'frame_skip', 'rep_prob'])

    # frame_skip: how many frame in one step, should be 1 or 2 or 4.
    # rep_prob: the probability of rerun the previous action in this step, should be 0 or 0.25.

    def __init__(self, cfg):
        self._cfg = cfg
        self.frameskip = 1
        self.rep_prob = 0
        self._action_helper = PongRawActionRunner()
        self._reward_helper = PongRewardRunner()
        self._obs_helper = PongObsRunner()
        self._wrap_frame = cfg.get('wrap_frame', False)
        self._wrap_frame_height = cfg.get('wrap_frame_height', 84)
        self._wrap_frame_width = cfg.get('wrap_frame_width', 84)

        if cfg != {}:
            self._game = cfg.get('game', None)
            self._mode = cfg.get('mode', None)
            self._difficulty = cfg.get('difficulty', None)
            self._obs_type = cfg.get('obs_type', None)
            self.frameskip = cfg.get('frameskip', 1)
            self.rep_prob = cfg.get('rep_prob', 0)

        self._isGameover = False
        self._launch_env_flag = False
        if self.rep_prob == 0.25:
            self.rep_name = '-v0'
        elif self.rep_prob == 0:
            self.rep_name = '-v4'
        else:
            raise NotImplementedError
        if self.frameskip == 4:
            self.frame_name = 'Deterministic'
        elif self.frameskip == 2:
            self.frame_name = ''
        elif self.frameskip == 1:
            self.frame_name = 'NoFrameskip'
        else:
            raise NotImplementedError
        self._env = gym.make("Pong" + self.frame_name + self.rep_name).unwrapped
        self._launch_env_flag = True

    def _launch_env(self):
        self._env = gym.make("Pong" + self.frame_name + self.rep_name).unwrapped
        self._launch_env_flag = True

    def reset(self):
        if not self._launch_env_flag:
            self._launch_env()
        ret = self._env.reset().transpose((2, 0, 1))
        self._reward_helper.reset()
        self._obs_helper.reset()
        self._action_helper.reset()
        ret = torch.from_numpy(ret).float()
        if self._wrap_frame:
            ret = transform(self._wrap_frame_height, self._wrap_frame_width)(ret)
        return ret

    def close(self):
        self._env.close()

    def step(self, action: torch.tensor) -> 'PongEnv.timestep':
        assert self._launch_env_flag
        self.agent_action = action
        action = action.item()

        # env step
        self._pong_obs, self._reward_of_action, self._is_gameover, self._rest_life = self._env.step(action)
        self._pong_obs = self._pong_obs.transpose((2, 0, 1))

        # transform obs, reward and record statistics

        self.action = self._action_helper.get(self)
        self.reward = self._reward_helper.get(self)
        self.obs = self._obs_helper.get(self)
        if self._wrap_frame:
            self.obs = transform(self._wrap_frame_height, self._wrap_frame_width)(self.obs)

        info = {'cum_reward': self._reward_helper.cum_reward}

        return PongEnv.timestep(
            obs=self.obs, reward=self.reward, done=self._is_gameover, rest_lives=self._rest_life, info=info
        )

    def seed(self, seed: int) -> None:
        self._env.seed(seed)

    def info(self) -> 'PongEnv.info':
        info_data = {
            'obs_space': self._obs_helper.info,
            'act_space': self._action_helper.info,
            'rew_space': self._reward_helper.info,
            'frame_skip': self.frameskip,
            'rep_prob': self.rep_prob
        }
        return PongEnv.info_template(**info_data)

    def __repr__(self) -> str:
        return 'PongEnv:\n\
                \tobservation[{}]\n\
                \taction[{}]\n\
                \treward[{}]\n'.format(repr(self._obs_helper), repr(self._action_helper), repr(self._reward_helper))

    def pack(self, timesteps: List['PongEnv.timestep'] = None, obs: Any = None) -> 'PongEnv.timestep':
        assert not (timesteps is None and obs is None)
        assert not (timesteps is not None and obs is not None)
        if timesteps is not None:
            assert isinstance(timesteps, list)
            assert isinstance(timesteps[0], tuple)
            timestep_type = type(timesteps[0])
            items = [[getattr(timesteps[i], item) for i in range(len(timesteps))] for item in timesteps[0]._fields]
            return timestep_type(*items)
        if obs is not None:
            return obs

    def unpack(self, action: Any) -> List[Any]:
        return [{'action': act} for act in action]

    @property
    def cum_reward(self) -> torch.tensor:
        return self._reward_helper.cum_reward


PongTimestep = PongEnv.timestep
