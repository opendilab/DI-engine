import copy

import torch

from ding.envs.common import EnvElementRunner
from ding.envs.env.base_env import BaseEnv
from .gfootball_reward import GfootballReward


class GfootballRewardRunner(EnvElementRunner):

    def _init(self, cfg, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = GfootballReward(cfg)
        self._cum_reward = 0.0

    def get(self, engine: BaseEnv) -> torch.tensor:
        ret = copy.deepcopy(engine._reward_of_action)
        self._cum_reward += ret
        return self._core._to_agent_processor(ret)

    def reset(self) -> None:
        self._cum_reward = 0.0

    @property
    def cum_reward(self) -> torch.tensor:
        return torch.FloatTensor([self._cum_reward])
