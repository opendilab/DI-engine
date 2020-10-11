import copy

import torch

from nervex.envs.common import EnvElementRunner
from nervex.envs.env.base_env import BaseEnv
from .pong_reward import PongReward


class PongRewardRunner(EnvElementRunner):
    def _init(self, cfg, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = PongReward(cfg)
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
