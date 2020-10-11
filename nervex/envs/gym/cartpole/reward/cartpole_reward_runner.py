import copy

import torch

from nervex.envs.common import EnvElementRunner
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.gym.cartpole.reward.cartpole_reward import CartpoleReward


class CartpoleRewardRunner(EnvElementRunner):
    def _init(self, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = CartpoleReward()

    def get(self, engine: BaseEnv) -> torch.FloatTensor:
        ret = copy.deepcopy(engine.reward)
        self._cum_reward += ret
        return self._core._to_agent_processor(ret)

    def reset(self) -> None:
        self._cum_reward = 0.0

    @property
    def cum_reward(self) -> torch.tensor:
        return torch.FloatTensor([self._cum_reward])
