from typing import List, Tuple
import copy
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.common import EnvElementRunner
from nervex.envs.gym.pendulum.reward.pendulum_reward import PendulumReward


class PendulumRewardRunner(EnvElementRunner):
    def _init(self, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = PendulumReward()

    def get(self, engine: BaseEnv) -> float:
        reward_of_action = copy.deepcopy(engine.reward)
        ret = reward_of_action
        return ret

    def reset(self) -> None:
        pass
