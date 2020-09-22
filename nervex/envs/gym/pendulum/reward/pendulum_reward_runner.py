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
        ret = copy.deepcopy(engine.reward)
        return self._core._to_agent_processor(ret)

    def reset(self) -> None:
        pass
