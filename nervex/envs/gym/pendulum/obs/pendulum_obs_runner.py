import copy

from nervex.envs.common import EnvElementRunner
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.gym.pendulum.obs.pendulum_obs import PendulumObs


# done


class PendulumObsRunner(EnvElementRunner):
    def _init(self, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = PendulumObs()

    def get(self, engine: BaseEnv) -> int:
        ret = copy.deepcopy(engine.obs)
        return self._core._to_agent_processor(ret)

    def reset(self) -> None:
        pass
