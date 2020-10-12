import copy

from nervex.envs.common import EnvElementRunner
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.gym.cartpole.obs.cartpole_obs import CartpoleObs

# done


class CartpoleObsRunner(EnvElementRunner):

    def _init(self, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = CartpoleObs()

    def get(self, engine: BaseEnv) -> int:
        ret = copy.deepcopy(engine.obs)
        return self._core._to_agent_processor(ret)

    def reset(self) -> None:
        pass
