from typing import List, Tuple
import numpy as np
import copy
from sc2learner.envs.env.base_env import BaseEnv
from sc2learner.envs.common import EnvElementRunner
from .pong_obs import PongObs

# done
class PongObsRunner(EnvElementRunner):

    def _init(self, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = PongObs()

    def get(self, engine: BaseEnv) -> int:
        ram_obs = copy.deepcopy(engine.pong_obs)
        assert isinstance(ram_obs, np.ndarray)
        ret = ram_obs
        return ret

    #overriede
    def reset(self) -> None:
        pass