from typing import List, Tuple
import numpy as np
import copy
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.common import EnvElementRunner
from nervex.envs.sumo.obs.sumo_obs import SumoObs

# done


class SumoObsRunner(EnvElementRunner):

    def _init(self, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = SumoObs()

    def get(self, engine: BaseEnv) -> int:
        # obs = copy.deepcopy(engine.sumo_obs)
        return engine.sumo_obs

    #overriede
    def reset(self) -> None:
        pass

    # def info(self) -> dict:
    #     self._core.info_template
