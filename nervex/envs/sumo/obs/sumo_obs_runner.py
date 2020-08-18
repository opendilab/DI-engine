from typing import List, Tuple
import numpy as np
import copy
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.common import EnvElementRunner
from nervex.envs.sumo.obs.sumo_obs import SumoObs


class SumoObsRunner(EnvElementRunner):
    def _init(self, cfg: dict) -> None:
        # set self._core and other state variable
        self._core = SumoObs(cfg)

    def get(self, engine: BaseEnv) -> int:
        # obs = copy.deepcopy(engine.sumo_obs)
        return self._core._to_agent_processor()

    # override
    def reset(self) -> None:
        pass
