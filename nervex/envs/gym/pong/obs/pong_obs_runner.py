from typing import List, Tuple
import numpy as np
import copy
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.common import EnvElementRunner
from .pong_obs import PongObs


# done


class PongObsRunner(EnvElementRunner):
    def _init(self, cfg, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = PongObs(cfg)

    def get(self, engine: BaseEnv) -> int:
        ret = copy.deepcopy(engine._pong_obs)
        assert isinstance(ret, np.ndarray)
        return self._core._to_agent_processor(ret)

    def reset(self) -> None:
        pass
