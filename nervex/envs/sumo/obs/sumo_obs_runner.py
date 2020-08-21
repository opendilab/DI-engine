from typing import List, Tuple
import numpy as np
import copy
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.common import EnvElementRunner
from nervex.envs.sumo.obs.sumo_obs import SumoObs


class SumoObsRunner(EnvElementRunner):
    r"""
    Overview:
        runner that help to get the observation space
    Interface:
        _init, get, reset
    """
    def _init(self, cfg: dict) -> None:
        r"""
        Overview:
            init the sumo observation helper with the given config file
        Arguments:
            - cfg(:obj:`EasyDict`): config, you can refer to `env/sumo/sumo_env_default_config.yaml`
        """
        # set self._core and other state variable
        self._core = SumoObs(cfg)

    def get(self, engine: BaseEnv):
        r"""
        Overview:
            return the formated observation
        Returns:
            - obs(:obj:`torch.Size([380])` or :obj:`dict`{junction : obs}): the returned observation,
            :obj:`torch.Size([380]) if used centerlized_obs, else :obj:`dict`
        """
        # obs = copy.deepcopy(engine.sumo_obs)
        return self._core._to_agent_processor()

    # override
    def reset(self) -> None:
        r"""
        Overview:
            reset obs runner, and return the initial obs
        """
        return self._core._to_agent_processor()
