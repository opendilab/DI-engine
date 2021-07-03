import copy

import numpy as np

from ding.envs.common import EnvElementRunner
from ding.envs.env.base_env import BaseEnv
from .gfootball_action import GfootballRawAction


class GfootballRawActionRunner(EnvElementRunner):

    def _init(self, cfg, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = GfootballRawAction(cfg)

    def get(self, engine: BaseEnv) -> np.array:
        agent_action = copy.deepcopy(engine.agent_action)
        return agent_action

    def reset(self) -> None:
        pass
