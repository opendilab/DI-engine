import copy

import torch

from nervex.envs.common import EnvElementRunner
from nervex.envs.env.base_env import BaseEnv
from .pong_action import PongRawAction


class PongRawActionRunner(EnvElementRunner):
    def _init(self, cfg, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = PongRawAction(cfg)

    def get(self, engine: BaseEnv) -> torch.tensor:
        agent_action = copy.deepcopy(engine.agent_action)
        return agent_action

    def reset(self) -> None:
        pass
