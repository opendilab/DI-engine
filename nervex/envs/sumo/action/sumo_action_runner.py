from typing import List, Tuple
import copy
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.common import EnvElementRunner
from nervex.envs.sumo.action.sumo_action import SumoRawAction


class SumoRawActionRunner(EnvElementRunner):
    def _init(self, cfg) -> None:
        # set self._core and other state variable
        self._core = SumoRawAction(cfg)
        self._last_action = None

    def get(self, engine: BaseEnv):
        action = engine.action
        if self._last_action is None:
            self._last_action = [None for _ in range(len(action))]
        data = {}
        for tl, act, last_act in zip(self._core._tls, engine.action, self._last_action):
            data[tl] = {'action': act, 'last_action': last_act}
        raw_action = self._core._from_agent_processor(data)
        self._last_action = action
        return raw_action

    #override
    def reset(self) -> None:
        self._last_action = None
