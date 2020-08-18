from typing import List, Tuple
import copy
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.common import EnvElementRunner
from nervex.envs.sumo.action.sumo_action import SumoRawAction

class SumoRawActionRunner(EnvElementRunner):

    def _init(self, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = SumoRawAction()

    def get(self, engine: BaseEnv) -> int:
        agent_action = copy.deepcopy(engine.agent_action)
        assert isinstance(agent_action, int)
        ret = agent_action
        return ret

    #override
    def reset(self) -> None:
        pass

    # def info(self) -> dict:
    #     self._core.info_template
