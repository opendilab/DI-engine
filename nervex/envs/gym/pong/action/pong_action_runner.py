from typing import List, Tuple
import copy
from sc2learner.envs.env.base_env import BaseEnv
from sc2learner.envs.common import EnvElementRunner
from .pong_action import PongRawAction

# done
class PongRawActionRunner(EnvElementRunner):

    def _init(self, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = PongRawAction()

    def get(self, engine: BaseEnv) -> int:
        agent_action = copy.deepcopy(engine.agent_action)
        assert isinstance(agent_action, int)
        ret = agent_action
        return ret

    #override
    def reset(self) -> None:
        pass