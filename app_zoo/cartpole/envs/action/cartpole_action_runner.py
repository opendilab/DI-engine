from nervex.envs.common import EnvElementRunner
from nervex.envs.env.base_env import BaseEnv
from .cartpole_action import CartpoleRawAction


class CartpoleRawActionRunner(EnvElementRunner):

    def _init(self, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = CartpoleRawAction()

    def get(self, engine: BaseEnv) -> list:
        raw_action = self._core._from_agent_processor(engine.action)
        return raw_action

    def reset(self) -> None:
        pass
