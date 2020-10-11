from nervex.envs.common import EnvElementRunner
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.gym.pendulum.action.pendulum_action import PendulumRawAction


class PendulumRawActionRunner(EnvElementRunner):
    def _init(self, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = PendulumRawAction()

    def get(self, engine: BaseEnv) -> list:
        raw_action = self._core._from_agent_processor(engine.action, engine.frameskip)
        return raw_action

    def reset(self) -> None:
        pass
