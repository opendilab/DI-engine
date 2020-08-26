from nervex.envs.gym.pong.pong_env import PongEnv
from nervex.worker.actor.env_manager.vec_env_manager import SubprocessEnvManager


class PongEnvManager(SubprocessEnvManager):
    def _init(self) -> None:
        self._envs = [PongEnv(self._cfg) for _ in range(self.env_num)]
