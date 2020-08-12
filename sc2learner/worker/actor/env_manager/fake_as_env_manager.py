from .vec_env_manager import SubprocessEnvManager
from sc2learner.envs.alphastar import FakeAlphaStarEnv


class FakeASEnvManager(SubprocessEnvManager):
    # override
    def _init(self) -> None:
        self._envs = [FakeAlphaStarEnv({}) for _ in range(self.env_num)]
