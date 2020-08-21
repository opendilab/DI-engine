from nervex.envs.sumo.sumo_env import SumoWJ3Env
from collections import namedtuple
from nervex.worker.actor.env_manager.base_env_manager import BaseEnvManager
from nervex.worker.actor.env_manager.vec_env_manager import SubprocessEnvManager


class SumoEnvManager(SubprocessEnvManager):
    def _init(self) -> None:
        self._envs = [SumoWJ3Env({}) for _ in range(self.env_num)]

    def reset(self, *args, **kwargs) -> dict:
        obs = super().reset(*args, **kwargs)
        assert isinstance(obs, list)
        # agent_num, env_num
        return obs

    def step(self, action: list):
        r"""
        Arguments:
            - action(:obj:`list`): the input action to step, len(action) = env_num
        """
        assert (len(action) == self.env_num)
        timestep = super().step(action)
        return timestep
