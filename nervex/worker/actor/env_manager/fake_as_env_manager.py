from collections import namedtuple
from nervex.worker.actor.env_manager.vec_env_manager import SubprocessEnvManager
from nervex.envs.alphastar import FakeAlphaStarEnv


class FakeASEnvManager(SubprocessEnvManager):
    # override
    def _init(self) -> None:
        self._envs = [FakeAlphaStarEnv({}) for _ in range(self.env_num)]

    # override
    def reset(self, *args, **kwargs) -> dict:
        obs = super().reset(*args, **kwargs)
        assert isinstance(obs, list)
        # agent_num, env_num
        return list(zip(*obs))

    # override
    def step(self, action) -> 'AlphaStarEnv.timestep':  # noqa
        # agent_num, env_num -> env_num, agent_num
        action = list(zip(*action))
        timestep = super().step(action)
        assert isinstance(timestep, list)
        assert isinstance(timestep[0], tuple)
        timestep_type = type(timestep[0])
        def pair(data):
            if not (isinstance(data, list) or isinstance(data, tuple)):
                return [data, data]
            else:
                return data
        # item, agent_num, env_num
        items = [list(zip(*[pair(getattr(timestep[i], item)) for i in range(len(timestep))])) for item in timestep[0]._fields]
        return timestep_type(*items)
