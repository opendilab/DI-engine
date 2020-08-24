from nervex.worker.actor.env_manager import SubprocessEnvManager
from nervex.envs.sumo import SumoWJ3Env


class SumoWJ3EnvManager(SubprocessEnvManager):
    # override
    def _init(self):
        self._envs = [SumoWJ3Env(self._cfg) for _ in range(self.env_num)]

    # override
    def step(self, action) -> 'SumoWJ3Env.timestep':  # noqa
        timestep = super().step(action)
        assert isinstance(timestep, list)
        assert isinstance(timestep[0], tuple)
        timestep_type = type(timestep[0])

        # item, agent_num, env_num
        items = [[getattr(timestep[i], item) for i in range(len(timestep))] for item in timestep[0]._fields]
        return timestep_type(*items)
