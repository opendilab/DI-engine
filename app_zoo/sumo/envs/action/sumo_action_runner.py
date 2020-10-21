from nervex.envs.common import EnvElementRunner
from nervex.envs.env.base_env import BaseEnv
from .sumo_action import SumoRawAction


class SumoRawActionRunner(EnvElementRunner):
    r"""
    Overview:
        runner that help to get the action space
    Interface:
        _init, get, reset
    """

    def _init(self, cfg) -> None:
        r"""
        Overview:
            init the sumo action helper with the given config file
        Arguments:
            - cfg(:obj:`EasyDict`): config, you can refer to `envs/sumo/sumo_env_default_config.yaml`
        """
        # set self._core and other state variable
        self._core = SumoRawAction(cfg)
        self._last_action = None

    def get(self, engine: BaseEnv):
        r"""
        Overview:
            return the raw_action
        Arguments:
            - engine(:obj:`BaseEnv`): the sumo_env
        Returns:
            - raw_action(:obj:`dict`): the returned raw_action
        """
        action = engine.action
        if self._last_action is None:
            self._last_action = [None for _ in range(len(action))]
        data = {}
        for tl, act, last_act in zip(self._core._tls, engine.action, self._last_action):
            data[tl] = {'action': act, 'last_action': last_act}
        raw_action = self._core._from_agent_processor(data)
        self._last_action = action
        return raw_action

    # override
    def reset(self) -> None:
        r"""
        Overview:
            reset the stored last_action
        """
        self._last_action = None
