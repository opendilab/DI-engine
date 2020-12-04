from nervex.envs.common import EnvElementRunner
from nervex.envs.env.base_env import BaseEnv
from .sumo_reward import SumoReward


class SumoRewardRunner(EnvElementRunner):
    r"""
    Overview:
        runner that help to get the reward space
    Interface:
        _init, get, reset
    """

    def _init(self, cfg: dict) -> None:
        r"""
        Overview:
            init the sumo reward helper with the given config file
        Arguments:
            - cfg(:obj:`EasyDict`): config, you can refer to `envs/sumo/sumo_env_default_config.yaml`
        """
        # set self._core and other state variable
        self._core = SumoReward(cfg)
        self._reward_weight = cfg.reward_weight
        self._reward_type = self._core._reward_type
        self._last_wait_time = 0
        self._last_vehicle_info = {}
        self._cum_reward = {k: 0 for k in self._reward_type}

    def reset(self) -> None:
        self._last_wait_time = 0
        self._last_vehicle_info = {}
        self._cum_reward = {k: 0 for k in self._reward_type}

    def get(self, engine: BaseEnv) -> float:
        r"""
        Overview:
            return the raw_action
        Arguments:
            - engine(:obj:`BaseEnv`): the sumo_env
        Returns:
            - reward(:obj:`float`): the reward of current env
        """
        assert isinstance(engine, BaseEnv)
        inputs_data = {}
        for k in self._reward_type:
            if k == 'wait_time':
                inputs_data[k] = {'last_wait_time': self._last_wait_time}
            elif k == 'queue_len':
                inputs_data[k] = {}
            elif k == 'delay_time':
                inputs_data[k] = {'last_vehicle_info': self._last_vehicle_info}

        output_data = self._core._to_agent_processor(inputs_data)
        reward = {}
        for k in self._reward_type:
            if k == 'wait_time':
                reward[k] = output_data[k][0]
                self._last_wait_time = output_data[k][1]
            elif k == 'delay_time':
                reward[k] = output_data[k][0]
                self._last_vehicle_info = output_data[k][1]
            elif k == 'queue_len':
                reward[k] = output_data[k]

        for k in self._reward_type:
            self._cum_reward[k] += reward[k]
        total_reward = 0.
        for k, w in self._reward_weight.items():
            total_reward += reward[k] * w
        return total_reward

    @property
    def cum_reward(self) -> dict:
        return self._cum_reward
