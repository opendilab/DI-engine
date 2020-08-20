from typing import List, Tuple
import copy
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.common import EnvElementRunner
from nervex.envs.sumo.reward.sumo_reward import SumoReward


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
            - cfg(:obj:`EasyDict`): config, you can refer to `env/sumo/sumo_env_default_config.yaml`
        """
        # set self._core and other state variable
        self._core = SumoReward(cfg)
        self._reward_type = self._core._reward_type
        self._last_wait_time = 0
        self._last_vehicle_info = {}

    def reset(self) -> None:
        self._last_wait_time = 0
        self._last_vehicle_info = {}

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
                reward[k] = output_data

        return reward
