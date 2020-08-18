from typing import List, Tuple
import copy
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.common import EnvElementRunner
from nervex.envs.sumo.reward.sumo_reward import SumoReward


class SumoRewardRunner(EnvElementRunner):

    def _init(self, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._core = SumoReward(*args)
        self.last_total_wait = 0
        
    def get(self, engine: BaseEnv) -> float:
        assert isinstance(engine, BaseEnv)
        # reward_of_action = copy.deepcopy(engine.reward_of_action)
        # ret = reward_of_action
        self.current_total_wait = engine._collect_waiting_times()
        self.wait_time_reward = self.last_total_wait - self.current_total_wait
        self.last_total_wait = self.current_total_wait

        return self._core._to_agent_processor(
            engine.reward_type,
            self.wait_time_reward,
            engine._get_queue_length(),
            engine._collect_delay_time()
        )

    #overriede
    def reset(self) -> None:
        self.last_total_wait = 0
        pass

    # def info(self) -> dict:
    #     return self._core.info_template
