import copy

import numpy as np

from ding.envs.common import EnvElementRunner, EnvElement
from ding.envs.env.base_env import BaseEnv
from .gfootball_obs import PlayerObs, MatchObs
from ding.utils import deep_merge_dicts


class GfootballObsRunner(EnvElementRunner):

    def _init(self, cfg, *args, **kwargs) -> None:
        # set self._core and other state variable
        self._obs_match = MatchObs(cfg)
        self._obs_player = PlayerObs(cfg)
        self._core = self._obs_player  # placeholder

    def get(self, engine: BaseEnv) -> dict:
        ret = copy.deepcopy(engine._football_obs)
        # print(ret, type(ret))
        assert isinstance(ret, dict)
        match_obs = self._obs_match._to_agent_processor(ret)
        players_obs = self._obs_player._to_agent_processor(ret)
        return deep_merge_dicts(match_obs, players_obs)

    def reset(self) -> None:
        pass

    # override
    @property
    def info(self):
        return {'match': self._obs_match.info, 'player': self._obs_player.info}
