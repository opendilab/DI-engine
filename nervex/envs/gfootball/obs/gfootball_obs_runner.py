import copy

import numpy as np

from nervex.envs.common import EnvElementRunner, EnvElement
from nervex.envs.env.base_env import BaseEnv
from .gfootball_obs import PlayerObs, MatchObs
from nervex.utils import merge_dicts


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
        return merge_dicts(match_obs, players_obs)

    def reset(self) -> None:
        pass

    # override
    def info(self) -> 'EnvElement.info_template':
        return [self._obs_match.info, self._obs_player.info]
