from time import time
import copy
import pytest
from easydict import EasyDict
import numpy as np
from dizoo.gobigger.envs import GoBiggerEnv


@pytest.mark.unittest
class TestGoBiggerEnv:

    @pytest.mark.parametrize('use_spatial', [True, False])
    def test_naive(self, use_spatial):
        cfg = copy.deepcopy(EasyDict(GoBiggerEnv.config))
        cfg.use_spatial = use_spatial
        env = GoBiggerEnv(cfg)
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        act_val = env.info().act_space.value
        min_val, max_val = act_val['min'], act_val['max']
        player_num = env._player_num
        team_num = env._team_num
        player_num_per_team = env._player_num_per_team
        for i in range(10):
            random_action = [np.random.randint(min_val, max_val, size=(player_num_per_team, )) for _ in range(team_num)]
            timestep = env.step(random_action)
            for k, v in timestep.obs[0][0].items():
                if isinstance(v, list):
                    print('obs', k, len(v), v[0].shape)
                elif isinstance(v, np.ndarray):
                    print('obs', k, v.shape)
            print('reward', timestep.reward)
            reward = timestep.reward
            assert isinstance(timestep.obs, list)
            assert len(timestep.obs) == team_num
            for o in obs:
                assert len(o) == player_num_per_team
            assert isinstance(timestep.done, bool)
            assert isinstance(reward, list)
            assert len(reward) == team_num
            assert isinstance(timestep, tuple)
        # print(env.info())
        env.close()
