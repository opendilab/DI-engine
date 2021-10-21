import competitive_rl
import pytest
import numpy as np
from easydict import EasyDict
from dizoo.competitive_rl.envs.competitive_rl_env import CompetitiveRlEnv


@pytest.mark.envtest
class TestCompetitiveRlEnv:

    def test_pong_single(self):
        cfg = dict(
            opponent_type="builtin",
            is_evaluator=True,
            env_id='cPongDouble-v0',
        )
        cfg = EasyDict(cfg)
        env = CompetitiveRlEnv(cfg)
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == env.info().obs_space.shape
        # act_shape = env.info().act_space.shape
        act_val = env.info().act_space.value
        min_val, max_val = act_val['min'], act_val['max']
        np.random.seed(314)
        i = 0
        while True:
            random_action = np.random.randint(min_val, max_val, size=(1, ))
            timestep = env.step(random_action)
            if timestep.done:
                print(timestep)
                print('Env episode has {} steps'.format(i))
                break
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == env.info().obs_space.shape
            assert timestep.reward.shape == env.info().rew_space.shape
            assert timestep.reward >= env.info().rew_space.value['min']
            assert timestep.reward <= env.info().rew_space.value['max']
            i += 1
        print(env.info())
        env.close()

    def test_pong_double(self):
        cfg = dict(env_id='cPongDouble-v0', )
        cfg = EasyDict(cfg)
        env = CompetitiveRlEnv(cfg)
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == env.info().obs_space.shape
        act_val = env.info().act_space.value
        min_val, max_val = act_val['min'], act_val['max']
        np.random.seed(314)
        i = 0
        while True:
            random_action = [np.random.randint(min_val, max_val, size=(1, )) for _ in range(2)]
            timestep = env.step(random_action)
            if timestep.done:
                print(timestep)
                print('Env episode has {} steps'.format(i))
                break
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == env.info().obs_space.shape
            assert timestep.reward.shape == env.info().rew_space.shape
            i += 1
        print(env.info())
        env.close()
