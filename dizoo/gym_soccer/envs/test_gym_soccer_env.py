import numpy as np
import pytest
from dizoo.gym_soccer.envs.gym_soccer_env import GymSoccerEnv
from easydict import EasyDict


@pytest.mark.envtest
class TestGymSoccerEnv:

    def test_naive(self):
        env = GymSoccerEnv(EasyDict({'env_id': 'Soccer-v0', 'act_scale': True}))
        # env.enable_save_replay('./video')
        env.seed(25, dynamic_seed=False)
        assert env._seed == 25
        obs = env.reset()
        assert obs.shape == (59, )
        for i in range(1000):
            random_action = env.get_random_action()
            # print('random_action', random_action)
            timestep = env.step(random_action)
            # env.render()
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (59, )
            # print(timestep.obs)
            assert timestep.reward.shape == (1, )
            assert timestep.info['action_args_mask'].shape == (3, 5)
            if timestep.done:
                print('reset env')
                env.reset()
                assert env._final_eval_reward == 0
        print(env.info())
        # env.replay_log("./video/20211019011053-base_left_0-vs-base_right_0.rcg")
        env.close()
