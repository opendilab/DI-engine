import numpy as np
import pytest
from dizoo.taxi.envs import TaxiV3Env

@pytest.mark.envtest
class TestTaxiV3Env:
    def test_naive(self):
        env = TaxiV3Env({})
        env.seed(314, dynamic_seed=False)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == ()
        for _ in range(5):
            env.reset()
            np.random.seed(314)
            print('=' * 60)
            for i in range(10):
                # Both ``env.random_action()``, and utilizing ``np.random`` as well as action space,
                # can generate legal random action.
                if i < 5:
                    random_action = np.array([env.action_space.sample()])
                else:
                    random_action = env.random_action()
                timestep = env.step(random_action)
                print(f"你本次的Timestep是：{timestep}")
                assert isinstance(timestep.obs, np.ndarray)
                assert isinstance(timestep.done, bool)
                assert timestep.obs.shape == ()
                assert timestep.reward.shape == (1, )
                assert timestep.reward >= env.reward_space.low
                assert timestep.reward <= env.reward_space.high
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()
        
        
#& 为了测试一下可视化的结果，我还这么测试了一个definite的序列，就是能接到乘客且对号入座甩客的actions
#& 结果显示可以生成一个gif图像。当然在代码里设定只有done = True时才行。        
# env = TaxiV3Env({})
# env.seed(314, dynamic_seed=False)
# assert env._seed == 314
# #* seed通过检查
# env.enable_save_replay()
# #* replay可视化通过检查

# obs = env.reset()
# actions = [1, 1, 3, 3, 3, 1, 1, 4, 0, 0, 2, 2, 2, 2, 1, 1, 5]
# for action in actions:
#     timestep = env.step(np.array([action]))
#     print(f"你本次的Timestep是：{timestep}")
# env.close()