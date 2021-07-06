import pytest
from easydict import EasyDict
import numpy as np
from dizoo.classic_control.bitflip.envs import BitFlipEnv


@pytest.mark.unittest
def test_bitfilp_env():
    n_bits = 10
    env = BitFlipEnv(EasyDict({'n_bits': n_bits}))
    env.seed(314)
    assert env._seed == 314
    obs = env.reset()
    assert obs.shape == (2 * n_bits, )
    for i in range(10):
        action = np.random.randint(0, n_bits, size=(1, ))
        timestep = env.step(action)
        assert timestep.obs.shape == (2 * n_bits, )
        assert timestep.reward.shape == (1, )
