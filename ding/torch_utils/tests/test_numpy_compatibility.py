"""Runtime contracts required by NumPy 1 and 2 training clients."""
import gymnasium as gym
import numpy as np
import pytest
import torch

from ding.envs import DingEnvWrapper
from ding.torch_utils import to_ndarray, to_tensor


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64, np.bool_])
def test_numpy_tensor_roundtrip(dtype):
    value = np.array([[0, 1], [1, 0]], dtype=dtype)
    tensor = to_tensor({'obs': value})['obs']
    assert isinstance(tensor, torch.Tensor)
    np.testing.assert_array_equal(to_ndarray(tensor), value)
    if dtype == np.float32:
        tensor[0, 0] = 2
        assert value[0, 0] == 2


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v1'])
def test_gymnasium_environment(env_id):
    env = DingEnvWrapper(gym.make(env_id))
    try:
        env.seed(0, dynamic_seed=False)
        assert np.isfinite(env.reset()).all()
        step = env.step(env.random_action())
        assert np.isfinite(step.obs).all()
        assert np.isfinite(step.reward).all()
    finally:
        env.close()
