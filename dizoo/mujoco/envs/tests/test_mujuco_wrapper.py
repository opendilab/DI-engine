import pytest
import numpy as np

# from dizoo.mujoco.envs.mujoco_wrappers import RunningMeanStd
from ding.envs import RunningMeanStd


@pytest.mark.unittest
def test_rms():
    test_list = [np.random.randn(1) for _ in range(1000)]
    array = np.concatenate(test_list)
    static_mean = array.mean()
    static_std = array.std()
    rms = RunningMeanStd(shape=(1, ))
    for i in test_list:
        rms.update(i)
    rms_mean = rms.mean
    rms_std = rms.std
    assert abs(static_mean - rms_mean) <= 1e-5
    assert abs(static_std - static_std) <= 1e-5
