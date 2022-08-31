import pytest
import gym


@pytest.mark.envtest
def test_shapes():
    from dizoo.mujoco.envs import mujoco_gym_env
    ant = gym.make('AntTruncatedObs-v2')
    assert ant.observation_space.shape == (27, )
    assert ant.action_space.shape == (8, )
    humanoid = gym.make('HumanoidTruncatedObs-v2')
    assert humanoid.observation_space.shape == (45, )
    assert humanoid.action_space.shape == (17, )
