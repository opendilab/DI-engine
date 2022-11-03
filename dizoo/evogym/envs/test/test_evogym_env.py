import pytest
import numpy as np
from easydict import EasyDict

from ding.utils import set_pkg_seed
from dizoo.evogym.envs import EvoGymEnv


@pytest.mark.envtest
@pytest.mark.parametrize('robot', ['speed_bot', 'random'])
def test_evogym_env_eval_episode_return(robot):
    set_pkg_seed(1234, use_cuda=False)
    env = EvoGymEnv(EasyDict({'env_id': 'Walker-v0', 'robot': robot, 'robot_dir': '../'}))
    env.seed(1234)
    env.reset()
    action_dim = env.action_space.shape
    eval_episode_return = np.array([0.], dtype=np.float32)
    if robot == 'speed_bot':
        assert env.observation_space.shape == (58, )
        assert action_dim == (10, )

    while True:
        action = np.random.random(size=action_dim)
        timestep = env.step(action)
        eval_episode_return += timestep.reward
        print("{}(dtype: {})".format(timestep.reward, timestep.reward.dtype))
        if timestep.done:
            print(
                "{}({}), {}({})".format(
                    timestep.info['eval_episode_return'], type(timestep.info['eval_episode_return']),
                    eval_episode_return, type(eval_episode_return)
                )
            )
            # timestep.reward and the cumulative reward in wrapper EvalEpisodeReturn are not the same.
            assert abs(timestep.info['eval_episode_return'].item() - eval_episode_return.item()) / \
                abs(timestep.info['eval_episode_return'].item()) < 1e-5
            break
