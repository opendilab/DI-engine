import pytest
import numpy as np
from easydict import EasyDict

from ding.utils import set_pkg_seed
from dizoo.evogym.envs import EvoGymEnv


@pytest.mark.envtest
@pytest.mark.parametrize('robot', ['speed_bot', 'random'])
def test_evogym_env_final_eval_reward(robot):
    set_pkg_seed(1234, use_cuda=False)
    env = EvoGymEnv(EasyDict({'env_id': 'Walker-v0', 'robot': robot}))
    env.seed(1234)
    env.reset()
    action_dim = env.action_space.shape
    final_eval_reward = np.array([0.], dtype=np.float32)
    if robot == 'speed_bot':
        assert env.observation_space.shape == (58,)
        assert action_dim == (10,)

    while True:
        action = np.random.random(size=action_dim)
        timestep = env.step(action)
        final_eval_reward += timestep.reward
        print("{}(dtype: {})".format(timestep.reward, timestep.reward.dtype))
        if timestep.done:
            print(
                "{}({}), {}({})".format(
                    timestep.info['final_eval_reward'], type(timestep.info['final_eval_reward']), final_eval_reward,
                    type(final_eval_reward)
                )
            )
            # timestep.reward and the cumulative reward in wrapper FinalEvalReward are not the same.
            assert abs(timestep.info['final_eval_reward'].item() - final_eval_reward.item()) / \
                abs(timestep.info['final_eval_reward'].item()) < 1e-5
            break


if __name__ == '__main__':
    set_pkg_seed(1234, use_cuda=False)
    env = EvoGymEnv(EasyDict({'env_id': 'Walker-v0', 'robot': 'random'}))
    env.seed(1234)
    env.reset()
    action_dim = env.action_space.shape
    print(action_dim)
    print(env.observation_space.shape)
    final_eval_reward = np.array([0.], dtype=np.float32)

    while True:
        action = np.random.random(size=action_dim)
        timestep = env.step(action)
        final_eval_reward += timestep.reward
        print("{}(dtype: {})".format(timestep.reward, timestep.reward.dtype))
        if timestep.done:
            print(
                "{}({}), {}({})".format(
                    timestep.info['final_eval_reward'], type(timestep.info['final_eval_reward']), final_eval_reward,
                    type(final_eval_reward)
                )
            )
            # timestep.reward and the cumulative reward in wrapper FinalEvalReward are not the same.
            assert abs(timestep.info['final_eval_reward'].item() - final_eval_reward.item()) / \
                   abs(timestep.info['final_eval_reward'].item()) < 1e-5
            break
