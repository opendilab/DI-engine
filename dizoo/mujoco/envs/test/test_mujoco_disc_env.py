import pytest
import numpy as np
from easydict import EasyDict

from ding.utils import set_pkg_seed
from dizoo.mujoco.envs import MujocoDiscEnv


@pytest.mark.envtest
def test_mujoco_env_final_eval_reward():
    set_pkg_seed(1234, use_cuda=False)
    each_dim_disc_size = 2
    env = MujocoDiscEnv(
        EasyDict(
            {
                'env_id': 'Ant-v3',
                'action_clip': False,
                'each_dim_disc_size': each_dim_disc_size,
                'delay_reward_step': 4,
                'save_replay_gif': False,
                'replay_path_gif': None
            }
        )
    )
    env.seed(1234)
    env.reset()
    action_dim = env._raw_action_space.shape
    final_eval_reward = np.array([0.], dtype=np.float32)
    while True:
        action = np.random.randint(0, each_dim_disc_size ** action_dim[0], 1)
        timestep = env.step(action)
        final_eval_reward += timestep.reward
        # print("{}(dtype: {})".format(timestep.reward, timestep.reward.dtype))
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
