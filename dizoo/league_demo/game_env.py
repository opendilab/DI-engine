import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep


class GameEnv(BaseEnv):

    def __init__(self):
        pass

    def seed(self, seed, dynamic_seed=False):
        pass

    def reset(self):
        return np.array([[0, 1], [1, 0]]).astype(np.float32)  # trivial observation

    def step(step, actions):
        if actions == [0, 0]:
            rewards = -10, -10
        elif actions == [0, 1]:
            rewards = +1, -1
        elif actions == [1, 0]:
            rewards = -1, +1
        elif actions == [1, 1]:
            rewards = 0, 0
        else:
            raise RuntimeError("invalid actions: {}".format(actions))
        observations = np.array([[0, 1], [1, 0]]).astype(np.float32)
        rewards = np.array(rewards).astype(np.float32)
        rewards = rewards[..., np.newaxis]
        dones = True, True
        infos = {}, {}
        return BaseEnvTimestep(
            observations, rewards, True, [{
                'final_eval_reward': rewards[0]
            }, {
                'final_eval_reward': rewards[1]
            }]
        )

    def close(self):
        pass
