import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep


class GameEnv(BaseEnv):

    def __init__(self, game_type='prisoner_dilemma'):
        self.game_type = game_type
        assert self.game_type in ['zero_sum', 'prisoner_dilemma']

    def seed(self, seed, dynamic_seed=False):
        pass

    def reset(self):
        return np.array([[0, 1], [1, 0]]).astype(np.float32)  # trivial observation

    def step(self, actions):
        if self.game_type == 'zero_sum':
            if actions == [0, 0]:
                rewards = 3, -3
                results = "win", "lose"
            elif actions == [0, 1]:
                rewards = -2, 2
                results = "lose", "win"
            elif actions == [1, 0]:
                rewards = -2, 2
                results = "lose", "win"
            elif actions == [1, 1]:
                rewards = 1, -1
                results = "win", "lose"
            else:
                raise RuntimeError("invalid actions: {}".format(actions))
        elif self.game_type == 'prisoner_dilemma':
            if actions == [0, 0]:
                rewards = -1, -1
                results = "draw", "draw"
            elif actions == [0, 1]:
                rewards = -20, 0
                results = "win", "lose"
            elif actions == [1, 0]:
                rewards = 0, -20
                results = "loss", "win"
            elif actions == [1, 1]:
                rewards = -10, -10
                results = 'draw', 'draw'
            else:
                raise RuntimeError("invalid actions: {}".format(actions))
        observations = np.array([[0, 1], [1, 0]]).astype(np.float32)
        rewards = np.array(rewards).astype(np.float32)
        rewards = rewards[..., np.newaxis]
        dones = True, True
        infos = {'result': results[0]}, {'result': results[1]}
        return BaseEnvTimestep(
            observations, rewards, True, [{
                'final_eval_reward': rewards[0]
            }, {
                'final_eval_reward': rewards[1]
            }]
        )

    def close(self):
        pass

    def __repr__(self):
        return "DI-engine League Demo GameEnv"

    def info(self):
        pass
