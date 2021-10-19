import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep


class GameEnv(BaseEnv):

    def __init__(self, game_type='prisoner_dilemma'):
        self.game_type = game_type
        assert self.game_type in ['zero_sum', 'prisoner_dilemma']
        if self.game_type == 'prisoner_dilemma':
            self.optimal_policy = [0, 1]
        elif self.game_type == 'zero_sum':
            self.optimal_policy = [0.375, 0.625]

    def seed(self, seed, dynamic_seed=False):
        pass

    def reset(self):
        return np.array([[0, 1], [1, 0]]).astype(np.float32)  # trivial observation

    def step(self, actions):
        if self.game_type == 'zero_sum':
            if actions == [0, 0]:
                rewards = 3, -3
                results = "wins", "losses"
            elif actions == [0, 1]:
                rewards = -2, 2
                results = "losses", "wins"
            elif actions == [1, 0]:
                rewards = -2, 2
                results = "losses", "wins"
            elif actions == [1, 1]:
                rewards = 1, -1
                results = "wins", "losses"
            else:
                raise RuntimeError("invalid actions: {}".format(actions))
        elif self.game_type == 'prisoner_dilemma':
            if actions == [0, 0]:
                rewards = -1, -1
                results = "draws", "draws"
            elif actions == [0, 1]:
                rewards = -20, 0
                results = "losses", "wins"
            elif actions == [1, 0]:
                rewards = 0, -20
                results = "wins", "losses"
            elif actions == [1, 1]:
                rewards = -10, -10
                results = 'draws', 'draws'
            else:
                raise RuntimeError("invalid actions: {}".format(actions))
        observations = np.array([[0, 1], [1, 0]]).astype(np.float32)
        rewards = np.array(rewards).astype(np.float32)
        rewards = rewards[..., np.newaxis]
        dones = True, True
        infos = {
            'result': results[0],
            'final_eval_reward': rewards[0]
        }, {
            'result': results[1],
            'final_eval_reward': rewards[1]
        }
        return BaseEnvTimestep(observations, rewards, True, infos)

    def close(self):
        pass

    def __repr__(self):
        return "DI-engine League Demo GameEnv"

    def info(self):
        pass
