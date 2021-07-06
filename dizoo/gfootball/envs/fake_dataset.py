import random
import numpy as np

from dizoo.gfootball.envs.obs.gfootball_obs import PlayerObs, MatchObs
from ding.utils.data import default_collate


def generate_data(player_obs: dict) -> np.array:
    dim = player_obs['dim']
    min = player_obs['value']['min']
    max = player_obs['value']['max']
    dinfo = player_obs['value']['dinfo']
    if dinfo in ['one-hot', 'boolean vector']:
        data = np.zeros((dim, ), dtype=np.float32)
        data[random.randint(0, dim - 1)] = 1
        return data
    elif dinfo == 'float':
        data = np.random.rand(dim)
        for dim_idx in range(dim):
            data[dim_idx] = min[dim_idx] + (max[dim_idx] - min[dim_idx]) * data[dim_idx]
        return data


class FakeGfootballDataset:

    def __init__(self):
        match_obs = MatchObs({})
        player_obs = PlayerObs({})
        self.match_obs_info = match_obs.template
        self.player_obs_info = player_obs.template
        self.action_dim = 19
        self.batch_size = 4
        del match_obs, player_obs

    def __len__(self) -> int:
        return self.batch_size

    def get_random_action(self) -> np.array:
        return np.random.randint(0, self.action_dim - 1, size=(1, ))

    def get_random_obs(self) -> dict:
        inputs = {}
        for match_obs in self.match_obs_info:
            key = match_obs['ret_key']
            data = generate_data(match_obs)
            inputs[key] = data
        players_list = []
        for _ in range(22):
            one_player = {}
            for player_obs in self.player_obs_info:
                key = player_obs['ret_key']
                data = generate_data(player_obs)
                one_player[key] = data
            players_list.append(one_player)
        inputs['players'] = players_list
        return inputs

    def get_batched_obs(self, bs: int) -> dict:
        batch = []
        for _ in range(bs):
            batch.append(self.get_random_obs())
        return default_collate(batch)

    def get_random_reward(self) -> np.array:
        return np.array([random.random() - 0.5])

    def get_random_terminals(self) -> int:
        sample = random.random()
        if sample > 0.99:
            return 1
        return 0

    def get_batch_sample(self, bs: int) -> list:
        batch = []
        for _ in range(bs):
            step = {}
            step['obs'] = self.get_random_obs()
            step['next_obs'] = self.get_random_obs()
            step['action'] = self.get_random_action()
            step['done'] = self.get_random_terminals()
            step['reward'] = self.get_random_reward()
            batch.append(step)
        return batch
