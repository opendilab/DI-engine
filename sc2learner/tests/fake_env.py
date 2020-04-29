import random
from sc2learner.data.fake_dataset import FakeReplayDataset, get_z


class DummyStat:
    def __init__(self):
        pass

    def get_z(self, idx):
        return get_z()


class FakeEnv:
    def __init__(self, num_agents, *args, **kwargs):
        self.dataset = FakeReplayDataset(dict(trajectory_len=1))
        self.num_agents = num_agents
        self.game_loop = 0
        self.loaded_eval_stat = DummyStat()

    def _get_obs(self):
        obs = []
        for _ in range(self.num_agents):
            stepdata = random.choice(self.dataset)[0]
            del stepdata['actions']
            obs.append(stepdata)
        return obs

    def reset(self):
        self.game_loop = 0
        return self._get_obs()

    def step(self, *args, **kwargs):
        self.game_loop += 1
        due = [True] * self.num_agents
        obs = self._get_obs()
        reward = [0.0] * self.num_agents
        done = False
        # This fake z is generated from get_z, it's useless to check it against itself
        # use real environment to check the format of z
        episode_stat = [get_z() for i in range(self.num_agents)]
        info = {}
        return self.game_loop, due, obs, reward, done, episode_stat, info

    def load_stat(self, stat, agent_no):
        pass

    def get_target_z(self, agent_no, game_loop):
        return None
