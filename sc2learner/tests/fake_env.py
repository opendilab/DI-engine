import random
from sc2learner.data.fake_dataset import FakeReplayDataset


class FakeEnv:
    def __init__(self, num_agents, *args, **kwargs):
        self.dataset = FakeReplayDataset(dict(trajectory_len=1))
        self.num_agents = num_agents
        self.game_step = 0

    def _get_obs(self):
        return [random.choice(self.dataset)[0] for _ in range(self.num_agents)]

    def reset(self):
        self.game_step = 0
        return self._get_obs()

    def step(self, *args, **kwargs):
        self.game_step += 1
        due = [True] * self.num_agents
        obs = self._get_obs()
        reward = [0.0] * self.num_agents
        done = False
        episode_stat = [{} for i in range(self.num_agents)]
        info = {}
        return self.game_step, due, obs, reward, done, episode_stat, info

    def load_stat(self, stat, agent_no):
        pass