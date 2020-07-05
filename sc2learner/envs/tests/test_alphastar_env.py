import pytest
import torch
from sc2learner.data.fake_dataset import fake_stat_processed_professional_player
from sc2learner.envs import AlphaStarEnv


@pytest.fixture(scope='function')
def setup_as_stat():
    return fake_stat_processed_professional_player()


@pytest.mark.envtest
class TestAlphaStarEnv:
    def _get_random_action(self, agent_num):
        action = {
            'action_type': torch.LongTensor([0]),
            'delay': torch.LongTensor([0]),
            'queued': None,
            'selected_units': None,
            'target_units': None,
            'target_location': None
        }
        return [action] * agent_num

    def test_naive(self, setup_as_config, setup_as_stat):
        env = AlphaStarEnv(setup_as_config.env)
        agent_num = env.agent_num
        obs = env.reset([setup_as_stat for _ in range(agent_num)])

        S = 10
        for s in range(S):
            action = self._get_random_action(agent_num)
            timestep = env.step(action)
            print('step', s)
            assert isinstance(timestep, env.timestep)
