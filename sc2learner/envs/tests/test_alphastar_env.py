import pytest
import numpy as np
import torch
from sc2learner.data.fake_dataset import fake_stat_processed_professional_player, get_random_action
from sc2learner.envs import AlphaStarEnv


@pytest.fixture(scope='function')
def setup_as_stat():
    return fake_stat_processed_professional_player()


@pytest.mark.envtest
class TestAlphaStarEnv:
    def _get_random_action(self, agent_num, map_size):
        action = get_random_action()
        if action['target_location'] is not None:
            action['target_location'][0] = min(action['target_location'][0], map_size[0] - 1)
            action['target_location'][1] = min(action['target_location'][1], map_size[1] - 1)
        selected_units = action['selected_units'].tolist() if isinstance(action['selected_units'], torch.Tensor) else []
        target_units = action['target_units'].tolist() if isinstance(action['target_units'], torch.Tensor) else []
        set_units = set(selected_units).union(set(target_units))
        if len(set_units) > 0:
            max_units_idx = max(set_units)
            entity_raw = {'id': [int(4e9) + np.random.randint(0, 10000) for _ in range(max_units_idx + 1)]}
        else:
            entity_raw = {'id': []}
        action_data = {'action': action, 'entity_raw': entity_raw}
        return [action_data] * agent_num

    def test_naive(self, setup_as_config, setup_as_stat):
        env = AlphaStarEnv(setup_as_config.env)
        print(env)
        env_info = env.info()
        assert env_info.agent_num == env._agent_num
        print({k: v.shape for k, v in env_info.obs_space.items()})
        print(env_info.act_space.shape)
        print(env_info.rew_space.value)
        agent_num = env._agent_num
        map_size = env._map_size
        obs = env.reset([setup_as_stat for _ in range(agent_num)])

        S = 5
        for s in range(S):
            action = self._get_random_action(agent_num, map_size)
            timestep = env.step(action)
            print('step', s)
            assert isinstance(timestep, env.timestep)
