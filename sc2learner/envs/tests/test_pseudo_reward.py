import pytest
from sc2learner.data.fake_dataset import fake_stat_processed_professional_player
from sc2learner.envs.statistics import GameLoopStatistics, RealTimeStatistics
from sc2learner.envs.rewards import RewardHelper


@pytest.fixture(scope='function')
def setup_loaded_stats():
    stat = fake_stat_processed_professional_player()
    return [GameLoopStatistics(stat, 20)] * 2


@pytest.mark.unittest
class TestPseudoReward:
    def test_global(self, setup_loaded_stats):
        reward_helper = RewardHelper(2, 'global', 1.)
        episode_stats = [RealTimeStatistics() for _ in range(2)]
        # case 1
        rewards = [0, 1]
        actions = [
            {
                'action_type': 197,
                'target_location': [19, 20]
            },
            {
                'action_type': 1,
                'target_location': [18, 21]
            },
        ]  # build_hatchery_pt, smart_pt
        action_types = [a['action_type'] for a in actions]
        episode_stats[0].update_stat(actions[0], None, 22)
        episode_stats[1].update_stat(actions[1], None, 22)
        battle_values = RewardHelper.BattleValues(150, 100, 250, 350)
        rewards = reward_helper.get_pseudo_rewards(
            rewards, action_types, episode_stats, setup_loaded_stats, 22, battle_values
        )
        assert rewards['winloss'].tolist() == [0, 1]
        assert rewards['build_order'].tolist() == pytest.approx([-19.8, -0])
        assert rewards['built_unit'].tolist() == [-15, -0]
        assert rewards['effect'].tolist() == [-0, -0]
        assert rewards['upgrade'].tolist() == [-0, -0]
        assert rewards['battle'].tolist() == [-150, 150]
        # case 2
        rewards = [0, 0]
        actions = [{'action_type': 197, 'target_location': [29, 33]}, {'action_type': 197, 'target_location': [17, 40]}]
        action_types = [a['action_type'] for a in actions]
        episode_stats[0].update_stat(actions[0], None, 44)
        episode_stats[1].update_stat(actions[1], None, 44)
        battle_values = RewardHelper.BattleValues(50, 100, 550, 350)
        rewards = reward_helper.get_pseudo_rewards(
            rewards, action_types, episode_stats, setup_loaded_stats, 44, battle_values
        )
        assert rewards['winloss'].tolist() == [0, 0]
        assert rewards['build_order'].tolist() == pytest.approx([-19.6, -19.8])
        assert rewards['built_unit'].tolist() == [-0, -15]
        assert rewards['effect'].tolist() == [-0, -0]
        assert rewards['upgrade'].tolist() == [-0, -0]
        assert rewards['battle'].tolist() == [250, -250]
        rewards = reward_helper.get_pseudo_rewards(
            [0, 0], action_types, episode_stats, setup_loaded_stats, 44, battle_values, return_list=True
        )
        assert isinstance(rewards, list) and len(rewards) == 2
        for k in rewards[0].keys():
            assert rewards[0][k].shape == (1, )

    def test_immediate(self, setup_loaded_stats):
        reward_helper = RewardHelper(2, 'immediate', 1.)
