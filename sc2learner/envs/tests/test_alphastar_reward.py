import pytest
from collections import namedtuple
from sc2learner.data.fake_dataset import fake_stat_processed_professional_player
from sc2learner.envs.stat.alphastar_statistics import GameLoopStatistics, RealTimeStatistics
from sc2learner.envs.reward.alphastar_reward import AlphaStarReward

TmpAction = namedtuple('TmpAction', ['action_type', 'target_location'])


@pytest.fixture(scope='function')
def setup_loaded_stats():
    stat = fake_stat_processed_professional_player()
    return [GameLoopStatistics(stat, 20)] * 2


@pytest.mark.unittest
class TestPseudoReward:
    def test_global(self, setup_loaded_stats):
        reward_helper = AlphaStarReward(2, 'global', 1.)
        episode_stats = [RealTimeStatistics() for _ in range(2)]
        # case 1
        rewards = [-1, 1]
        actions = [TmpAction(197, [19, 20]), TmpAction(1, [18, 21])]  # build_hatchery_pt, smart_pt
        action_types = [a.action_type for a in actions]
        episode_stats[0].update_stat(actions[0], None, 22)
        episode_stats[1].update_stat(actions[1], None, 22)
        battle_values = AlphaStarReward.BattleValues(150, 200, 250, 350)
        rewards = reward_helper.get_pseudo_rewards(
            rewards, action_types, episode_stats, setup_loaded_stats, 22, battle_values
        )
        assert rewards['winloss'].tolist() == [-1, 1]
        assert rewards['build_order'].tolist() == pytest.approx([-0.8, -0])
        assert rewards['built_unit'].tolist() == [-15, -0]
        assert rewards['effect'].tolist() == [-0, -0]
        assert rewards['upgrade'].tolist() == [-0, -0]
        assert rewards['battle'].tolist() == [-50, 50]
        # case 2
        rewards = [0, 0]
        actions = [TmpAction(217, [17, 19]), TmpAction(197, [36, 13])]
        action_types = [a.action_type for a in actions]
        episode_stats[0].update_stat(actions[0], None, 44)
        episode_stats[1].update_stat(actions[1], None, 44)
        battle_values = AlphaStarReward.BattleValues(50, 100, 550, 600)
        rewards = reward_helper.get_pseudo_rewards(
            rewards, action_types, episode_stats, setup_loaded_stats, 44, battle_values
        )
        assert rewards['winloss'].tolist() == [0, 0]
        assert rewards['build_order'].tolist() == pytest.approx([-0.8, -0])
        assert rewards['built_unit'].tolist() == [-14, -15]
        assert rewards['effect'].tolist() == [-0, -0]
        assert rewards['upgrade'].tolist() == [-0, -0]
        assert rewards['battle'].tolist() == [0, -0]
        rewards = reward_helper.get_pseudo_rewards(
            [0, 0], action_types, episode_stats, setup_loaded_stats, 44, battle_values, return_list=True
        )
        assert isinstance(rewards, list) and len(rewards) == 2
        for k in rewards[0].keys():
            assert rewards[0][k].shape == (1, )
        # case 3
        episode_stats = [RealTimeStatistics() for _ in range(2)]
        win_loss_rewards = [0, 0]
        battle_values = AlphaStarReward.BattleValues(50, 100, 550, 600)
        actions = [
            TmpAction(217, [17, 19]),
            TmpAction(217, [17, 20]),
            TmpAction(197, [36, 13]),
            TmpAction(193, None),
            TmpAction(315, None),
            TmpAction(437, None),
            TmpAction(495, None),
            TmpAction(296, None),
            TmpAction(185, None),
        ]
        action_types = [197, 217]
        episode_stats[0].update_stat(actions[2], None, 1)
        episode_stats[1].update_stat(actions[0], None, 1)
        rewards = reward_helper.get_pseudo_rewards(
            win_loss_rewards, action_types, episode_stats, setup_loaded_stats, 1, battle_values
        )
        assert rewards['build_order'].tolist() == [0, -1]
        assert rewards['built_unit'].tolist() == [-15, -15]

        action_types = [217, 217]
        episode_stats[0].update_stat(actions[0], None, 1)
        episode_stats[1].update_stat(actions[1], None, 1)
        rewards = reward_helper.get_pseudo_rewards(
            win_loss_rewards, action_types, episode_stats, setup_loaded_stats, 1, battle_values
        )
        assert rewards['build_order'].tolist() == pytest.approx([0, -1.4])
        assert rewards['built_unit'].tolist() == [-14, 0]

        action_types = [193, 197]
        episode_stats[0].update_stat(actions[3], None, 1)
        episode_stats[1].update_stat(actions[2], None, 1)
        rewards = reward_helper.get_pseudo_rewards(
            win_loss_rewards, action_types, episode_stats, setup_loaded_stats, 1, battle_values
        )
        assert rewards['build_order'].tolist() == pytest.approx([0, -2.4])
        assert rewards['built_unit'].tolist() == [-13, -14]

        action_types = [315, 437]
        episode_stats[0].update_stat(actions[4], None, 1)
        episode_stats[1].update_stat(actions[5], None, 1)
        rewards = reward_helper.get_pseudo_rewards(
            win_loss_rewards, action_types, episode_stats, setup_loaded_stats, 1, battle_values
        )
        assert rewards['effect'].tolist() == pytest.approx([0, 0])
        assert rewards['upgrade'].tolist() == pytest.approx([0, -2])

        action_types = [296, 185]
        episode_stats[0].update_stat(actions[7], None, 1)
        episode_stats[1].update_stat(actions[8], None, 1)
        rewards = reward_helper.get_pseudo_rewards(
            win_loss_rewards, action_types, episode_stats, setup_loaded_stats, 1, battle_values
        )
        assert rewards['effect'].tolist() == pytest.approx([-1, 0])
        assert rewards['built_unit'].tolist() == pytest.approx([0, -15])
        assert rewards['upgrade'].tolist() == pytest.approx([0, 0])

    def test_immediate(self, setup_loaded_stats):
        reward_helper = AlphaStarReward(2, 'immediate', 1.)
        episode_stats = [RealTimeStatistics() for _ in range(2)]
        # case 1
        rewards = [-1, 1]
        actions = [TmpAction(197, [19, 20]), TmpAction(503, None)]  # build_hatchery_pt, smart_pt
        action_types = [a.action_type for a in actions]
        episode_stats[0].update_stat(actions[0], None, 22)
        episode_stats[1].update_stat(actions[1], None, 22)
        battle_values = AlphaStarReward.BattleValues(150, 200, 250, 350)
        rewards = reward_helper.get_pseudo_rewards(
            rewards, action_types, episode_stats, setup_loaded_stats, 22, battle_values
        )
        assert rewards['winloss'].tolist() == [-1, 1]
        assert rewards['build_order'].tolist() == pytest.approx([-0.8, 0])
        assert rewards['built_unit'].tolist() == [-3, -1]
        assert rewards['effect'].tolist() == [-0, -0]
        assert rewards['upgrade'].tolist() == [-0, -0]
        assert rewards['battle'].tolist() == [-50, 50]
        # case 2
        rewards = [-1, 1]
        actions = [TmpAction(217, [17, 19]), TmpAction(515, None)]  # build_hatchery_pt, smart_pt
        action_types = [a.action_type for a in actions]
        episode_stats[0].update_stat(actions[0], None, 44)
        episode_stats[1].update_stat(actions[1], None, 44)
        battle_values = AlphaStarReward.BattleValues(150, 200, 250, 350)
        rewards = reward_helper.get_pseudo_rewards(
            rewards, action_types, episode_stats, setup_loaded_stats, 44, battle_values
        )
        assert rewards['winloss'].tolist() == [-1, 1]
        assert rewards['build_order'].tolist() == pytest.approx([-1.8, 0])
        assert rewards['built_unit'].tolist() == [-4, 0]
        assert rewards['effect'].tolist() == [-0, -0]
        assert rewards['upgrade'].tolist() == [-0, -0]
        assert rewards['battle'].tolist() == [-50, 50]
        # case 3
        reward_helper = AlphaStarReward(2, 'immediate', 1.)
        episode_stats = [RealTimeStatistics() for _ in range(2)]
        battle_values = AlphaStarReward.BattleValues(150, 200, 250, 350)
        win_loss_rewards = [0, 0]
        actions = [TmpAction(197, [1, 1]), TmpAction(217, [17, 19]), TmpAction(516, None), TmpAction(193, None)]

        action_types = [197, 197]
        episode_stats[0].update_stat(actions[0], None, 1)
        episode_stats[1].update_stat(actions[0], None, 1)
        rewards = reward_helper.get_pseudo_rewards(
            win_loss_rewards, action_types, episode_stats, setup_loaded_stats, 1000, battle_values
        )
        assert rewards['build_order'].tolist() == pytest.approx([-0.8, -0.8])
        action_types = [217, 217]
        episode_stats[0].update_stat(actions[1], None, 1)
        episode_stats[1].update_stat(actions[1], None, 1)
        rewards = reward_helper.get_pseudo_rewards(
            win_loss_rewards, action_types, episode_stats, setup_loaded_stats, 1532, battle_values
        )
        assert rewards['build_order'].tolist() == pytest.approx([-0.8, -0.8])
        action_types = [516, 516]
        episode_stats[0].update_stat(actions[2], None, 1)
        episode_stats[1].update_stat(actions[2], None, 1)
        rewards = reward_helper.get_pseudo_rewards(
            win_loss_rewards, action_types, episode_stats, setup_loaded_stats, 1570, battle_values
        )
        assert rewards['build_order'].tolist() == pytest.approx([-1.8, -1.8])
        action_types = [193, 193]
        episode_stats[0].update_stat(actions[3], None, 1)
        episode_stats[1].update_stat(actions[3], None, 1)
        rewards = reward_helper.get_pseudo_rewards(
            win_loss_rewards, action_types, episode_stats, setup_loaded_stats, 2590, battle_values
        )
        assert rewards['build_order'].tolist() == pytest.approx([-2.8, -2.8])


if __name__ == '__main__':
    pytest.main(['test_pseudo_reward.py', '-s'])
