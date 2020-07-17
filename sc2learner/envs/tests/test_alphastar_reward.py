import pytest
from collections import namedtuple
from sc2learner.data.fake_dataset import fake_stat_processed_professional_player
from sc2learner.envs.stat.alphastar_statistics import GameLoopStatistics, RealTimeStatistics
from sc2learner.envs.reward.alphastar_reward import AlphaStarReward
from sc2learner.envs.reward.alphastar_reward_runner import AlphaStarRewardRunner
from sc2learner.envs.env.base_env import BaseEnv

TmpAction = namedtuple('TmpAction', ['action_type', 'target_location'])


@pytest.fixture(scope='function')
def setup_loaded_stats():
    stat = fake_stat_processed_professional_player()
    return [GameLoopStatistics(stat, 20)] * 2


@pytest.fixture(scope='function')
def setup_fake_env(setup_loaded_stats):
    class FakeEnv(BaseEnv):
        def __init__(self):
            self.agent_num = 2
            self.loaded_eval_stat = setup_loaded_stats
            self.episode_stat = [RealTimeStatistics() for _ in range(self.agent_num)]

        def __repr__(self):
            pass

        def close(self):
            pass

        def info(self):
            pass

        def reset(self):
            pass

        def step(self):
            pass

        def seed(self):
            pass

            #method_names = ['__repr__', 'close', 'info', 'reset', 'seed', 'step']
            #for item in method_names:
            #    setattr(self, item, lambda x: x)

    return FakeEnv()


@pytest.mark.unittest
class TestPseudoReward:
    def test_global(self, setup_loaded_stats):
        reward_helper = AlphaStarReward(2, 'global', 1.)
        episode_stat = [RealTimeStatistics() for _ in range(2)]
        # case 1
        reward = [-1, 1]
        action = [TmpAction(197, [19, 20]), TmpAction(1, [18, 21])]  # build_hatchery_pt, smart_pt
        action_types = [a.action_type for a in action]
        episode_stat[0].update_stat(action[0], None, 22)
        episode_stat[1].update_stat(action[1], None, 22)
        battle_value = AlphaStarReward.BattleValues(150, 200, 250, 350)
        reward = reward_helper.get_pseudo_rewards(
            reward, action_types, episode_stat, setup_loaded_stats, 22, battle_value
        )
        assert reward['winloss'].tolist() == [-1, 1]
        assert reward['build_order'].tolist() == pytest.approx([-0.8, -0])
        assert reward['built_unit'].tolist() == [-15, -0]
        assert reward['effect'].tolist() == [-0, -0]
        assert reward['upgrade'].tolist() == [-0, -0]
        assert reward['battle'].tolist() == [-50, 50]
        # case 2
        reward = [0, 0]
        action = [TmpAction(217, [17, 19]), TmpAction(197, [36, 13])]
        action_types = [a.action_type for a in action]
        episode_stat[0].update_stat(action[0], None, 44)
        episode_stat[1].update_stat(action[1], None, 44)
        battle_value = AlphaStarReward.BattleValues(50, 100, 550, 600)
        reward = reward_helper.get_pseudo_rewards(
            reward, action_types, episode_stat, setup_loaded_stats, 44, battle_value
        )
        assert reward['winloss'].tolist() == [0, 0]
        assert reward['build_order'].tolist() == pytest.approx([-0.8, -0])
        assert reward['built_unit'].tolist() == [-14, -15]
        assert reward['effect'].tolist() == [-0, -0]
        assert reward['upgrade'].tolist() == [-0, -0]
        assert reward['battle'].tolist() == [0, -0]
        reward = reward_helper.get_pseudo_rewards(
            [0, 0], action_types, episode_stat, setup_loaded_stats, 44, battle_value, return_list=True
        )
        assert isinstance(reward, list) and len(reward) == 2
        for k in reward[0].keys():
            assert reward[0][k].shape == (1, )
        # case 3
        episode_stat = [RealTimeStatistics() for _ in range(2)]
        win_loss_reward = [0, 0]
        battle_value = AlphaStarReward.BattleValues(50, 100, 550, 600)
        action = [
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
        episode_stat[0].update_stat(action[2], None, 1)
        episode_stat[1].update_stat(action[0], None, 1)
        reward = reward_helper.get_pseudo_rewards(
            win_loss_reward, action_types, episode_stat, setup_loaded_stats, 1, battle_value
        )
        assert reward['build_order'].tolist() == [0, -1]
        assert reward['built_unit'].tolist() == [-15, -15]

        action_types = [217, 217]
        episode_stat[0].update_stat(action[0], None, 1)
        episode_stat[1].update_stat(action[1], None, 1)
        reward = reward_helper.get_pseudo_rewards(
            win_loss_reward, action_types, episode_stat, setup_loaded_stats, 1, battle_value
        )
        assert reward['build_order'].tolist() == pytest.approx([0, -1.4])
        assert reward['built_unit'].tolist() == [-14, 0]

        action_types = [193, 197]
        episode_stat[0].update_stat(action[3], None, 1)
        episode_stat[1].update_stat(action[2], None, 1)
        reward = reward_helper.get_pseudo_rewards(
            win_loss_reward, action_types, episode_stat, setup_loaded_stats, 1, battle_value
        )
        assert reward['build_order'].tolist() == pytest.approx([0, -2.4])
        assert reward['built_unit'].tolist() == [-13, -14]

        action_types = [315, 437]
        episode_stat[0].update_stat(action[4], None, 1)
        episode_stat[1].update_stat(action[5], None, 1)
        reward = reward_helper.get_pseudo_rewards(
            win_loss_reward, action_types, episode_stat, setup_loaded_stats, 1, battle_value
        )
        assert reward['effect'].tolist() == pytest.approx([0, 0])
        assert reward['upgrade'].tolist() == pytest.approx([0, -2])

        action_types = [296, 185]
        episode_stat[0].update_stat(action[7], None, 1)
        episode_stat[1].update_stat(action[8], None, 1)
        reward = reward_helper.get_pseudo_rewards(
            win_loss_reward, action_types, episode_stat, setup_loaded_stats, 1, battle_value
        )
        assert reward['effect'].tolist() == pytest.approx([-1, 0])
        assert reward['built_unit'].tolist() == pytest.approx([0, -15])
        assert reward['upgrade'].tolist() == pytest.approx([0, 0])

    def test_immediate(self, setup_loaded_stats):
        reward_helper = AlphaStarReward(2, 'immediate', 1.)
        episode_stat = [RealTimeStatistics() for _ in range(2)]
        # case 1
        reward = [-1, 1]
        action = [TmpAction(197, [19, 20]), TmpAction(503, None)]  # build_hatchery_pt, smart_pt
        action_types = [a.action_type for a in action]
        episode_stat[0].update_stat(action[0], None, 22)
        episode_stat[1].update_stat(action[1], None, 22)
        battle_value = AlphaStarReward.BattleValues(150, 200, 250, 350)
        reward = reward_helper.get_pseudo_rewards(
            reward, action_types, episode_stat, setup_loaded_stats, 22, battle_value
        )
        assert reward['winloss'].tolist() == [-1, 1]
        assert reward['build_order'].tolist() == pytest.approx([-0.8, 0])
        assert reward['built_unit'].tolist() == [-3, -1]
        assert reward['effect'].tolist() == [-0, -0]
        assert reward['upgrade'].tolist() == [-0, -0]
        assert reward['battle'].tolist() == [-50, 50]
        # case 2
        reward = [-1, 1]
        action = [TmpAction(217, [17, 19]), TmpAction(515, None)]  # build_hatchery_pt, smart_pt
        action_types = [a.action_type for a in action]
        episode_stat[0].update_stat(action[0], None, 44)
        episode_stat[1].update_stat(action[1], None, 44)
        battle_value = AlphaStarReward.BattleValues(150, 200, 250, 350)
        reward = reward_helper.get_pseudo_rewards(
            reward, action_types, episode_stat, setup_loaded_stats, 44, battle_value
        )
        assert reward['winloss'].tolist() == [-1, 1]
        assert reward['build_order'].tolist() == pytest.approx([-1.8, 0])
        assert reward['built_unit'].tolist() == [-4, 0]
        assert reward['effect'].tolist() == [-0, -0]
        assert reward['upgrade'].tolist() == [-0, -0]
        assert reward['battle'].tolist() == [-50, 50]
        # case 3
        reward_helper = AlphaStarReward(2, 'immediate', 1.)
        episode_stat = [RealTimeStatistics() for _ in range(2)]
        battle_value = AlphaStarReward.BattleValues(150, 200, 250, 350)
        win_loss_reward = [0, 0]
        action = [TmpAction(197, [1, 1]), TmpAction(217, [17, 19]), TmpAction(516, None), TmpAction(193, None)]

        action_types = [197, 197]
        episode_stat[0].update_stat(action[0], None, 1)
        episode_stat[1].update_stat(action[0], None, 1)
        reward = reward_helper.get_pseudo_rewards(
            win_loss_reward, action_types, episode_stat, setup_loaded_stats, 1000, battle_value
        )
        assert reward['build_order'].tolist() == pytest.approx([-0.8, -0.8])
        action_types = [217, 217]
        episode_stat[0].update_stat(action[1], None, 1)
        episode_stat[1].update_stat(action[1], None, 1)
        reward = reward_helper.get_pseudo_rewards(
            win_loss_reward, action_types, episode_stat, setup_loaded_stats, 1532, battle_value
        )
        assert reward['build_order'].tolist() == pytest.approx([-0.8, -0.8])
        action_types = [516, 516]
        episode_stat[0].update_stat(action[2], None, 1)
        episode_stat[1].update_stat(action[2], None, 1)
        reward = reward_helper.get_pseudo_rewards(
            win_loss_reward, action_types, episode_stat, setup_loaded_stats, 1570, battle_value
        )
        assert reward['build_order'].tolist() == pytest.approx([-1.8, -1.8])
        action_types = [193, 193]
        episode_stat[0].update_stat(action[3], None, 1)
        episode_stat[1].update_stat(action[3], None, 1)
        reward = reward_helper.get_pseudo_rewards(
            win_loss_reward, action_types, episode_stat, setup_loaded_stats, 2590, battle_value
        )
        assert reward['build_order'].tolist() == pytest.approx([-2.8, -2.8])

    def test_reward_runner(self, setup_fake_env):
        env = setup_fake_env
        reward_helper = AlphaStarRewardRunner(2, 'global', 1, return_list=False)
        reward_helper.reset()
        # case 1
        action = [TmpAction(197, [19, 20]), TmpAction(1, [18, 21])]  # build_hatchery_pt, smart_pt
        action_types = [a.action_type for a in action]
        env.episode_stat[0].update_stat(action[0], None, 22)
        env.episode_stat[1].update_stat(action[1], None, 22)
        env.episode_steps = 22
        env.action = action
        env.reward = [-1, 1]
        reward_helper._last_battle_value = [100, 150]
        env.battle_value = [100, 200]
        reward = reward_helper.get(env)
        assert reward['winloss'].tolist() == [-1, 1]
        assert reward['build_order'].tolist() == pytest.approx([-0.8, -0])
        assert reward['built_unit'].tolist() == [-15, -0]
        assert reward['effect'].tolist() == [-0, -0]
        assert reward['upgrade'].tolist() == [-0, -0]
        assert reward['battle'].tolist() == [-50, 50]
        print(repr(reward_helper))
        print(reward_helper.info)
        # case 2
        reward = [0, 0]
        action = [TmpAction(217, [17, 19]), TmpAction(197, [36, 13])]
        action_types = [a.action_type for a in action]
        env.episode_stat[0].update_stat(action[0], None, 44)
        env.episode_stat[1].update_stat(action[1], None, 44)
        env.episode_steps = 44
        env.action = action
        env.reward = [0, 0]
        assert reward_helper._last_battle_value == [100, 200]
        env.battle_value = [150, 250]
        reward = reward_helper.get(env)
        assert reward['winloss'].tolist() == [0, 0]
        assert reward['build_order'].tolist() == pytest.approx([-0.8, -0])
        assert reward['built_unit'].tolist() == [-14, -15]
        assert reward['effect'].tolist() == [-0, -0]
        assert reward['upgrade'].tolist() == [-0, -0]
        assert reward['battle'].tolist() == [0, -0]
