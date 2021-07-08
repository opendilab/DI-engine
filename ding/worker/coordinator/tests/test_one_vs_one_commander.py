import time
import pytest
import os


@pytest.mark.unittest
class Test1v1Commander:

    def test_init(self, setup_1v1commander):
        # basic
        assert not setup_1v1commander._end_flag
        # task space
        assert setup_1v1commander._collector_task_space.cur == setup_1v1commander._collector_task_space.min_val == 0
        assert setup_1v1commander._collector_task_space.max_val == 2
        assert setup_1v1commander._learner_task_space.cur == setup_1v1commander._learner_task_space.min_val == 0
        assert setup_1v1commander._learner_task_space.max_val == 1
        # league
        league = setup_1v1commander._league
        active_players = league.active_players
        assert len(active_players) == 1
        active_player = active_players[0]
        assert active_player.player_id == setup_1v1commander._active_player.player_id
        # policy
        assert 'eps' in setup_1v1commander._policy.get_setting_collect({'learner_step': 100, 'envstep': 10000})

    def test_get_task(self, setup_1v1commander):
        # Must fist learner, then collector.
        assert setup_1v1commander.get_collector_task() is None

        # Get learner task
        learner_task_info = setup_1v1commander.get_learner_task()
        assert setup_1v1commander._learner_task_space.cur == 1
        learner_task_id = learner_task_info['task_id']
        assert learner_task_id.startswith('learner_task_'), learner_task_info['task_id']
        assert len(setup_1v1commander._current_policy_id) == 1
        assert learner_task_info['policy_id'] == setup_1v1commander._current_policy_id[0]
        assert learner_task_info['buffer_id'] == setup_1v1commander._current_buffer_id
        assert setup_1v1commander.get_learner_task() is None

        # Get evaluator task
        # Only after evaluator task is finished, can get collector task.
        evaluator_task_info = setup_1v1commander.get_collector_task()
        assert setup_1v1commander._collector_task_space.cur == 1
        evaluator_task_id = evaluator_task_info['task_id']
        assert evaluator_task_id.startswith('evaluator_task_'), evaluator_task_info['task_id']
        assert evaluator_task_info['collector_cfg'].eval_flag
        env_kwargs = evaluator_task_info['collector_cfg'].env
        assert env_kwargs.eval_opponent == setup_1v1commander._league.active_players[0]._eval_opponent_difficulty[0]
        assert len(evaluator_task_info['collector_cfg'].policy) == 1

        # Finish evaluator task, not reach stop value
        finished_task_dict = {
            'eval_flag': True,
            'game_result': [['losses', 'losses'], ['losses', 'draws']],
            'train_iter': 0,
            'real_episode_count': 4,
            'step_count': 4 * 120,
            'avg_time_per_episode': 1.89,
            'avg_time_per_step': 1.89 / 120,
            'avg_step_per_episode': 120.,
            'reward_mean': -10.3,
            'reward_std': 3.4,
        }
        assert not setup_1v1commander.finish_collector_task(evaluator_task_id, finished_task_dict)
        assert setup_1v1commander._collector_task_space.cur == 0

        # Get collector_task
        collector_task_info = setup_1v1commander.get_collector_task()
        assert setup_1v1commander._collector_task_space.cur == 1
        collector_task_id = collector_task_info['task_id']
        assert collector_task_id.startswith('collector_task_'), collector_task_info['task_id']
        assert collector_task_info['buffer_id'] == learner_task_info['buffer_id']
        assert 'eps' in collector_task_info['collector_cfg'].collect_setting
        policy_update_path = collector_task_info['collector_cfg'].policy_update_path
        assert len(policy_update_path) == 2
        assert policy_update_path[0] == policy_update_path[1]
        policy_update_flag = collector_task_info['collector_cfg'].policy_update_flag
        assert policy_update_flag[0] == policy_update_flag[1]
        assert not collector_task_info['collector_cfg'].eval_flag
        assert len(collector_task_info['collector_cfg'].policy) == 2

        # Finish collector_task
        finished_task_dict = {
            'eval_flag': False,
            'game_result': [['losses', 'losses'], ['losses', 'losses']],
            'step_count': 400,
            'train_iter': 20,
            'real_episode_count': 8,
            'avg_time_per_episode': 1.33,
            'avg_time_per_step': 1.33 / 500,
            'avg_step_per_episode': 50.,
            'reward_mean': 11.,
            'reward_std': 3.,
        }
        assert not setup_1v1commander.finish_collector_task(collector_task_id, finished_task_dict)
        assert setup_1v1commander._collector_task_space.cur == 0

        # Update learner info
        for i in range(0, 101, 10):
            learner_info = {
                'learner_step': i,
            }
            setup_1v1commander.update_learner_info('some_task_id', learner_info)

        # Get evaluator task; Finish evaluator task and reach stop value.
        time.sleep(5 + 0.1)
        evaluator_task_info = setup_1v1commander.get_collector_task()
        evaluator_task_id = evaluator_task_info['task_id']
        assert setup_1v1commander._collector_task_space.cur == 1
        assert evaluator_task_info['collector_cfg'].eval_flag
        finished_task_dict = {
            'eval_flag': True,
            'game_result': [['wins', 'wins'], ['wins', 'wins']],
            'train_iter': 100,
            'real_episode_count': 4,
            'step_count': 4 * 120,
            'avg_time_per_episode': 1.89,
            'avg_time_per_step': 1.89 / 120,
            'avg_step_per_episode': 120.,
            'reward_mean': 20.,
            'reward_std': 0.,
        }
        assert setup_1v1commander.finish_collector_task(evaluator_task_id, finished_task_dict)
        assert setup_1v1commander._end_flag
        assert setup_1v1commander._collector_task_space.cur == 0

        # Finish learner task
        finished_task_dict = {'buffer_id': setup_1v1commander._current_buffer_id}
        setup_1v1commander.finish_learner_task(learner_task_id, finished_task_dict)
        assert setup_1v1commander._learner_task_space.cur == 0

    @pytest.mark.notify
    def test_notify(self, setup_1v1commander):
        _ = setup_1v1commander.get_learner_task()
        setup_1v1commander.notify_fail_learner_task({})
        time.sleep(0.01)
        assert setup_1v1commander._learner_task_space.cur == 0
        _ = setup_1v1commander.get_collector_task()
        setup_1v1commander.notify_fail_collector_task({})
        time.sleep(0.01)
        assert setup_1v1commander._collector_task_space.cur == 0

        os.popen('rm -rf log')
        os.popen('rm -rf total_config.py')
