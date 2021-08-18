import os
import copy
import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from functools import partial

from ding.config import compile_config
from ding.worker import BaseLearner, Episode1v1Collector, NaiveReplayBuffer, BaseSerialEvaluator
from ding.envs import BaseEnvManager
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from dizoo.league_demo.demo_league import DemoLeague
from dizoo.slime_volley.envs import SlimeVolleyEnv
from dizoo.slime_volley.config import slime_volley_league_ppo_config


def main(cfg, seed=0, max_iterations=int(1e10)):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        Episode1v1Collector,
        BaseSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env_cfg = copy.deepcopy(cfg.env)
    collector_env_cfg.is_evaluator = False
    evaluator_env_cfg = copy.deepcopy(cfg.env)
    evaluator_env_cfg.is_evaluator = True
    collector_env = BaseEnvManager(env_fn=[partial(SlimeVolleyEnv, collector_env_cfg)
                                           for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[partial(SlimeVolleyEnv, evaluator_env_cfg)
                                           for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    league = DemoLeague(cfg.policy.other.league)
    policies = {}
    learners = {}
    collectors = {}
    for player_id in league.active_players_ids:
        # default set the same arch model(different init weight)
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)
        policies[player_id] = policy
        collector_env = BaseEnvManager(env_fn=[partial(SlimeVolleyEnv, cfg.env) for _ in range(collector_env_num)], cfg=cfg.env.manager)
        collector_env.seed(seed)

        learners[player_id] = BaseLearner(
            cfg.policy.learn.learner,
            policy.learn_mode,
            tb_logger=tb_logger,
            exp_name=cfg.exp_name,
            instance_name=player_id + '_learner'
        )
        collectors[player_id] = Episode1v1Collector(
            cfg.policy.collect.collector,
            collector_env,
            tb_logger=tb_logger,
            exp_name=cfg.exp_name,
            instance_name=player_id + '_colllector',
        )
    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    policies['historical'] = policy

    main_key = [k for k in learners.keys() if k.startswith('main_player')][0]
    main_learner = learners[main_key]
    main_collector = collectors[main_key]
    # collect_mode ppo use multinomial sample for selecting action
    evaluator_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
    evaluator_cfg.stop_value = cfg.env.stop_value
    evaluator = BaseSerialEvaluator(
        evaluator_cfg,
        evaluator_env,
        policies[main_key].collect_mode,
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='fixed_evaluator'
    )

    for player_id, player_ckpt_path in zip(league.active_players_ids, league.active_players_ckpts):
        torch.save(policies[player_id].collect_mode.state_dict(), player_ckpt_path)
        league.judge_snapshot(player_id, force=True)

    stop_flag = False
    for run_iter in range(max_iterations):
        if evaluator.should_eval(main_learner.train_iter):
            stop_flag, reward = evaluator.eval(
                main_learner.save_checkpoint, main_learner.train_iter, main_collector.envstep
            )
            tb_logger.add_scalar('fixed_evaluator_step/reward_mean', reward, main_collector.envstep)
        if stop_flag:
            break
        for player_id, player_ckpt_path in zip(league.active_players_ids, league.active_players_ckpts):
            collector, learner = collectors[player_id], learners[player_id]
            job = league.get_job_info(player_id)
            opponent_player_id = job['player_id'][1]
            # print('job player: {}'.format(job['player_id']))
            if 'historical' in opponent_player_id:
                opponent_policy = policies['historical'].collect_mode
                opponent_path = job['checkpoint_path'][1]
                opponent_policy.load_state_dict(torch.load(opponent_path, map_location='cpu'))
            else:
                opponent_policy = policies[opponent_player_id].collect_mode
            collector.reset_policy([policies[player_id].collect_mode, opponent_policy])
            train_data, episode_info = collector.collect(train_iter=learner.train_iter)
            train_data, episode_info = train_data[0], episode_info[0]  # only use launer player data for training
            for d in train_data:
                d['adv'] = d['reward']

            for i in range(cfg.policy.learn.update_per_collect):
                learner.train(train_data, collector.envstep)
            torch.save(learner.policy.state_dict(), player_ckpt_path)

            player_info = learner.learn_info
            player_info['player_id'] = player_id
            league.update_active_player(player_info)
            league.judge_snapshot(player_id)
            job_finish_info = {
                'launch_player': job['launch_player'],
                'player_id': job['player_id'],
                'result': [e['result'] for e in episode_info],
            }
            league.finish_job(job_finish_info)
        if run_iter % 100 == 0:
            print(repr(league.payoff))


if __name__ == "__main__":
    main(slime_volley_league_ppo_config)
