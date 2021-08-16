import os
import copy
import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from functools import partial

from ding.config import compile_config
from ding.worker import BaseLearner, Episode1v1Collector, OnevOneEvaluator, NaiveReplayBuffer
from ding.envs import BaseEnvManager
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from dizoo.league_demo.demo_league import DemoLeague
from dizoo.slime_volley.envs import SlimeVolleyEnv
from dizoo.slime_volley.config import slime_volley_league_ppo_config


class EvalPolicy1:

    def forward(self, data: dict) -> dict:
        return {env_id: {'action': torch.zeros(1)} for env_id in data.keys()}

    def reset(self, data_id: list = []) -> None:
        pass


class EvalPolicy2:

    def forward(self, data: dict) -> dict:
        return {env_id: {'action': torch.from_numpy(np.random.randint(0, 6, (1, )))} for env_id in data.keys()}

    def reset(self, data_id: list = []) -> None:
        pass


def main(cfg, seed=0, max_iterations=int(1e10)):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        Episode1v1Collector,
        OnevOneEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    evaluator_env1 = BaseEnvManager(env_fn=[partial(SlimeVolleyEnv, cfg.env) for _ in range(evaluator_env_num)], cfg=cfg.env.manager)
    evaluator_env2 = BaseEnvManager(env_fn=[partial(SlimeVolleyEnv, cfg.env) for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    evaluator_env1.seed(seed, dynamic_seed=False)
    evaluator_env2.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    league = DemoLeague(cfg.policy.other.league)
    eval_policy1 = EvalPolicy1()
    eval_policy2 = EvalPolicy2()
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
    evaluator1_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
    evaluator1_cfg.stop_value = cfg.env.stop_value[0]
    evaluator1 = OnevOneEvaluator(
        evaluator1_cfg,
        evaluator_env1, [policies[main_key].collect_mode, eval_policy1],
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='fixed_evaluator'
    )
    evaluator2_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
    evaluator2_cfg.stop_value = cfg.env.stop_value[1]
    evaluator2 = OnevOneEvaluator(
        evaluator2_cfg,
        evaluator_env2, [policies[main_key].collect_mode, eval_policy2],
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='uniform_evaluator'
    )

    for player_id, player_ckpt_path in zip(league.active_players_ids, league.active_players_ckpts):
        torch.save(policies[player_id].collect_mode.state_dict(), player_ckpt_path)
        league.judge_snapshot(player_id, force=True)

    for run_iter in range(max_iterations):
        if evaluator1.should_eval(main_learner.train_iter):
            stop_flag1, reward = evaluator1.eval(
                main_learner.save_checkpoint, main_learner.train_iter, main_collector.envstep
            )
            tb_logger.add_scalar('fixed_evaluator_step/reward_mean', reward, main_collector.envstep)
        if evaluator2.should_eval(main_learner.train_iter):
            stop_flag2, reward = evaluator2.eval(
                main_learner.save_checkpoint, main_learner.train_iter, main_collector.envstep
            )
            tb_logger.add_scalar('uniform_evaluator_step/reward_mean', reward, main_collector.envstep)
        if stop_flag1 and stop_flag2:
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
