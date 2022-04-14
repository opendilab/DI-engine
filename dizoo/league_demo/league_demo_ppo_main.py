import os
import copy
import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from easydict import EasyDict

from ding.config import compile_config
from ding.worker import BaseLearner, BattleInteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed, Scheduler, deep_merge_dicts
from dizoo.league_demo.game_env import GameEnv
from dizoo.league_demo.demo_league import DemoLeague
from dizoo.league_demo.league_demo_collector import LeagueDemoCollector
from dizoo.league_demo.league_demo_ppo_config import league_demo_ppo_config


class EvalPolicy1:

    def __init__(self, optimal_policy: list) -> None:
        assert len(optimal_policy) == 2
        self.optimal_policy = optimal_policy

    def forward(self, data: dict) -> dict:
        return {
            env_id: {
                'action': torch.from_numpy(np.random.choice([0, 1], p=self.optimal_policy, size=(1, )))
            }
            for env_id in data.keys()
        }

    def reset(self, data_id: list = []) -> None:
        pass


class EvalPolicy2:

    def forward(self, data: dict) -> dict:
        return {
            env_id: {
                'action': torch.from_numpy(np.random.choice([0, 1], p=[0.5, 0.5], size=(1, )))
            }
            for env_id in data.keys()
        }

    def reset(self, data_id: list = []) -> None:
        pass


def main(cfg, seed=0, max_train_iter=int(1e8), max_env_step=int(1e8)):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        LeagueDemoCollector,
        BattleInteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    env_type = cfg.env.env_type
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    evaluator_env1 = BaseEnvManager(
        env_fn=[lambda: GameEnv(env_type) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )
    evaluator_env2 = BaseEnvManager(
        env_fn=[lambda: GameEnv(env_type) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )
    evaluator_env3 = BaseEnvManager(
        env_fn=[lambda: GameEnv(env_type) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    evaluator_env1.seed(seed, dynamic_seed=False)
    evaluator_env2.seed(seed, dynamic_seed=False)
    evaluator_env3.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    league = DemoLeague(cfg.policy.other.league)
    eval_policy1 = EvalPolicy1(evaluator_env1._env_ref.optimal_policy)
    eval_policy2 = EvalPolicy2()
    policies = {}
    learners = {}
    collectors = {}

    for player_id in league.active_players_ids:
        # default set the same arch model(different init weight)
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)
        policies[player_id] = policy
        collector_env = BaseEnvManager(
            env_fn=[lambda: GameEnv(env_type) for _ in range(collector_env_num)], cfg=cfg.env.manager
        )
        collector_env.seed(seed)

        learners[player_id] = BaseLearner(
            cfg.policy.learn.learner,
            policy.learn_mode,
            tb_logger=tb_logger,
            exp_name=cfg.exp_name,
            instance_name=player_id + '_learner'
        )
        collectors[player_id] = LeagueDemoCollector(
            cfg.policy.collect.collector,
            collector_env,
            tb_logger=tb_logger,
            exp_name=cfg.exp_name,
            instance_name=player_id + '_collector',
        )

    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    policies['historical'] = policy
    # use initial policy as another eval_policy
    eval_policy3 = PPOPolicy(cfg.policy, model=copy.deepcopy(model)).collect_mode

    main_key = [k for k in learners.keys() if k.startswith('main_player')][0]
    main_player = league.get_player_by_id(main_key)
    main_learner = learners[main_key]
    main_collector = collectors[main_key]
    # collect_mode ppo use multinomial sample for selecting action
    evaluator1_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
    evaluator1_cfg.stop_value = cfg.env.stop_value[0]
    evaluator1 = BattleInteractionSerialEvaluator(
        evaluator1_cfg,
        evaluator_env1, [policies[main_key].collect_mode, eval_policy1],
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='fixed_evaluator'
    )
    evaluator2_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
    evaluator2_cfg.stop_value = cfg.env.stop_value[1]
    evaluator2 = BattleInteractionSerialEvaluator(
        evaluator2_cfg,
        evaluator_env2, [policies[main_key].collect_mode, eval_policy2],
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='uniform_evaluator'
    )
    evaluator3_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
    evaluator3_cfg.stop_value = 99999999  # stop_value of evaluator3 is a placeholder
    evaluator3 = BattleInteractionSerialEvaluator(
        evaluator3_cfg,
        evaluator_env3, [policies[main_key].collect_mode, eval_policy3],
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='init_evaluator'
    )

    def load_checkpoint_fn(player_id: str, ckpt_path: str):
        state_dict = torch.load(ckpt_path)
        policies[player_id].learn_mode.load_state_dict(state_dict)

    torch.save(policies['historical'].learn_mode.state_dict(), league.reset_checkpoint_path)
    league.load_checkpoint = load_checkpoint_fn
    # snapshot the initial player as the first historial player
    for player_id, player_ckpt_path in zip(league.active_players_ids, league.active_players_ckpts):
        torch.save(policies[player_id].collect_mode.state_dict(), player_ckpt_path)
        league.judge_snapshot(player_id, force=True)
    init_main_player_rating = league.metric_env.create_rating(mu=0)

    schedule_flag = cfg.policy.learn.scheduler.schedule_flag
    if schedule_flag:
        user_scheduler_config = cfg.policy.learn.scheduler
        merged_scheduler_config = EasyDict(deep_merge_dicts(Scheduler.config, user_scheduler_config))
        param_scheduler = Scheduler(merged_scheduler_config)

    count = 0
    while True:
        if evaluator1.should_eval(main_learner.train_iter):
            stop_flag1, episode_info = evaluator1.eval(
                main_learner.save_checkpoint, main_learner.train_iter, main_collector.envstep
            )
            win_loss_result = [e['result'] for e in episode_info[0]]
            # set fixed NE policy trueskill(exposure) equal 10
            main_player.rating = league.metric_env.rate_1vsC(
                main_player.rating, league.metric_env.create_rating(mu=10, sigma=1e-8), win_loss_result
            )

        if evaluator2.should_eval(main_learner.train_iter):
            stop_flag2, episode_info = evaluator2.eval(
                main_learner.save_checkpoint, main_learner.train_iter, main_collector.envstep
            )
            win_loss_result = [e['result'] for e in episode_info[0]]
            # set random(uniform) policy trueskill(exposure) equal 0
            main_player.rating = league.metric_env.rate_1vsC(
                main_player.rating, league.metric_env.create_rating(mu=0, sigma=1e-8), win_loss_result
            )
        if evaluator3.should_eval(main_learner.train_iter):
            _, episode_info = evaluator3.eval(
                main_learner.save_checkpoint, main_learner.train_iter, main_collector.envstep
            )
            win_loss_result = [e['result'] for e in episode_info[0]]
            # use init main player as another evaluator metric
            main_player.rating, init_main_player_rating = league.metric_env.rate_1vs1(
                main_player.rating, init_main_player_rating, win_loss_result
            )
            tb_logger.add_scalar(
                'league/init_main_player_trueskill', init_main_player_rating.exposure, main_collector.envstep
            )
        if stop_flag1 and stop_flag2:
            break

        for player_id, player_ckpt_path in zip(league.active_players_ids, league.active_players_ckpts):
            tb_logger.add_scalar(
                'league/{}_trueskill'.format(player_id),
                league.get_player_by_id(player_id).rating.exposure, main_collector.envstep
            )
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
            train_data, episode_info = train_data[0], episode_info[0]  # only use launch player data for training
            for d in train_data:
                d['adv'] = d['reward']

            for i in range(cfg.policy.learn.update_per_collect):
                learner.train(train_data, collector.envstep)
            torch.save(learner.policy.state_dict(), player_ckpt_path)

            player_info = learner.learn_info
            player_info['player_id'] = player_id
            league.update_active_player(player_info)
            league.judge_snapshot(player_id)
            # set eval_flag=True to enable trueskill update
            job_finish_info = {
                'eval_flag': True,
                'launch_player': job['launch_player'],
                'player_id': job['player_id'],
                'result': [e['result'] for e in episode_info],
            }
            league.finish_job(job_finish_info)

            if schedule_flag:
                metrics = float(main_player.rating.exposure)
                entropy_weight = learner.policy.get_attribute('entropy_weight')
                entropy_weight = param_scheduler.step(metrics, entropy_weight)
                learner.policy.set_attribute('entropy_weight', entropy_weight)

        if main_collector.envstep >= max_env_step or main_learner.train_iter >= max_train_iter:
            break
        if count % 100 == 0:
            print(repr(league.payoff))
        count += 1


if __name__ == "__main__":
    main(league_demo_ppo_config)
