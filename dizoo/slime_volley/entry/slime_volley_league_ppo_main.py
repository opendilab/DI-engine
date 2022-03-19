import os
import gym
import numpy as np
import copy
import shutil
import torch
from tensorboardX import SummaryWriter
from functools import partial
from easydict import EasyDict

from ding.config import compile_config
from ding.worker import BaseLearner, BattleEpisodeSerialCollector, NaiveReplayBuffer, InteractionSerialEvaluator
from ding.envs import SyncSubprocessEnvManager
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from ding.league import BaseLeague, ActivePlayer
from dizoo.slime_volley.envs import SlimeVolleyEnv
from dizoo.slime_volley.config.slime_volley_league_ppo_config import main_config


class MyLeague(BaseLeague):
    # override
    def _get_job_info(self, player: ActivePlayer, eval_flag: bool = False) -> dict:
        assert isinstance(player, ActivePlayer), player.__class__
        player_job_info = EasyDict(player.get_job(eval_flag))
        return {
            'agent_num': 2,
            'launch_player': player.player_id,
            'player_id': [player.player_id, player_job_info.opponent.player_id],
            'checkpoint_path': [player.checkpoint_path, player_job_info.opponent.checkpoint_path],
            'player_active_flag': [isinstance(p, ActivePlayer) for p in [player, player_job_info.opponent]],
        }

    # override
    def _mutate_player(self, player: ActivePlayer):
        # no mutate operation
        pass

    # override
    def _update_player(self, player: ActivePlayer, player_info: dict) -> None:
        assert isinstance(player, ActivePlayer)
        if 'learner_step' in player_info:
            player.total_agent_step = player_info['learner_step']

    # override
    @staticmethod
    def save_checkpoint(src_checkpoint_path: str, dst_checkpoint_path: str) -> None:
        shutil.copy(src_checkpoint_path, dst_checkpoint_path)


def main(cfg, seed=0):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        PPOPolicy,
        BaseLearner,
        BattleEpisodeSerialCollector,
        InteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env_cfg = copy.deepcopy(cfg.env)
    collector_env_cfg.agent_vs_agent = True
    evaluator_env_cfg = copy.deepcopy(cfg.env)
    evaluator_env_cfg.agent_vs_agent = False
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(SlimeVolleyEnv, evaluator_env_cfg) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )
    evaluator_env.seed(seed, dynamic_seed=False)

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    league = MyLeague(cfg.policy.other.league)
    policies, learners, collectors = {}, {}, {}

    for player_id in league.active_players_ids:
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)
        policies[player_id] = policy
        collector_env = SyncSubprocessEnvManager(
            env_fn=[partial(SlimeVolleyEnv, collector_env_cfg) for _ in range(collector_env_num)], cfg=cfg.env.manager
        )
        collector_env.seed(seed)

        learners[player_id] = BaseLearner(
            cfg.policy.learn.learner,
            policy.learn_mode,
            tb_logger,
            exp_name=cfg.exp_name,
            instance_name=player_id + '_learner'
        )
        collectors[player_id] = BattleEpisodeSerialCollector(
            cfg.policy.collect.collector,
            collector_env, [policy.collect_mode, policy.collect_mode],
            tb_logger,
            exp_name=cfg.exp_name,
            instance_name=player_id + '_collector'
        )
    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    policies['historical'] = policy
    main_key = [k for k in learners.keys() if k.startswith('main_player')][0]
    main_player = league.get_player_by_id(main_key)
    main_learner = learners[main_key]
    main_collector = collectors[main_key]

    # eval vs bot
    evaluator_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
    evaluator_cfg.stop_value = cfg.env.stop_value
    evaluator = InteractionSerialEvaluator(
        evaluator_cfg,
        evaluator_env,
        policy.eval_mode,
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='builtin_ai_evaluator'
    )

    def load_checkpoint_fn(player_id: str, ckpt_path: str):
        state_dict = torch.load(ckpt_path)
        policies[player_id].learn_mode.load_state_dict(state_dict)

    league.load_checkpoint = load_checkpoint_fn
    # snapshot the initial player as the first historial player
    for player_id, player_ckpt_path in zip(league.active_players_ids, league.active_players_ckpts):
        torch.save(policies[player_id].collect_mode.state_dict(), player_ckpt_path)
        league.judge_snapshot(player_id, force=True)

    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    count = 0
    while True:
        if evaluator.should_eval(main_learner.train_iter):
            stop_flag, eval_episode_info = evaluator.eval(
                main_learner.save_checkpoint, main_learner.train_iter, main_collector.envstep
            )
            win_loss_result = [e['result'] for e in eval_episode_info]
            # set eval bot rating as 100
            main_player.rating = league.metric_env.rate_1vsC(
                main_player.rating, league.metric_env.create_rating(mu=100, sigma=1e-8), win_loss_result
            )
            if stop_flag:
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

            new_data, episode_info = collector.collect(
                train_iter=learner.train_iter, n_episode=cfg.policy.collect.n_episode
            )
            train_data = sum(new_data[0], [])  # sum all episodes
            learner.train(train_data, collector.envstep)

            player_info = learner.learn_info
            player_info['player_id'] = player_id
            league.update_active_player(player_info)
            league.judge_snapshot(player_id)
            # set eval_flag=True to enable trueskill update
            job_finish_info = {
                'eval_flag': True,
                'launch_player': job['launch_player'],
                'player_id': job['player_id'],
                # result is from `info` returned from env.step
                'result': [e['result'] for e in episode_info[0]],
            }
            league.finish_job(job_finish_info)
        if count % 50 == 0:
            payoff_string = repr(league.payoff)
            rank_string = league.player_rank(string=True)
            tb_logger.add_text('payoff_step', payoff_string, main_collector.envstep)
            tb_logger.add_text('rank_step', rank_string, main_collector.envstep)
        count += 1


if __name__ == "__main__":
    main(main_config)
