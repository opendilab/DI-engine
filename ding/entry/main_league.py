"""
BUG: Stop in 2 iters
"""
from ding.framework import Task, Context
from rich import print
import time
import os
import copy
import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.worker import BaseLearner, BattleEpisodeSerialCollector, BattleInteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed, Scheduler
from dizoo.league_demo.game_env import GameEnv
from dizoo.league_demo.demo_league import DemoLeague
from dizoo.league_demo.league_demo_ppo_config import league_demo_ppo_config
from easydict import EasyDict
from ding.utils.default_helper import deep_merge_dicts
from ding.utils import DistributedWriter


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


class LeaguePolicy:
    pass


def league_dispatching(task: Task, cfg, tb_logger, league, policies):

    def update_active_player(player_info):
        league.update_active_player(player_info)
        league.judge_snapshot(player_info["player_id"])

    task.on("update_active_player", update_active_player)

    def _league(ctx):
        print("League dispatching on node {}".format(task.router.node_id))
        time.sleep(1)
        # One episode each round
        i = ctx.total_step % len(league.active_players_ids)
        player_id, player_ckpt_path = league.active_players_ids[i], league.active_players_ckpts[i]

        job = league.get_job_info(player_id)
        opponent_player_id = job['player_id'][1]

        if 'historical' in opponent_player_id:
            opponent_policy = policies['historical'].collect_mode
            opponent_path = job['checkpoint_path'][1]
            opponent_policy.load_state_dict(torch.load(opponent_path, map_location='cpu'))
        else:
            opponent_policy = policies[opponent_player_id].collect_mode
        # Watch out that in parallel mode, we should not send functions between processes, instead,
        # we should send the objects needed by the policies.
        collect_session = {
            "policies": [policies[player_id].collect_mode, opponent_policy],
            "player_id": player_id,
            "player_ckpt_path": player_ckpt_path
        }
        print("Player ID", collect_session["player_id"])

        task.emit("set_collect_session", collect_session)

        yield

        job_finish_info = {
            'eval_flag': True,
            'launch_player': job['launch_player'],
            'player_id': job['player_id'],
            'result': [e['result'] for e in ctx.episode_info],
        }

        league.finish_job(job_finish_info)

    return _league


def collecting(task: Task, cfg, tb_logger, player_ids):
    collectors = {}

    def _collect(ctx):
        collect_session = task.wait_for("set_collect_session")[0][0]
        print("  Collecting on node {}".format(task.router.node_id))

        if not collectors:
            for player_id in player_ids:
                collector_env = BaseEnvManager(
                    env_fn=[lambda: GameEnv(cfg.env.env_type) for _ in range(cfg.env.collector_env_num)],
                    cfg=cfg.env.manager
                )
                collector_env.seed(0)
                collectors[player_id] = BattleEpisodeSerialCollector(
                    cfg.policy.collect.collector,
                    collector_env,
                    tb_logger=tb_logger,
                    exp_name=cfg.exp_name,
                    instance_name=player_id + '_colllector',
                )

        collector = collectors[collect_session["player_id"]]
        collector.reset_policy(collect_session["policies"])
        train_data, episode_info = collector.collect()  # TODO Do we need train_iter?
        train_data, episode_info = train_data[0], episode_info[0]  # only use launch player data for training
        ctx.episode_info = episode_info
        for d in train_data:
            d['adv'] = d['reward']

        learn_session = {
            "player_id": collect_session["player_id"],
            "train_data": train_data,
            "envstep": collector.envstep,
            "player_ckpt_path": collect_session["player_ckpt_path"]
        }
        task.emit_remote("set_learn_session", learn_session)  # Shoot and forget
        task.emit("set_learn_session", learn_session)  # Shoot and forget

    return _collect


def learning(task: Task, cfg, tb_logger, player_ids, policies):
    learners = {}

    def _learn(ctx):
        learn_session = task.wait_for("set_learn_session")[0][0]
        print("    Learning on node {}".format(task.router.node_id))
        time.sleep(1)

        if not learners:
            for player_id in player_ids:
                policy = policies[player_id]
                learners[player_id] = BaseLearner(
                    cfg.policy.learn.learner,
                    policy.learn_mode,
                    tb_logger=tb_logger,
                    exp_name=cfg.exp_name,
                    instance_name=player_id + '_learner'
                )

        learner = learners[learn_session["player_id"]]
        state_dict = learner.policy.state_dict()
        torch.save(state_dict, learn_session["player_ckpt_path"])  # Save to local

        player_info = learner.learn_info
        player_info['player_id'] = learn_session["player_id"]
        player_info["train_iter"] = learner.train_iter

        task.emit_remote("update_active_player", player_info)  # Broadcast to other middleware
        task.emit("update_active_player", player_info)  # Broadcast to other middleware

    return _learn


def evaluating(task: Task, cfg, tb_logger, player_ids, policies):
    evaluator_env1, eval_policy1, evaluator1 = None, None, None
    learn_session = None

    def set_learn_session(remote_learn_session):
        if "main_player" in remote_learn_session["player_id"]:
            nonlocal learn_session
            learn_session = remote_learn_session

    task.on("set_learn_session", set_learn_session)

    def _evaluate(ctx):
        print("      Evaluating on node {}".format(task.router.node_id))
        time.sleep(1)

        nonlocal evaluator_env1, eval_policy1, evaluator1
        if evaluator_env1 is None:
            evaluator_env_num = cfg.env.evaluator_env_num
            env_type = cfg.env.env_type
            evaluator_env1 = BaseEnvManager(
                env_fn=[lambda: GameEnv(env_type) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
            )
            eval_policy1 = EvalPolicy1(evaluator_env1._env_ref.optimal_policy)
            evaluator1_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
            evaluator1_cfg.stop_value = cfg.env.stop_value[0]
            main_key = [k for k in player_ids if k.startswith('main_player')][0]
            evaluator1 = BattleInteractionSerialEvaluator(
                evaluator1_cfg,
                evaluator_env1, [policies[main_key].collect_mode, eval_policy1],
                tb_logger,
                exp_name=cfg.exp_name,
                instance_name='fixed_evaluator'
            )

        nonlocal learn_session

        if ctx.total_step == 0:
            train_iter = 0
        else:
            player_info = task.wait_for("update_active_player")
            if "main_player" not in learn_session["player_id"]:
                return
            train_iter = learn_session["train_iter"]

        if evaluator1.should_eval(train_iter):
            # main_player =
            stop_flag1, reward, episode_info = evaluator1.eval(None, train_iter, learn_session["envstep"])
            win_loss_result = [e['result'] for e in episode_info[0]]

            task.emit("win_loss_result", win_loss_result)
            task.emit_remote("win_loss_result", win_loss_result)
            # # set fixed NE policy trueskill(exposure) equal 10
            # main_player.rating = league.metric_env.rate_1vsC(
            #     main_player.rating, league.metric_env.create_rating(mu=10, sigma=1e-8), win_loss_result
            # )
            # tb_logger.add_scalar('fixed_evaluator_step/reward_mean', reward, learn_session["envstep"])

    return _evaluate


def main():
    cfg = compile_config(
        league_demo_ppo_config,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        BattleEpisodeSerialCollector,
        BattleInteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    set_pkg_seed(0, use_cuda=cfg.policy.cuda)
    tb_logger = DistributedWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    league = DemoLeague(cfg.policy.other.league)
    policies = {}
    for player_id in league.active_players_ids:
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)
        policies[player_id] = policy

    # set load_checkpoint_fn in league, add a random policy player as reset player
    def load_checkpoint_fn(player_id: str, ckpt_path: str):
        state_dict = torch.load(ckpt_path)
        policies[player_id].learn_mode.load_state_dict(state_dict)

    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    policies['historical'] = policy
    torch.save(policies['historical'].learn_mode.state_dict(), league.reset_checkpoint_path)
    league.load_checkpoint = load_checkpoint_fn

    # snapshot the initial player as the first historial player
    for player_id, player_ckpt_path in zip(league.active_players_ids, league.active_players_ckpts):
        torch.save(policies[player_id].collect_mode.state_dict(), player_ckpt_path)
        league.judge_snapshot(player_id, force=True)

    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    policies['historical'] = policy

    with Task(async_mode=True, n_async_workers=4, auto_sync_ctx=False) as task:
        if task.match_labels(["standalone", "league"]):
            task.use(league_dispatching(task, cfg=cfg, tb_logger=tb_logger, league=league, policies=policies))
        if task.match_labels(["standalone", "collect"]):
            task.use(collecting(task, cfg=cfg, tb_logger=tb_logger, player_ids=league.active_players_ids))
        if task.match_labels(["standalone", "learn"]):
            task.use(
                learning(task, cfg=cfg, tb_logger=tb_logger, player_ids=league.active_players_ids, policies=policies)
            )
        if task.match_labels(["standalone", "evaluate"]):
            task.use(
                evaluating(task, cfg=cfg, tb_logger=tb_logger, player_ids=league.active_players_ids, policies=policies)
            )
        task.run(100)


if __name__ == "__main__":
    main()
