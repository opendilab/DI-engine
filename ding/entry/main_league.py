"""
BUG: Stop in 2 iters
"""
from ding.framework import Task, Context
from rich import print
import time
import torch
from ding.utils.default_helper import set_pkg_seed
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


def league_dispatching(task: Task, cfg, tb_logger, league, policies):

    def update_active_player(player_info):
        league.update_active_player(player_info)
        league.judge_snapshot(player_info["player_id"])

    task.on("update_active_player", update_active_player)
    task.router.register_rpc("update_active_player", update_active_player)

    def _league(ctx):
        import random
        num = random.random()
        print("League dispatching on node {}".format(num))
        time.sleep(1)
        # One episode each round
        i = ctx.total_step % len(league.active_players_ids)
        player_id, player_ckpt_path = league.active_players_ids[i], league.active_players_ckpts[i]

        job = league.get_job_info(player_id)
        print("League finish", num)
        opponent_player_id = job['player_id'][1]

        if 'historical' in opponent_player_id:
            opponent_policy = policies['historical'].collect_mode
            opponent_path = job['checkpoint_path'][1]
            opponent_policy.load_state_dict(torch.load(opponent_path, map_location='cpu'))
        else:
            opponent_policy = policies[opponent_player_id].collect_mode
        # Watch out that in parallel mode, we should not send functions between processes, instead,
        # we should send the objects needed by the policies.
        session = {
            "policies": [policies[player_id].collect_mode, opponent_policy],
            "player_id": player_id,
            "player_ckpt_path": player_ckpt_path
        }
        print("Player ID", session["player_id"])
        task.emit("set_session", session)

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
    session = None
    collectors = {}

    def set_session(new_session):
        nonlocal session
        session = new_session

    task.on("set_session", set_session)

    def _collect(ctx):
        print("  Collecting on node {}".format(task.router.node_id))
        time.sleep(1)

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

        # session = task.wait_for("set_session")
        nonlocal session
        while not session:  # Waiting
            time.sleep(0.1)

        collector = collectors[session["player_id"]]
        collector.reset_policy(session["policies"])
        train_data, episode_info = collector.collect()  # TODO Do we need train_iter?
        train_data, episode_info = train_data[0], episode_info[0]  # only use launch player data for training
        ctx.episode_info = episode_info
        for d in train_data:
            d['adv'] = d['reward']

        session["train_data"] = train_data
        session["envstep"] = collector.envstep
        task.emit("set_session", session)  # Shoot and forget
        del session["policies"]
        task.router.send_rpc("set_session", session)

        session = None

    return _collect


def learning(task: Task, cfg, tb_logger, player_ids, policies):
    learners = {}
    session = None

    def set_session(new_session):
        nonlocal session
        session = new_session

    task.on("set_session", set_session)
    task.router.register_rpc("set_session", set_session)

    def _learn(ctx):
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

        nonlocal session
        while not (session and session.get("train_data")):  # TODO: Use task.wait_for() to replace local variable
            time.sleep(0.1)

        learner = learners[session["player_id"]]
        state_dict = learner.policy.state_dict()
        torch.save(state_dict, session["player_ckpt_path"])  # Save to local

        player_info = learner.learn_info
        player_info['player_id'] = session["player_id"]

        task.router.send_rpc("update_active_player", player_info)  # Broadcast to other processes
        task.emit("update_active_player", player_info)  # Broadcast to other middleware

        session = None

    return _learn


def evaluating(task: Task, cfg, tb_logger):

    def _evaluate(ctx):
        print("      Evaluating on node {}".format(task.router.node_id))
        time.sleep(1)

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

    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    policies['historical'] = policy

    with Task(async_mode=True, n_async_workers=4) as task:
        task.use(
            league_dispatching(task, cfg=cfg, tb_logger=tb_logger, league=league, policies=policies),
            filter_labels=["standalone", "league"]
        )
        task.use(
            collecting(task, cfg=cfg, tb_logger=tb_logger, player_ids=league.active_players_ids),
            filter_labels=["standalone", "collect"]
        )
        task.use(
            learning(task, cfg=cfg, tb_logger=tb_logger, player_ids=league.active_players_ids, policies=policies),
            filter_labels=["standalone", "learn"]
        )
        task.use(evaluating(task, cfg=cfg, tb_logger=tb_logger), filter_labels=["standalone", "evaluate"])
        task.run(100)


if __name__ == "__main__":
    main()
    # Parallel.runner(n_parallel_workers=2, labels=["league"])(main)
