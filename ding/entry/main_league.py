from ding.framework import Task
import logging
import os
import torch

from ding.config import compile_config
from ding.worker import BaseLearner, BattleEpisodeSerialCollector, BattleInteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import BaseEnvManager
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from dizoo.league_demo.demo_league import DemoLeague
from dizoo.league_demo.league_demo_ppo_config import league_demo_ppo_config
from ding.utils import DistributedWriter
from ding.framework.middleware import league_learner, league_evaluator, league_dispatcher, league_collector


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

    with Task(async_mode=True, n_async_workers=3, auto_sync_ctx=False) as task:
        if not task.router.is_active:
            logging.info("League should be executed in parallel mode, use `main_league.sh` to execute league!")
            exit(1)
        if task.match_labels(["league"]):
            task.use(league_dispatcher(task, cfg=cfg, tb_logger=tb_logger, league=league, policies=policies))
        if task.match_labels(["collect"]):
            task.use(league_collector(task, cfg=cfg, tb_logger=tb_logger, player_ids=league.active_players_ids))
        if task.match_labels(["learn"]):
            # Distribute learners on different nodes
            player_idx = task.router.node_id % len(league.active_players_ids)
            task.use(
                league_learner(
                    task,
                    cfg=cfg,
                    tb_logger=tb_logger,
                    player_id=league.active_players_ids[player_idx],
                    policies=policies
                )
            )
        if task.match_labels(["evaluate"]):
            task.use(
                league_evaluator(
                    task, cfg=cfg, tb_logger=tb_logger, player_ids=league.active_players_ids, policies=policies
                )
            )
        task.run(100)


if __name__ == "__main__":
    main()
