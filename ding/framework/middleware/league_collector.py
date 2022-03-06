from time import sleep
from ding.envs.env_manager.base_env_manager import BaseEnvManager
from ding.worker.collector.battle_episode_serial_collector import BattleEpisodeSerialCollector
from dizoo.league_demo.game_env import GameEnv
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from ding.framework import Task, Context
    from ding.utils.log_writer_helper import DistributedWriter


def league_collector(task: "Task", cfg: dict, tb_logger: "DistributedWriter", player_ids: List[str]):
    collectors = {}

    def _collect(ctx: "Context"):
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
        task.emit("set_learn_session", learn_session)  # Shoot and forget

    return _collect
