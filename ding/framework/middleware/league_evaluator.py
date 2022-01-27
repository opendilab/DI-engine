import torch
import copy
import numpy as np
from ding.envs.env_manager.base_env_manager import BaseEnvManager
from ding.worker.collector.battle_interaction_serial_evaluator import BattleInteractionSerialEvaluator
from dizoo.league_demo.game_env import GameEnv
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from ding.framework import Task, Context
    from ding.utils.log_writer_helper import DistributedWriter


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


def league_evaluator(
    task: "Task", cfg: dict, tb_logger: "DistributedWriter", player_ids: List[str], policies: List[str]
):
    evaluator_env1, eval_policy1, evaluator1 = None, None, None
    learn_session = None

    def set_learn_session(remote_learn_session):
        if "main_player" in remote_learn_session["player_id"]:
            nonlocal learn_session
            learn_session = remote_learn_session

    task.on("set_learn_session", set_learn_session)

    def _evaluate(ctx: "Context"):
        print("      Evaluating on node {}".format(task.router.node_id))
        import time
        time.sleep(10)

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
            # TODO Handle these events in league
            # # set fixed NE policy trueskill(exposure) equal 10
            # main_player.rating = league.metric_env.rate_1vsC(
            #     main_player.rating, league.metric_env.create_rating(mu=10, sigma=1e-8), win_loss_result
            # )
            # tb_logger.add_scalar('fixed_evaluator_step/reward_mean', reward, learn_session["envstep"])

    return _evaluate
