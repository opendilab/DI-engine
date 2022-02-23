import torch
from ding.worker.learner.base_learner import BaseLearner
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from ding.framework import Task, Context
    from ding.utils.log_writer_helper import DistributedWriter


def league_learner(task: "Task", cfg: dict, tb_logger: "DistributedWriter", player_id: str, policies: List[str]):
    learner = None

    task.emit("learner_online", player_id)

    def _learn(ctx: "Context"):
        learn_session = task.wait_for("set_learn_session")[0][0]
        print("    Learning on node: {}, player: {}".format(task.router.node_id, player_id))
        if learn_session["player_id"] != player_id:
            return

        nonlocal learner
        if not learner:
            policy = policies[player_id]
            learner = BaseLearner(
                cfg.policy.learn.learner,
                policy.learn_mode,
                tb_logger=tb_logger,
                exp_name=cfg.exp_name,
                instance_name=player_id + '_learner'
            )

        for _ in range(cfg.policy.learn.update_per_collect):
            learner.train(learn_session["train_data"], learn_session["envstep"])

        state_dict = learner.policy.state_dict()
        torch.save(state_dict, learn_session["player_ckpt_path"])  # Save to local

        player_info = learner.learn_info
        player_info['player_id'] = learn_session["player_id"]
        player_info["train_iter"] = learner.train_iter

        task.emit("update_active_player", player_info)  # Broadcast to other middleware

    return _learn
