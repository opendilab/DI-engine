from ditk import logging
import os
from dataclasses import dataclass
from collections import deque
from time import sleep
from typing import TYPE_CHECKING

from ding.framework import task, EventEnum
from ding.framework.storage import FileStorage
from ding.league.player import PlayerMeta
from ding.utils.sparse_logging import log_every_sec

if TYPE_CHECKING:
    from ding.policy import Policy
    from ding.framework import BattleContext
    from ding.framework.middleware.league_actor import ActorData
    from ding.league import ActivePlayer


@dataclass
class LearnerModel:
    player_id: str
    state_dict: dict
    train_iter: int = 0


class LeagueLearnerCommunicator:

    def __init__(self, cfg: dict, policy: "Policy", player: "ActivePlayer") -> None:
        self.cfg = cfg
        self._cache = deque(maxlen=20)
        self.player = player
        self.player_id = player.player_id
        self.policy = policy
        self.prefix = '{}/ckpt'.format(cfg.exp_name)
        if not os.path.exists(self.prefix):
            os.makedirs(self.prefix)
        task.on(EventEnum.ACTOR_SEND_DATA.format(player=self.player_id), self._push_data)

    def _push_data(self, data: "ActorData"):
        log_every_sec(
            logging.INFO, 5,
            "[Learner {}] receive data of player {} from actor! \n".format(task.router.node_id, self.player_id)
        )
        for env_trajectories in data.train_data:
            for traj in env_trajectories.trajectories:
                self._cache.append(traj)

    def __call__(self, ctx: "BattleContext"):
        ctx.trajectories = list(self._cache)
        self._cache.clear()
        sleep(0.0001)
        yield
        log_every_sec(logging.INFO, 20, "[Learner {}] ctx.train_iter {}".format(task.router.node_id, ctx.train_iter))
        self.player.total_agent_step = ctx.train_iter
        if self.player.is_trained_enough():
            logging.info('{1} [Learner {0}] trained enough! {1} \n\n'.format(task.router.node_id, "-" * 40))
            storage = FileStorage(
                path=os.path.join(self.prefix, "{}_{}_ckpt.pth".format(self.player_id, ctx.train_iter))
            )
            storage.save(self.policy.state_dict())
            task.emit(
                EventEnum.LEARNER_SEND_META,
                PlayerMeta(player_id=self.player_id, checkpoint=storage, total_agent_step=ctx.train_iter)
            )

            learner_model = LearnerModel(
                player_id=self.player_id, state_dict=self.policy.state_dict(), train_iter=ctx.train_iter
            )
            task.emit(EventEnum.LEARNER_SEND_MODEL, learner_model)
