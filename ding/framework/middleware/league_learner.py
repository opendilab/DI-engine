import os
import logging
from dataclasses import dataclass
from threading import Lock
from time import sleep
from typing import TYPE_CHECKING, Callable, Optional

from ding.framework import task, EventEnum
from ding.framework.storage import Storage, FileStorage
from ding.league.player import PlayerMeta
from ding.worker.learner.base_learner import BaseLearner

if TYPE_CHECKING:
    from ding.framework import Context
    from ding.framework.middleware.league_actor import ActorData
    from ding.league import ActivePlayer

@dataclass
class LearnerModel:
    player_id: str
    state_dict: dict
    train_iter: int = 0

class LeagueLearner:

    def __init__(self, cfg: dict, policy_fn: Callable, player: "ActivePlayer") -> None:
        self.cfg = cfg
        self.policy_fn = policy_fn
        self.player = player
        self.player_id = player.player_id
        self.checkpoint_prefix = cfg.policy.other.league.path_policy
        self._learner = self._get_learner()
        self._lock = Lock()
        task.on(EventEnum.ACTOR_SEND_DATA.format(player=self.player_id), self._on_actor_data)
        self._step = 0

    def _on_actor_data(self, actor_data: "ActorData"):
        print("receive data from actor!")
        with self._lock:
            cfg = self.cfg
            for _ in range(cfg.policy.learn.update_per_collect):
                print("train model")
                self._learner.train(actor_data.train_data, actor_data.env_step)

        self.player.total_agent_step = self._learner.train_iter
        print("save checkpoint")
        checkpoint = self._save_checkpoint() if self.player.is_trained_enough() else None
        task.emit(
            EventEnum.LEARNER_SEND_META,
            PlayerMeta(player_id=self.player_id, checkpoint=checkpoint, total_agent_step=self._learner.train_iter)
        )

        print("pack model")
        learner_model = LearnerModel(
            player_id=self.player_id, state_dict=self._learner.policy.state_dict(), train_iter=self._learner.train_iter
        )
        task.emit(EventEnum.LEARNER_SEND_MODEL, learner_model)

    def _get_learner(self) -> BaseLearner:
        policy = self.policy_fn().learn_mode
        learner = BaseLearner(
            self.cfg.policy.learn.learner,
            policy,
            exp_name=self.cfg.exp_name,
            instance_name=self.player_id + '_learner'
        )
        return learner

    def _save_checkpoint(self) -> Optional[Storage]:
        if not os.path.exists(self.checkpoint_prefix):
            os.makedirs(self.checkpoint_prefix)
        storage = FileStorage(
            path=os.path.join(self.checkpoint_prefix, "{}_{}_ckpt.pth".format(self.player_id, self._learner.train_iter))
        )
        storage.save(self._learner.policy.state_dict())
        return storage

    def __call__(self, _: "Context") -> None:
        sleep(1)
        logging.info("{} Step: {}".format(self.__class__, self._step))
        self._step += 1
