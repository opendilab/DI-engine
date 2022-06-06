from dataclasses import dataclass
from time import sleep
from os import path as osp
from ding.framework import task, EventEnum
from ding.framework.storage import Storage, FileStorage
from ding.league.player import PlayerMeta
from ding.worker.learner.base_learner import BaseLearner
from typing import TYPE_CHECKING, Callable, Optional
from threading import Lock
if TYPE_CHECKING:
    from ding.framework import Context
    from ding.framework.middleware.league_actor import ActorData
    from ding.league import ActivePlayer


@dataclass
class LearnerModel:
    player_id: str
    state_dict: dict
    train_iter: int = 0


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
        task.on(EventEnum.ACTOR_SEND_DATA.format(player=self.player_id), self._on_actor_data)
        self._lock = Lock()

    def _on_actor_data(self, actor_data: "ActorData"):
        with self._lock:
            cfg = self.cfg
            for _ in range(cfg.policy.learn.update_per_collect):
                self._learner.train(actor_data.train_data, actor_data.env_step)

        self.player.total_agent_step = self._learner.train_iter
        checkpoint = self._save_checkpoint() if self.player.is_trained_enough() else None
        task.emit(
            EventEnum.LEARNER_SEND_META,
            PlayerMeta(player_id=self.player_id, checkpoint=checkpoint, total_agent_step=self._learner.train_iter)
        )

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
        storage = FileStorage(
            path=osp.join(self.checkpoint_prefix, "{}_{}_ckpt.pth".format(self.player_id, self._learner.train_iter))
        )
        storage.save(self._learner.policy.state_dict())
        return storage

    def __call__(self, _: "Context") -> None:
        sleep(1)
