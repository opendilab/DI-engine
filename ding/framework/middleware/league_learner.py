from ditk import logging
import os
from dataclasses import dataclass
from collections import deque
from threading import Lock
from time import time, sleep
from typing import TYPE_CHECKING, Callable, Optional

from ding.data.buffer.deque_buffer import DequeBuffer
from ding.framework import task, EventEnum
from ding.framework.middleware import OffPolicyLearner, CkptSaver, data_pusher
from ding.framework.storage import Storage, FileStorage
from ding.league.player import PlayerMeta
from ding.utils import DistributedWriter
from ding.utils.sparse_logging import log_every_sec
from ding.worker.learner.base_learner import BaseLearner
from ding.torch_utils import to_device

if TYPE_CHECKING:
    from ding.policy import Policy
    from ding.framework import Context, BattleContext
    from ding.framework.middleware.league_actor import ActorData
    from ding.league import ActivePlayer

PRE_RECV_TIME = 10 * 60
SEND_MODEL_FREQ_TIME = 30

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

        self._writer = DistributedWriter.get_instance()
        
        self.traj_num = 0
        self.total_recv_time = 0
        self.last_time = None
        self.pre_collect_finished = False

        self.last_train_iter = 0
        self.total_network_delay = 0

        self.last_send_model_time = None
        self.total_send_model_time = 0
        self.send_model_num = 0

    def _push_data(self, data: "ActorData"):
        if self.last_time is None:
            self.last_time = time()

        current_recv_traj = 0

        for env_trajectories in data.train_data:
            for traj in env_trajectories.trajectories:
                self._cache.append(traj)
                current_recv_traj += 1

        if self.total_recv_time >= PRE_RECV_TIME and self.pre_collect_finished is False:
            self.pre_collect_finished = True
            self.total_recv_time = 0
            self.traj_num = 0

        if current_recv_traj > 0:
            self.traj_num += current_recv_traj
            this_time = time()
            current_time = this_time - self.last_time
            self.total_recv_time += current_time
            self.last_time = this_time

            # TODO
            network_delay = this_time - data.meta.send_wall_time
            self.total_network_delay += network_delay

            if self.pre_collect_finished:
                log_every_sec(
                    logging.INFO, 5,
                    "[Learner {}] receive {} trajectories of player {} from actor! Current recv speed: {} traj/s, Total recv speed: {} traj/s, Current network_delay: {} traj/s, Total network_delay: {} traj/s \n"
                    .format(
                        task.router.node_id,
                        current_recv_traj,
                        self.player_id,
                        current_recv_traj / current_time,
                        self.traj_num /self.total_recv_time,
                        current_recv_traj / network_delay,
                        self.traj_num / self.total_network_delay,
                    )
                )

                self._writer.add_scalar(
                    "current_recv_traj_speed____traj/s-traj",
                    current_recv_traj / current_time, 
                    self.traj_num
                )
                self._writer.add_scalar(
                    "traj_num_speed____traj/s-traj", 
                    self.traj_num /self.total_recv_time,
                    self.traj_num
                )
                self._writer.add_scalar(
                    "current_network_delay____traj/s-traj", 
                    current_recv_traj / network_delay, 
                    self.traj_num
                )
                self._writer.add_scalar(
                    "total_network_delay____traj/s-traj", 
                    self.traj_num / self.total_network_delay,
                    self.traj_num
                )
                
            else:
                print('we are now pre collecting')
                print(self.total_recv_time)
        # if isinstance(data.train_data, list):
        #     self._cache.extend(data.train_data)
        # else:
        #     self._cache.append(data.train_data)

    def __call__(self, ctx: "BattleContext"):
        # log_every_sec(logging.INFO, 5, "[Learner {}] pour data into the ctx".format(task.router.node_id))
        ctx.trajectories = list(self._cache)
        if ctx.train_iter > self.last_train_iter:
            self.last_train_iter = ctx.train_iter
            logging.info(
                '[Learner {}] cache size is {}, train_iter is {}'.format(
                    task.router.node_id, len(self._cache), ctx.train_iter
                )
            )
            self._writer.add_scalar("cache_length-train_iter", len(self._cache), ctx.train_iter)
        self._cache.clear()
        sleep(0.0001)
        yield
        if self.last_send_model_time is None:
            self.last_send_model_time = time()
        
        this_send_model_time = time()
        current_send_model_time = this_send_model_time - self.last_send_model_time
        self.total_send_model_time += current_send_model_time
        self.last_send_model_time = this_send_model_time

        log_every_sec(logging.INFO, 20, "[Learner {}] ctx.train_iter {}".format(task.router.node_id, ctx.train_iter))
        self.player.total_agent_step = ctx.train_iter
        
        if self.total_send_model_time // SEND_MODEL_FREQ_TIME > self.send_model_num:
            logging.info('{1} [Learner {0}] send model! {1} \n\n'.format(task.router.node_id, "-" * 40))
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
            self.send_model_num = self.total_send_model_time // SEND_MODEL_FREQ_TIME


# class OffPolicyLeagueLearner:

#     def __init__(self, cfg: dict, policy_fn: Callable, player: "ActivePlayer") -> None:
#         self._buffer = DequeBuffer(size=10000)
#         self._policy = policy_fn().learn_mode
#         self.player_id = player.player_id
#         task.on(EventEnum.ACTOR_SEND_DATA.format(player=self.player_id), self._push_data)
#         self._learner = OffPolicyLearner(cfg, self._policy, self._buffer)
#         # self._ckpt_handler = CkptSaver(cfg, self._policy, train_freq=100)

#     def _push_data(self, data: "ActorData"):
#         print("push data into the buffer!")
#         self._buffer.push(data.train_data)

#     def __call__(self, ctx: "Context"):
#         print("num of objects in buffer:", self._buffer.count())
#         self._learner(ctx)
#         checkpoint = None

#         sleep(2)
#         print('learner send player meta\n', flush=True)
#         task.emit(
#             EventEnum.LEARNER_SEND_META,
#             PlayerMeta(player_id=self.player_id, checkpoint=checkpoint, total_agent_step=0)
#         )

#         learner_model = LearnerModel(
#             player_id=self.player_id,
#             state_dict=self._policy.state_dict(),
#             train_iter=ctx.train_iter  # self._policy.state_dict()
#         )
#         print('learner send model\n', flush=True)
#         task.emit(EventEnum.LEARNER_SEND_MODEL, learner_model)

# class LeagueLearner:

#     def __init__(self, cfg: dict, policy_fn: Callable, player: "ActivePlayer") -> None:
#         self.cfg = cfg
#         self.policy_fn = policy_fn
#         self.player = player
#         self.player_id = player.player_id
#         self.checkpoint_prefix = cfg.policy.other.league.path_policy
#         self._learner = self._get_learner()
#         self._lock = Lock()
#         task.on(EventEnum.ACTOR_SEND_DATA.format(player=self.player_id), self._on_actor_send_data)
#         self._step = 0

#     def _on_actor_send_data(self, actor_data: "ActorData"):
#         logging.info("learner {} receive data from actor! \n".format(task.router.node_id), flush=True)
#         with self._lock:
#             cfg = self.cfg
#             for _ in range(cfg.policy.learn.update_per_collect):
#                 pass
#                 # print("train model")
#                 # print(actor_data.train_data)
#                 # self._learner.train(actor_data.train_data, actor_data.env_step)

#         self.player.total_agent_step = self._learner.train_iter
#         # print("save checkpoint")
#         checkpoint = self._save_checkpoint() if self.player.is_trained_enough() else None

#         print('learner {} send player meta {}\n'.format(task.router.node_id, self.player_id), flush=True)
#         task.emit(
#             EventEnum.LEARNER_SEND_META,
#             PlayerMeta(player_id=self.player_id, checkpoint=checkpoint, total_agent_step=self._learner.train_iter)
#         )

#         # print("pack model")
#         learner_model = LearnerModel(
#             player_id=self.player_id, state_dict=self._learner.policy.state_dict(), train_iter=self._learner.train_iter
#         )
#         print('learner {} send model\n'.format(task.router.node_id), flush=True)
#         task.emit(EventEnum.LEARNER_SEND_MODEL, learner_model)

#     def _get_learner(self) -> BaseLearner:
#         policy = self.policy_fn().learn_mode
#         learner = BaseLearner(
#             self.cfg.policy.learn.learner,
#             policy,
#             exp_name=self.cfg.exp_name,
#             instance_name=self.player_id + '_learner'
#         )
#         return learner

#     def _save_checkpoint(self) -> Optional[Storage]:
#         if not os.path.exists(self.checkpoint_prefix):
#             os.makedirs(self.checkpoint_prefix)
#         storage = FileStorage(
#             path=os.path.
#             join(self.checkpoint_prefix, "{}_{}_ckpt.pth".format(self.player_id, self._learner.train_iter))
#         )
#         storage.save(self._learner.policy.state_dict())
#         return storage

#     def __del__(self):
#         print('task finished, learner {} closed\n'.format(task.router.node_id), flush=True)

#     def __call__(self, _: "Context") -> None:
#         sleep(1)
