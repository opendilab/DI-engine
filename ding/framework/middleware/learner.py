from typing import TYPE_CHECKING, Callable, List, Tuple, Union, Dict, Optional
from easydict import EasyDict
from collections import deque

from ding.framework import task
from ding.data import Buffer
from .functional import trainer, offpolicy_data_fetcher, reward_estimator, her_data_enhancer
from ditk import logging
from ding.utils import DistributedWriter

if TYPE_CHECKING:
    from ding.framework import Context, OnlineRLContext

import time


class OffPolicyLearner:
    """
    Overview:
        The class of the off-policy learner, including data fetching and model training. Use \
            the `__call__` method to execute the whole learning process.
    """

    def __init__(
            self,
            cfg: EasyDict,
            policy,
            buffer_: Union[Buffer, List[Tuple[Buffer, float]], Dict[str, Buffer]],
            reward_model=None
    ) -> None:
        """
        Arguments:
            - cfg (:obj:`EasyDict`): Config.
            - policy (:obj:`Policy`): The policy to be trained.
            - buffer\_ (:obj:`Buffer`): The replay buffer to store the data for training.
            - reward_model (:obj:`nn.Module`): Additional reward estimator likes RND, ICM, etc. \
                default to None.
        """
        self.cfg = cfg
        self._fetcher = task.wrap(offpolicy_data_fetcher(cfg, buffer_))
        self._trainer = task.wrap(trainer(cfg, policy))
        if reward_model is not None:
            self._reward_estimator = task.wrap(reward_estimator(cfg, reward_model))
        else:
            self._reward_estimator = None

        self.last_iter_time = None
        self.total_iter_time = 0
        self.total_trainning_time = 0

        self.last_train_iter = 0

        self._writer = DistributedWriter.get_instance()

    def __call__(self, ctx: "OnlineRLContext") -> None:
        """
        Output of ctx:
            - train_output (:obj:`Deque`): The training output in deque.
        """
        if self.last_iter_time is None:
            self.last_iter_time = time.time()
        begin_trainning_time = time.time()

        train_output_queue = []
        for _ in range(self.cfg.policy.learn.update_per_collect):
            self._fetcher(ctx)
            if ctx.train_data is None:
                break
            if self._reward_estimator:
                self._reward_estimator(ctx)
            self._trainer(ctx)
            train_output_queue.append(ctx.train_output)
        ctx.train_output = train_output_queue

        finish_iter_time = time.time()
        current_iter_time = finish_iter_time - self.last_iter_time
        self.total_iter_time += current_iter_time

        self.total_trainning_time += finish_iter_time - begin_trainning_time
        self.last_iter_time = finish_iter_time

        if ctx.train_iter == 0:
            self.total_iter_time = 0
            self.total_trainning_time = 0
        elif ctx.train_iter > self.last_train_iter:
            logging.info(
                "[Learner {}] total epoch speed is {} iter/s, current epoch speed is {} iter/s, total_training speed is {} iter/s, current training speed is {} iter/s"
                .format(
                    task.router.node_id, ctx.train_iter / self.total_iter_time,
                    (ctx.train_iter - self.last_train_iter) / current_iter_time,
                    ctx.train_iter / self.total_trainning_time,
                    (ctx.train_iter - self.last_train_iter) / (finish_iter_time - begin_trainning_time)
                )
            )
            self._writer.add_scalar(
                "total_epoch_speed____train_iter/s-train_iter", ctx.train_iter / self.total_iter_time, ctx.train_iter
            )
            self._writer.add_scalar(
                "current_epoch_speed____train_iter/s-train_iter",
                (ctx.train_iter - self.last_train_iter) / current_iter_time, ctx.train_iter
            )
            self._writer.add_scalar(
                "total_trainning_speed____train_iter/s-train_iter", 
                ctx.train_iter / self.total_trainning_time, ctx.train_iter
            )
            self._writer.add_scalar(
                "current_trainning_speed____train_iter/s-train_iter",
                (ctx.train_iter - self.last_train_iter) / (finish_iter_time - begin_trainning_time), ctx.train_iter
            )
            self.last_train_iter = ctx.train_iter


class HERLearner:
    """
    Overview:
        The class of the learner with the Hindsight Experience Replay (HER). \
            Use the `__call__` method to execute the data featching and training \
            process.
    """

    def __init__(
            self,
            cfg: EasyDict,
            policy,
            buffer_: Union[Buffer, List[Tuple[Buffer, float]], Dict[str, Buffer]],
            her_reward_model,
    ) -> None:
        """
        Arguments:
            - cfg (:obj:`EasyDict`): Config.
            - policy (:obj:`Policy`): The policy to be trained.
            - buffer\_ (:obj:`Buffer`): The replay buffer to store the data for training.
            - her_reward_model (:obj:`HerRewardModel`): HER reward model.
        """
        self.cfg = cfg
        self._fetcher = task.wrap(her_data_enhancer(cfg, buffer_, her_reward_model))
        self._trainer = task.wrap(trainer(cfg, policy))

    def __call__(self, ctx: "OnlineRLContext") -> None:
        """
        Output of ctx:
            - train_output (:obj:`Deque`): The deque of training output.
        """
        train_output_queue = []
        for _ in range(self.cfg.policy.learn.update_per_collect):
            self._fetcher(ctx)
            if ctx.train_data is None:
                break
            self._trainer(ctx)
            train_output_queue.append(ctx.train_output)
        ctx.train_output = train_output_queue
