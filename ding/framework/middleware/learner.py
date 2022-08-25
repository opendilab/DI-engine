from typing import TYPE_CHECKING, Callable, List, Tuple, Union, Dict, Optional
from easydict import EasyDict
from collections import deque

from ding.framework import task
from ding.data import Buffer
from .functional import trainer, offpolicy_data_fetcher, reward_estimator, her_data_enhancer

if TYPE_CHECKING:
    from ding.framework import Context, OnlineRLContext


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

    def __call__(self, ctx: "OnlineRLContext") -> None:
        """
        Output of ctx:
            - train_output (:obj:`Deque`): The training output in deque.
        """
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
