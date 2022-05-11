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
        The middleware that executes the whole learning process, including data fetching and model training.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - policy (:obj:`Policy`): The policy to be trained.
        - buffer_: The replay buffer to store the data for training.
        - reward_model: The reward estimator in model-based setting, default to None.
    """

    def __init__(
            self,
            cfg: EasyDict,
            policy,
            buffer_: Union[Buffer, List[Tuple[Buffer, float]], Dict[str, Buffer]],
            reward_model=None
    ) -> None:
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
            - train_output (:obj:`Deque`): The deque of training output.
        """

        train_output_queue = deque()
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
        The middleware that executes the whole learning process with HER reward.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - policy (:obj:`Policy`): The policy to be trained.
        - buffer_: The replay buffer to store the data for training.
        - her_reward_model: HER reward model.
    """

    def __init__(
            self,
            cfg: EasyDict,
            policy,
            buffer_: Union[Buffer, List[Tuple[Buffer, float]], Dict[str, Buffer]],
            her_reward_model,
    ) -> None:
        self.cfg = cfg
        self._fetcher = task.wrap(her_data_enhancer(cfg, buffer_, her_reward_model))
        self._trainer = task.wrap(trainer(cfg, policy))

    def __call__(self, ctx: "OnlineRLContext") -> None:
        """
        Output of ctx:
            - train_output (:obj:`Deque`): The deque of training output.
        """

        train_output_queue = deque()
        for _ in range(self.cfg.policy.learn.update_per_collect):
            self._fetcher(ctx)
            if ctx.train_data is None:
                break
            self._trainer(ctx)
            train_output_queue.append(ctx.train_output)
        ctx.train_output = train_output_queue
