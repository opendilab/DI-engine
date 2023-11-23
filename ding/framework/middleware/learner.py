from typing import TYPE_CHECKING, Callable, List, Tuple, Union, Dict, Optional
from easydict import EasyDict
from collections import deque

from ding.framework import task
from ding.data import Buffer
from .functional import trainer, offpolicy_data_fetcher, reward_estimator, her_data_enhancer, offpolicy_data_fetcher_v2

if TYPE_CHECKING:
    from ding.framework import Context, OnlineRLContext
    from ding.policy import Policy
    from ding.reward_model import BaseRewardModel

from queue import Queue
import time
from threading import Thread
from ding.policy.common_utils import fast_preprocess_learn


def data_process_func(
    data_queue_input: Queue,
    data_queue_output: Queue,
    use_priority: bool = False,
    use_priority_IS_weight: bool = False,
    use_nstep: bool = False,
    cuda: bool = True,
    device: str = "cuda:0",
):
    while True:
        if data_queue_input.empty():
            time.sleep(0.001)
        else:
            data = data_queue_input.get()
            if data is None:
                break
            else:
                output_data = fast_preprocess_learn(
                    data,
                    use_priority=use_priority,
                    use_priority_IS_weight=use_priority_IS_weight,
                    use_nstep=use_nstep,
                    cuda=cuda,
                    device=device,
                )
            data_queue_output.put(output_data)


class OffPolicyLearner:
    """
    Overview:
        The class of the off-policy learner, including data fetching and model training. Use \
            the `__call__` method to execute the whole learning process.
    """

    def __new__(cls, *args, **kwargs):
        if task.router.is_active and not task.has_role(task.role.LEARNER):
            return task.void()
        return super(OffPolicyLearner, cls).__new__(cls)

    def __init__(
            self,
            cfg: EasyDict,
            policy: 'Policy',
            buffer_: Union[Buffer, List[Tuple[Buffer, float]], Dict[str, Buffer]],
            reward_model: Optional['BaseRewardModel'] = None,
            log_freq: int = 100,
    ) -> None:
        """
        Arguments:
            - cfg (:obj:`EasyDict`): Config.
            - policy (:obj:`Policy`): The policy to be trained.
            - buffer (:obj:`Buffer`): The replay buffer to store the data for training.
            - reward_model (:obj:`BaseRewardModel`): Additional reward estimator likes RND, ICM, etc. \
                default to None.
            - log_freq (:obj:`int`): The frequency (iteration) of showing log.
        """
        self.cfg = cfg
        self._fetcher = task.wrap(offpolicy_data_fetcher(cfg, buffer_))
        self._trainer = task.wrap(trainer(cfg, policy, log_freq=log_freq))
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
            ctx.train_output_for_post_process = ctx.train_output
        ctx.train_output = train_output_queue


class EnvpoolOffPolicyLearner:
    """
    Overview:
        The class of the off-policy learner, including data fetching and model training. Use \
            the `__call__` method to execute the whole learning process.
    """

    def __new__(cls, *args, **kwargs):
        if task.router.is_active and not task.has_role(task.role.LEARNER):
            return task.void()
        return super(EnvpoolOffPolicyLearner, cls).__new__(cls)

    def __init__(
            self,
            cfg: EasyDict,
            policy: 'Policy',
            buffer_: Union[Buffer, List[Tuple[Buffer, float]], Dict[str, Buffer]],
            reward_model: Optional['BaseRewardModel'] = None,
            log_freq: int = 100,
    ) -> None:
        """
        Arguments:
            - cfg (:obj:`EasyDict`): Config.
            - policy (:obj:`Policy`): The policy to be trained.
            - buffer (:obj:`Buffer`): The replay buffer to store the data for training.
            - reward_model (:obj:`BaseRewardModel`): Additional reward estimator likes RND, ICM, etc. \
                default to None.
            - log_freq (:obj:`int`): The frequency (iteration) of showing log.
        """
        self.cfg = cfg

        self._fetcher = task.wrap(offpolicy_data_fetcher_v2(cfg, buffer_))

        self._data_queue_input = Queue()
        self._data_queue_output = Queue()

        self.thread_worker = Thread(
            target=data_process_func,
            args=(
                self._data_queue_input,
                self._data_queue_output,
                cfg.policy.priority,
                cfg.policy.priority_IS_weight,
                cfg.policy.nstep > 1,
                cfg.policy.cuda,
                policy._device,
            )
        )
        self.thread_worker.start()

        self._trainer = task.wrap(trainer(cfg, policy.learn_mode, log_freq=log_freq))
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
        data_counter = 0
        for _ in range(self.cfg.policy.learn.update_per_collect):
            self._fetcher(ctx)
            if ctx.train_data_sample is None:
                break
            self._data_queue_input.put(ctx.train_data_sample)
            data_counter += 1

        for _ in range(data_counter):
            while True:
                if self._data_queue_output.empty():
                    time.sleep(0.001)
                    continue
                else:
                    ctx.train_data = self._data_queue_output.get()
                    break
            if self._reward_estimator:
                self._reward_estimator(ctx)
            self._trainer(ctx)

            train_output_queue.append(ctx.train_output)
            ctx.train_output_for_post_process = ctx.train_output

        ctx.train_output = train_output_queue

        yield

        if task.finish:
            self._data_queue_input.put(None)
            self.thread_worker.join()


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
