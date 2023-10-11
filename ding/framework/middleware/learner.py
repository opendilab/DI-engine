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
import torch.multiprocessing as mp
from threading import Thread
from ding.policy.common_utils import default_preprocess_learn, fast_preprocess_learn


def data_process_func(data_queue_input, data_queue_output):
    while True:
        data = data_queue_input.get()
        if data is None:
            break
        else:
            #print("get one data")
            output_data = fast_preprocess_learn(
                data,
                use_priority=False,  #policy._cfg.priority,
                use_priority_IS_weight=False,  #policy._cfg.priority_IS_weight,
                cuda=True,  #policy._cuda,
                device="cuda:0",  #policy._device,
            )
        data_queue_output.put(output_data)
        #print("put one data, queue size:{}".format(data_queue_output.qsize()))


def data_process_func_v2(data_queue_input, data_queue_output):
    while True:
        if data_queue_input.empty():
            time.sleep(0.001)
        else:
            data = data_queue_input.get()
            if data is None:
                break
            else:
                #print("get one data")
                output_data = fast_preprocess_learn(
                    data,
                    use_priority=False,  #policy._cfg.priority,
                    use_priority_IS_weight=False,  #policy._cfg.priority_IS_weight,
                    cuda=True,  #policy._cuda,
                    device="cuda:0",  #policy._device,
                )
            data_queue_output.put(output_data)
            #print("put one data, queue size:{}".format(data_queue_output.qsize()))


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
        start = time.time()
        time_fetcher = 0.0
        time_trainer = 0.0
        train_output_queue = []
        for _ in range(self.cfg.policy.learn.update_per_collect):
            start_fetcher = time.time()
            self._fetcher(ctx)
            time_fetcher += time.time() - start_fetcher
            if ctx.train_data is None:
                break
            if self._reward_estimator:
                self._reward_estimator(ctx)
            start_trainer = time.time()
            self._trainer(ctx)
            time_trainer += time.time() - start_trainer
            train_output_queue.append(ctx.train_output)
            ctx.train_output_for_post_process = ctx.train_output
        ctx.train_output = train_output_queue
        ctx.learner_time += time.time() - start
        print("time_trainer:time_fetcher={}:{}={}".format(time_trainer, time_fetcher, time_trainer / time_fetcher))


class OffPolicyLearnerV2:
    """
    Overview:
        The class of the off-policy learner, including data fetching and model training. Use \
            the `__call__` method to execute the whole learning process.
    """

    def __new__(cls, *args, **kwargs):
        if task.router.is_active and not task.has_role(task.role.LEARNER):
            return task.void()
        return super(OffPolicyLearnerV2, cls).__new__(cls)

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
        #self._data_queue_input = mp.Queue()
        #self._data_queue_output = mp.Queue()

        self._data_queue_input = Queue()
        self._data_queue_output = Queue()

        self.thread_worker = Thread(target=data_process_func_v2, args=(self._data_queue_input, self._data_queue_output))
        self.thread_worker.start()

        #self._fetcher_worker_process = mp.Process(target=data_process_func, args=(self._data_queue_input, self._data_queue_output))
        #self._fetcher_worker_process.start()

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
        start = time.time()
        time_fetcher = 0.0
        time_trainer = 0.0
        time_fetch_data = 0.0
        time_get_data = 0.0

        train_output_queue = []
        data_counter = 0

        start_fetcher = time.time()
        for _ in range(self.cfg.policy.learn.update_per_collect):
            start_fetch_data = time.time()
            self._fetcher(ctx)
            time_fetch_data += time.time() - start_fetch_data
            if ctx.train_data_sample is None:
                break
            self._data_queue_input.put(ctx.train_data_sample)
            data_counter += 1
        time_fetcher += time.time() - start_fetcher

        start_trainer = time.time()
        for _ in range(data_counter):
            start_get_data = time.time()
            while True:
                if self._data_queue_output.empty():
                    time.sleep(0.001)
                    continue
                else:
                    ctx.train_data = self._data_queue_output.get()
                    break
            time_get_data += time.time() - start_get_data
            if self._reward_estimator:
                self._reward_estimator(ctx)
            self._trainer(ctx)

            train_output_queue.append(ctx.train_output)
            ctx.train_output_for_post_process = ctx.train_output
        time_trainer += time.time() - start_trainer

        ctx.train_output = train_output_queue
        ctx.learner_time += time.time() - start
        #print("time_fetcher:time_fetch_data={}:{}={}".format(time_fetcher, time_fetch_data, time_fetcher / time_fetch_data))
        #print("time_trainer:time_get_data={}:{}={}".format(time_trainer, time_get_data, time_trainer / time_get_data))
        #print("time_trainer:time_fetcher={}:{}={}".format(time_trainer, time_fetcher, time_trainer / time_fetcher))


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
