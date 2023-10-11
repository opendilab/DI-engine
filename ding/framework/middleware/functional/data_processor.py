import os
from typing import TYPE_CHECKING, Callable, List, Union, Tuple, Dict, Optional
from easydict import EasyDict
from ditk import logging
import torch
from ding.data import Buffer, Dataset, DataLoader, offline_data_save_type
from ding.data.buffer.middleware import PriorityExperienceReplay
from ding.framework import task
from ding.utils import get_rank

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext, OfflineRLContext


def data_pusher(cfg: EasyDict, buffer_: Buffer, group_by_env: Optional[bool] = None):
    """
    Overview:
        Push episodes or trajectories into the buffer.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - buffer (:obj:`Buffer`): Buffer to push the data in.
    """
    if task.router.is_active and not task.has_role(task.role.LEARNER):
        return task.void()

    def _push(ctx: "OnlineRLContext"):
        """
        Overview:
            In ctx, either `ctx.trajectories` or `ctx.episodes` should not be None.
        Input of ctx:
            - trajectories (:obj:`List[Dict]`): Trajectories.
            - episodes (:obj:`List[Dict]`): Episodes.
        """

        if ctx.trajectories is not None:  # each data in buffer is a transition
            if group_by_env:
                for i, t in enumerate(ctx.trajectories):
                    buffer_.push(t, {'env': t.env_data_id.item()})
            else:
                for t in ctx.trajectories:
                    buffer_.push(t)
            ctx.trajectories = None
        elif ctx.episodes is not None:  # each data in buffer is a episode
            for t in ctx.episodes:
                buffer_.push(t)
            ctx.episodes = None
        else:
            raise RuntimeError("Either ctx.trajectories or ctx.episodes should be not None.")

    return _push


def buffer_saver(cfg: EasyDict, buffer_: Buffer, every_envstep: int = 1000, replace: bool = False):
    """
    Overview:
        Save current buffer data.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - buffer (:obj:`Buffer`): Buffer to push the data in.
        - every_envstep (:obj:`int`): save at every env step.
        - replace (:obj:`bool`): Whether replace the last file.
    """

    buffer_saver_env_counter = -every_envstep

    def _save(ctx: "OnlineRLContext"):
        """
        Overview:
            In ctx, `ctx.env_step` should not be None.
        Input of ctx:
            - env_step (:obj:`int`): env step.
        """
        nonlocal buffer_saver_env_counter
        if ctx.env_step is not None:
            if ctx.env_step >= every_envstep + buffer_saver_env_counter:
                buffer_saver_env_counter = ctx.env_step
                if replace:
                    buffer_.save_data(os.path.join(cfg.exp_name, "replaybuffer", "data_latest.hkl"))
                else:
                    buffer_.save_data(
                        os.path.join(cfg.exp_name, "replaybuffer", "data_envstep_{}.hkl".format(ctx.env_step))
                    )
        else:
            raise RuntimeError("buffer_saver only supports collecting data by step rather than episode.")

    return _save


def offpolicy_data_fetcher(
        cfg: EasyDict,
        buffer_: Union[Buffer, List[Tuple[Buffer, float]], Dict[str, Buffer]],
        data_shortage_warning: bool = False,
) -> Callable:
    """
    Overview:
        The return function is a generator which meanly fetch a batch of data from a buffer, \
        a list of buffers, or a dict of buffers.
    Arguments:
        - cfg (:obj:`EasyDict`): Config which should contain the following keys: `cfg.policy.learn.batch_size`.
        - buffer (:obj:`Union[Buffer, List[Tuple[Buffer, float]], Dict[str, Buffer]]`): \
            The buffer where the data is fetched from. \
            ``Buffer`` type means a buffer.\
            ``List[Tuple[Buffer, float]]`` type means a list of tuple. In each tuple there is a buffer and a float. \
            The float defines, how many batch_size is the size of the data \
            which is sampled from the corresponding buffer.\
            ``Dict[str, Buffer]`` type means a dict in which the value of each element is a buffer. \
            For each key-value pair of dict, batch_size of data will be sampled from the corresponding buffer \
            and assigned to the same key of `ctx.train_data`.
        - data_shortage_warning (:obj:`bool`): Whether to output warning when data shortage occurs in fetching.
    """

    def _fetch(ctx: "OnlineRLContext"):
        """
        Input of ctx:
            - train_output (:obj:`Union[Dict, Deque[Dict]]`): This attribute should exist \
                if `buffer_` is of type Buffer and if `buffer_` use the middleware `PriorityExperienceReplay`. \
                The meta data `priority` of the sampled data in the `buffer_` will be updated \
                to the `priority` attribute of `ctx.train_output` if `ctx.train_output` is a dict, \
                or the `priority` attribute of `ctx.train_output`'s popped element \
                if `ctx.train_output` is a deque of dicts.
        Output of ctx:
            - train_data (:obj:`Union[List[Dict], Dict[str, List[Dict]]]`): The fetched data. \
                ``List[Dict]`` type means a list of data.
                    `train_data` is of this type if the type of `buffer_` is Buffer or List.
                ``Dict[str, List[Dict]]]`` type means a dict, in which the value of each key-value pair
                    is a list of data. `train_data` is of this type if the type of `buffer_` is Dict.
        """
        try:
            unroll_len = cfg.policy.collect.unroll_len
            if isinstance(buffer_, Buffer):
                if unroll_len > 1:
                    buffered_data = buffer_.sample(
                        cfg.policy.learn.batch_size, groupby="env", unroll_len=unroll_len, replace=True
                    )
                    ctx.train_data = [[t.data for t in d] for d in buffered_data]  # B, unroll_len
                else:
                    buffered_data = buffer_.sample(cfg.policy.learn.batch_size)
                    ctx.train_data = [d.data for d in buffered_data]
            elif isinstance(buffer_, List):  # like sqil, r2d3
                assert unroll_len == 1, "not support"
                buffered_data = []
                for buffer_elem, p in buffer_:
                    data_elem = buffer_elem.sample(int(cfg.policy.learn.batch_size * p))
                    assert data_elem is not None
                    buffered_data.append(data_elem)
                buffered_data = sum(buffered_data, [])
                ctx.train_data = [d.data for d in buffered_data]
            elif isinstance(buffer_, Dict):  # like ppg_offpolicy
                assert unroll_len == 1, "not support"
                buffered_data = {k: v.sample(cfg.policy.learn.batch_size) for k, v in buffer_.items()}
                ctx.train_data = {k: [d.data for d in v] for k, v in buffered_data.items()}
            else:
                raise TypeError("not support buffer argument type: {}".format(type(buffer_)))

            assert buffered_data is not None
        except (ValueError, AssertionError):
            if data_shortage_warning:
                # You can modify data collect config to avoid this warning, e.g. increasing n_sample, n_episode.
                # Fetcher will skip this this attempt.
                logging.warning(
                    "Replay buffer's data is not enough to support training, so skip this training to wait more data."
                )
            ctx.train_data = None
            return

        yield

        if isinstance(buffer_, Buffer):
            if any([isinstance(m, PriorityExperienceReplay) for m in buffer_._middleware]):
                index = [d.index for d in buffered_data]
                meta = [d.meta for d in buffered_data]
                # such as priority
                if isinstance(ctx.train_output, List):
                    priority = ctx.train_output.pop()['priority']
                else:
                    priority = ctx.train_output['priority']
                for idx, m, p in zip(index, meta, priority):
                    m['priority'] = p
                    buffer_.update(index=idx, data=None, meta=m)

    return _fetch


def offline_data_fetcher_from_mem(cfg: EasyDict, dataset: Dataset) -> Callable:

    from threading import Thread
    from queue import Queue
    import time
    stream = torch.cuda.Stream()

    def producer(queue, dataset, batch_size, device):
        torch.set_num_threads(4)
        nonlocal stream
        idx_iter = iter(range(len(dataset)))
        with torch.cuda.stream(stream):
            while True:
                if queue.full():
                    time.sleep(0.1)
                else:
                    try:
                        start_idx = next(idx_iter)
                    except StopIteration:
                        del idx_iter
                        idx_iter = iter(range(len(dataset)))
                        start_idx = next(idx_iter)
                    data = [dataset.__getitem__(idx) for idx in range(start_idx, start_idx + batch_size)]
                    data = [[i[j] for i in data] for j in range(len(data[0]))]
                    data = [torch.stack(x).to(device) for x in data]
                    queue.put(data)

    queue = Queue(maxsize=50)
    device = 'cuda:{}'.format(get_rank() % torch.cuda.device_count()) if cfg.policy.cuda else 'cpu'
    producer_thread = Thread(
        target=producer, args=(queue, dataset, cfg.policy.batch_size, device), name='cuda_fetcher_producer'
    )

    def _fetch(ctx: "OfflineRLContext"):
        nonlocal queue, producer_thread
        if not producer_thread.is_alive():
            time.sleep(5)
            producer_thread.start()
        while queue.empty():
            time.sleep(0.001)
        ctx.train_data = queue.get()

    return _fetch


def offline_data_fetcher(cfg: EasyDict, dataset: Dataset) -> Callable:
    """
    Overview:
        The outer function transforms a Pytorch `Dataset` to `DataLoader`. \
        The return function is a generator which each time fetches a batch of data from the previous `DataLoader`.\
        Please refer to the link https://pytorch.org/tutorials/beginner/basics/data_tutorial.html \
        and https://pytorch.org/docs/stable/data.html for more details.
    Arguments:
        - cfg (:obj:`EasyDict`): Config which should contain the following keys: `cfg.policy.learn.batch_size`.
        - dataset (:obj:`Dataset`): The dataset of type `torch.utils.data.Dataset` which stores the data.
    """
    # collate_fn is executed in policy now
    dataloader = DataLoader(dataset, batch_size=cfg.policy.learn.batch_size, shuffle=True, collate_fn=lambda x: x)
    dataloader = iter(dataloader)

    def _fetch(ctx: "OfflineRLContext"):
        """
        Overview:
            Every time this generator is iterated, the fetched data will be assigned to ctx.train_data. \
            After the dataloader is empty, the attribute `ctx.train_epoch` will be incremented by 1.
        Input of ctx:
            - train_epoch (:obj:`int`): Number of `train_epoch`.
        Output of ctx:
            - train_data (:obj:`List[Tensor]`): The fetched data batch.
        """
        nonlocal dataloader
        try:
            ctx.train_data = next(dataloader)  # noqa
        except StopIteration:
            ctx.train_epoch += 1
            del dataloader
            dataloader = DataLoader(
                dataset, batch_size=cfg.policy.learn.batch_size, shuffle=True, collate_fn=lambda x: x
            )
            dataloader = iter(dataloader)
            ctx.train_data = next(dataloader)
        # TODO apply data update (e.g. priority) in offline setting when necessary
        ctx.trained_env_step += len(ctx.train_data)

    return _fetch


def offline_data_saver(data_path: str, data_type: str = 'hdf5') -> Callable:
    """
    Overview:
        Save the expert data of offline RL in a directory.
    Arguments:
        - data_path (:obj:`str`): File path where the expert data will be written into, which is usually ./expert.pkl'.
        - data_type (:obj:`str`): Define the type of the saved data. \
            The type of saved data is pkl if `data_type == 'naive'`. \
            The type of saved data is hdf5 if `data_type == 'hdf5'`.
    """

    def _save(ctx: "OnlineRLContext"):
        """
        Input of ctx:
            - trajectories (:obj:`List[Tensor]`): The expert data to be saved.
        """
        data = ctx.trajectories
        offline_data_save_type(data, data_path, data_type)
        ctx.trajectories = None

    return _save


def sqil_data_pusher(cfg: EasyDict, buffer_: Buffer, expert: bool) -> Callable:
    """
    Overview:
        Push trajectories into the buffer in sqil learning pipeline.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - buffer (:obj:`Buffer`): Buffer to push the data in.
        - expert (:obj:`bool`): Whether the pushed data is expert data or not. \
            In each element of the pushed data, the reward will be set to 1 if this attribute is `True`, otherwise 0.
    """

    def _pusher(ctx: "OnlineRLContext"):
        """
        Input of ctx:
            - trajectories (:obj:`List[Dict]`): The trajectories to be pushed.
        """
        for t in ctx.trajectories:
            if expert:
                t.reward = torch.ones_like(t.reward)
            else:
                t.reward = torch.zeros_like(t.reward)
            buffer_.push(t)
        ctx.trajectories = None

    return _pusher
