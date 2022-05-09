from typing import TYPE_CHECKING, Callable, List, Union, Tuple, Dict
from easydict import EasyDict
from collections import deque
import logging
import torch
from ding.data import Buffer, Dataset, DataLoader, offline_data_save_type
from ding.data.buffer.middleware import PriorityExperienceReplay
from ding.framework import task

if TYPE_CHECKING:
    from ding.framework import Context, OnlineRLContext, OfflineRLContext


def data_pusher(cfg: EasyDict, buffer_: Buffer):
    """
    Overview:
        Push a episode or a trajectory into the buffer.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - buffer_ (:obj:`Buffer`): Buffer to push the data in.
    """
    def _push(ctx: "OnlineRLContext"):
        """
        Overview:
            In ctx, either ctx.trajectories or ctx.episodes should not be None.
        Input of ctx:
            - trajectories (:obj:`List[Dict]`): A trajectory.
            - episodes (:obj:`List[Dict]`): A episode.
        """
        if ctx.trajectories is not None:
            for t in ctx.trajectories:
                buffer_.push(t)
            ctx.trajectories = None
        elif ctx.episodes is not None:
            for t in ctx.episodes:
                buffer_.push(t)
            ctx.episodes = None
        else:
            raise RuntimeError("Either ctx.trajectories or ctx.episodes should be not None.")

    return _push


def offpolicy_data_fetcher(
        cfg: EasyDict, buffer_: Union[Buffer, List[Tuple[Buffer, float]], Dict[str, Buffer]]
) -> Callable:
    """
    Overview:
        The return function of this function is a generator which meanly fetch a batch of data from a buffer, \
        or a list of buffers, or a dict of buffers.
    Arguments:
        - cfg (:obj:`EasyDict`): Config which should define cfg.policy.learn.batch_size.
        - buffer_ (:obj:`Union[Buffer, List[Tuple[Buffer, float]], Dict[str, Buffer]]`): \
            The buffer where the data is fetched from. \
            ``Buffer`` type means a buffer.\
            ``List[Tuple[Buffer, float]]`` type means a list of tuple. In each tuple the first element is a buffer,\
            and the second element is a float which define how many times of batch_size the data will be fetched.\
            E.g. if buffer_[0] is a tuple (buffer_elem, 3), that means 3*batch_size of data will be fetched from buffer_elem.\
            ``Dict[str, Buffer]`` type means a dict in which each value is a buffer. For each key-value pair, \
            batch_size of data will be fetched and put under the same key of ctx.train_data, \
            which is also a dict and has the same keys with buffer_.
    """
    def _fetch(ctx: "OnlineRLContext"):
        """
        Input of ctx:
            - train_output (:obj:`Union[Dict, Deque[Dict]]`): This attribute should exist \
                if buffer_ is of type Buffer and if buffer_ use the middleware PriorityExperienceReplay. \
                The meta data ``priority`` of buffer_ will be updated after yield, \
                by the ``priority`` attribute of ctx.train_output if it is a dict, \
                or the ``priority`` attribute of the popped element of ctx.train_output if it is a deque of dicts.
        Output of ctx:
            - train_data (:obj:`Union[List[Dict], Dict[str, List[Dict]]]`): the fetched data. \
                ``List[Dict]`` type means a list of data.
                     train_data is of this type if the type of buffer_ is Buffer or List.
                ``Dict[str, List[Dict]]]`` type means a dict, in which the value of each key-value pair 
                    is a list of data. train_data is of this type if the type of buffer_ is Dict.
        """
        try:
            if isinstance(buffer_, Buffer):
                buffered_data = buffer_.sample(cfg.policy.learn.batch_size)
                ctx.train_data = [d.data for d in buffered_data]
            elif isinstance(buffer_, List):  # like sqil, r2d3
                buffered_data = []
                for buffer_elem, p in buffer_:
                    data_elem = buffer_elem.sample(int(cfg.policy.learn.batch_size * p))
                    assert data_elem is not None
                    buffered_data.append(data_elem)
                buffered_data = sum(buffered_data, [])
                ctx.train_data = [d.data for d in buffered_data]
            elif isinstance(buffer_, Dict):  # like ppg_offpolicy
                buffered_data = {k: v.sample(cfg.policy.learn.batch_size) for k, v in buffer_.items()}
                ctx.train_data = {k: [d.data for d in v] for k, v in buffered_data.items()}
            else:
                raise TypeError("not support buffer argument type: {}".format(type(buffer_)))

            assert buffered_data is not None
        except (ValueError, AssertionError):
            # You can modify data collect config to avoid this warning, e.g. increasing n_sample, n_episode.
            logging.warning(
                "Replay buffer's data is not enough to support training, so skip this training for waiting more data."
            )
            ctx.train_data = None
            return

        yield

        if isinstance(buffer_, Buffer):
            if any([isinstance(m, PriorityExperienceReplay) for m in buffer_._middleware]):
                index = [d.index for d in buffered_data]
                meta = [d.meta for d in buffered_data]
                # such as priority
                if isinstance(ctx.train_output, deque):
                    priority = ctx.train_output.pop()['priority']
                else:
                    priority = ctx.train_output['priority']
                for idx, m, p in zip(index, meta, priority):
                    m['priority'] = p
                    buffer_.update(index=idx, data=None, meta=m)

    return _fetch


def offline_data_fetcher(cfg: EasyDict, dataset: Dataset) -> Callable:
    """
    Overview:
        The outer function of this function transform a Pytorch Dataset to DataLoader, \
        and the return function is a generator which each time fetch a batch of data from a Pytorch DataLoader.\
        Please refer to the link https://pytorch.org/tutorials/beginner/basics/data_tutorial.html \
        and https://pytorch.org/docs/stable/data.html for more details.
    Arguments:
        - cfg (:obj:`EasyDict`): Config which should define cfg.policy.learn.batch_size.
        - dataset (:obj:`Dataset`): The dataset of type torch.utils.data.Dataset which store the data. 
    """
    # collate_fn is executed in policy now
    dataloader = DataLoader(dataset, batch_size=cfg.policy.learn.batch_size, shuffle=True, collate_fn=lambda x: x)

    def _fetch(ctx: "OfflineRLContext"):
        """
        Overview:
            Every time this generator is iterated, the fetched data will be assigned to ctx.train_data. \
            After all the data in the dataloader is fetched, the attribute ctx.train_data will be incremented by 1. 
        Output of ctx:
            - train_data (:obj:`List[Tenosr]`): The fetched data batch.
            - train_epoch (:obj:`int`): Number of train_epoch.
        """
        while True:
            for i, data in enumerate(dataloader):
                ctx.train_data = data
                yield
            ctx.train_epoch += 1
        # TODO apply data update (e.g. priority) in offline setting when necessary

    return _fetch


def offline_data_saver(cfg: EasyDict, data_path: str, data_type: str = 'hdf5') -> Callable:
    """
    Overview:
        Save the expert data of offline RL.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - data_path (:obj:`str`): File path of the expert data will be written to, which is usually ./expert.pkl'.
        - data_type (:obj:`str`): To define the type of the saved data. \
            The type of saved data is pkl if data_type == 'naive'. \
            The type of saved data is hdf5 if data_type == 'hdf5'.
    """
    def _save(ctx: "OnlineRLContext"):
        """
        Input of ctx:
            - trajectories (:obj:`List[Tenosr]`): The expert data to be saved.
        """
        data = ctx.trajectories
        offline_data_save_type(data, data_path, data_type)
        ctx.trajectories = None

    return _save


def sqil_data_pusher(cfg: EasyDict, buffer_: Buffer, expert: bool) -> Callable:
    """
    Overview:
        Push a trajectory into buffer in sqil training pipeline. The reward of each element will be one \
        if expert is True, zero if expert is False
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - buffer_ (:obj:`Buffer`): Buffer to push the data in.
        - expert (:obj:`bool`): Wether the pushed data is expert data or normal data. 
    """
    def _pusher(ctx: "OnlineRLContext"):
        """
        Input of ctx:
            - trajectories (:obj:`List[Tenosr]`): The data to be saved.
        """
        for t in ctx.trajectories:
            if expert:
                t.reward = torch.ones_like(t.reward)
            else:
                t.reward = torch.zeros_like(t.reward)
            buffer_.push(t)
        ctx.trajectories = None

    return _pusher
