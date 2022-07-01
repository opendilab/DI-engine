from typing import TYPE_CHECKING, Callable, List, Union, Tuple, Dict, Optional
from easydict import EasyDict
from ditk import logging
import torch
from ding.data import Buffer, Dataset, DataLoader, offline_data_save_type
from ding.data.buffer.middleware import PriorityExperienceReplay

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext, OfflineRLContext


def data_pusher(cfg: EasyDict, buffer_: Buffer, group_by_env: Optional[bool] = None):
    """
    Overview:
        Push episodes or trajectories into the buffer.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - buffer\_ (:obj:`Buffer`): Buffer to push the data in.
    """

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


def offpolicy_data_fetcher(
        cfg: EasyDict, buffer_: Union[Buffer, List[Tuple[Buffer, float]], Dict[str, Buffer]]
) -> Callable:
    """
    Overview:
        The return function is a generator which meanly fetch a batch of data from a buffer, \
        a list of buffers, or a dict of buffers.
    Arguments:
        - cfg (:obj:`EasyDict`): Config which should contain the following keys: `cfg.policy.learn.batch_size`.
        - buffer\_ (:obj:`Union[Buffer, List[Tuple[Buffer, float]], Dict[str, Buffer]]`): \
            The buffer where the data is fetched from. \
            ``Buffer`` type means a buffer.\
            ``List[Tuple[Buffer, float]]`` type means a list of tuple. In each tuple there is a buffer and a float. \
            The float defines, how many batch_size is the size of the data \
            which is sampled from the corresponding buffer.\
            ``Dict[str, Buffer]`` type means a dict in which the value of each element is a buffer. \
            For each key-value pair of dict, batch_size of data will be sampled from the corresponding buffer \
            and assigned to the same key of `ctx.train_data`.
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
                if isinstance(ctx.train_output, List):
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
        Save the expert data of offline RL in a directory.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
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
        - buffer\_ (:obj:`Buffer`): Buffer to push the data in.
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
