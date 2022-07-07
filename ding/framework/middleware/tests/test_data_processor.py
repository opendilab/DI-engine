import pytest

from ding.data.buffer import DequeBuffer

from ding.framework import Context, OnlineRLContext, OfflineRLContext
from ding.framework.middleware.functional.data_processor import \
    data_pusher, offpolicy_data_fetcher, offline_data_fetcher, offline_data_saver, sqil_data_pusher

from ding.data.buffer.middleware import PriorityExperienceReplay

from easydict import EasyDict
from ding.data import Dataset
from collections import deque
import torch
import math
import os
import copy

from unittest.mock import patch


@pytest.mark.unittest
def test_data_pusher():
    buffer = DequeBuffer(size=10)
    ctx = OnlineRLContext()
    ctx.trajectories = [i for i in range(5)]
    data_pusher(cfg=None, buffer_=buffer)(ctx)
    assert buffer.count() == 5

    buffer = DequeBuffer(size=10)
    ctx = OnlineRLContext()
    ctx.episodes = [i for i in range(5)]
    data_pusher(cfg=None, buffer_=buffer)(ctx)
    assert buffer.count() == 5

    buffer = DequeBuffer(size=10)
    ctx = OnlineRLContext()
    with pytest.raises(RuntimeError) as exc_info:
        data_pusher(cfg=None, buffer_=buffer)(ctx)
    assert str(exc_info.value) == "Either ctx.trajectories or ctx.episodes should be not None."


def offpolicy_data_fetcher_type_buffer_helper(priority=0.5, use_list=True):
    cfg = EasyDict({'policy': {'learn': {'batch_size': 20}, 'collect': {'unroll_len': 1}}})
    buffer = DequeBuffer(size=20)
    buffer.use(PriorityExperienceReplay(buffer=buffer))
    for i in range(20):
        buffer.push({'obs': i, 'reward': 1, 'info': 'xxx'})
    ctx = OnlineRLContext()

    if use_list:
        ctx.train_output = [{'priority': [priority for _ in range(20)]}]
    else:
        ctx.train_output = {'priority': [priority for _ in range(20)]}

    func_generator = offpolicy_data_fetcher(cfg=cfg, buffer_=buffer)(ctx)
    next(func_generator)
    assert len(ctx.train_data) == cfg.policy.learn.batch_size
    assert all(d['obs'] >= 0 and i < 20 and isinstance(i, int) for d in ctx.train_data)
    assert [d['obs'] for d in ctx.train_data] == [i for i in range(20)]
    assert [d['reward'] for d in ctx.train_data] == [1 for i in range(20)]
    assert [d['info'] for d in ctx.train_data] == ['xxx' for i in range(20)]
    assert [d['priority_IS'] for d in ctx.train_data] == [torch.tensor([1]) for i in range(20)]
    assert buffer.export_data()[0].meta['priority'] == 1.0
    # assert sorted(ctx.train_data) == [i for i in range(20)]

    try:
        next(func_generator)
    except StopIteration:
        pass
    assert buffer.export_data()[0].meta['priority'] == priority


def call_offpolicy_data_fetcher_type_buffer():
    # if isinstance(buffer_, Buffer):
    offpolicy_data_fetcher_type_buffer_helper(priority=0.5, use_list=True)
    offpolicy_data_fetcher_type_buffer_helper(priority=0.3, use_list=False)


def call_offpolicy_data_fetcher_type_list():
    #elif isinstance(buffer_, List)
    cfg = EasyDict({'policy': {'learn': {'batch_size': 5}, 'collect': {'unroll_len': 1}}})
    buffer = DequeBuffer(size=20)
    for i in range(20):
        buffer.push(i)
    ctx = OnlineRLContext()
    buffer1 = copy.deepcopy(buffer)
    buffer2 = copy.deepcopy(buffer)
    buffer3 = copy.deepcopy(buffer)
    buffer_list = [(buffer1, 1), (buffer2, 2), (buffer3, 3)]

    next(offpolicy_data_fetcher(cfg=cfg, buffer_=buffer_list)(ctx))
    assert len(ctx.train_data) == cfg.policy.learn.batch_size * (1 + 2 + 3)
    assert all(i >= 0 and i < 20 and isinstance(i, int) for i in ctx.train_data)


def call_offpolicy_data_fetcher_type_dict():
    #elif isinstance(buffer_, Dict)
    cfg = EasyDict({'policy': {'learn': {'batch_size': 5}, 'collect': {'unroll_len': 1}}})
    buffer = DequeBuffer(size=20)
    for i in range(20):
        buffer.push(i)
    ctx = OnlineRLContext()
    buffer1 = copy.deepcopy(buffer)
    buffer2 = copy.deepcopy(buffer)
    buffer3 = copy.deepcopy(buffer)
    buffer_dict = {'key1': buffer1, 'key2': buffer2, 'key3': buffer3}

    next(offpolicy_data_fetcher(cfg=cfg, buffer_=buffer_dict)(ctx))
    assert all(len(v) == cfg.policy.learn.batch_size for k, v in ctx.train_data.items())
    assert all(all(i >= 0 and i < 20 and isinstance(i, int) for i in v) for k, v in ctx.train_data.items())


def call_offpolicy_data_fetcher_type_int():
    # else catch TypeError
    cfg = EasyDict({'policy': {'learn': {'batch_size': 5}, 'collect': {'unroll_len': 1}}})
    ctx = OnlineRLContext()
    with pytest.raises(TypeError) as exc_info:
        next(offpolicy_data_fetcher(cfg=cfg, buffer_=1)(ctx))
    assert str(exc_info.value) == "not support buffer argument type: {}".format(type(1))


@pytest.mark.unittest
def test_offpolicy_data_fetcher():
    call_offpolicy_data_fetcher_type_buffer()
    call_offpolicy_data_fetcher_type_list()
    call_offpolicy_data_fetcher_type_dict()
    call_offpolicy_data_fetcher_type_int()


@pytest.mark.unittest
def test_offline_data_fetcher():
    cfg = EasyDict({'policy': {'learn': {'batch_size': 5}}})
    dataset_size = 10
    num_batch = math.ceil(dataset_size / cfg.policy.learn.batch_size)
    data = torch.linspace(11, 20, dataset_size)
    data_list = list(data)

    class MyDataset(Dataset):

        def __init__(self):
            self.x = data
            self.len = len(self.x)

        def __getitem__(self, index):
            return self.x[index]

        def __len__(self):
            return self.len

    ctx = OfflineRLContext()
    ctx.train_epoch = 0

    data_tmp = []
    for i, _ in enumerate(offline_data_fetcher(cfg, MyDataset())(ctx)):
        assert i // num_batch == ctx.train_epoch
        data_tmp.extend(ctx.train_data)

        if i % num_batch == num_batch - 1:
            assert sorted(data_tmp) == data_list
            data_tmp = []
        if i >= num_batch * 5 - 1:
            break


@pytest.mark.unittest
def test_offline_data_saver():
    transition = {}
    transition['obs'] = torch.zeros((3, 1))
    transition['next_obs'] = torch.zeros((3, 1))
    transition['action'] = torch.zeros((1, 1))
    transition['reward'] = torch.tensor((1, ))
    transition['done'] = False
    transition['collect_iter'] = 0

    fake_data = [transition for i in range(32)]

    ctx = OnlineRLContext()
    ctx.trajectories = fake_data
    data_path_ = './expert.pkl'

    def mock_offline_data_save_type(exp_data, expert_data_path, data_type):
        assert exp_data == fake_data
        assert expert_data_path == data_path_
        assert data_type == 'naive'

    with patch("ding.framework.middleware.functional.data_processor.offline_data_save_type",
               mock_offline_data_save_type):
        offline_data_saver(cfg=None, data_path=data_path_, data_type='naive')(ctx)

    assert ctx.trajectories is None

    ctx = OnlineRLContext()
    ctx.trajectories = fake_data

    def mock_offline_data_save_type(exp_data, expert_data_path, data_type):
        assert exp_data == fake_data
        assert expert_data_path == data_path_
        assert data_type == 'hdf5'

    with patch("ding.framework.middleware.functional.data_processor.offline_data_save_type",
               mock_offline_data_save_type):
        offline_data_saver(cfg=None, data_path=data_path_, data_type='hdf5')(ctx)

    assert ctx.trajectories is None


@pytest.mark.unittest
def test_sqil_data_pusher():
    transition = {}
    transition['obs'] = torch.zeros((3, 1))
    transition['next_obs'] = torch.zeros((3, 1))
    transition['action'] = torch.zeros((1, 1))
    transition['reward'] = torch.tensor((2, ))
    transition['done'] = False
    transition['collect_iter'] = 0
    transition = EasyDict(transition)

    fake_data = [transition for i in range(5)]

    # expert = True
    ctx = OnlineRLContext()
    ctx.trajectories = copy.deepcopy(fake_data)
    buffer = DequeBuffer(size=10)
    sqil_data_pusher(cfg=None, buffer_=buffer, expert=True)(ctx)
    assert buffer.count() == 5
    assert all(t.data.reward == 1 for t in buffer.export_data())

    # expert = False
    ctx = OnlineRLContext()
    ctx.trajectories = copy.deepcopy(fake_data)
    buffer = DequeBuffer(size=10)
    sqil_data_pusher(cfg=None, buffer_=buffer, expert=False)(ctx)
    assert buffer.count() == 5
    assert all(t.data.reward == 0 for t in buffer.export_data())
