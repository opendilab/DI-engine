import pytest
import threading
import time
import torch
import torch.nn as nn
from functools import partial
from itertools import product

from ding.utils import EasyTimer
from ding.utils.data import AsyncDataLoader

batch_size_args = [3, 6]
num_workers_args = [0, 4]
chunk_size_args = [1, 3]
args = [item for item in product(*[batch_size_args, num_workers_args, chunk_size_args])]
unittest_args = [item for item in product(*[[3], [2], [1]])]


class Dataset(object):

    def __init__(self):
        self.data = torch.randn(256, 256)

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        time.sleep(0.5)
        return [self.data, idx]


class TestAsyncDataLoader:

    def get_data_source(self):
        dataset = Dataset()

        def data_source_fn(batch_size):
            return [partial(dataset.__getitem__, idx=i) for i in range(batch_size)]

        return data_source_fn

    def get_model(self):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.main = [nn.Linear(256, 256) for _ in range(10)]
                self.main = nn.Sequential(*self.main)

            def forward(self, x):
                idx = x[1]
                x = self.main(x[0])
                time.sleep(1)
                return [x, idx]

        return Model()

    # @pytest.mark.unittest
    @pytest.mark.parametrize('batch_size, num_workers, chunk_size', unittest_args)
    def test_cpu(self, batch_size, num_workers, chunk_size):
        self.entry(batch_size, num_workers, chunk_size, use_cuda=False)

    @pytest.mark.cudatest
    @pytest.mark.parametrize('batch_size, num_workers, chunk_size', args)
    def test_gpu(self, batch_size, num_workers, chunk_size):
        self.entry(batch_size, num_workers, chunk_size, use_cuda=True)
        torch.cuda.empty_cache()

    def entry(self, batch_size, num_workers, chunk_size, use_cuda):
        model = self.get_model()
        if use_cuda:
            model.cuda()
        timer = EasyTimer()
        data_source = self.get_data_source()
        device = 'cuda' if use_cuda else 'cpu'
        dataloader = AsyncDataLoader(data_source, batch_size, device, num_workers=num_workers, chunk_size=chunk_size)
        count = 0
        total_data_time = 0.
        while True:
            with timer:
                data = next(dataloader)
            data_time = timer.value
            if count > 2:  # ignore start-3 time
                total_data_time += data_time
            with timer:
                with torch.no_grad():
                    _, idx = model(data)
                if use_cuda:
                    idx = idx.cpu()
                sorted_idx = torch.sort(idx)[0]
                assert sorted_idx.eq(torch.arange(batch_size)).sum() == batch_size, idx
            model_time = timer.value
            print('count {}, data_time: {}, model_time: {}'.format(count, data_time, model_time))
            count += 1
            if count == 10:
                break
        if num_workers < 1:
            assert total_data_time <= 7 * batch_size * 0.5 + 7 * 0.01 - 7 * 1
        else:
            assert total_data_time <= 7 * 0.008
        dataloader.__del__()
        time.sleep(0.5)
        assert len(threading.enumerate()) <= 2, threading.enumerate()
