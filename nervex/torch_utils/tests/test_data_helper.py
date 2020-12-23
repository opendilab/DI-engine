import pytest
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nervex.torch_utils import CudaFetcher, to_device, to_tensor, to_dtype, tensor_to_list, same_shape
from nervex.utils import EasyTimer


@pytest.mark.cudatest
class TestCudaFetcher:

    def get_dataloader(self):

        class Dataset(object):

            def __init__(self):
                self.data = torch.randn(2560, 2560)

            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return self.data

        return DataLoader(Dataset(), batch_size=3)

    def get_model(self):

        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.main = [nn.Linear(2560, 2560) for _ in range(100)]
                self.main = nn.Sequential(*self.main)

            def forward(self, x):
                x = self.main(x)
                return x

        return Model()

    def test_naive(self):
        model = self.get_model()
        model.cuda()
        timer = EasyTimer()
        dataloader = iter(self.get_dataloader())
        dataloader = CudaFetcher(dataloader, device='cuda', sleep=0.1)
        dataloader.run()

        count = 0
        while True:
            with timer:
                data = next(dataloader)
                model(data)
            print('count {}, run_time: {}'.format(count, timer.value))
            count += 1
            if count == 10:
                break

        dataloader.close()


@pytest.mark.unittest
class TestDataFunc:

    def test_tensor_list(self):
        t = torch.randn(3, 5)
        tlist1 = tensor_to_list(t)
        assert len(tlist1) == 3
        assert len(tlist1[0]) == 5

        t = torch.randn(3, )
        tlist1 = tensor_to_list(t)
        assert len(tlist1) == 3

        tback = to_tensor(tlist1, torch.float)
        assert (tback == t)[0]

        t = [torch.randn(5, ) for i in range(3)]
        tlist1 = tensor_to_list(t)
        assert len(tlist1) == 3
        assert len(tlist1[0]) == 5

        td = {'t': t}
        tdlist1 = tensor_to_list(td)

        assert len(tdlist1['t']) == 3
        assert len(tdlist1['t'][0]) == 5

    def test_same_shape(self):
        tlist = [torch.randn(3, 5) for i in range(5)]
        assert same_shape(tlist)
        tlist = [torch.randn(3, 5), torch.randn(4, 5)]
        assert not same_shape(tlist)

    def test_to_dtype(self):
        t = torch.randint(0, 10, (3, 5))
        tfloat = to_dtype(t, torch.float)
        assert tfloat.dtype == torch.float
        tlist = [t]
        tlfloat = to_dtype(tlist, torch.float)
        assert tlfloat[0].dtype == torch.float
        tdict = {'t': t}
        tdictf = to_dtype(tdict, torch.float)
        assert tdictf['t'].dtype == torch.float

    def test_to_tensor(self):
        i = 10
        t = to_tensor(i, torch.int)
        assert t.item() == i
        d = {'i': i}
        dt = to_tensor(d, torch.int)
        assert dt['i'].item() == i
        with pytest.raises(TypeError):
            _ = to_tensor({1, 2}, torch.int)

        data_type = namedtuple('data_type', ['x', 'y'])
        inputs = data_type(np.random.random(3), 4)
        outputs = to_tensor(inputs, torch.float32)
        assert type(outputs) == data_type
        assert isinstance(outputs.x, torch.Tensor)
        assert isinstance(outputs.y, torch.Tensor)
        assert outputs.x.dtype == torch.float32
        assert outputs.y.dtype == torch.float32
