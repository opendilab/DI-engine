import pytest
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ding.torch_utils import CudaFetcher, to_device, to_dtype, to_tensor, to_ndarray, to_list, \
    tensor_to_list, same_shape, build_log_buffer, get_tensor_data
from ding.utils import EasyTimer


@pytest.fixture(scope='function')
def setup_data_dict():
    return {
        'tensor': torch.randn(4),
        'list': [True, False, False],
        'tuple': (4, 5, 6),
        'bool': True,
        'int': 10,
        'float': 10.,
        'array': np.random.randn(4),
        'str': "asdf",
        'none': None,
    }


@pytest.mark.unittest
class TestDataFunction:

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
        with pytest.raises(TypeError):
            to_dtype(EasyTimer(), torch.float)

    def test_to_tensor(self, setup_data_dict):
        i = 10
        t = to_tensor(i)
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

        transformed_tensor = to_tensor(setup_data_dict)
        with pytest.raises(TypeError):
            to_tensor(EasyTimer(), torch.float)

    def test_to_ndarray(self, setup_data_dict):
        t = torch.randn(3, 5)
        tarray1 = to_ndarray(t)
        assert tarray1.shape == (3, 5)
        assert isinstance(tarray1, np.ndarray)

        t = [torch.randn(5, ) for i in range(3)]
        tarray1 = to_ndarray(t, np.float32)
        assert isinstance(tarray1, list)
        assert tarray1[0].shape == (5, )
        assert isinstance(tarray1[0], np.ndarray)

        transformed_array = to_ndarray(setup_data_dict)
        with pytest.raises(TypeError):
            to_ndarray(EasyTimer(), np.float32)

    def test_to_list(self, setup_data_dict):
        # tensor_to_list
        t = torch.randn(3, 5)
        tlist1 = tensor_to_list(t)
        assert len(tlist1) == 3
        assert len(tlist1[0]) == 5

        t = torch.randn(3, )
        tlist1 = tensor_to_list(t)
        assert len(tlist1) == 3

        t = [torch.randn(5, ) for i in range(3)]
        tlist1 = tensor_to_list(t)
        assert len(tlist1) == 3
        assert len(tlist1[0]) == 5

        td = {'t': t}
        tdlist1 = tensor_to_list(td)
        assert len(tdlist1['t']) == 3
        assert len(tdlist1['t'][0]) == 5

        tback = to_tensor(tlist1, torch.float)
        for i in range(3):
            assert (tback[i] == t[i]).all()

        with pytest.raises(TypeError):
            tensor_to_list(EasyTimer())

        # to_list
        transformed_list = to_list(setup_data_dict)
        with pytest.raises(TypeError):
            to_ndarray(EasyTimer())

    def test_same_shape(self):
        tlist = [torch.randn(3, 5) for i in range(5)]
        assert same_shape(tlist)
        tlist = [torch.randn(3, 5), torch.randn(4, 5)]
        assert not same_shape(tlist)

    def test_get_tensor_data(self):
        a = {
            'tensor': torch.tensor([1, 2, 3.], requires_grad=True),
            'list': [torch.tensor([1, 2, 3.], requires_grad=True) for _ in range(2)],
            'none': None
        }
        tensor_a = get_tensor_data(a)
        assert not tensor_a['tensor'].requires_grad
        for t in tensor_a['list']:
            assert not t.requires_grad
        with pytest.raises(TypeError):
            get_tensor_data(EasyTimer())


@pytest.mark.unittest
def test_log_dict():
    log_buffer = build_log_buffer()
    log_buffer['not_tensor'] = torch.randn(3)
    assert isinstance(log_buffer['not_tensor'], list)
    assert len(log_buffer['not_tensor']) == 3
    log_buffer.update({'not_tensor': 4, 'a': 5})
    assert log_buffer['not_tensor'] == 4


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


@pytest.mark.cudatest
def test_to_device_cuda(setup_data_dict):
    setup_data_dict['module'] = nn.Linear(3, 5)
    device = 'cuda'
    cuda_d = to_device(setup_data_dict, device, ignore_keys=['module'])
    assert cuda_d['module'].weight.device == torch.device('cpu')
    other = EasyTimer()
    with pytest.raises(TypeError):
        to_device(other)


@pytest.mark.unittest
def test_to_device_cpu(setup_data_dict):
    setup_data_dict['module'] = nn.Linear(3, 5)
    device = 'cpu'
    cuda_d = to_device(setup_data_dict, device, ignore_keys=['module'])
    assert cuda_d['module'].weight.device == torch.device('cpu')
    other = EasyTimer()
    with pytest.raises(TypeError):
        to_device(other)
