import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nervex.torch_utils import CudaFetcher
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
