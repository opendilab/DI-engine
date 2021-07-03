import numpy as np
import pytest
import torch

from ding.torch_utils.network import SoftArgmax


@pytest.mark.unittest
class TestSoftArgmax:

    def test(self):
        H, W = (48, 64)
        B = 4
        N = 10
        model = SoftArgmax()
        # data case 1
        for _ in range(N):
            test_h = np.random.randint(0, H, size=(B, ))
            test_w = np.random.randint(0, W, size=(B, ))
            test_location = torch.LongTensor([test_h, test_w]).permute(1, 0).contiguous()
            assert test_location.shape == (B, 2)
            data = torch.full((B, 1, H, W), -1e8)
            for idx, (h, w) in enumerate(test_location):
                data[idx, 0, h, w] = 1

            pred_location = model(data)
            assert pred_location.shape == (B, 2)
            assert torch.abs(pred_location - test_location).sum() < 1e-6
        # data case 2
        pseudo_gauss_kernel = torch.FloatTensor([1, 3, 1, 3, 5, 3, 1, 3, 1]).reshape(3, 3)
        for _ in range(N):
            test_h = np.random.randint(1, H - 1, size=(B, ))
            test_w = np.random.randint(1, W - 1, size=(B, ))
            test_location = torch.LongTensor([test_h, test_w]).permute(1, 0).contiguous()
            assert test_location.shape == (B, 2)
            data = torch.full((B, 1, H, W), -1e8)
            for idx, (h, w) in enumerate(test_location):
                data[idx, 0, h - 1:h + 2, w - 1:w + 2] = pseudo_gauss_kernel

            pred_location = model(data)
            assert pred_location.shape == (B, 2)
            assert torch.abs(pred_location - test_location).sum() < 1e-4
