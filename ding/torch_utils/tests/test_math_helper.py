import numpy as np
import pytest
import torch

from ding.torch_utils.math_helper import cov


@pytest.mark.unittest
class TestMathHelper:

    def test_cov(self):
        r'''
        Overview:
            Test the conv
        '''
        # test 1D
        # test dtype and rowvar
        x1 = np.array([1, 2, 3])
        cov1 = np.cov(x1, rowvar=False)
        x1_tensor = torch.FloatTensor(x1)
        cov1_tensor = cov(x1_tensor, rowvar=False).detach().numpy()
        assert (np.abs(cov1 - cov1_tensor) < 1e-6).any()

        # test 2D
        x2 = np.array([[0., 2.], [1., 1.], [2., 0.]]).T
        cov2 = np.cov(x2, rowvar=True)
        x2_tensor = torch.FloatTensor(x2)
        cov2_tensor = cov(x2_tensor, rowvar=True).detach().numpy()
        assert (np.abs(cov2 - cov2_tensor) < 1e-6).any()

        # test bias
        cov3 = np.cov(x2, rowvar=True, bias=True)
        cov3_tensor = cov(x2_tensor, rowvar=True, bias=True).detach().numpy()
        assert (np.abs(cov3 - cov3_tensor) < 1e-6).any()

        # test ddof
        aweights = np.array([1., 2., 3.])
        cov4 = np.cov(x2, rowvar=True, ddof=0, aweights=aweights)
        cov4_tensor = cov(x2_tensor, rowvar=True, ddof=0, aweights=aweights).detach().numpy()
        assert (np.abs(cov4 - cov4_tensor) < 1e-6).any()

        # test aweights
        cov5 = np.cov(x2, rowvar=True, aweights=aweights)
        aweights_tensor = torch.FloatTensor(aweights)
        cov5_tensor = cov(x2_tensor, rowvar=True, aweights=aweights_tensor).detach().numpy()
        assert (np.abs(cov5 - cov5_tensor) < 1e-6).any()
