import pytest
import torch
from itertools import product
from ding.world_model.model.networks import DenseHead

# arguments
shape = [255, (255, ), ()]
# to do
# dist = ['normal', 'huber', 'binary', 'twohot_symlog']
dist = ['twohot_symlog']
args = list(product(*[shape, dist]))


@pytest.mark.unittest
@pytest.mark.parametrize('shape, dist', args)
def test_DenseHead(shape, dist):
    in_dim, layer_num, units, B, time = 1536, 2, 512, 16, 64
    head = DenseHead(in_dim, shape, layer_num, units, dist=dist)
    x = torch.randn(B, time, in_dim)
    a = torch.randn(B, time, 1)
    y = head(x)
    assert y.mode().shape == (B, time, 1)
    assert y.log_prob(a).shape == (B, time)
