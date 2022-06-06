import pytest
import torch
from dizoo.distar.model.value import ValueBaseline


@pytest.mark.envtest
def test_value_baseline():

    class CFG:

        def __init__(self):
            self.activation = 'relu'
            self.norm_type = 'LN'
            self.input_dim = 1024
            self.res_dim = 256
            self.res_num = 16
            self.use_value_feature = False
            self.atan = True

    model = ValueBaseline(CFG())
    inputs = torch.randn(4, 1024)
    output = model(inputs)
    assert (output.shape == (4, ))
