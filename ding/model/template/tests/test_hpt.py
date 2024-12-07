import pytest
import torch
from itertools import product
from ding.model.template.hpt import HPT
from ding.torch_utils import is_differentiable

T, B = 3, 4
obs_shape = [4, (8, )]
act_shape = [3, (6, )]
args = list(product(*[obs_shape, act_shape]))


@pytest.mark.unittest
class TestHPT:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, list):
            loss = sum([t.sum() for t in outputs])
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_hpt(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs = torch.randn(B, obs_shape)
            state_dim = obs_shape
        else:
            inputs = torch.randn(B, *obs_shape)
            state_dim = obs_shape[0]

        if isinstance(act_shape, int):
            action_dim = act_shape
        else:
            action_dim = len(act_shape)

        model = HPT(state_dim=state_dim, action_dim=action_dim)
        outputs = model(inputs)

        assert isinstance(outputs, torch.Tensor)

        if isinstance(act_shape, int):
            assert outputs.shape == (B, act_shape)
        elif len(act_shape) == 1:
            assert outputs.shape == (B, *act_shape)
        else:
            for i, s in enumerate(act_shape):
                assert outputs[i].shape == (B, s)

        self.output_check(model, outputs)
