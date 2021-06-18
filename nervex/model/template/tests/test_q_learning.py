import pytest
from itertools import product
import torch
from nervex.model.template import DQN, RainbowDQN
from nervex.torch_utils import is_differentiable

B = 4
obs_shape = [4, (8, ), (4, 64, 64)]
act_shape = [3, (6, ), [2, 3, 6]]
args = list(product(*[obs_shape, act_shape]))


@pytest.mark.unittest
class TestQLearning:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, list):
            loss = sum([t.sum() for t in outputs])
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

    @pytest.mark.parametrize('obs_space, act_space', args)
    def test_dqn(self, obs_space, act_space):
        if isinstance(obs_space, int):
            inputs = torch.randn(B, obs_space)
        else:
            inputs = torch.randn(B, *obs_space)
        model = DQN(obs_space, act_space)
        outputs = model(inputs)
        assert isinstance(outputs, dict)
        if isinstance(act_space, int):
            assert outputs['logit'].shape == (B, act_space)
        elif len(act_space) == 1:
            assert outputs['logit'].shape == (B, *act_space)
        else:
            for i, s in enumerate(act_space):
                assert outputs['logit'][i].shape == (B, s)
        self.output_check(model, outputs['logit'])

    @pytest.mark.rainbow
    @pytest.mark.parametrize('obs_space, act_space', args)
    def test_rainbowdqn(self, obs_space, act_space):
        if isinstance(obs_space, int):
            inputs = torch.randn(B, obs_space)
        else:
            inputs = torch.randn(B, *obs_space)
        model = RainbowDQN(obs_space, act_space, n_atom=41)
        outputs = model(inputs)
        assert isinstance(outputs, dict)
        if isinstance(act_space, int):
            assert outputs['logit'].shape == (B, act_space)
            assert outputs['distribution'].shape == (B, act_space, 41)
        elif len(act_space) == 1:
            assert outputs['logit'].shape == (B, *act_space)
            assert outputs['distribution'].shape == (B, *act_space, 41)
        else:
            for i, s in enumerate(act_space):
                assert outputs['logit'][i].shape == (B, s)
                assert outputs['distribution'][i].shape == (B, s, 41)
        self.output_check(model, outputs['logit'])
