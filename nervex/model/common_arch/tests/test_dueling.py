import torch
import numpy as np
import pytest

from nervex.model import DuelingHead
from nervex.torch_utils import is_differentiable

B = 4
T = 6
embedding_dim = 64
action_dim = 12


@pytest.mark.unittest
class TestDuelingHead:

    def output_check(self, model, outputs, act_dim):
        if isinstance(act_dim, tuple):
            loss = sum([t.sum() for t in outputs])
        elif np.isscalar(act_dim):
            loss = outputs.sum()
        is_differentiable(loss, model)

    def test_dueling(self):
        inputs = torch.randn(B, embedding_dim)
        model = DuelingHead(embedding_dim, action_dim, 3, 3)
        outputs = model(inputs)
        self.output_check(model, outputs, action_dim)
        assert outputs.shape == (B, action_dim)
