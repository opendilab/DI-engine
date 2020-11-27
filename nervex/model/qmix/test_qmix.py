import pytest
import torch
from nervex.torch_utils import is_differentiable
from .qmix import Mixer


@pytest.mark.unittest
def test_mixer():
    agent_num, bs, embedding_dim = 4, 3, 32
    agent_q = torch.randn(agent_num, bs)
    state_embedding = torch.randn(bs, embedding_dim)
    mixer = Mixer(agent_num, embedding_dim)
    total_q = mixer(agent_q, state_embedding)
    assert total_q.shape == (bs, )
    loss = total_q.mean()
    is_differentiable(loss, mixer)
