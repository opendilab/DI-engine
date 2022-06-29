import pytest
import torch
from ding.policy.mbpolicy.utils import q_evaluation


@pytest.mark.unittest
def test_q_evaluation():
    T, B, O, A = 10, 20, 100, 30
    obss = torch.randn(T, B, O)
    actions = torch.randn(T, B, A)

    def fake_q_fn(obss, actions):
        # obss:    flatten_B * O
        # actions: flatten_B * A
        # return:  flatten_B
        return obss.sum(-1)

    q_value = q_evaluation(obss, actions, fake_q_fn)
    assert q_value.shape == (T, B)
