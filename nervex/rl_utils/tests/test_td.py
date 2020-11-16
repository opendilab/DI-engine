import pytest
import torch
from nervex.rl_utils import q_nstep_td_data, q_nstep_td_error, q_1step_td_data, q_1step_td_error, td_lambda_data,\
    td_lambda_error


@pytest.mark.unittest
def test_q_nstep_td():
    batch_size = 4
    action_dim = 3
    next_q = torch.randn(batch_size, action_dim)
    done = torch.randn(batch_size)
    action = torch.randint(0, action_dim, size=(batch_size, ))
    for nstep in range(1, 10):
        q = torch.randn(batch_size, action_dim).requires_grad_(True)
        reward = torch.rand(nstep, batch_size)
        data = q_nstep_td_data(q, next_q, action, reward, done)
        loss = q_nstep_td_error(data, 0.95, nstep=nstep)
        assert loss.shape == ()
        assert q.grad is None
        loss.backward()
        assert isinstance(q.grad, torch.Tensor)
        print(loss)


@pytest.mark.unittest
def test_1step_compatible():
    batch_size = 4
    action_dim = 3
    next_q = torch.randn(batch_size, action_dim)
    done = torch.randn(batch_size)
    action = torch.randint(0, action_dim, size=(batch_size, ))
    q = torch.randn(batch_size, action_dim).requires_grad_(True)
    reward = torch.rand(batch_size)
    nstep_data = q_nstep_td_data(q, next_q, action, reward.unsqueeze(0), done)
    onestep_data = q_1step_td_data(q, next_q, action, reward, done)
    nstep_loss = q_nstep_td_error(nstep_data, 0.99, nstep=1)
    onestep_loss = q_1step_td_error(onestep_data, 0.99)
    assert nstep_loss.item() == onestep_loss.item()


@pytest.mark.unittest
def test_td_lambda():
    T, B = 8, 4
    value = torch.randn(T+1, B).requires_grad_(True)
    reward = torch.rand(T, B)
    loss = td_lambda_error(td_lambda_data(value, reward, None))
    assert loss.shape == ()
    assert value.grad is None
    loss.backward()
    assert isinstance(value.grad, torch.Tensor)
