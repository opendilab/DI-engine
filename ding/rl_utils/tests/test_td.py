import pytest
import torch
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error, q_1step_td_data, q_1step_td_error, td_lambda_data,\
    td_lambda_error, q_nstep_td_error_with_rescale, dist_1step_td_data, dist_1step_td_error, dist_nstep_td_data, \
    dist_nstep_td_error, v_1step_td_data, v_1step_td_error, v_nstep_td_data, v_nstep_td_error


@pytest.mark.unittest
def test_q_nstep_td():
    batch_size = 4
    action_dim = 3
    next_q = torch.randn(batch_size, action_dim)
    done = torch.randn(batch_size)
    action = torch.randint(0, action_dim, size=(batch_size, ))
    next_action = torch.randint(0, action_dim, size=(batch_size, ))
    for nstep in range(1, 10):
        q = torch.randn(batch_size, action_dim).requires_grad_(True)
        reward = torch.rand(nstep, batch_size)
        data = q_nstep_td_data(q, next_q, action, next_action, reward, done, None)
        loss, td_error_per_sample = q_nstep_td_error(data, 0.95, nstep=nstep)
        assert td_error_per_sample.shape == (batch_size, )
        assert loss.shape == ()
        assert q.grad is None
        loss.backward()
        assert isinstance(q.grad, torch.Tensor)
        print(loss)


@pytest.mark.unittest
def test_dist_1step_td():
    batch_size = 4
    action_dim = 3
    n_atom = 51
    v_min = -10.0
    v_max = 10.0
    dist = torch.randn(batch_size, action_dim, n_atom).abs().requires_grad_(True)
    next_dist = torch.randn(batch_size, action_dim, n_atom).abs()
    done = torch.randn(batch_size)
    action = torch.randint(0, action_dim, size=(batch_size, ))
    next_action = torch.randint(0, action_dim, size=(batch_size, ))
    reward = torch.randn(batch_size)
    data = dist_1step_td_data(dist, next_dist, action, next_action, reward, done, None)
    loss = dist_1step_td_error(data, 0.95, v_min, v_max, n_atom)
    assert loss.shape == ()
    assert dist.grad is None
    loss.backward()
    assert isinstance(dist.grad, torch.Tensor)


@pytest.mark.unittest
def test_q_1step_compatible():
    batch_size = 4
    action_dim = 3
    next_q = torch.randn(batch_size, action_dim)
    done = torch.randn(batch_size)
    action = torch.randint(0, action_dim, size=(batch_size, ))
    next_action = torch.randint(0, action_dim, size=(batch_size, ))
    q = torch.randn(batch_size, action_dim).requires_grad_(True)
    reward = torch.rand(batch_size)
    nstep_data = q_nstep_td_data(q, next_q, action, next_action, reward.unsqueeze(0), done, None)
    onestep_data = q_1step_td_data(q, next_q, action, next_action, reward, done, None)
    nstep_loss, _ = q_nstep_td_error(nstep_data, 0.99, nstep=1)
    onestep_loss = q_1step_td_error(onestep_data, 0.99)
    assert pytest.approx(nstep_loss.item(), onestep_loss.item())


@pytest.mark.unittest
def test_dist_nstep_td():
    batch_size = 4
    action_dim = 3
    n_atom = 51
    v_min = -10.0
    v_max = 10.0
    nstep = 5
    dist = torch.randn(batch_size, action_dim, n_atom).abs().requires_grad_(True)
    next_n_dist = torch.randn(batch_size, action_dim, n_atom).abs()
    done = torch.randn(batch_size)
    action = torch.randint(0, action_dim, size=(batch_size, ))
    next_action = torch.randint(0, action_dim, size=(batch_size, ))
    reward = torch.randn(nstep, batch_size)
    data = dist_nstep_td_data(dist, next_n_dist, action, next_action, reward, done, None)
    loss, _ = dist_nstep_td_error(data, 0.95, v_min, v_max, n_atom, nstep)
    assert loss.shape == ()
    assert dist.grad is None
    loss.backward()
    assert isinstance(dist.grad, torch.Tensor)


@pytest.mark.unittest
def test_q_nstep_td_with_rescale():
    batch_size = 4
    action_dim = 3
    next_q = torch.randn(batch_size, action_dim)
    done = torch.randn(batch_size)
    action = torch.randint(0, action_dim, size=(batch_size, ))
    next_action = torch.randint(0, action_dim, size=(batch_size, ))
    for nstep in range(1, 10):
        q = torch.randn(batch_size, action_dim).requires_grad_(True)
        reward = torch.rand(nstep, batch_size)
        data = q_nstep_td_data(q, next_q, action, next_action, reward, done, None)
        loss, _ = q_nstep_td_error_with_rescale(data, 0.95, nstep=nstep)
        assert loss.shape == ()
        assert q.grad is None
        loss.backward()
        assert isinstance(q.grad, torch.Tensor)
        print(loss)


@pytest.mark.unittest
def test_dist_1step_compatible():
    batch_size = 4
    action_dim = 3
    n_atom = 51
    v_min = -10.0
    v_max = 10.0
    dist = torch.randn(batch_size, action_dim, n_atom).abs().requires_grad_(True)
    next_dist = torch.randn(batch_size, action_dim, n_atom).abs()
    done = torch.randn(batch_size)
    action = torch.randint(0, action_dim, size=(batch_size, ))
    next_action = torch.randint(0, action_dim, size=(batch_size, ))
    reward = torch.randn(batch_size)
    onestep_data = dist_1step_td_data(dist, next_dist, action, next_action, reward, done, None)
    nstep_data = dist_nstep_td_data(dist, next_dist, action, next_action, reward.unsqueeze(0), done, None)
    onestep_loss = dist_1step_td_error(onestep_data, 0.95, v_min, v_max, n_atom)
    nstep_loss, _ = dist_nstep_td_error(nstep_data, 0.95, v_min, v_max, n_atom, nstep=1)
    assert pytest.approx(nstep_loss.item(), onestep_loss.item())


@pytest.mark.unittest
def test_td_lambda():
    T, B = 8, 4
    value = torch.randn(T + 1, B).requires_grad_(True)
    reward = torch.rand(T, B)
    loss = td_lambda_error(td_lambda_data(value, reward, None))
    assert loss.shape == ()
    assert value.grad is None
    loss.backward()
    assert isinstance(value.grad, torch.Tensor)


@pytest.mark.unittest
def test_v_1step_td():
    batch_size = 5
    v = torch.randn(batch_size).requires_grad_(True)
    next_v = torch.randn(batch_size)
    reward = torch.rand(batch_size)
    done = torch.zeros(batch_size)
    data = v_1step_td_data(v, next_v, reward, done, None)
    loss, td_error_per_sample = v_1step_td_error(data, 0.99)
    assert loss.shape == ()
    assert v.grad is None
    loss.backward()
    assert isinstance(v.grad, torch.Tensor)
    data = v_1step_td_data(v, next_v, reward, None, None)
    loss, td_error_per_sample = v_1step_td_error(data, 0.99)


@pytest.mark.unittest
def test_v_nstep_td():
    batch_size = 5
    v = torch.randn(batch_size).requires_grad_(True)
    next_v = torch.randn(batch_size)
    reward = torch.rand(5, batch_size)
    done = torch.zeros(batch_size)
    data = v_nstep_td_data(v, next_v, reward, done, 0.9, 0.99)
    loss, td_error_per_sample = v_nstep_td_error(data, 0.99, 5)
    assert loss.shape == ()
    assert v.grad is None
    loss.backward()
    assert isinstance(v.grad, torch.Tensor)
    data = v_nstep_td_data(v, next_v, reward, done, 0.9, 0.99)
    loss, td_error_per_sample = v_nstep_td_error(data, 0.99, 5)
