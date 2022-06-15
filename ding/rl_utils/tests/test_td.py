import pytest
import torch
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error, q_1step_td_data, q_1step_td_error, td_lambda_data,\
    td_lambda_error, q_nstep_td_error_with_rescale, dist_1step_td_data, dist_1step_td_error, dist_nstep_td_data,\
    dqfd_nstep_td_data, dqfd_nstep_td_error, dist_nstep_td_error, v_1step_td_data, v_1step_td_error, v_nstep_td_data,\
    v_nstep_td_error, q_nstep_sql_td_error, iqn_nstep_td_data, iqn_nstep_td_error,\
    fqf_nstep_td_data, fqf_nstep_td_error, qrdqn_nstep_td_data, qrdqn_nstep_td_error
from ding.rl_utils.td import shape_fn_dntd, shape_fn_qntd, shape_fn_td_lambda, shape_fn_qntd_rescale


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
        data = q_nstep_td_data(q, next_q, action, next_action, reward, done, None)
        loss, td_error_per_sample = q_nstep_td_error(data, 0.95, nstep=nstep, cum_reward=True)
        value_gamma = torch.tensor(0.9)
        data = q_nstep_td_data(q, next_q, action, next_action, reward, done, None)
        loss, td_error_per_sample = q_nstep_td_error(data, 0.95, nstep=nstep, cum_reward=True, value_gamma=value_gamma)
        loss.backward()
        assert isinstance(q.grad, torch.Tensor)


@pytest.mark.unittest
def test_q_nstep_td_ngu():
    batch_size = 4
    action_dim = 3
    next_q = torch.randn(batch_size, action_dim)
    done = torch.randn(batch_size)
    action = torch.randint(0, action_dim, size=(batch_size, ))
    next_action = torch.randint(0, action_dim, size=(batch_size, ))
    gamma = [torch.tensor(0.95) for i in range(batch_size)]

    for nstep in range(1, 10):
        q = torch.randn(batch_size, action_dim).requires_grad_(True)
        reward = torch.rand(nstep, batch_size)
        data = q_nstep_td_data(q, next_q, action, next_action, reward, done, None)
        loss, td_error_per_sample = q_nstep_td_error(data, gamma, nstep=nstep)
        assert td_error_per_sample.shape == (batch_size, )
        assert loss.shape == ()
        assert q.grad is None
        loss.backward()
        assert isinstance(q.grad, torch.Tensor)


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
    assert pytest.approx(nstep_loss.item()) == onestep_loss.item()


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
    weight = torch.tensor([0.9])
    value_gamma = torch.tensor(0.9)
    data = dist_nstep_td_data(dist, next_n_dist, action, next_action, reward, done, weight)
    loss, _ = dist_nstep_td_error(data, 0.95, v_min, v_max, n_atom, nstep, value_gamma)
    assert loss.shape == ()
    loss.backward()
    assert isinstance(dist.grad, torch.Tensor)


@pytest.mark.unittest
def test_dist_nstep_multi_agent_td():
    batch_size = 4
    action_dim = 3
    agent_num = 2
    n_atom = 51
    v_min = -10.0
    v_max = 10.0
    nstep = 5
    dist = torch.randn(batch_size, agent_num, action_dim, n_atom).abs().requires_grad_(True)
    next_n_dist = torch.randn(batch_size, agent_num, action_dim, n_atom).abs()
    done = torch.randint(0, 2, (batch_size, ))
    action = torch.randint(
        0, action_dim, size=(
            batch_size,
            agent_num,
        )
    )
    next_action = torch.randint(
        0, action_dim, size=(
            batch_size,
            agent_num,
        )
    )
    reward = torch.randn(nstep, batch_size)
    data = dist_nstep_td_data(dist, next_n_dist, action, next_action, reward, done, None)
    loss, _ = dist_nstep_td_error(data, 0.95, v_min, v_max, n_atom, nstep)
    assert loss.shape == ()
    assert dist.grad is None
    loss.backward()
    assert isinstance(dist.grad, torch.Tensor)
    weight = 0.9
    value_gamma = 0.9
    data = dist_nstep_td_data(dist, next_n_dist, action, next_action, reward, done, weight)
    loss, _ = dist_nstep_td_error(data, 0.95, v_min, v_max, n_atom, nstep, value_gamma)
    assert loss.shape == ()
    loss.backward()
    assert isinstance(dist.grad, torch.Tensor)
    agent_total_loss = 0
    for i in range(agent_num):
        data = dist_nstep_td_data(
            dist[:, i, ], next_n_dist[:, i, ], action[:, i, ], next_action[:, i, ], reward, done, weight
        )
        agent_loss, _ = dist_nstep_td_error(data, 0.95, v_min, v_max, n_atom, nstep, value_gamma)
        agent_total_loss = agent_total_loss + agent_loss
    agent_average_loss = agent_total_loss / agent_num
    assert abs(agent_average_loss.item() - loss.item()) < 1e-5


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
def test_q_nstep_td_with_rescale_ngu():
    batch_size = 4
    action_dim = 3
    next_q = torch.randn(batch_size, action_dim)
    done = torch.randn(batch_size)
    action = torch.randint(0, action_dim, size=(batch_size, ))
    next_action = torch.randint(0, action_dim, size=(batch_size, ))
    gamma = [torch.tensor(0.95) for i in range(batch_size)]
    for nstep in range(1, 10):
        q = torch.randn(batch_size, action_dim).requires_grad_(True)
        reward = torch.rand(nstep, batch_size)
        data = q_nstep_td_data(q, next_q, action, next_action, reward, done, None)
        loss, _ = q_nstep_td_error_with_rescale(data, gamma, nstep=nstep)
        assert loss.shape == ()
        assert q.grad is None
        loss.backward()
        assert isinstance(q.grad, torch.Tensor)
        print(loss)


@pytest.mark.unittest
def test_qrdqn_nstep_td():
    batch_size = 4
    action_dim = 3
    tau = 3
    next_q = torch.randn(batch_size, action_dim, tau)
    done = torch.randn(batch_size)
    action = torch.randint(0, action_dim, size=(batch_size, ))
    next_action = torch.randint(0, action_dim, size=(batch_size, ))
    for nstep in range(1, 10):
        q = torch.randn(batch_size, action_dim, tau).requires_grad_(True)
        reward = torch.rand(nstep, batch_size)
        data = qrdqn_nstep_td_data(q, next_q, action, next_action, reward, done, tau, None)
        loss, td_error_per_sample = qrdqn_nstep_td_error(data, 0.95, nstep=nstep)
        assert td_error_per_sample.shape == (batch_size, )
        assert loss.shape == ()
        assert q.grad is None
        loss.backward()
        assert isinstance(q.grad, torch.Tensor)
        loss, td_error_per_sample = qrdqn_nstep_td_error(data, 0.95, nstep=nstep, value_gamma=torch.tensor(0.9))
        assert td_error_per_sample.shape == (batch_size, )


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
    assert pytest.approx(nstep_loss.item()) == onestep_loss.item()


@pytest.mark.unittest
def test_dist_1step_multi_agent_td():
    batch_size = 4
    action_dim = 3
    agent_num = 2
    n_atom = 51
    v_min = -10.0
    v_max = 10.0
    dist = torch.randn(batch_size, agent_num, action_dim, n_atom).abs().requires_grad_(True)
    next_dist = torch.randn(batch_size, agent_num, action_dim, n_atom).abs()
    done = torch.randint(0, 2, (batch_size, ))
    action = torch.randint(
        0, action_dim, size=(
            batch_size,
            agent_num,
        )
    )
    next_action = torch.randint(
        0, action_dim, size=(
            batch_size,
            agent_num,
        )
    )
    reward = torch.randn(batch_size)
    data = dist_1step_td_data(dist, next_dist, action, next_action, reward, done, None)
    loss = dist_1step_td_error(data, 0.95, v_min, v_max, n_atom)
    assert loss.shape == ()
    assert dist.grad is None
    loss.backward()
    assert isinstance(dist.grad, torch.Tensor)
    agent_total_loss = 0
    for i in range(agent_num):
        data = dist_1step_td_data(
            dist[:, i, ], next_dist[:, i, ], action[:, i, ], next_action[:, i, ], reward, done, None
        )
        agent_loss = dist_1step_td_error(data, 0.95, v_min, v_max, n_atom)
        agent_total_loss = agent_total_loss + agent_loss
    agent_average_loss = agent_total_loss / agent_num
    assert abs(agent_average_loss.item() - loss.item()) < 1e-5


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
    loss.backward()
    assert isinstance(v.grad, torch.Tensor)


@pytest.mark.unittest
def test_v_1step_multi_agent_td():
    batch_size = 5
    agent_num = 2
    v = torch.randn(batch_size, agent_num).requires_grad_(True)
    next_v = torch.randn(batch_size, agent_num)
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
    loss.backward()
    assert isinstance(v.grad, torch.Tensor)


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
    data = v_nstep_td_data(v, next_v, reward, done, None, 0.99)
    loss, td_error_per_sample = v_nstep_td_error(data, 0.99, 5)
    loss.backward()
    assert isinstance(v.grad, torch.Tensor)


@pytest.mark.unittest
def test_dqfd_nstep_td():
    batch_size = 4
    action_dim = 3
    next_q = torch.randn(batch_size, action_dim)
    done = torch.randn(batch_size)
    done_1 = torch.randn(batch_size)
    next_q_one_step = torch.randn(batch_size, action_dim)
    action = torch.randint(0, action_dim, size=(batch_size, ))
    next_action = torch.randint(0, action_dim, size=(batch_size, ))
    next_action_one_step = torch.randint(0, action_dim, size=(batch_size, ))
    is_expert = torch.ones((batch_size))
    for nstep in range(1, 10):
        q = torch.randn(batch_size, action_dim).requires_grad_(True)
        reward = torch.rand(nstep, batch_size)
        data = dqfd_nstep_td_data(
            q, next_q, action, next_action, reward, done, done_1, None, next_q_one_step, next_action_one_step, is_expert
        )
        loss, td_error_per_sample, loss_statistics = dqfd_nstep_td_error(
            data, 0.95, lambda_n_step_td=1, lambda_supervised_loss=1, margin_function=0.8, nstep=nstep
        )
        assert td_error_per_sample.shape == (batch_size, )
        assert loss.shape == ()
        assert q.grad is None
        loss.backward()
        assert isinstance(q.grad, torch.Tensor)
        print(loss)


@pytest.mark.unittest
def test_q_nstep_sql_td():
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
        loss, td_error_per_sample, record_target_v = q_nstep_sql_td_error(data, 0.95, 1.0, nstep=nstep)
        assert td_error_per_sample.shape == (batch_size, )
        assert loss.shape == ()
        assert q.grad is None
        loss.backward()
        assert isinstance(q.grad, torch.Tensor)
        data = q_nstep_td_data(q, next_q, action, next_action, reward, done, None)
        loss, td_error_per_sample, record_target_v = q_nstep_sql_td_error(data, 0.95, 0.5, nstep=nstep, cum_reward=True)
        value_gamma = torch.tensor(0.9)
        data = q_nstep_td_data(q, next_q, action, next_action, reward, done, None)
        loss, td_error_per_sample, record_target_v = q_nstep_sql_td_error(
            data, 0.95, 0.5, nstep=nstep, cum_reward=True, value_gamma=value_gamma
        )
        loss.backward()
        assert isinstance(q.grad, torch.Tensor)


@pytest.mark.unittest
def test_iqn_nstep_td():
    batch_size = 4
    action_dim = 3
    tau = 3
    next_q = torch.randn(tau, batch_size, action_dim)
    done = torch.randn(batch_size)
    action = torch.randint(0, action_dim, size=(batch_size, ))
    next_action = torch.randint(0, action_dim, size=(batch_size, ))
    for nstep in range(1, 10):
        q = torch.randn(tau, batch_size, action_dim).requires_grad_(True)
        replay_quantile = torch.randn([tau, batch_size, 1])
        reward = torch.rand(nstep, batch_size)
        data = iqn_nstep_td_data(q, next_q, action, next_action, reward, done, replay_quantile, None)
        loss, td_error_per_sample = iqn_nstep_td_error(data, 0.95, nstep=nstep)
        assert td_error_per_sample.shape == (batch_size, )
        assert loss.shape == ()
        assert q.grad is None
        loss.backward()
        assert isinstance(q.grad, torch.Tensor)
        loss, td_error_per_sample = iqn_nstep_td_error(data, 0.95, nstep=nstep, value_gamma=torch.tensor(0.9))
        assert td_error_per_sample.shape == (batch_size, )


@pytest.mark.unittest
def test_fqf_nstep_td():
    batch_size = 4
    action_dim = 3
    tau = 3
    next_q = torch.randn(batch_size, tau, action_dim)
    done = torch.randn(batch_size)
    action = torch.randint(0, action_dim, size=(batch_size, ))
    next_action = torch.randint(0, action_dim, size=(batch_size, ))
    for nstep in range(1, 10):
        q = torch.randn(batch_size, tau, action_dim).requires_grad_(True)
        quantiles_hats = torch.randn([batch_size, tau])
        reward = torch.rand(nstep, batch_size)
        data = fqf_nstep_td_data(q, next_q, action, next_action, reward, done, quantiles_hats, None)
        loss, td_error_per_sample = fqf_nstep_td_error(data, 0.95, nstep=nstep)
        assert td_error_per_sample.shape == (batch_size, )
        assert loss.shape == ()
        assert q.grad is None
        loss.backward()
        assert isinstance(q.grad, torch.Tensor)
        loss, td_error_per_sample = fqf_nstep_td_error(data, 0.95, nstep=nstep, value_gamma=torch.tensor(0.9))
        assert td_error_per_sample.shape == (batch_size, )


@pytest.mark.unittest
def test_shape_fn_qntd():
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
        tmp = shape_fn_qntd([data, 0.95, 1], {})
        assert tmp[0] == reward.shape[0]
        assert tmp[1] == q.shape[0]
        assert tmp[2] == q.shape[1]
        tmp = shape_fn_qntd([], {'gamma': 0.95, 'nstep': 1, 'data': data})
        assert tmp[0] == reward.shape[0]
        assert tmp[1] == q.shape[0]
        assert tmp[2] == q.shape[1]


@pytest.mark.unittest
def test_shape_fn_dntd():
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
    tmp = shape_fn_dntd([data, 0.9, v_min, v_max, n_atom, nstep], {})
    assert tmp[0] == reward.shape[0]
    assert tmp[1] == dist.shape[0]
    assert tmp[2] == dist.shape[1]
    assert tmp[3] == n_atom
    tmp = shape_fn_dntd([], {'data': data, 'gamma': 0.9, 'v_min': v_min, 'v_max': v_max, 'n_atom': n_atom, 'nstep': 5})
    assert tmp[0] == reward.shape[0]
    assert tmp[1] == dist.shape[0]
    assert tmp[2] == dist.shape[1]
    assert tmp[3] == n_atom


@pytest.mark.unittest
def test_shape_fn_qntd_rescale():
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
        tmp = shape_fn_qntd_rescale([data, 0.95, 1], {})
        assert tmp[0] == reward.shape[0]
        assert tmp[1] == q.shape[0]
        assert tmp[2] == q.shape[1]
        tmp = shape_fn_qntd_rescale([], {'gamma': 0.95, 'nstep': 1, 'data': data})
        assert tmp[0] == reward.shape[0]
        assert tmp[1] == q.shape[0]
        assert tmp[2] == q.shape[1]


@pytest.mark.unittest
def test_fn_td_lambda():
    T, B = 8, 4
    value = torch.randn(T + 1, B).requires_grad_(True)
    reward = torch.rand(T, B)
    data = td_lambda_data(value, reward, None)
    tmp = shape_fn_td_lambda([], {'data': data})
    assert tmp == reward.shape[0]
    tmp = shape_fn_td_lambda([data], {})
    assert tmp == reward.shape
