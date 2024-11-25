import pytest
import torch
from ding.torch_utils import is_differentiable
from ding.model.template import Mixer, QMix


@pytest.mark.unittest
def test_mixer():
    agent_num, bs, embedding_dim = 4, 3, 32
    agent_q = torch.randn(bs, agent_num)
    state_embedding = torch.randn(bs, embedding_dim)
    mixer = Mixer(agent_num, embedding_dim, 64)
    total_q = mixer(agent_q, state_embedding)
    assert total_q.shape == (bs, )
    loss = total_q.mean()
    is_differentiable(loss, mixer)


@pytest.mark.unittest
def test_qmix():
    use_mixer = [True, False]
    agent_num, bs, T = 4, 3, 8
    obs_dim, global_obs_dim, action_dim = 32, 32 * 4, 9
    embedding_dim = 64
    for mix in use_mixer:
        qmix_model = QMix(agent_num, obs_dim, global_obs_dim, action_dim, [128, embedding_dim], mix)
        data = {
            'obs': {
                'agent_state': torch.randn(T, bs, agent_num, obs_dim),
                'global_state': torch.randn(T, bs, global_obs_dim),
                'action_mask': torch.randint(0, 2, size=(T, bs, agent_num, action_dim))
            },
            'prev_state': [[None for _ in range(agent_num)] for _ in range(bs)],
            'action': torch.randint(0, action_dim, size=(T, bs, agent_num))
        }
        output = qmix_model(data, single_step=False)
        assert set(output.keys()) == set(['total_q', 'logit', 'next_state', 'action_mask'])
        assert output['total_q'].shape == (T, bs)
        assert output['logit'].shape == (T, bs, agent_num, action_dim)
        assert len(output['next_state']) == bs and all([len(n) == agent_num for n in output['next_state']])
        print(output['next_state'][0][0]['h'].shape)
        loss = output['total_q'].sum()
        is_differentiable(loss, qmix_model)
        data.pop('action')
        output = qmix_model(data, single_step=False)


@pytest.mark.unittest
def test_qmix_process_global_state():
    # Test the behavior of the _process_global_state method with different global_obs_shape types
    agent_num, obs_dim, global_obs_dim, action_dim = 4, 32, 32 * 4, 9
    embedding_dim = 64

    # Case 1: Test "flat" type global_obs_shape
    global_obs_shape = global_obs_dim  # Flat global_obs_shape
    qmix_model_flat = QMix(agent_num, obs_dim, global_obs_shape, action_dim, [64, 128, embedding_dim], mixer=True)

    # Simulate input for the "flat" type global_state
    batch_size, time_steps = 3, 8
    global_state_flat = torch.randn(batch_size, time_steps, global_obs_dim)
    processed_flat = qmix_model_flat._process_global_state(global_state_flat)

    # Ensure the output shape is correct [batch_size, time_steps, embedding_dim]
    assert processed_flat.shape == (batch_size, time_steps, global_obs_dim)

    # Case 2: Test "image" type global_obs_shape
    global_obs_shape = [3, 64, 64]  # Image-shaped global_obs_shape (C, H, W)
    qmix_model_image = QMix(agent_num, obs_dim, global_obs_shape, action_dim, [64, 128, embedding_dim], mixer=True)

    # Simulate input for the "image" type global_state
    C, H, W = global_obs_shape
    global_state_image = torch.randn(batch_size, time_steps, C, H, W)
    processed_image = qmix_model_image._process_global_state(global_state_image)

    # Ensure the output shape is correct [batch_size, time_steps, embedding_dim]
    assert processed_image.shape == (batch_size, time_steps, embedding_dim)
