import pytest
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F

from ding.model.template import DecisionTransformer
from ding.torch_utils import is_differentiable

action_space = ['continuous', 'discrete']
state_encoder = [None, nn.Sequential(nn.Flatten(), nn.Linear(8, 8), nn.Tanh())]
args = list(product(*[action_space, state_encoder]))
args.pop(1)


@pytest.mark.unittest
@pytest.mark.parametrize('action_space, state_encoder', args)
def test_decision_transformer(action_space, state_encoder):
    B, T = 4, 6
    if state_encoder:
        state_dim = (2, 2, 2)
    else:
        state_dim = 3
    act_dim = 2
    DT_model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        state_encoder=state_encoder,
        n_blocks=3,
        h_dim=8,
        context_len=T,
        n_heads=2,
        drop_p=0.1,
        continuous=(action_space == 'continuous')
    )
    DT_model.configure_optimizers(1.0, 0.0003)

    is_continuous = True if action_space == 'continuous' else False
    if state_encoder:
        timesteps = torch.randint(0, 100, [B, 3 * T - 1, 1], dtype=torch.long)  # B x T
    else:
        timesteps = torch.randint(0, 100, [B, T], dtype=torch.long)  # B x T
    if isinstance(state_dim, int):
        states = torch.randn([B, T, state_dim])  # B x T x state_dim
    else:
        states = torch.randn([B, T, *state_dim])  # B x T x state_dim
    if action_space == 'continuous':
        actions = torch.randn([B, T, act_dim])  # B x T x act_dim
        action_target = torch.randn([B, T, act_dim])
    else:
        actions = torch.randint(0, act_dim, [B, T, 1])
        action_target = torch.randint(0, act_dim, [B, T, 1])
    returns_to_go_sample = torch.tensor([1, 0.8, 0.6, 0.4, 0.2, 0.])
    returns_to_go = returns_to_go_sample.repeat([B, 1]).unsqueeze(-1)  # B x T x 1

    # all ones since no padding
    traj_mask = torch.ones([B, T], dtype=torch.long)  # B x T

    if is_continuous:
        assert action_target.shape == (B, T, act_dim)
    else:
        assert action_target.shape == (B, T, 1)
        actions = actions.squeeze(-1)

    returns_to_go = returns_to_go.float()
    state_preds, action_preds, return_preds = DT_model.forward(
        timesteps=timesteps, states=states, actions=actions, returns_to_go=returns_to_go
    )
    if state_encoder:
        assert state_preds is None
        assert return_preds is None
    else:
        assert state_preds.shape == (B, T, state_dim)
        assert return_preds.shape == (B, T, 1)
    assert action_preds.shape == (B, T, act_dim)

    # only consider non padded elements
    if state_encoder:
        action_preds = action_preds.reshape(-1, act_dim)
    else:
        action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1, ) > 0]

    if is_continuous:
        action_target = action_target.view(-1, act_dim)[traj_mask.view(-1, ) > 0]
    else:
        action_target = action_target.view(-1)[traj_mask.view(-1, ) > 0]

    if is_continuous:
        action_loss = F.mse_loss(action_preds, action_target)
    else:
        action_loss = F.cross_entropy(action_preds, action_target)

    if state_encoder:
        is_differentiable(
            action_loss, [DT_model.transformer, DT_model.embed_action, DT_model.embed_rtg, DT_model.state_encoder]
        )
    else:
        is_differentiable(
            action_loss, [
                DT_model.transformer, DT_model.embed_action, DT_model.predict_action, DT_model.embed_rtg,
                DT_model.embed_state
            ]
        )
