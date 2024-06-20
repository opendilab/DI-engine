import pytest
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F

from ding.model.template.elastic_decision_transformer import ElasticDecisionTransformer
from ding.torch_utils import is_differentiable

@pytest.mark.unittest
def test_elastic_decision_transformer():
    B, T = 4, 6
    state_dim = 3
    act_dim = 2
    num_bin = 120
    model = ElasticDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        h_dim=8,
        context_len=T,
        num_bin=num_bin,
        n_blocks=3,
        n_heads=2, #! H must be divisible by n_heads
        drop_p=0.1,
        env_name="example_env",
        num_inputs=3, #! must be 3 or 4
        is_continuous=True
    )

    timesteps = torch.randint(0, 100, (B, T))
    
    states = torch.randn(B, T, state_dim)
    
    actions = torch.randn(B, T, act_dim)
    action_target = torch.randn([B, T, act_dim])
    
    returns_to_go_sample = torch.tensor([1, 0.8, 0.6, 0.4, 0.2, 0.])
    returns_to_go = returns_to_go_sample.repeat([B, 1]).unsqueeze(-1)  # B x T x 1
    
    rewards = torch.randn(B, T, 1)

    traj_mask = torch.ones([B, T], dtype=torch.long)

    assert action_target.shape == (B, T, act_dim)
    returns_to_go = returns_to_go.float()
    # Forward
    state_preds, action_preds, return_preds, return_preds2, reward_preds = model(
        timesteps, states, actions, returns_to_go, rewards = rewards
    )
    assert state_preds.shape == torch.Size([B, T, state_dim])
    assert action_preds.shape == torch.Size([B, T, act_dim])
    assert return_preds.shape == torch.Size([B, T, num_bin])
    assert return_preds2.shape == torch.Size([B, T, 1])
    assert reward_preds.shape == torch.Size([B, T, 1])
    
    action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1, ) > 0]
    action_target = action_target.view(-1, act_dim)[traj_mask.view(-1, ) > 0]
    
    action_loss = F.mse_loss(action_preds, action_target)
    
    is_differentiable(
                action_loss, [
                    model.transformer, model.embed_action, model.predict_action, model.embed_rtg,
                    model.embed_state
                ]
            )

# if __name__ == "__main__":
#     test_elastic_decision_transformer()
#     print(f"Finished！")
