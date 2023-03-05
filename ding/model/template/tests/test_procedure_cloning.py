import torch
import pytest

from ding.model.template import ProcedureCloningMCTS

B = 4
T = 15
obs_shape = (64, 64, 3)
hidden_shape = (9, 9, 64)
action_dim = 9
obs_embeddings = 256


@pytest.mark.unittest
def test_procedure_cloning():
    inputs = {
        'states': torch.randn(B, *obs_shape),
        'hidden_states': torch.randn(B, T, *hidden_shape),
        'actions': torch.randn(B, action_dim)
    }
    model = ProcedureCloningMCTS(obs_shape=obs_shape, hidden_shape=hidden_shape,
                                 seq_len=T, action_dim=action_dim)

    print(model)

    hidden_state_preds, action_preds, target_hidden_state = model(inputs['states'], inputs['hidden_states'])
    assert hidden_state_preds.shape == (B, T, obs_embeddings)
    assert action_preds.shape == (B, action_dim)

    action_eval = model.forward_eval(inputs['states'])
    assert action_eval.shape == (B, action_dim)
