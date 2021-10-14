import pytest
import torch
from dizoo.gobigger.model import GoBiggerHybridNetwork
B, A = 3, 2


@pytest.mark.envtest
def test_hybrid_network():
    model = GoBiggerHybridNetwork((3, 144, 144), 21, 42, 16, rnn=True)
    print(model)
    team_obs = []
    for i in range(A):
        scalar_obs = torch.randn(B, 21)
        unit_obs = torch.randn(B, 30, 42)
        spatial_obs = torch.randn(B, 3, 144, 144)
        prev_state = [[torch.randn(1, 1, 128) for __ in range(2)] for _ in range(B)]
        obs = {'spatial_obs': spatial_obs, 'scalar_obs': scalar_obs, 'unit_obs': unit_obs, 'prev_state': prev_state}
        team_obs.append(obs)
    output = model(team_obs)
    assert output['logit'].shape == (B, A, 16)
    assert len(output['next_state']) == B and len(output['next_state'][0]) == A
