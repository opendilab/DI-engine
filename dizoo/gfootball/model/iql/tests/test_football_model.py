import pytest
import copy
import torch
from dizoo.gfootball.model.iql.iql_network import FootballIQL, ScalarEncoder, PlayerEncoder, FootballHead
from ding.utils import deep_merge_dicts
from ding.torch_utils import to_tensor, to_dtype
from dizoo.gfootball.envs.fake_dataset import FakeGfootballDataset
import pprint


@pytest.mark.envtest
class TestModel:

    def test_encoder(self, setup_config):
        B = 4
        # scalar_encoder_arch = setup_config.model.encoder.match_scalar
        # player_attr_dim = setup_config.model.encoder.player.transformer.player_attr_dim
        # action_dim = setup_config.model.policy.action_dim
        cfg = copy.deepcopy(setup_config)

        for t in ['transformer', 'spatial']:
            cfg.model.encoder.player.encoder_type = t

            # inputs = {}
            # for k, v in scalar_encoder_arch.items():
            #     inputs[k] = torch.randn(B, v['input_dim'])
            # inputs['players'] = []
            # for _ in range(22):
            #     inputs['players'].append({k: torch.randn(B, v) for k, v in player_attr_dim.items()})
            fake_dataset = FakeGfootballDataset()
            inputs = fake_dataset.get_batched_obs(bs=B)
            pp = pprint.PrettyPrinter(indent=2)
            print('observation: ')
            pp.pprint(inputs)

            model = FootballIQL(cfg)
            assert isinstance(model, torch.nn.Module)
            inputs = to_dtype(inputs, torch.float32)
            inputs = to_tensor(inputs)
            action = model(inputs)
            assert action.shape == (B, 19)
