import pytest
import torch
import yaml
import os
from easydict import EasyDict
from sc2learner.agent.model.alphastar.obs_encoder import ScalarEncoder
from sc2learner.envs.observations import transform_scalar_data


@pytest.fixture(scope='function')
def setup_config():
    with open(os.path.join(os.path.dirname(__file__), '../actor_critic_default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg


@pytest.mark.unitest
class TestEncoder:
    def test_scalar_encoder(self, setup_config):
        B = 4
        template_obs, template_replay, template_action = transform_scalar_data()
        model = ScalarEncoder(setup_config.model.encoder.obs_encoder.scalar_encoder)
        assert isinstance(model, torch.nn.Module)

        inputs = {}
        for item in template_obs + template_replay + template_action:
            if 'input_dim' in item.keys() and 'output_dim' in item.keys():
                if item['key'] == 'beginning_build_order':
                    N = setup_config.model.encoder.obs_encoder.scalar_encoder.begin_num
                    inputs[item['key']] = torch.randn(B, N, item['input_dim'])
                else:
                    inputs[item['key']] = torch.randn(B, item['input_dim'])
        cumulative_stat_input = {}
        for k, v in template_replay[1]['input_dims'].items():
            cumulative_stat_input[k] = torch.randn(B, v)
        inputs['cumulative_stat'] = cumulative_stat_input

        embedded_scalar, scalar_context, baseline_feature, cumulative_stat = model(inputs)
        assert (embedded_scalar.shape == (B, 1280))
        assert (scalar_context.shape == (B, 256))
        assert (baseline_feature.shape == (B, 352))
        for k, v in cumulative_stat.items():
            assert v.shape == (B, template_replay[1]['output_dim'])

        loss = embedded_scalar.mean()
        for p in model.parameters():
            assert p.grad is None
        loss.backward()
        for p in model.parameters():
            assert isinstance(p.grad, torch.Tensor)
