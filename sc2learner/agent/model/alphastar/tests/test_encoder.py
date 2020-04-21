import pytest
import torch
import numpy as np
import yaml
import os
from easydict import EasyDict
from sc2learner.agent.model.alphastar.obs_encoder import ScalarEncoder, EntityEncoder, SpatialEncoder
from sc2learner.envs.observations import transform_scalar_data


@pytest.fixture(scope='function')
def setup_config():
    with open(os.path.join(os.path.dirname(__file__), '../actor_critic_default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg


def is_differentiable(loss, model):
    assert isinstance(loss, torch.Tensor)
    assert isinstance(model, torch.nn.Module)
    for p in model.parameters():
        assert p.grad is None
    loss.backward()
    for k, p in model.named_parameters():
        assert isinstance(p.grad, torch.Tensor), k


@pytest.mark.unittest
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
        is_differentiable(loss, model)

    def test_entity_encoder(self, setup_config):
        B = 4
        handle = setup_config.model.encoder.obs_encoder.entity_encoder
        model = EntityEncoder(handle)
        assert isinstance(model, torch.nn.Module)

        entity_num = np.random.randint(200, 300)
        # input_dim: 2102, output_dim: 256
        inputs = torch.randn(B, entity_num, handle.input_dim)
        entity_embeddings, embedded_entity = model(inputs)
        assert entity_embeddings.shape == (B, entity_num, handle.output_dim)
        assert embedded_entity.shape == (B, handle.output_dim)
        loss = embedded_entity.mean() + entity_embeddings.mean()
        is_differentiable(loss, model)

    def test_entity_encoder_input_list(self, setup_config):
        B = 4
        handle = setup_config.model.encoder.obs_encoder.entity_encoder
        model = EntityEncoder(handle)
        assert isinstance(model, torch.nn.Module)

        entity_nums = []
        inputs = []  # list or tuple
        for b in range(B):
            entity_num = np.random.randint(200, 300)
            entity_nums.append(entity_num)
            inputs.append(torch.randn(entity_num, handle.input_dim))
        entity_embeddings, embedded_entity = model(inputs)
        assert isinstance(entity_embeddings, tuple)
        for entity_embedding, entity_num in zip(entity_embeddings, entity_nums):
            assert isinstance(entity_embedding, torch.Tensor)
            assert entity_embedding.shape == (entity_num, handle.output_dim)
        assert embedded_entity.shape == (B, handle.output_dim)

    def test_spatial_encoder(self, setup_config):
        B = 5
        handle = setup_config.model.encoder.obs_encoder.spatial_encoder
        model = SpatialEncoder(handle)
        assert isinstance(model, torch.nn.Module)

        H = np.random.randint(20, 25) * 8
        W = np.random.randint(20, 25) * 8
        # input_dim: 52(20 + 32)
        inputs = torch.randn(B, handle.input_dim, H, W)
        map_size = [[H, W] for _ in range(B)]
        output, map_skip = model(inputs, map_size)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (B, handle.fc_dim)
        assert isinstance(map_skip, list)
        # resblock_num: 4
        assert len(map_skip) == handle.resblock_num
        for m in map_skip:
            assert len(m) == B
            for t in m:
                assert t.shape == (handle.down_channels[-1], H // 8, W // 8)
        loss = output.mean()
        is_differentiable(loss, model)

    def test_spatial_encoder_input_list(self, setup_config):
        B = 5
        handle = setup_config.model.encoder.obs_encoder.spatial_encoder
        model = SpatialEncoder(handle)
        assert isinstance(model, torch.nn.Module)

        for n in range(10):
            inputs = []
            map_size = []
            for _ in range(B):
                H = np.random.randint(20, 25) * 8
                W = np.random.randint(20, 25) * 8
                inputs.append(torch.randn(handle.input_dim, H, W))
                map_size.append([H, W])
            if n % 2 == 0:
                # fake pad input
                max_h, max_w = max(list(zip(*map_size))[0]), max(list(zip(*map_size))[1])
                inputs = torch.randn(B, handle.input_dim, max_h, max_w)
            output, map_skip = model(inputs, map_size)
            assert isinstance(output, torch.Tensor)
            assert output.shape == (B, handle.fc_dim)
            assert isinstance(map_skip, list)
            # resblock_num: 4
            assert len(map_skip) == handle.resblock_num
            for m in map_skip:
                assert len(m) == B
                for t, (H, W) in zip(m, map_size):
                    assert t.shape == (handle.down_channels[-1], H // 8, W // 8)
