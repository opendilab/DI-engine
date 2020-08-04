import pytest
import torch
import numpy as np
from sc2learner.model.alphastar.obs_encoder import ScalarEncoder, EntityEncoder, SpatialEncoder
from sc2learner.model.alphastar.encoder import Encoder
from sc2learner.model.alphastar.tests.conftest import is_differentiable


@pytest.mark.unittest
class TestEncoder:
    def test_scalar_encoder(self, setup_config, setup_env_info):
        B = 4
        handle = setup_env_info.obs_space['scalar'].shape
        cfg = setup_config.model.encoder.obs_encoder.scalar_encoder
        # merge input dim
        for k, v in cfg.module.items():
            cfg.module[k].input_dim = handle[k]

        model = ScalarEncoder(setup_config.model.encoder.obs_encoder.scalar_encoder)
        assert isinstance(model, torch.nn.Module)

        inputs = {}
        for k, v in handle.items():
            if k == 'beginning_build_order':
                N = setup_config.model.encoder.obs_encoder.scalar_encoder.begin_num
                # subtract seq info dim, which is automatedly added by bo network
                inputs[k] = torch.randn(B, N, v - 20)
            elif k == 'cumulative_stat':
                inputs[k] = {k1: torch.randn(B, v1) for k1, v1 in v.items()}
            else:
                inputs[k] = torch.randn(B, v)

        embedded_scalar, scalar_context, baseline_feature, cumulative_stat = model(inputs)
        assert (embedded_scalar.shape == (B, 1280))
        assert (scalar_context.shape == (B, 256))
        assert (baseline_feature.shape == (B, 352))
        for k, v in cumulative_stat.items():
            assert v.shape == (B, 32)

        loss = embedded_scalar.mean()
        is_differentiable(loss, model)

    def test_entity_encoder_input_list(self, setup_config, setup_env_info):
        B = 4
        handle = setup_config.model.encoder.obs_encoder.entity_encoder
        handle_env = setup_env_info.obs_space['entity']
        handle.input_dim = handle_env.shape[-1]
        model = EntityEncoder(handle)
        assert isinstance(model, torch.nn.Module)

        entity_nums = []
        inputs = []  # list or tuple
        # input_dim: 1340, output_dim: 256
        for b in range(B):
            entity_num = np.random.randint(200, 600)
            entity_nums.append(entity_num)
            inputs.append(torch.randn(entity_num, handle.input_dim))
        entity_embeddings, embedded_entity = model(inputs)
        assert isinstance(entity_embeddings, list)
        for entity_embedding, entity_num in zip(entity_embeddings, entity_nums):
            assert isinstance(entity_embedding, torch.Tensor)
            assert entity_embedding.shape == (entity_num, handle.output_dim)
        assert embedded_entity.shape == (B, handle.output_dim)
        loss = embedded_entity.mean() + sum([t.mean() for t in entity_embedding])
        is_differentiable(loss, model)

    def test_spatial_encoder(self, setup_config, setup_env_info):
        B = 5
        handle = setup_config.model.encoder.obs_encoder.spatial_encoder
        handle_env = setup_env_info.obs_space['spatial']
        handle.input_dim = handle_env.shape[0]
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

    def test_spatial_encoder_input_list(self, setup_config, setup_env_info):
        B = 5
        handle = setup_config.model.encoder.obs_encoder.spatial_encoder
        handle_env = setup_env_info.obs_space['spatial']
        handle.input_dim = handle_env.shape[0]
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

    def test_scatter_connection(self, setup_config, setup_env_info):
        B = 5
        handle = setup_config.model.encoder
        handle.obs_encoder.spatial_encoder.input_dim = setup_env_info.obs_space['spatial'].shape[0]
        handle.obs_encoder.entity_encoder.input_dim = setup_env_info.obs_space['entity'].shape[-1]
        for k, v in handle.obs_encoder.scalar_encoder.module.items():
            handle.obs_encoder.scalar_encoder.module[k].input_dim = setup_env_info.obs_space['scalar'].shape[k]
        handle.score_cumulative.input_dim = setup_env_info.obs_space['scalar'].shape['score_cumulative']
        model = Encoder(handle)
        assert isinstance(model, torch.nn.Module)
        map_size = (200, 180)

        spatial_info = torch.randn(B, 8, *map_size)
        entity_num = [np.random.randint(200, 300) for _ in range(B)]
        entity_raw = [
            {
                'location': [
                    [np.random.randint(0, map_size[0]),
                     np.random.randint(0, map_size[1])] for _ in range(entity_num[b])
                ]
            } for b in range(B)
        ]
        entity_embedding = [torch.randn(entity_num[b], handle.scatter.input_dim) for b in range(B)]

        output = model._scatter_connection(spatial_info, entity_embedding, entity_raw)
        assert output.shape == (B, 8 + handle.scatter.output_dim, *map_size)
        loss = output.mean()
        is_differentiable(loss, model.scatter_project)
