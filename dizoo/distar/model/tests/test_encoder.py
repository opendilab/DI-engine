from easydict import EasyDict
import pytest
import torch
from ding.utils import read_yaml_config
from ding.utils.data import default_collate
from dizoo.distar.model.encoder import Encoder
from dizoo.distar.envs.fake_data import scalar_info, entity_info, spatial_info


@pytest.mark.envtest
def test_encoder():
    B, M = 4, 512
    cfg = read_yaml_config('../actor_critic_default_config.yaml')
    cfg = EasyDict(cfg)
    encoder = Encoder(cfg)
    print(encoder)

    spatial_info_data = default_collate([spatial_info() for _ in range(B)])
    entity_info_data = default_collate([entity_info() for _ in range(B)])
    scalar_info_data = default_collate([scalar_info() for _ in range(B)])
    entity_num = torch.randint(M // 2, M, size=(B, ))
    entity_num[-1] = M

    lstm_input, scalar_context, baseline_feature, entity_embeddings, map_skip = encoder(
        spatial_info_data, entity_info_data, scalar_info_data, entity_num
    )
    assert lstm_input.shape == (B, 1536)
    assert scalar_context.shape == (B, 448)
    assert baseline_feature.shape == (B, 512)
    assert entity_embeddings.shape == (B, 512, 256)
    assert isinstance(map_skip, list) and len(map_skip) == 7
