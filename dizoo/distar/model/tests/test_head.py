import pytest
import torch
from easydict import EasyDict
from ding.utils import read_yaml_config
from dizoo.distar.model.head import ActionTypeHead, DelayHead, SelectedUnitsHead, TargetUnitHead, LocationHead, \
    QueuedHead
B, ENTITY = 4, 128
total_config = EasyDict(read_yaml_config('../actor_critic_default_config.yaml'))


@pytest.mark.envtest
class TestHead():

    def test_action_type_head(self):
        cfg = total_config.model.policy.head.action_type_head
        head = ActionTypeHead(cfg)
        lstm_output = torch.randn(B, cfg.input_dim)
        scalar_context = torch.randn(B, cfg.context_dim)

        logit, action_type, embedding = head(lstm_output, scalar_context)
        assert logit.shape == (B, 327)
        assert action_type.shape == (B, )
        assert embedding.shape == (B, cfg.gate_dim)

        logit, output_action_type, embedding = head(lstm_output, scalar_context, action_type)
        assert logit.shape == (B, 327)
        assert embedding.shape == (B, cfg.gate_dim)
        assert output_action_type.eq(action_type).all()

    def test_delay_head(self):
        cfg = total_config.model.policy.head.delay_head
        head = DelayHead(cfg)
        inputs = torch.randn(B, cfg.input_dim)

        logit, delay, embedding = head(inputs)
        assert logit.shape == (B, cfg.delay_dim)
        assert delay.shape == (B, )
        assert embedding.shape == (B, cfg.input_dim)

        logit, output_delay, embedding = head(inputs, delay)
        assert logit.shape == (B, cfg.delay_dim)
        assert embedding.shape == (B, cfg.input_dim)
        assert output_delay.eq(delay).all()

    def test_queued_head(self):
        cfg = total_config.model.policy.head.queued_head
        head = QueuedHead(cfg)
        inputs = torch.randn(B, cfg.input_dim)

        logit, queued, embedding = head(inputs)
        assert logit.shape == (B, cfg.queued_dim)
        assert queued.shape == (B, )
        assert embedding.shape == (B, cfg.input_dim)

        logit, output_queued, embedding = head(inputs, queued)
        assert logit.shape == (B, cfg.queued_dim)
        assert embedding.shape == (B, cfg.input_dim)
        assert output_queued.eq(queued).all()

    def test_target_unit_head(self):
        cfg = total_config.model.policy.head.target_unit_head
        head = TargetUnitHead(cfg)
        inputs = torch.randn(B, cfg.input_dim)
        entity_embedding = torch.rand(B, ENTITY, cfg.entity_embedding_dim)
        entity_num = torch.randint(ENTITY // 2, ENTITY, size=(B, ))
        entity_num[-1] = ENTITY

        logit, target_unit = head(inputs, entity_embedding, entity_num)
        assert logit.shape == (B, ENTITY)
        assert target_unit.shape == (B, )

        logit, output_target_unit = head(inputs, entity_embedding, entity_num, target_unit)
        assert logit.shape == (B, ENTITY)
        assert output_target_unit.eq(target_unit).all()

    def test_location_head(self):
        cfg = total_config.model.policy.head.location_head
        head = LocationHead(cfg)
        inputs = torch.randn(B, cfg.input_dim)
        Y, X = cfg.spatial_y, cfg.spatial_x
        map_skip = [torch.randn(B, cfg.res_dim, Y // 8, X // 8) for _ in range(cfg.res_num)]

        logit, location = head(inputs, map_skip)
        assert logit.shape == (B, Y * X)
        assert location.shape == (B, )

        logit, output_location = head(inputs, map_skip, location)
        assert logit.shape == (B, Y * X)
        assert output_location.eq(location).all()

    def test_selected_units_head(self):
        cfg = total_config.model.policy.head.selected_units_head
        head = SelectedUnitsHead(cfg)
        inputs = torch.randn(B, cfg.input_dim)
        entity_embedding = torch.rand(B, ENTITY, cfg.entity_embedding_dim)
        entity_num = torch.randint(ENTITY // 2, ENTITY, size=(B, ))
        entity_num[-1] = ENTITY
        su_mask = torch.randint(0, 2, size=(B, )).bool()
        su_mask[-1] = 1

        logit, selected_units, embedding, selected_units_num, extra_units = head(
            inputs, entity_embedding, entity_num, su_mask=su_mask
        )
        N = selected_units_num.max()
        assert embedding.shape == (B, cfg.input_dim)
        assert logit.shape == (B, N, ENTITY + 1)
        assert selected_units.shape == (B, N)
        assert selected_units_num.shape == (B, )
        assert extra_units.shape == (B, ENTITY + 1)

        selected_units_num = selected_units_num.clamp(0, 64)
        logit, output_selected_units, embedding, selected_units_num, extra_units = head(
            inputs, entity_embedding, entity_num, selected_units_num, selected_units, su_mask
        )
        N = selected_units_num.max()
        assert embedding.shape == (B, cfg.input_dim)
        assert logit.shape == (B, N, ENTITY + 1)
        assert extra_units is None
        assert output_selected_units.eq(selected_units).all()
