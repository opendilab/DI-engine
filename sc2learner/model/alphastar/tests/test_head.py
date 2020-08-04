import pytest
import math
import numpy as np
import torch
from sc2learner.model.alphastar.head import ActionTypeHead, DelayHead, QueuedHead, SelectedUnitsHead,\
    TargetUnitHead, LocationHead
from sc2learner.model.alphastar.tests.conftest import is_differentiable


@pytest.mark.unittest
class TestHead:
    def test_action_type_head(self, setup_config):
        B = 4
        handle = setup_config.model.policy.head.action_type_head
        model = ActionTypeHead(handle)

        lstm_output = torch.randn(B, handle.input_dim)
        scalar_context = torch.randn(B, handle.context_dim)
        mask = torch.ones(B, handle.action_num)
        # no action_type
        logits, action, embedding = model(lstm_output, scalar_context, mask)
        assert logits.shape == (B, handle.action_num)
        assert action.shape == (B, )
        assert action.dtype == torch.long
        assert embedding.shape == (B, handle.gate_dim)
        loss = logits.mean() + embedding.mean()
        is_differentiable(loss, model)

        # indicated action_type
        action_type = torch.randint(0, handle.action_num, size=(B, ))
        logits, action_type_output, embedding = model(lstm_output, scalar_context, mask, action_type=action_type)
        assert logits.shape == (B, handle.action_num)
        assert embedding.shape == (B, handle.gate_dim)
        assert action_type_output.dtype == torch.long
        assert action_type_output.eq(action_type).sum() == B

    def test_delay_head(self, setup_config):
        B = 4
        handle = setup_config.model.policy.head.delay_head
        model = DelayHead(handle)
        # input_dim: 1024
        autoregressive_embedding = torch.randn(B, handle.input_dim)
        # no delay
        output, delay, embedding = model(autoregressive_embedding)
        assert output.shape == (B, handle.delay_dim)
        assert output.dtype == torch.float
        assert delay.shape == (B, )
        assert delay.dtype == torch.long
        assert embedding.shape == (B, handle.input_dim)

        loss = output.mean() + embedding.mean()
        is_differentiable(loss, model)
        # indicated delay
        delay = torch.randint(0, handle.delay_dim, size=(B, ))
        output, delay_pred, embedding = model(autoregressive_embedding, delay)
        assert delay_pred.dtype == torch.long
        assert delay_pred.shape == (B, )
        assert delay_pred.eq(delay).sum() == B

    def test_queued_head(self, setup_config):
        B = 4
        handle = setup_config.model.policy.head.queued_head
        model = QueuedHead(handle)
        autoregressive_embedding = torch.randn(B, handle.input_dim)
        # no queued
        output, queued, embedding = model(autoregressive_embedding, temperature=0.8)
        assert output.shape == (B, handle.queued_dim)
        assert output.dtype == torch.float
        assert queued.shape == (B, )
        assert queued.dtype == torch.long
        assert embedding.shape == (B, handle.input_dim)
        # indicated queued
        queued = torch.randint(0, handle.queued_dim, size=(B, ))
        output, queued_pred, embedding = model(autoregressive_embedding, queued=queued)
        assert queued_pred.eq(queued).sum() == B

    def test_selected_units_head(self, setup_config):
        B = 4
        handle = setup_config.model.policy.head.selected_units_head
        model = SelectedUnitsHead(handle)
        assert isinstance(model, torch.nn.Module)
        autoregressive_embedding = torch.randn(B, handle.input_dim)
        unit_type_mask = torch.ones(B, handle.unit_type_dim)
        max_entity_num = handle.max_entity_num
        # no selected_units
        entity_num = [np.random.randint(200, 300) for _ in range(B)]
        unit_mask = []
        for b in range(B):
            mask = torch.ones(entity_num[b])
            masked_index = np.random.choice(entity_num[b], size=(100, ), replace=False)
            mask[masked_index] = 0
            unit_mask.append(mask)
        entity_embedding = [torch.randn(entity_num[b], handle.entity_embedding_dim) for b in range(B)]
        logits, selected_units, embedding = model(
            autoregressive_embedding, unit_type_mask, unit_mask, entity_embedding, temperature=0.8
        )
        assert isinstance(logits, list) and len(logits) == B
        assert isinstance(selected_units, list) and len(selected_units) == B
        assert embedding.shape == (B, handle.input_dim)
        loss = embedding.mean()
        for idx, (logit, selected_unit) in enumerate(zip(logits, selected_units)):
            assert isinstance(logit, torch.FloatTensor)
            assert isinstance(selected_unit, torch.LongTensor)
            assert logit.shape == (min(max_entity_num, selected_unit.shape[0] + 1), entity_num[idx] + 1)
            loss += logit.mean()
        is_differentiable(loss, model)
        # indicated selected_units
        selected_units = []
        for _ in range(B):
            num_b = np.random.randint(1, max_entity_num)
            unit_index = torch.LongTensor(np.random.choice(entity_num[b], size=(num_b, ), replace=False))
            unit_index = torch.sort(unit_index).values
            selected_units.append(unit_index)
        logits, selected_units_output, embedding = model(
            autoregressive_embedding,
            unit_type_mask,
            unit_mask,
            entity_embedding,
            temperature=0.8,
            selected_units=selected_units
        )
        assert len(selected_units) == len(selected_units_output)
        for su, suo in zip(selected_units, selected_units_output):
            assert su.ne(suo).sum() == 0
        for idx, (logit, selected_unit) in enumerate(zip(logits, selected_units_output)):
            assert logit.shape == (selected_unit.shape[0] + 1, entity_num[idx] + 1)

    def test_target_unit_head(self, setup_config):
        B = 4
        handle = setup_config.model.policy.head.target_unit_head
        model = TargetUnitHead(handle)
        assert isinstance(model, torch.nn.Module)

        autoregressive_embedding = torch.randn(B, handle.input_dim)
        # entity_num can be different among a batch
        entity_num = [np.random.randint(200, 300) for _ in range(B)]
        unit_type_mask = torch.ones(B, handle.unit_type_dim)
        unit_mask = [torch.ones(entity_num[b]) for b in range(B)]
        entity_embedding = [torch.randn(entity_num[b], handle.entity_embedding_dim) for b in range(B)]
        logits, target_unit = model(
            autoregressive_embedding, unit_type_mask, unit_mask, entity_embedding, temperature=0.8
        )
        assert isinstance(logits, list) and len(logits) == B
        assert isinstance(target_unit, torch.LongTensor)
        assert all([logits[b].shape[0] == entity_num[b] for b in range(B)])
        assert target_unit.shape == (B, )
        assert target_unit.min() >= 0 and all([target_unit[b] < entity_num[b] for b in range(B)])
        loss = sum([l.mean() for l in logits])
        is_differentiable(loss, model)

    def _location_head(self, handle):
        B = 4
        model = LocationHead(handle)
        assert isinstance(model, torch.nn.Module)
        autoregressive_embedding = torch.randn(B, handle.input_dim)
        map_size = np.random.randint(20, 25, size=(2, )) * 8
        map_skip = [torch.randn(B, handle.res_dim, *(map_size // 8)) for _ in range(handle.res_num)]
        location_mask = torch.randn(B, 1, *map_size)
        outputs, location = model(autoregressive_embedding, map_skip, location_mask, temperature=0.8)
        assert isinstance(outputs, torch.FloatTensor)
        assert isinstance(location, torch.LongTensor)
        if handle.output_type == 'cls':
            assert outputs.shape == (B, map_size[0], map_size[1])
        elif handle.output_type == 'soft_argmax':
            assert outputs.shape == (B, 2)
        assert location.shape == (B, 2)
        assert location[..., 0].min() >= 0 and location[..., 0].max() < map_size[0]
        assert location[..., 1].min() >= 0 and location[..., 1].max() < map_size[1]
        loss = outputs.mean()
        is_differentiable(loss, model)

    def test_target_location_head(self, setup_config):
        handle = setup_config.model.policy.head.location_head
        self._location_head(handle)

    def test_target_location_head_soft_argmax(self, setup_config):
        handle = setup_config.model.policy.head.location_head
        handle.output_type = 'soft_argmax'
        self._location_head(handle)
