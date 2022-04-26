import pytest
import torch
from ding.framework import OnlineRLContext
from ding.data.buffer import DequeBuffer
from easydict import EasyDict
from typing import Any, List, Dict, Optional
import numpy as np
import copy
from ding.framework.middleware.functional.enhancer import her_data_enhancer
from unittest.mock import Mock, patch
from ding.framework.middleware.tests import MockHerRewardModel, CONFIG


@pytest.mark.unittest
def test_her_data_enhancer():
    cfg = copy.deepcopy(CONFIG)
    ctx = OnlineRLContext()

    with patch("ding.reward_model.HerRewardModel", MockHerRewardModel):
        mock_her_reward_model = MockHerRewardModel()
        buffer = DequeBuffer(mock_her_reward_model.episode_size)

        train_data = [
            [
                {
                    'action': torch.randint(low=0, high=5, size=(1, )),
                    'collect_train_iter': torch.tensor([0]),
                    'done': torch.tensor(False),
                    'next_obs': torch.randint(low=0, high=2, size=(10, ), dtype=torch.float32),
                    'obs': torch.randint(low=0, high=2, size=(10, ), dtype=torch.float32),
                    'reward': torch.randint(low=0, high=2, size=(1, ), dtype=torch.float32),
                } for _ in range(np.random.choice([1, 4, 5], size=1)[0])
            ] for _ in range(mock_her_reward_model.episode_size)
        ]

        for d in train_data:
            buffer.push(d)

        her_data_enhancer(cfg=cfg, buffer_=buffer, her_reward_model=MockHerRewardModel())(ctx)
    assert len(ctx.train_data) == mock_her_reward_model.episode_size * mock_her_reward_model.episode_element_size
    assert len(ctx.train_data[0]) == 6
