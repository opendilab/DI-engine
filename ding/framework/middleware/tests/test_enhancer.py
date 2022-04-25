import pytest
import torch
from ding.framework import OnlineRLContext
from ding.data.buffer import DequeBuffer
from easydict import EasyDict
from typing import Any, List, Dict, Optional
import numpy as np
from ding.framework.middleware.functional.enhancer import her_data_enhancer
from unittest.mock import Mock


class MockHerRewardModel(Mock):
    def __init__(self) -> None:
        super(MockHerRewardModel, self).__init__()
        self.episode_size = 32
    
    def estimate(self, episode: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [[episode[0] for _ in range(4)]]


@pytest.mark.unittest
def test_her_data_enhancer():
    cfg = EasyDict({'policy':{'learn':{'batch_size':32}}})
    buffer = DequeBuffer(cfg.policy.learn.batch_size)
    ctx = OnlineRLContext()

    train_data = [
        [
            {
                'action': torch.randint(low=0,high=5,size=(1,)),
                'collect_train_iter': torch.tensor([0]),
                'done': torch.tensor(False),
                'next_obs': torch.randint(low=0,high=2,size=(10,),dtype=torch.float32),
                'obs': torch.randint(low=0,high=2,size=(10,),dtype=torch.float32),
                'reward':torch.randint(low=0,high=2,size=(1,),dtype=torch.float32),
            }
            for _ in range(np.random.choice([1,4,5],size=1)[0])
        ]
        for _ in range(cfg.policy.learn.batch_size)
    ]

    for d in train_data:
        buffer.push(d)

    # TODO
    her_data_enhancer(cfg = cfg, buffer_ = buffer, her_reward_model = MockHerRewardModel())(ctx)
    assert len(ctx.train_data) == 32 * 4
    assert len(ctx.train_data[0]) == 6

if __name__ == '__main__':
    test_her_data_enhancer()