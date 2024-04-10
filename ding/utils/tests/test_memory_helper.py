import pytest
import os
import time
import torch
from ding.utils import SimpleMemoryProfiler
from ding.utils.memory_helper import multi_chunk_test
from ding.model import DRQN


@pytest.mark.unittest
def test_memory_profiler():
    multi_chunk_test()
    assert os.path.exists('test_simple_memory_profiler_multi_chunk')
    assert os.path.exists('test_simple_memory_profiler_multi_chunk/memory.log')
    assert os.path.exists('test_simple_memory_profiler_multi_chunk/summary_sunburst.html')

    model = DRQN(
        obs_shape=(4, 84, 84),
        action_shape=6,
        encoder_hidden_size_list=[128, 256, 512],
        head_layer_num=3,
        norm_type='LN',
    )
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    profiler = SimpleMemoryProfiler(model, optimizer, 'test_simple_memory_profiler_drqn', total_steps=1)

    x = torch.randn(3, 8, 4, 84, 84)
    inputs = {'obs': x, 'prev_state': None}
    y = model(inputs)['logit']

    optimizer.zero_grad()
    y.mean().backward()
    optimizer.step()

    profiler.step()

    time.sleep(0.3)
    os.popen('rm -rf test_simple_memory_profiler*')
