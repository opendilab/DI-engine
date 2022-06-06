import pytest
from ding.framework import OnlineRLContext, OfflineRLContext, ding_init
from ding.framework.middleware.functional import online_logger, offline_logger
from easydict import EasyDict
import os
from os import path
import shutil
from collections import deque
from unittest.mock import Mock, patch
from ding.utils import DistributedWriter
import copy

test_folder = "test_exp"
test_path = path.join(os.getcwd(), test_folder)
cfg = EasyDict({"exp_name": "test_exp"})

@pytest.fixture(scope='function')
def online_ctx_output_dict():
    ctx = OnlineRLContext()
    ctx.eval_value = -10000
    ctx.train_iter = 34
    ctx.env_step = 78
    ctx.train_output = {
        'priority': [107],
        '[histogram]test_histogram': [1,2,3,4,5,6],
        'td_error': 15
    }
    return ctx

@pytest.fixture(scope='function')
def online_ctx_output_deque():
    ctx = OnlineRLContext()
    ctx.eval_value = -600
    ctx.train_iter = 24
    ctx.env_step = 30
    ctx.train_output = deque([
        {
            'priority': [107],
            '[histogram]test_histogram': [1,2,3,4,5,6],
            'td_error': 15
        }, 
        {
            'priority': [108],
            '[histogram]test_histogram': [1,2,3,4,5,6],
            'td_error': 30
        }
    ])
    return ctx

@pytest.fixture(scope='function')
def online_ctx_output_list():
    ctx = OnlineRLContext()
    ctx.eval_value = -1000000
    ctx.train_iter = 23232
    ctx.env_step = 33333
    ctx.train_output = [
        {
            'priority': [107],
            '[histogram]test_histogram': [1,2,3,4,5,6],
            'td_error': 15
        }, 
        {
            'priority': [108],
            '[histogram]test_histogram': [1,2,3,4,5,6],
            'td_error': 30
        }
    ]
    return ctx

@pytest.fixture(scope='function')
def online_scalar_ctx():
    ctx = OfflineRLContext()
    ctx.eval_value = -777888
    ctx.train_iter = 2233
    ctx.env_step = 32323
    ctx.train_output = {
        '[scalars]': 1
    }
    return ctx


@pytest.mark.zms
class TestOnlineLogger:

    def test_online_logger_output_dict(self, online_ctx_output_dict):
        ding_init(cfg)
        online_logger()(online_ctx_output_dict)

    def test_online_logger_record_output_dict(self, online_ctx_output_dict):
        ding_init(cfg)
        online_logger(record_train_iter=True)(online_ctx_output_dict)

    def test_online_logger_record_output_deque(self, online_ctx_output_deque):
        ding_init(cfg)
        online_logger()(online_ctx_output_deque)
    
    def test_online_logger_record_output_list(self, online_ctx_output_list):
        ding_init(cfg)
        with pytest.raises(NotImplementedError) as exc_info:
            online_logger()(online_ctx_output_list)
    
    def test_online_logger_scalars(self, online_scalar_ctx):
        ding_init(cfg)
        with pytest.raises(NotImplementedError) as exc_info:
            online_logger()(online_scalar_ctx)

def get_offline_ctx():
    ctx = OfflineRLContext()
    ctx.eval_value = -10000000000
    ctx.train_iter = 3333
    ctx.train_output = {
        'priority': [107],
        '[histogram]test_histogram': [1,2,3,4,5,6],
        'td_error': 15
    }
    return ctx

@pytest.fixture(scope='function')
def offline_ctx_output_dict():
    ctx = get_offline_ctx()
    return ctx

@pytest.fixture(scope='function')
def offline_scalar_ctx():
    ctx = get_offline_ctx()
    ctx.train_output = {
        '[scalars]': 1
    }
    return ctx

class MockWriter:

    def __init__(self):
        self.ctx = get_offline_ctx()
    
    def add_scalar(self, tag, scalar_value, global_step):
        assert global_step == self.ctx.train_iter
        if tag == 'basic/eval_episode_reward_mean-train_iter':
            assert scalar_value == self.ctx.eval_value
        elif tag == 'basic/train_td_error-train_iter':
            assert scalar_value == self.ctx.train_output['td_error']
        else:
            raise NotImplementedError('tag should be in the tags defined')

    def add_histogram(self, tag, values, global_step):
        assert tag == 'test_histogram'
        assert values == [1,2,3,4,5,6]
        assert global_step == self.ctx.train_iter

def mock_get_instance():
    return MockWriter()


@pytest.mark.offline
class TestOfflineLogger:

    def test_offline_logger_no_scalars(self, offline_ctx_output_dict):
        with patch.object(DistributedWriter, 'get_instance', new=mock_get_instance):
            offline_logger()(offline_ctx_output_dict)
    
    def test_offline_logger_scalars(self, offline_scalar_ctx):
        with patch.object(DistributedWriter, 'get_instance', new=mock_get_instance):
            with pytest.raises(NotImplementedError) as exc_info:
                offline_logger()(offline_scalar_ctx)


