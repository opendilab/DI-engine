from os import path
import os
import copy
from easydict import EasyDict
from collections import deque
import pytest
import shutil
import wandb
import h5py
import torch.nn as nn
from unittest.mock import MagicMock
from unittest.mock import Mock, patch

from ding.utils import DistributedWriter
from ding.framework.middleware.tests import MockPolicy, CONFIG
from ding.framework import OnlineRLContext, OfflineRLContext
from ding.framework.middleware.functional import online_logger, offline_logger, wandb_online_logger, \
    wandb_offline_logger

test_folder = "test_exp"
test_path = path.join(os.getcwd(), test_folder)
cfg = EasyDict({"exp_name": "test_exp"})


def get_online_ctx():
    ctx = OnlineRLContext()
    ctx.eval_value = -10000
    ctx.train_iter = 34
    ctx.env_step = 78
    ctx.train_output = {'priority': [107], '[histogram]test_histogram': [1, 2, 3, 4, 5, 6], 'td_error': 15}
    return ctx


@pytest.fixture(scope='function')
def online_ctx_output_dict():
    ctx = get_online_ctx()
    return ctx


@pytest.fixture(scope='function')
def online_ctx_output_deque():
    ctx = get_online_ctx()
    ctx.train_output = deque([ctx.train_output])
    return ctx


@pytest.fixture(scope='function')
def online_ctx_output_list():
    ctx = get_online_ctx()
    ctx.train_output = [ctx.train_output]
    return ctx


@pytest.fixture(scope='function')
def online_scalar_ctx():
    ctx = get_online_ctx()
    ctx.train_output = {'[scalars]': 1}
    return ctx


class MockOnlineWriter:

    def __init__(self):
        self.ctx = get_online_ctx()

    def add_scalar(self, tag, scalar_value, global_step):
        if tag in ['basic/eval_episode_return_mean-env_step', 'basic/eval_episode_return_mean']:
            assert scalar_value == self.ctx.eval_value
            assert global_step == self.ctx.env_step
        elif tag == 'basic/eval_episode_return_mean-train_iter':
            assert scalar_value == self.ctx.eval_value
            assert global_step == self.ctx.train_iter
        elif tag in ['basic/train_td_error-env_step', 'basic/train_td_error']:
            assert scalar_value == self.ctx.train_output['td_error']
            assert global_step == self.ctx.env_step
        elif tag == 'basic/train_td_error-train_iter':
            assert scalar_value == self.ctx.train_output['td_error']
            assert global_step == self.ctx.train_iter
        else:
            raise NotImplementedError('tag should be in the tags defined')

    def add_histogram(self, tag, values, global_step):
        assert tag == 'test_histogram'
        assert values == [1, 2, 3, 4, 5, 6]
        assert global_step in [self.ctx.train_iter, self.ctx.env_step]

    def close(self):
        pass


def mock_get_online_instance():
    return MockOnlineWriter()


@pytest.mark.unittest
class TestOnlineLogger:

    def test_online_logger_output_dict(self, online_ctx_output_dict):
        with patch.object(DistributedWriter, 'get_instance', new=mock_get_online_instance):
            online_logger()(online_ctx_output_dict)

    def test_online_logger_record_output_dict(self, online_ctx_output_dict):
        with patch.object(DistributedWriter, 'get_instance', new=mock_get_online_instance):
            online_logger(record_train_iter=True)(online_ctx_output_dict)

    def test_online_logger_record_output_deque(self, online_ctx_output_deque):
        with patch.object(DistributedWriter, 'get_instance', new=mock_get_online_instance):
            online_logger()(online_ctx_output_deque)


def get_offline_ctx():
    ctx = OfflineRLContext()
    ctx.eval_value = -10000000000
    ctx.train_iter = 3333
    ctx.train_output = {'priority': [107], '[histogram]test_histogram': [1, 2, 3, 4, 5, 6], 'td_error': 15}
    return ctx


@pytest.fixture(scope='function')
def offline_ctx_output_dict():
    ctx = get_offline_ctx()
    return ctx


@pytest.fixture(scope='function')
def offline_scalar_ctx():
    ctx = get_offline_ctx()
    ctx.train_output = {'[scalars]': 1}
    return ctx


class MockOfflineWriter:

    def __init__(self):
        self.ctx = get_offline_ctx()

    def add_scalar(self, tag, scalar_value, global_step):
        assert global_step == self.ctx.train_iter
        if tag == 'basic/eval_episode_return_mean-train_iter':
            assert scalar_value == self.ctx.eval_value
        elif tag == 'basic/train_td_error-train_iter':
            assert scalar_value == self.ctx.train_output['td_error']
        else:
            raise NotImplementedError('tag should be in the tags defined')

    def add_histogram(self, tag, values, global_step):
        assert tag == 'test_histogram'
        assert values == [1, 2, 3, 4, 5, 6]
        assert global_step == self.ctx.train_iter

    def close(self):
        pass


def mock_get_offline_instance():
    return MockOfflineWriter()


class TestOfflineLogger:

    def test_offline_logger_no_scalars(self, offline_ctx_output_dict):
        with patch.object(DistributedWriter, 'get_instance', new=mock_get_offline_instance):
            offline_logger()(offline_ctx_output_dict)

    def test_offline_logger_scalars(self, offline_scalar_ctx):
        with patch.object(DistributedWriter, 'get_instance', new=mock_get_offline_instance):
            with pytest.raises(NotImplementedError) as exc_info:
                offline_logger()(offline_scalar_ctx)


class TheModelClass(nn.Module):

    def state_dict(self):
        return 'fake_state_dict'


class TheEnvClass(Mock):

    def enable_save_replay(self, replay_path):
        return


class TheObsDataClass(Mock):

    def __getitem__(self, index):
        return [[1, 1, 1]] * 50


class The1DDataClass(Mock):

    def __getitem__(self, index):
        return [[1]] * 50


@pytest.mark.unittest
def test_wandb_online_logger():
    record_path = './video_qbert_dqn'
    cfg = EasyDict(
        dict(
            gradient_logger=True,
            plot_logger=True,
            action_logger=True,
            return_logger=True,
            video_logger=True,
        )
    )
    env = TheEnvClass()
    ctx = OnlineRLContext()
    ctx.train_output = [{'reward': 1, 'q_value': [1.0]}]
    model = TheModelClass()
    wandb.init(config=cfg, anonymous="must")

    def mock_metric_logger(data, step):
        metric_list = [
            "q_value",
            "target q_value",
            "loss",
            "lr",
            "entropy",
            "reward",
            "q value",
            "video",
            "q value distribution",
            "train iter",
            "episode return mean",
            "env step",
            "action",
            "actions_of_trajectory_0",
            "actions_of_trajectory_1",
            "actions_of_trajectory_2",
            "actions_of_trajectory_3",
            "return distribution",
        ]
        assert set(data.keys()) <= set(metric_list)

    def mock_gradient_logger(input_model, log, log_freq, log_graph):
        assert input_model == model

    def test_wandb_online_logger_metric():
        with patch.object(wandb, 'log', new=mock_metric_logger):
            wandb_online_logger(record_path, cfg, env=env, model=model, anonymous=True)(ctx)

    def test_wandb_online_logger_gradient():
        with patch.object(wandb, 'watch', new=mock_gradient_logger):
            wandb_online_logger(record_path, cfg, env=env, model=model, anonymous=True)(ctx)

    test_wandb_online_logger_metric()
    test_wandb_online_logger_gradient()


@pytest.mark.unittest
def test_wandb_offline_logger():
    record_path = './video_pendulum_cql'
    cfg = EasyDict(dict(gradient_logger=True, plot_logger=True, action_logger=True, vis_dataset=True))
    env = TheEnvClass()
    ctx = OfflineRLContext()
    ctx.train_output = [{'reward': 1, 'q_value': [1.0]}]
    model = TheModelClass()
    wandb.init(config=cfg, anonymous="must")
    exp_config = EasyDict(dict(dataset_path='dataset.h5'))

    def mock_metric_logger(data, step=None):
        metric_list = [
            "q_value", "target q_value", "loss", "lr", "entropy", "reward", "q value", "video", "q value distribution",
            "train iter", 'dataset'
        ]
        assert set(data.keys()) < set(metric_list)

    def mock_gradient_logger(input_model, log, log_freq, log_graph):
        assert input_model == model

    def mock_image_logger(imagepath):
        assert os.path.splitext(imagepath)[-1] == '.png'

    def test_wandb_offline_logger_gradient():
        cfg.vis_dataset = False
        print(cfg)
        with patch.object(wandb, 'watch', new=mock_gradient_logger):
            wandb_offline_logger(
                record_path=record_path, cfg=cfg, exp_config=exp_config, env=env, model=model, anonymous=True
            )(ctx)

    def test_wandb_offline_logger_dataset():
        cfg.vis_dataset = True
        m = MagicMock()
        m.__enter__.return_value = {'obs': TheObsDataClass(), 'action': The1DDataClass(), 'reward': The1DDataClass()}
        with patch.object(wandb, 'log', new=mock_metric_logger):
            with patch.object(wandb, 'Image', new=mock_image_logger):
                with patch('h5py.File', return_value=m):
                    wandb_offline_logger(
                        record_path=record_path, cfg=cfg, exp_config=exp_config, env=env, model=model, anonymous=True
                    )(ctx)

    test_wandb_offline_logger_gradient()
    test_wandb_offline_logger_dataset()
