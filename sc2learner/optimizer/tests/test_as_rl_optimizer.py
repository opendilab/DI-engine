import pytest
import os
import torch
from torch.utils.data import DataLoader
import yaml
from easydict import EasyDict
from sc2learner.optimizer import AlphaStarRLOptimizer
from sc2learner.data import build_dataset, build_dataloader
from sc2learner.agent import AlphaStarAgent
from sc2learner.torch_utils import to_device


@pytest.fixture(scope='class')
def setup_config():
    with open(os.path.join(os.path.dirname(__file__), '../alphastar_rl_optimizer_default_config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


def get_dataloader(batch_size):
    class CFG:
        dataset_type = 'fake_actor'

    dataset = build_dataset(CFG(), True)
    return build_dataloader(dataset, 'fake_actor', batch_size=batch_size, use_distributed=False)


@pytest.mark.unittest
class TestASRLOptimizer:
    def test_rl_training(self, setup_config):
        use_cuda = setup_config.train.use_cuda
        dataloader = get_dataloader(setup_config.train.batch_size)
        agent = AlphaStarAgent(setup_config.model, use_cuda=use_cuda)
        optimizer = AlphaStarRLOptimizer(agent, setup_config)
        assert isinstance(optimizer.agent.model, torch.nn.Module)
        assert optimizer.agent.model.training

        assert list(optimizer.agent.model.parameters())[0].grad is None
        for idx, data in enumerate(dataloader):
            if use_cuda:
                data = to_device(data, 'cuda')
            losses, times = optimizer.learn(data)
            print('*' * 15 + 'Training Iteration {}'.format(idx) + '*' * 15)
            print('loss: {}'.format(losses))
            print('time: {}'.format(times))
            assert list(optimizer.agent.model.parameters())[0].grad is not None
            if idx == 10:
                break
