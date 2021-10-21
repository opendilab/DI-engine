from ding.worker import MetricSerialEvaluator, IMetric
from torch.utils.data import DataLoader
import pytest
import torch.utils.data as data

import torch.nn as nn
from ding.torch_utils import to_tensor
import torch
from easydict import EasyDict
from ding.worker.collector.tests.fake_cls_policy import fake_policy

fake_cls_config = dict(
    exp_name='fake_config_for_test_metric_serial_evaluator',
    policy=dict(
        on_policy=False,
        cuda=False,
        eval=dict(batch_size=1, evaluator=dict(eval_freq=1, multi_gpu=False, stop_value=dict(acc=75.0))),
    ),
    env=dict(),
)

cfg = EasyDict(fake_cls_config)


class fake_eval_dataset(data.Dataset):

    def __init__(self) -> None:
        self.data = [i for i in range(5)]  # [0, 1, 2, 3, 4, 5]
        self.target = [2 * i + 1 for i in range(5)]  # [0, 3, 5, 7, 9, 11]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        data = self.data[index]
        target = self.target[index]
        return data, target


class fake_model(nn.Module):  # y = 2*x+1

    def __init__(self) -> None:
        super(fake_model, self).__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.constant_(self.linear.bias, 1)
        nn.init.constant_(self.linear.weight, 2)

    def forward(self, x):
        x = to_tensor(x).float()
        return self.linear(x)


class fake_ClassificationMetric(IMetric):

    @staticmethod
    def accuracy(inputs: torch.Tensor, label: torch.Tensor) -> dict:
        batch_size = label.size(0)
        correct = inputs.eq(label)
        return {'acc': correct.reshape(-1).float().sum(0) * 100. / batch_size}

    def eval(self, inputs: torch.Tensor, label: torch.Tensor) -> dict:
        output = self.accuracy(inputs, label)
        for k in output:
            output[k] = output[k].item()
        return output

    def reduce_mean(self, inputs) -> dict:
        L = len(inputs)
        output = {}
        for k in inputs[0].keys():
            output[k] = sum([t[k] for t in inputs]) / L
        return output

    def gt(self, metric1: dict, metric2: dict) -> bool:
        if metric2 is None:
            return True
        for k in metric1:
            if metric1[k] < metric2[k]:
                return False
        return True


@pytest.mark.unittest
@pytest.mark.parametrize('cfg', [cfg])
def test_evaluator(cfg):
    model = fake_model()
    eval_dataset = fake_eval_dataset()
    eval_dataloader = DataLoader(eval_dataset, cfg.policy.eval.batch_size, num_workers=2)
    policy = fake_policy(cfg.policy, model=model, enable_field=['eval'])
    eval_metric = fake_ClassificationMetric()
    evaluator = MetricSerialEvaluator(
        cfg.policy.eval.evaluator, [eval_dataloader, eval_metric], policy.eval_mode, exp_name=cfg.exp_name
    )

    cur_iter = 0
    assert evaluator.should_eval(cur_iter)

    evaluator._last_eval_iter = 0
    cur_iter = 1
    stop, reward = evaluator.eval(None, cur_iter, 0)
    assert stop
    assert reward['acc'] == 100
