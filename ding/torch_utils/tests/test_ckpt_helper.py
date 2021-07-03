import os
import time

import pytest
import torch
import torch.nn as nn
import uuid

from ding.torch_utils.checkpoint_helper import auto_checkpoint, build_checkpoint_helper, CountVar
from ding.utils import read_file, save_file


class DstModel(nn.Module):

    def __init__(self):
        super(DstModel, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 8)
        self.fc_dst = nn.Linear(3, 6)


class SrcModel(nn.Module):

    def __init__(self):
        super(SrcModel, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 8)
        self.fc_src = nn.Linear(3, 7)


class HasStateDict(object):

    def __init__(self, name):
        self._name = name
        self._state_dict = name + str(uuid.uuid4())

    def state_dict(self):
        old = self._state_dict
        self._state_dict = self._name + str(uuid.uuid4())
        return old

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict


@pytest.mark.unittest
class TestCkptHelper:

    def test_load_model(self):
        path = 'model.pt'
        os.popen('rm -rf ' + path)
        time.sleep(1)

        dst_model = DstModel()
        src_model = SrcModel()
        ckpt_state_dict = {'model': src_model.state_dict()}
        torch.save(ckpt_state_dict, path)

        ckpt_helper = build_checkpoint_helper({})
        with pytest.raises(RuntimeError):
            ckpt_helper.load(path, dst_model, strict=True)

        ckpt_helper.load(path, dst_model, strict=False)
        assert torch.abs(dst_model.fc1.weight - src_model.fc1.weight).max() < 1e-6
        assert torch.abs(dst_model.fc1.bias - src_model.fc1.bias).max() < 1e-6

        dst_model = DstModel()
        src_model = SrcModel()
        assert torch.abs(dst_model.fc1.weight - src_model.fc1.weight).max() > 1e-6
        src_optimizer = HasStateDict('src_optimizer')
        dst_optimizer = HasStateDict('dst_optimizer')
        src_last_epoch = CountVar(11)
        dst_last_epoch = CountVar(5)
        src_last_iter = CountVar(110)
        dst_last_iter = CountVar(50)
        src_dataset = HasStateDict('src_dataset')
        dst_dataset = HasStateDict('dst_dataset')
        src_collector_info = HasStateDict('src_collect_info')
        dst_collector_info = HasStateDict('dst_collect_info')
        ckpt_helper.save(
            path,
            src_model,
            optimizer=src_optimizer,
            dataset=src_dataset,
            collector_info=src_collector_info,
            last_iter=src_last_iter,
            last_epoch=src_last_epoch,
            prefix_op='remove',
            prefix="f"
        )
        ckpt_helper.load(
            path,
            dst_model,
            dataset=dst_dataset,
            optimizer=dst_optimizer,
            last_iter=dst_last_iter,
            last_epoch=dst_last_epoch,
            collector_info=dst_collector_info,
            strict=False,
            state_dict_mask=['fc1'],
            prefix_op='add',
            prefix="f"
        )
        assert dst_dataset.state_dict().startswith('src')
        assert dst_optimizer.state_dict().startswith('src')
        assert dst_collector_info.state_dict().startswith('src')
        assert dst_last_iter.val == 110
        for k, v in dst_model.named_parameters():
            assert k.startswith('fc')
        print('==dst', dst_model.fc2.weight)
        print('==src', src_model.fc2.weight)
        assert torch.abs(dst_model.fc2.weight - src_model.fc2.weight).max() < 1e-6
        assert torch.abs(dst_model.fc1.weight - src_model.fc1.weight).max() > 1e-6

        checkpoint = read_file(path)
        checkpoint.pop('dataset')
        checkpoint.pop('optimizer')
        checkpoint.pop('last_iter')
        save_file(path, checkpoint)
        ckpt_helper.load(
            path,
            dst_model,
            dataset=dst_dataset,
            optimizer=dst_optimizer,
            last_iter=dst_last_iter,
            last_epoch=dst_last_epoch,
            collector_info=dst_collector_info,
            strict=True,
            state_dict_mask=['fc1'],
            prefix_op='add',
            prefix="f"
        )
        with pytest.raises(NotImplementedError):
            ckpt_helper.load(
                path,
                dst_model,
                strict=False,
                lr_schduler='lr_scheduler',
                last_iter=dst_last_iter,
            )

        with pytest.raises(KeyError):
            ckpt_helper.save(path, src_model, prefix_op='key_error', prefix="f")
            ckpt_helper.load(path, dst_model, strict=False, prefix_op='key_error', prefix="f")

        os.popen('rm -rf ' + path + '*')


@pytest.mark.unittest
def test_count_var():
    var = CountVar(0)
    var.add(5)
    assert var.val == 5
    var.update(3)
    assert var.val == 3


@pytest.mark.unittest
def test_auto_checkpoint():

    class AutoCkptCls:

        def __init__(self):
            pass

        @auto_checkpoint
        def start(self):
            for i in range(10):
                if i < 5:
                    time.sleep(0.2)
                else:
                    raise Exception("There is an exception")
                    break

        def save_checkpoint(self, ckpt_path):
            print('Checkpoint is saved successfully in {}!'.format(ckpt_path))

    auto_ckpt = AutoCkptCls()
    auto_ckpt.start()


if __name__ == '__main__':
    test = TestCkptHelper()
    test.test_load_model()
