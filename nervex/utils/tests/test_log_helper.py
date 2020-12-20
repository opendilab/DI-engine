import random
from collections import deque

import numpy as np
import pytest

from easydict import EasyDict

from nervex.utils.log_helper import AverageMeter, build_logger, build_logger_naive, pretty_print
from nervex.utils.file_helper import remove_file

cfg = EasyDict(
    {
        'env': {},
        'env_num': 4,
        'common': {
            'save_path': "./summary_log",
            'load_path': '',
            'name': 'fakeLog',
            'only_evaluate': False,
        },
        'logger': {
            'print_freq': 10,
            'save_freq': 200,
            'eval_freq': 200,
        },
        'data': {
            'train': {},
            'eval': {},
        },
        'learner': {
            'log_freq': 100,
        },
    }
)


@pytest.mark.unittest
class TestAverageMeter:

    def test_AverageMeter(self):
        handle = AverageMeter(length=1)
        handle.reset()
        assert handle.val == 0.0
        assert handle.avg == 0.0
        for _ in range(10):
            t = random.uniform(0, 1)
            handle.update(t)
            assert handle.val == t
            assert handle.avg == pytest.approx(t, abs=1e-6)

        handle = AverageMeter(length=5)
        handle.reset()
        assert handle.val == 0.0
        assert handle.avg == 0.0
        queue = deque(maxlen=5)
        for _ in range(10):
            t = random.uniform(0, 1)
            handle.update(t)
            queue.append(t)
            assert handle.val == t
            assert handle.avg == pytest.approx(np.mean(queue, axis=0))

    def test_pretty_print(self):
        pretty_print(cfg)

    def test_logger(self):
        logger, tb_logger, variable_record = build_logger(cfg, name="fake_test")
        variable_record.register_var("fake_loss")
        variable_record.register_var("fake_reward")
        build_logger_naive('.', 'name')
        for i in range(20):
            variable_record.update_var({"fake_loss": i + 1, "fake_reward": i - 1})
        variable_record.update_var({"fake_not_registered": 100})
        assert set(variable_record.get_var_names()) == set(['fake_loss', 'fake_reward', "fake_not_registered"])
        assert isinstance(variable_record.get_var_text('fake_loss'), str)
        assert isinstance(variable_record.get_vars_tb_format(['fake_loss'], 10), list)
        assert len(variable_record.get_vars_tb_format(['fake_loss', 'fake_reward'], 10)) == 2
        remove_file("./name")
        remove_file("summary_log*")
