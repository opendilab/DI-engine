import random
from collections import deque
import numpy as np
import pytest
from easydict import EasyDict
import logging

from nervex.utils.log_helper import build_logger, get_default_logger, pretty_print
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
class TestLogger:

    def test_pretty_print(self):
        pretty_print(cfg)

    def test_logger(self):
        default_logger = get_default_logger()
        assert isinstance(default_logger, logging.Logger)
        logger, tb_logger = build_logger(cfg.common.save_path, name="fake_test", need_tb=True)
        vars = {'aa': 3.0, 'bb': 4, 'cc': 3e4}
        # text logger
        logger.info("I'm an info")
        logger.debug("I'm a bug")
        logger.error("I'm an error")
        logger.print_vars(vars)
        # tensorboard logger
        for var in vars:
            tb_logger.register_var(var, 'scalar')
        for i in range(10):
            new_vars = {k: v * (i + random.random()) for k, v in vars.items()}
            tb_logger.print_vars(vars, i, 'scalar')
        assert tb_logger.scalar_var_names == list(vars.keys())
        remove_file(cfg.common.save_path)
        tb_logger.close()
