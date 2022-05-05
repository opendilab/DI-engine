from typing import TYPE_CHECKING, Callable
import pytest
import unittest
import pathlib as pl
import os
import shutil
import random
import time

from ding.framework import Parallel
from ding.framework import task, Context
from ding.utils import traffic


def clean_up(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


def assertIsFile(path):
    if not pl.Path(path).resolve().is_file():
        raise AssertionError("File does not exist: %s" % str(path))


def fn_record_something() -> Callable:
    dir = "./tmp_test/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    clean_up(dir)
    file_path = "./tmp_test/traffic_log.txt"
    traffic.set_config(file_path=file_path, online=True, router=Parallel())

    def _fn(ctx: "Context") -> None:
        while True:
            time.sleep(0.01)
            if task.finish:
                time.sleep(0.01)
                assertIsFile(file_path)
                clean_up(dir)
                traffic.close()
                break

    return _fn


def fn_send_something() -> Callable:
    traffic.set_config(router=Parallel())
    i = 0

    def _fn(ctx: "Context") -> None:
        nonlocal i
        i += 1
        traffic.record(train_iter=i, train_reward=random())
        traffic.record(eval_iter=i, eval_reward=random())

    return _fn


def parallel_main():
    with task.start(async_mode=True):
        if task.router.node_id > 0:
            task.use(fn_send_something())
        else:
            task.use(fn_record_something())
        task.run(max_step=5)


@pytest.mark.unittest
class TestTrafficModule:

    def test_local_mode(self):
        dir = "./tmp_test/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        clean_up(dir)
        file_path = "./tmp_test/traffic_log.txt"
        traffic.config(file_path=file_path, online=True)
        for train_iter in range(10):
            traffic.record(train_iter=train_iter, train_reward=random())
        for eval_iter in range(10):
            traffic.record(eval_iter=eval_iter, eval_reward=random())
        assertIsFile(file_path)
        assert traffic.df.size == 100
        clean_up(dir)

    def test_remote_mode(self):
        Parallel.runner(n_parallel_workers=2, topology="mesh")(parallel_main)
