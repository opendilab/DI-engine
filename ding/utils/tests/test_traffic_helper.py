from typing import TYPE_CHECKING, Callable
import pytest
import unittest
import pathlib as pl
import os
import shutil
import random
import time
import logging
import timeit
import numpy as np
import pandas as pd
from tabulate import tabulate

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
    traffic.set_config(file_path=file_path, is_writer=True, router=Parallel())

    def _fn(ctx: "Context") -> None:
        while True:
            time.sleep(0.01)
            if task.finish:
                time.sleep(0.01)
                assert len(traffic._data) == 10
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
        traffic.record(train_iter=i, train_reward=i)
        traffic.record(eval_iter=i, eval_reward=random.random())

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
        traffic.set_config(file_path=file_path, is_writer=True)
        for train_iter in range(10):
            traffic.record(train_iter=train_iter, train_reward=random.random())
        for eval_iter in range(10):
            traffic.record(eval_iter=eval_iter, eval_reward=random.random())
        assertIsFile(file_path)
        assert len(traffic._data) == 20
        assert traffic.df.size == 100
        clean_up(dir)

    def test_remote_mode(self):
        Parallel.runner(n_parallel_workers=2, topology="mesh")(parallel_main)


repeats = 100


def get_mean_std(res):
    # return the total time per 1000 ops
    return np.mean(res) * 1000.0 / repeats, np.std(res) * 1000.0 / repeats


class TrafficBenchmark:

    def __init__(self, file_path: str) -> None:
        self._traffic = traffic.set_config(file_path=file_path, is_writer=True)

    def record_info(self):
        self._traffic.record(other_iter=random.random(), other_reward=random.random(), __label="other")
        self._traffic.record(eval_iter=random.random(), eval_reward=random.random(), __label="evaluator")
        self._traffic.record(train_iter=random.random(), train_loss=random.random(), __label="learner")

    def log_and_show(self):
        self.record_info()
        L = self._traffic.df.replace('', np.nan).ffill().iloc[-1].tolist()

    def extract_df(self):
        self._traffic._cache.clear()
        L = self._traffic.df.ffill().iloc[-1].tolist()

    def aggregate(self):
        self._traffic._cache.clear()
        df_gb = self._traffic.df.groupby('__label')
        L = df_gb.agg(
            {
                'train_iter': ['last'],
                'train_loss': ['mean', 'std'],
                'other_iter': ['last'],
                'other_reward': ['mean', 'max', 'std'],
                'eval_iter': ['last'],
                'eval_reward': ['mean', 'max', 'std']
            }
        )


@pytest.mark.benchmark
def test_benchmark():

    test_traffic = TrafficBenchmark(file_path="./tmp_test/test_traffic_benchmark.txt")

    mean, std = get_mean_std(timeit.repeat(test_traffic.record_info, number=repeats))
    print("Record Test:  mean {:.4f} s, std {:.4f} s".format(mean, std))

    for i in range(5000):
        test_traffic.record_info()

    mean, std = get_mean_std(timeit.repeat(test_traffic.extract_df, number=repeats))
    print("Process extract df Test:  mean {:.4f} s, std {:.4f} s".format(mean, std))

    mean, std = get_mean_std(timeit.repeat(test_traffic.aggregate, number=repeats))
    print("Process aggregate df Test:  mean {:.4f} s, std {:.4f} s".format(mean, std))

    clean_up("./tmp_test/")
