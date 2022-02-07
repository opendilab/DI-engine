import pytest
import unittest
from unittest import mock
from unittest.mock import patch
import pathlib as pl
import os
import shutil
import time
from typing import Callable

from ding.framework import Task, Context
from ding.framework import Parallel
from ding.framework.middleware import pace_controller


def fn(task: "Task"):
    another_node_total_step = 0

    def _listen_total_step(total_step):
        nonlocal another_node_total_step
        another_node_total_step = total_step
        return

    task.on("total_step", _listen_total_step)

    time.sleep(1)

    def _fn(ctx: "Context"):
        nonlocal another_node_total_step
        assert ctx.total_step <= another_node_total_step + 1
        assert ctx.total_step >= another_node_total_step
        task.emit("total_step", ctx.total_step, only_remote=True)
        return

    return _fn


def fn_with_identity(task: "Task", is_less: bool):
    another_node_total_step = 0

    def _listen_total_step(total_step):
        nonlocal another_node_total_step
        another_node_total_step = total_step
        return

    task.on("total_step", _listen_total_step)

    time.sleep(1)

    def _fn(ctx: "Context"):
        nonlocal another_node_total_step
        if is_less:
            assert ctx.total_step >= another_node_total_step
        else:
            assert ctx.total_step <= another_node_total_step + 1
        task.emit("total_step", ctx.total_step, only_remote=True)
        return

    return _fn


def parallel_main():
    with Task(async_mode=True) as task:
        task.use(fn(task))
        task.use(pace_controller(task))
        task.run(max_step=100)


def parallel_main_with_theme():
    with Task(async_mode=True) as task:
        task.use(fn(task))
        task.use(pace_controller(task, theme="test"))
        task.run(max_step=100)


def parallel_main_with_identity():
    with Task(async_mode=True) as task:
        if task.router.node_id > 0:
            task.use(fn_with_identity(task, False))
            task.use(pace_controller(task, identity="1"))
        else:
            task.use(fn_with_identity(task, True))
            task.use(pace_controller(task, identity="0"))
        task.run(max_step=100)


def parallel_main_with_timeout():
    time_begin = time.time()
    with Task(async_mode=True) as task:
        task.use(pace_controller(task, timeout=1))
        task.run(max_step=10)
    time_end = time.time()
    assert time_end - time_begin > 10


def non_parallel_main_with_timeout():
    time_begin = time.time()
    with Task(async_mode=True) as task:
        assert not task.router.is_active
        task.use(pace_controller(task, timeout=1))
        task.run(max_step=10)
    time_end = time.time()
    assert time_end - time_begin < 1


@pytest.mark.unittest
class TestPaceControllerModule:

    def test_pace_controller(self):
        Parallel.runner(n_parallel_workers=2, topology="star")(parallel_main)

    def test_pace_controller_with_theme(self):
        Parallel.runner(n_parallel_workers=2, topology="star")(parallel_main_with_theme)

    def test_pace_controller_with_identity(self):
        Parallel.runner(n_parallel_workers=3, topology="star")(parallel_main_with_identity)

    def test_pace_controller_with_timeout(self):
        Parallel.runner(n_parallel_workers=1, topology="star")(parallel_main_with_timeout)

    def test_pace_controller_with_non_parallel_mode(self):
        non_parallel_main_with_timeout()
