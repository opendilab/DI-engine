import pytest
import unittest
from unittest import mock
from unittest.mock import patch
import math
import time
from typing import Callable

from ding.framework import Task, Context
from ding.framework import Parallel
from ding.framework.middleware import pace_controller


def fn(task: "Task"):

    def _fn(ctx: "Context"):
        time.sleep(0.01)

    return _fn


def parallel_main(theme: str = "", timeout: float = math.inf, if_test_identity: bool = False):
    with Task(async_mode=True) as task:
        max_step = 10

        def _listen_to_finish(value):
            if if_test_identity and task.router.node_id > 0:
                assert task.ctx.total_step <= max_step
            else:
                assert task.ctx.total_step >= max_step - 1 and task.ctx.total_step <= max_step

        task.on("finish", _listen_to_finish)

        identity = str(task.router.node_id)
        if not if_test_identity:
            identity = ""

        if task.router.node_id > 0:
            task.use(fn(task))
            task.use(pace_controller(task, theme=theme, identity=identity, timeout=timeout))
        else:
            task.use(pace_controller(task, theme=theme, identity=identity, timeout=timeout))
        task.run(max_step=max_step)


def main():
    with Task(async_mode=True) as task:
        assert not task.router.is_active
        task.use(pace_controller(task, timeout=1))
        task.run(max_step=10)


@pytest.mark.unittest
class TestPaceControllerModule:

    def test_pace_controller(self):
        Parallel.runner(n_parallel_workers=2, topology="star")(parallel_main)

    def test_pace_controller_with_theme(self):
        Parallel.runner(n_parallel_workers=2, topology="star")(parallel_main, theme="test")

    def test_pace_controller_with_identity(self):
        Parallel.runner(n_parallel_workers=3, topology="star")(parallel_main, if_test_identity=True)

    def test_pace_controller_with_timeout(self):
        time_begin = time.time()
        Parallel.runner(n_parallel_workers=1, topology="star")(parallel_main, timeout=0.1)
        time_end = time.time()
        assert time_end - time_begin > 1

    def test_pace_controller_in_non_parallel_mode(self):
        time_begin = time.time()
        main()
        time_end = time.time()
        assert time_end - time_begin < 1
