import pytest
import unittest
from unittest import mock
from unittest.mock import patch
import pathlib as pl
import os
import shutil
from typing import Callable

from ding.framework import Task, Context
from ding.framework import Parallel
from ding.framework.middleware import pace_controller


def parallel_main():

    def fn(task: "Task"):
        another_node_total_step = 0

        def _listen_total_step(total_step):
            nonlocal another_node_total_step
            another_node_total_step = total_step
            return

        task.on("total_step", _listen_total_step)

        def _fn(ctx: "Context"):
            nonlocal another_node_total_step
            assert ctx.total_step <= another_node_total_step + 1
            assert ctx.total_step >= another_node_total_step
            task.emit("total_step", ctx.total_step, only_local=True)
            return

        return _fn

    with Task(async_mode=True) as task:
        task.use(fn(task))
        task.use(pace_controller(task))
        task.run(max_step=100)


@pytest.mark.unittest
class TestPaceControllerModule:

    def test(self):
        Parallel.runner(n_parallel_workers=2, topology="star")(parallel_main)
