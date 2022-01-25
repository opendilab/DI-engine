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
from ding.framework.middleware import Pace_Controller


@pytest.mark.unittest
class TestPaceControllerModule:

    def test(self):

        def main_for_test():
            with Task(async_mode=True) as task:

                def _count_listen(signal):
                    task.ctx.setdefault("count1", 0)
                    task.ctx.keep("count1")
                    task.ctx.setdefault("count2", 0)
                    task.ctx.keep("count2")
                    if signal:
                        print("do 1")
                        task.ctx.count2 += 1
                    else:
                        print("do 2")
                        task.ctx.count1 += 1

                task.on("count", _count_listen)

                def count1(task: "Task"):

                    def _count(ctx: "Context"):
                        ctx.setdefault("count1", 0)
                        ctx.keep("count1")
                        ctx.setdefault("count2", 0)
                        ctx.keep("count2")
                        #ctx.count1 += 1
                        task.emit("count", True, only_local=True)
                        print("sent the count 1 !")

                    return _count

                def count2(task: "Task"):

                    def _count(ctx: "Context"):
                        ctx.setdefault("count1", 0)
                        ctx.keep("count1")
                        ctx.setdefault("count2", 0)
                        ctx.keep("count2")
                        #ctx.count2 += 1
                        task.emit("count", False, only_local=True)
                        print("sent the count 2 !")

                    return _count

                def exam1(task: "Task"):

                    def _exam(ctx: "Context"):
                        assert ctx.count1 <= ctx.total_step + 1
                        assert ctx.count1 >= ctx.total_step

                    return _exam

                def exam2(task: "Task"):

                    def _exam(ctx: "Context"):
                        assert ctx.count2 <= ctx.total_step + 1
                        assert ctx.count2 >= ctx.total_step

                    return _exam

                task.use(
                    task.sequence(
                        count1(task),
                        Pace_Controller(
                            task,
                            theme="Test",
                            active_state=True,
                            filter_the_similar=False,
                            start_up_delay=0.5,
                            #timeout=5
                        ),
                        exam1(task)
                    )
                )

                task.use(
                    task.sequence(
                        count2(task),
                        Pace_Controller(
                            task,
                            theme="Test",
                            active_state=True,
                            filter_the_similar=False,
                            start_up_delay=0.5,
                            #timeout=5
                        ),
                        exam2(task)
                    )
                )
                task.run(max_step=100)

        main_for_test()
