import pytest
import time
import tempfile
import shutil
import os
from os import path
from ding.framework import Parallel
from ding.framework.task import task
from ding.utils import DistributedWriter


def main_distributed_writer(tempdir):
    with task.start():
        time.sleep(task.router.node_id * 1)  # Sleep 0 and 1, write to different files

        tblogger = DistributedWriter(tempdir).plugin(task.router, is_writer=(task.router.node_id == 0))

        def _add_scalar(ctx):
            n = 10
            for i in range(n):
                tblogger.add_scalar(str(task.router.node_id), task.router.node_id, ctx.total_step * n + i)

        task.use(_add_scalar)
        task.use(lambda _: time.sleep(0.2))
        task.run(max_step=3)

        time.sleep(0.3 + (1 - task.router.node_id) * 2)


@pytest.mark.unittest
def test_distributed_writer():
    tempdir = path.join(tempfile.gettempdir(), "tblogger")
    try:
        Parallel.runner(n_parallel_workers=2)(main_distributed_writer, tempdir)
        assert path.exists(tempdir)
        assert len(os.listdir(tempdir)) == 1
    finally:
        if path.exists(tempdir):
            shutil.rmtree(tempdir)
