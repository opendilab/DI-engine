from mpire.pool import WorkerPool
import pytest
import time
import tempfile
import shutil
import os
from os import path
from ding.framework import Parallel
from ding.framework.task import Task
from ding.utils import DistributedWriter


def main_distributed_writer(tempdir):
    with Task() as task:
        time.sleep(task.router.node_id * 1)  # Sleep 0 and 1, write to different files

        tblogger = DistributedWriter(tempdir).plugin(task, is_writer=("node.0" in task.labels))

        def _add_scalar(ctx):
            tblogger.add_scalar(str(task.router.node_id), task.router.node_id, ctx.total_step)

        task.use(_add_scalar)
        task.use(lambda _: time.sleep(0.2))
        task.run(max_step=10)

        time.sleep(0.3 + (1 - task.router.node_id) * 2)


@pytest.mark.unittest
def test_distributed_writer():
    tempdir = path.join(tempfile.gettempdir(), "tblogger")
    try:
        Parallel.runner(n_parallel_workers=2)(main_distributed_writer, tempdir)
    finally:
        if path.exists(tempdir):
            print("DIRS", os.listdir(tempdir))
            shutil.rmtree(tempdir)
