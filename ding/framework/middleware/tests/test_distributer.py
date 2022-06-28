import shutil
from time import sleep
import pytest
import numpy as np
import tempfile
from ding.data.storage_loader import FileStorageLoader
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware.distributer import ContextExchanger
from ding.framework.parallel import Parallel
from os import path


def context_exchanger_main():
    with task.start(ctx=OnlineRLContext()):
        if task.router.node_id == 0:
            task.add_role(task.role.LEARNER)
        elif task.router.node_id == 1:
            task.add_role(task.role.COLLECTOR)

        task.use(ContextExchanger(skip_n_iter=1))

        if task.has_role(task.role.LEARNER):

            def learner_context(ctx: OnlineRLContext):
                assert len(ctx.trajectories) == 2
                assert len(ctx.trajectory_end_idx) == 4
                assert len(ctx.episodes) == 8
                assert ctx.env_step > 0
                assert ctx.env_episode > 0
                yield
                ctx.train_iter += 1

            task.use(learner_context)
        elif task.has_role(task.role.COLLECTOR):

            def collector_context(ctx: OnlineRLContext):
                if ctx.total_step > 0:
                    assert ctx.train_iter > 0
                yield
                ctx.trajectories = [np.random.rand(10, 10) for _ in range(2)]
                ctx.trajectory_end_idx = [1 for _ in range(4)]
                ctx.episodes = [np.random.rand(10, 10) for _ in range(8)]
                ctx.env_step += 1
                ctx.env_episode += 1

            task.use(collector_context)

        task.run(max_step=3)


@pytest.mark.unittest
def test_context_exchanger():
    Parallel.runner(n_parallel_workers=2)(context_exchanger_main)


def context_exchanger_with_storage_loader_main():
    with task.start(ctx=OnlineRLContext()):
        if task.router.node_id == 0:
            task.add_role(task.role.LEARNER)
        elif task.router.node_id == 1:
            task.add_role(task.role.COLLECTOR)

        tempdir = path.join(tempfile.gettempdir(), "test_storage_loader")
        storage_loader = FileStorageLoader(dirname=tempdir)
        try:
            task.use(ContextExchanger(skip_n_iter=1, storage_loader=storage_loader))

            if task.has_role(task.role.LEARNER):

                def learner_context(ctx: OnlineRLContext):
                    assert len(ctx.trajectories) == 2
                    assert len(ctx.trajectory_end_idx) == 4
                    assert len(ctx.episodes) == 8
                    assert ctx.env_step > 0
                    assert ctx.env_episode > 0
                    yield
                    ctx.train_iter += 1

                task.use(learner_context)
            elif task.has_role(task.role.COLLECTOR):

                def collector_context(ctx: OnlineRLContext):
                    if ctx.total_step > 0:
                        assert ctx.train_iter > 0
                    yield
                    ctx.trajectories = [np.random.rand(10, 10) for _ in range(2)]
                    ctx.trajectory_end_idx = [1 for _ in range(4)]
                    ctx.episodes = [np.random.rand(10, 10) for _ in range(8)]
                    ctx.env_step += 1
                    ctx.env_episode += 1

                task.use(collector_context)

            task.run(max_step=3)
        finally:
            storage_loader.shutdown()
            sleep(1)
            if path.exists(tempdir):
                shutil.rmtree(tempdir)


@pytest.mark.unittest
def test_context_exchanger_with_storage_loader():
    Parallel.runner(n_parallel_workers=2)(context_exchanger_with_storage_loader_main)
