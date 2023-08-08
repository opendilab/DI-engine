import shutil
from time import sleep
import pytest
import numpy as np
import tempfile

import torch
from ding.data.model_loader import FileModelLoader
from ding.data.storage_loader import FileStorageLoader
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware.distributer import ContextExchanger, ModelExchanger
from ding.framework.parallel import Parallel
from ding.utils.default_helper import set_pkg_seed
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


@pytest.mark.tmp
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


@pytest.mark.tmp
def test_context_exchanger_with_storage_loader():
    Parallel.runner(n_parallel_workers=2)(context_exchanger_with_storage_loader_main)


class MockPolicy:

    def __init__(self) -> None:
        self._model = self._get_model(10, 10)

    def _get_model(self, X_shape, y_shape) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(X_shape, 24), torch.nn.ReLU(), torch.nn.Linear(24, 24), torch.nn.ReLU(),
            torch.nn.Linear(24, y_shape)
        )

    def train(self, X, y):
        loss_fn = torch.nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01)
        y_pred = self._model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            return self._model(X)


def model_exchanger_main():
    with task.start(ctx=OnlineRLContext()):
        set_pkg_seed(0, use_cuda=False)
        policy = MockPolicy()
        X = torch.rand(10)
        y = torch.rand(10)

        if task.router.node_id == 0:
            task.add_role(task.role.LEARNER)
        else:
            task.add_role(task.role.COLLECTOR)

        task.use(ModelExchanger(policy._model))

        if task.has_role(task.role.LEARNER):

            def train(ctx):
                policy.train(X, y)
                sleep(0.3)

            task.use(train)
        else:
            y_pred1 = policy.predict(X)

            def pred(ctx):
                if ctx.total_step > 0:
                    y_pred2 = policy.predict(X)
                    # Ensure model is upgraded
                    assert any(y_pred1 != y_pred2)
                sleep(0.3)

            task.use(pred)

        task.run(2)


@pytest.mark.tmp
def test_model_exchanger():
    Parallel.runner(n_parallel_workers=2, startup_interval=0)(model_exchanger_main)


def model_exchanger_main_with_model_loader():
    with task.start(ctx=OnlineRLContext()):
        set_pkg_seed(0, use_cuda=False)
        policy = MockPolicy()
        X = torch.rand(10)
        y = torch.rand(10)

        if task.router.node_id == 0:
            task.add_role(task.role.LEARNER)
        else:
            task.add_role(task.role.COLLECTOR)

        tempdir = path.join(tempfile.gettempdir(), "test_model_loader")
        model_loader = FileModelLoader(policy._model, dirname=tempdir)
        task.use(ModelExchanger(policy._model, model_loader=model_loader))

        try:
            if task.has_role(task.role.LEARNER):

                def train(ctx):
                    policy.train(X, y)
                    sleep(0.3)

                task.use(train)
            else:
                y_pred1 = policy.predict(X)

                def pred(ctx):
                    if ctx.total_step > 0:
                        y_pred2 = policy.predict(X)
                        # Ensure model is upgraded
                        assert any(y_pred1 != y_pred2)
                    sleep(0.3)

                task.use(pred)
            task.run(2)
        finally:
            model_loader.shutdown()
            sleep(0.3)
            if path.exists(tempdir):
                shutil.rmtree(tempdir)


@pytest.mark.tmp
def test_model_exchanger_with_model_loader():
    Parallel.runner(n_parallel_workers=2, startup_interval=0)(model_exchanger_main_with_model_loader)
