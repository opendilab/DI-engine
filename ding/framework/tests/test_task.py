from concurrent.futures import thread
from os import spawnl
from attr.validators import instance_of
import pytest
import time
import copy
import random
from mpire import WorkerPool
from ding.framework import Task
from ding.framework.context import Context
from ding.framework.parallel import Parallel
from ding.utils.design_helper import SingletonMetaclass


@pytest.mark.unittest
def test_serial_pipeline():

    def step0(ctx):
        ctx.setdefault("pipeline", [])
        ctx.pipeline.append(0)

    def step1(ctx):
        ctx.pipeline.append(1)

    # Execute step1, step2 twice
    task = Task()
    for _ in range(2):
        task.forward(step0)
        task.forward(step1)
    assert task.ctx.pipeline == [0, 1, 0, 1]

    # Renew and execute step1, step2
    task.renew()
    assert task.ctx.total_step == 1
    task.forward(step0)
    task.forward(step1)
    assert task.ctx.pipeline == [0, 1]

    # Test context inheritance
    task.renew()


@pytest.mark.unittest
def test_serial_yield_pipeline():

    def step0(ctx):
        ctx.setdefault("pipeline", [])
        ctx.pipeline.append(0)
        yield
        ctx.pipeline.append(0)

    def step1(ctx):
        ctx.pipeline.append(1)

    task = Task()
    task.forward(step0)
    task.forward(step1)
    task.backward()
    assert task.ctx.pipeline == [0, 1, 0]
    assert len(task._backward_stack) == 0


@pytest.mark.unittest
def test_async_pipeline():

    def step0(ctx):
        ctx.setdefault("pipeline", [])
        ctx.pipeline.append(0)

    def step1(ctx):
        ctx.pipeline.append(1)

    # Execute step1, step2 twice
    task = Task(async_mode=True)
    for _ in range(2):
        task.forward(step0)
        time.sleep(0.1)
        task.forward(step1)
        time.sleep(0.1)
    task.backward()
    assert task.ctx.pipeline == [0, 1, 0, 1]
    task.renew()
    assert task.ctx.total_step == 1


@pytest.mark.unittest
def test_async_yield_pipeline():

    def step0(ctx):
        ctx.setdefault("pipeline", [])
        time.sleep(0.1)
        ctx.pipeline.append(0)
        yield
        ctx.pipeline.append(0)

    def step1(ctx):
        time.sleep(0.2)
        ctx.pipeline.append(1)

    task = Task(async_mode=True)
    task.forward(step0)
    task.forward(step1)
    time.sleep(0.3)
    task.backward().sync()
    assert task.ctx.pipeline == [0, 1, 0]
    assert len(task._backward_stack) == 0


def parallel_main():
    task = Task()
    sync_count = 0

    def on_sync_parallel_ctx(ctx):
        nonlocal sync_count
        assert isinstance(ctx, Context)
        sync_count += 1

    task.on("sync_parallel_ctx", on_sync_parallel_ctx)
    task.use(lambda _: time.sleep(0.01 + random.random() / 10))
    task.run(max_step=10)
    assert sync_count > 0


def parallel_main_eager():
    task = Task()
    sync_count = 0

    def on_sync_parallel_ctx(ctx):
        nonlocal sync_count
        assert isinstance(ctx, Context)
        sync_count += 1

    task.on("sync_parallel_ctx", on_sync_parallel_ctx)
    for _ in range(10):
        task.forward(lambda _: time.sleep(0.01 + random.random() / 10))
        task.renew()
    assert sync_count > 0


@pytest.mark.unittest
def test_parallel_pipeline():
    Parallel.runner(n_parallel_workers=2)(parallel_main_eager)
    Parallel.runner(n_parallel_workers=2)(parallel_main)


@pytest.mark.unittest
def test_copy_task():
    t1 = Task(async_mode=True, n_async_workers=1)
    t2 = copy.copy(t1)
    assert t2.async_mode
    assert t1 is not t2


def attach_mode_main_task():
    with Task() as task:
        task.use(lambda _: time.sleep(0.1))
        task.run(max_step=10)


def attach_mode_attach_task():
    ctx = None

    def attach_callback(new_ctx):
        nonlocal ctx
        ctx = new_ctx

    with Task(attach_callback=attach_callback) as task:
        task.use(lambda _: time.sleep(0.1))
        task.run(max_step=10)
    assert ctx is not None


def attach_mode_main(job):
    if job == "run_task":
        Parallel.runner(
            n_parallel_workers=2, protocol="tcp", address="127.0.0.1", ports=[50501, 50502]
        )(attach_mode_main_task)
    elif "run_attach_task":
        time.sleep(0.3)
        try:
            Parallel.runner(
                n_parallel_workers=1,
                protocol="tcp",
                address="127.0.0.1",
                ports=[50503],
                attach_to=["tcp://127.0.0.1:50501", "tcp://127.0.0.1:50502"]
            )(attach_mode_attach_task)
        finally:
            del SingletonMetaclass.instances[Parallel]
    else:
        raise Exception("Unknown task")


@pytest.mark.unittest
def test_attach_mode():
    with WorkerPool(n_jobs=2, daemon=False, start_method="spawn") as pool:
        pool.map(attach_mode_main, ["run_task", "run_attach_task"])


@pytest.mark.unittest
def test_label():
    task = Task()
    result = {}
    task.use(lambda _: result.setdefault("not_me", True), filter_labels=["async"])
    task.use(lambda _: result.setdefault("has_me", True))
    task.run(max_step=1)

    assert "not_me" not in result
    assert "has_me" in result


def sync_parallel_ctx_main():
    with Task() as task:
        task.use(lambda _: time.sleep(1))
        if task.router.node_id == 0:  # Fast
            task.run(max_step=2)
        else:  # Slow
            task.run(max_step=10)
    assert task.parallel_ctx
    assert task.ctx.finish
    assert task.ctx.total_step < 9


@pytest.mark.unittest
def test_sync_parallel_ctx():
    Parallel.runner(n_parallel_workers=2)(sync_parallel_ctx_main)
