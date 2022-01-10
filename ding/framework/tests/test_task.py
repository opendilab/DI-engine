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
    with Task() as task:
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

    with Task() as task:
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
    with Task(async_mode=True) as task:
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

    with Task(async_mode=True) as task:
        task.forward(step0)
        task.forward(step1)
        time.sleep(0.3)
        task.backward().sync()
        assert task.ctx.pipeline == [0, 1, 0]
        assert len(task._backward_stack) == 0


def parallel_main():
    sync_count = 0

    def on_sync_parallel_ctx(ctx):
        nonlocal sync_count
        assert isinstance(ctx, Context)
        sync_count += 1

    with Task() as task:
        task.on("sync_parallel_ctx", on_sync_parallel_ctx)
        task.use(lambda _: time.sleep(0.2 + random.random() / 10))
        task.run(max_step=10)
        assert sync_count > 0


def parallel_main_eager():
    sync_count = 0

    def on_sync_parallel_ctx(ctx):
        nonlocal sync_count
        assert isinstance(ctx, Context)
        sync_count += 1

    with Task() as task:
        task.on("sync_parallel_ctx", on_sync_parallel_ctx)
        for _ in range(10):
            task.forward(lambda _: time.sleep(0.2 + random.random() / 10))
            task.renew()
        assert sync_count > 0


@pytest.mark.unittest
def test_parallel_pipeline():
    Parallel.runner(n_parallel_workers=2)(parallel_main_eager)
    Parallel.runner(n_parallel_workers=2)(parallel_main)


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
    with Task() as task:
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


@pytest.mark.unittest
def test_emit():
    with Task() as task:
        greets = []
        task.on("Greeting", lambda msg: greets.append(msg))

        def step1(ctx):
            task.emit("Greeting", "Hi")

        task.use(step1)
        task.run(max_step=10)
    assert len(greets) == 10


def emit_remote_main():
    with Task() as task:
        time.sleep(0.3)  # Wait for bound
        greets = []
        if task.router.node_id == 0:
            task.on("Greeting", lambda msg: greets.append(msg))
        else:
            for _ in range(10):
                task.emit_remote("Greeting", "Hi")
                time.sleep(0.1)
        time.sleep(1.2)
        if task.router.node_id == 0:
            assert len(greets) > 5
        else:
            assert len(greets) == 0


@pytest.mark.unittest
def test_emit_remote():
    Parallel.runner(n_parallel_workers=2)(emit_remote_main)


@pytest.mark.unittest
def test_wait_for():
    # Wait for will only work in async or parallel mode
    with Task(async_mode=True, n_async_workers=2) as task:
        greets = []

        def step1(_):
            hi = task.wait_for("Greeting")[0][0]
            if hi:
                greets.append(hi)

        def step2(_):
            task.emit("Greeting", "Hi")

        task.use(step1)
        task.use(step2)
        task.run(max_step=10)

    assert len(greets) == 10
    assert all(map(lambda hi: hi == "Hi", greets))

    # Test timeout exception
    with Task(async_mode=True, n_async_workers=2) as task:

        def step1(_):
            task.wait_for("Greeting", timeout=0.3, ignore_timeout_exception=False)

        task.use(step1)
        with pytest.raises(TimeoutError):
            task.run(max_step=1)


@pytest.mark.unittest
def test_async_exception():
    with Task(async_mode=True, n_async_workers=2) as task:

        def step1(_):
            task.wait_for("any_event")  # Never end

        def step2(_):
            time.sleep(0.3)
            raise Exception("Oh")

        task.use(step1)
        task.use(step2)
        with pytest.raises(Exception):
            task.run(max_step=2)

        assert task.ctx.total_step == 0
