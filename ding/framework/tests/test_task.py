import pytest
import time
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

    def on_count():
        nonlocal sync_count
        sync_count += 1

    def counter(task):

        def _counter(ctx):
            time.sleep(0.2 + random.random() / 10)
            task.emit("count", only_remote=True)

        return _counter

    with Task() as task:
        task.on("count", on_count)
        task.use(counter(task))
        task.run(max_step=10)
        assert sync_count > 0


@pytest.mark.unittest
def test_parallel_pipeline():
    Parallel.runner(n_parallel_workers=2)(parallel_main)


@pytest.mark.unittest
def test_label():
    with Task() as task:
        result = {}
        task.use(lambda _: result.setdefault("not_me", True), filter_labels=["async"])
        task.use(lambda _: result.setdefault("has_me", True))
        task.run(max_step=1)

        assert "not_me" not in result
        assert "has_me" in result


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
                task.emit("Greeting", "Hi")
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


def early_stop_main():
    with Task() as task:
        task.use(lambda _: time.sleep(0.2))
        if task.match_labels("node.0"):
            task.run(max_step=10)
        else:
            task.run(max_step=2)
        assert task.ctx.total_step < 7


@pytest.mark.unittest
def test_early_stop():
    Parallel.runner(n_parallel_workers=2)(early_stop_main)
