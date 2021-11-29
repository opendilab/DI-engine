import pytest
import time
import copy
import random
from mpire import WorkerPool
from ding.framework import Task
from ding.framework.parallel import Parallel


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
    assert task.ctx.prev.total_step == 0
    task.forward(step0)
    task.forward(step1)
    assert task.ctx.pipeline == [0, 1]

    # Test context inheritance
    task.ctx.prev.old_prop = "old_prop"  # This prop should be kept to new_ctx.prev
    task.ctx.new_prop = "new_prop"  # The prop should also be kept to new_ctx.prev
    task.renew()
    assert task.ctx.prev.old_prop == "old_prop"
    assert task.ctx.prev.new_prop == "new_prop"
    assert "prev" not in task.ctx.prev


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


def parallel_counter():
    call_count = 0  # +1 when call _counter

    def _counter(ctx):
        nonlocal call_count
        assert ctx.total_step >= call_count
        call_count += 1
        time.sleep(0.1 + random.random() / 10)

    return _counter


def parallel_main():
    task = Task(async_mode=True)
    task.use(parallel_counter())
    task.run(max_step=10)


def parallel_main_eager():
    counter_ware = parallel_counter()
    task = Task(async_mode=True)
    for i in range(5):
        task.forward(counter_ware)
        task.renew()
    assert task.ctx.total_step > i


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

    def wait(ctx):
        time.sleep(0.1)

    with Task() as task:
        task.use(wait)
        task.run(max_step=10)


def attach_mode_attach_task():

    def attach_step(ctx):
        # Should get ctx from other process and start from the latest state
        assert ctx.total_step > 0

    with Task() as task:
        task.use(attach_step)
        task.run(max_step=10)


def attach_mode_main(job):
    if job == "run_task":
        Parallel.runner(
            n_parallel_workers=2, protocol="tcp", address="127.0.0.1", ports=[50501, 50502]
        )(attach_mode_main_task)
    elif "run_attach_task":
        time.sleep(0.3)
        Parallel.runner(
            n_parallel_workers=1,
            protocol="tcp",
            address="127.0.0.1",
            ports=[50503],
            attach_to=["tcp://127.0.0.1:50501", "tcp://127.0.0.1:50502"]
        )(attach_mode_attach_task)
    else:
        raise Exception("Unknown task")


@pytest.mark.unittest
def test_attach_mode():
    with WorkerPool(n_jobs=2, daemon=False) as pool:
        pool.map(attach_mode_main, ["run_task", "run_attach_task"])
