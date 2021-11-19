from mpire.pool import WorkerPool
import pytest
import time
import copy
import random
from ding.framework import Task


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


@pytest.mark.unittest
def test_parallel_pipeline():
    task = Task(async_mode=True, n_async_workers=1, parallel_mode=True, n_parallel_workers=2)

    def counter():
        call_count = 0  # +1 when call _counter

        def _counter(ctx):
            nonlocal call_count
            if call_count > 3:
                assert ctx.total_step > call_count
            call_count += 1
            time.sleep(0.1 + random.random() / 10)

        return _counter

    def _execute(task: Task):
        counter_ware = counter()
        for i in range(5):
            task.forward(counter_ware)
            task.renew()
        assert task.ctx.total_step > i

    # In eager mode
    task.parallel(_execute)

    # In pipeline mode
    task = Task(async_mode=True, n_async_workers=1, parallel_mode=True, n_parallel_workers=2)
    task.use(counter())
    task.run(max_step=5)


@pytest.mark.unittest
def test_copy_task():
    t1 = Task(async_mode=True, n_async_workers=1, parallel_mode=True, n_parallel_workers=2)
    t2 = copy.copy(t1)
    assert t2.parallel_mode
    assert t2.async_mode
    assert t1 is not t2


@pytest.mark.unittest
def test_attach_mode():

    def run_task():

        def wait(ctx):
            time.sleep(0.1)

        task = Task(
            parallel_mode=True,
            n_parallel_workers=2,
            protocol="tcp",
            bind_address="127.0.0.1",
            bind_ports=[50501, 50502]
        )
        task.use(wait)
        task.run(max_step=10)

    def run_attach_task():
        attach_task = Task(
            parallel_mode=True,
            n_parallel_workers=1,
            protocol="tcp",
            bind_address="127.0.0.1",
            bind_ports=[50503],
            attach_to=["tcp://127.0.0.1:50501", "tcp://127.0.0.1:50502"]
        )

        def attach_step(ctx):
            # Should get ctx from other process and start from the latest state
            assert ctx.total_step > 0

        time.sleep(0.2)
        attach_task.use(attach_step)
        attach_task.run(max_step=10)

    def main_test(job):
        if job == "run_task":
            run_task()
        elif "run_attach_task":
            run_attach_task()
        else:
            raise Exception("Unknown task")

    with WorkerPool(n_jobs=2, daemon=False) as pool:
        pool.map(main_test, ["run_task", "run_attach_task"])
