import pytest
import time
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
