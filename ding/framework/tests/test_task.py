import pytest
from ding.framework import Task


@pytest.mark.unittest
def test_serial_pipeline():

    def step1(ctx):
        ctx.setdefault("pipeline", [])
        ctx.pipeline.append(0)

    def step2(ctx):
        ctx.pipeline.append(1)

    # Execute step1, step2 twice
    task = Task()
    for _ in range(2):
        task.forward(step1)
        task.forward(step2)
    assert task.ctx.pipeline == [0, 1, 0, 1]

    # Renew and execute step1, step2
    task.renew()
    task.forward(step1)
    task.forward(step2)
    assert task.ctx.pipeline == [0, 1]


@pytest.mark.unittest
def test_serial_yield_pipeline():

    def step1(ctx):
        ctx.setdefault("pipeline", [])
        ctx.pipeline.append(0)
        yield
        ctx.pipeline.append(0)

    def step2(ctx):
        ctx.pipeline.append(1)

    task = Task()
    task.forward(step1)
    task.forward(step2)
    assert task.ctx.pipeline == [0, 1, 0]
