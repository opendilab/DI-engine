# In use mode
# In forward mode
# Wrapper in wrapper

import pytest
from ding.framework import task
from ding.framework.wrapper import StepTimer


@pytest.mark.unittest
def test_step_timer():

    def step1(_):
        1

    def step2(_):
        2

    def step3(_):
        3

    def step4(_):
        4

    step_timer = StepTimer()
    with task.start(async_mode=True):
        task.use_wrapper(step_timer)
        task.use(step1)
        task.use(step2)
        task.use(task.serial(step3, step4))
        assert len(task._middleware) == 3
        task.run(3)

    assert len(step_timer.records) == 3
    for records in step_timer.records.values():
        assert len(records) == 3

    # Wrapper in wrapper
    step_timer1 = StepTimer()
    step_timer2 = StepTimer()
    with task.start():
        task.use_wrapper(step_timer1)
        task.use_wrapper(step_timer2)
        task.use(step1)
        task.use(step2)
        assert len(task._middleware) == 2
        task.run(3)

    for records in step_timer1.records.values():
        assert len(records) == 3
    for records in step_timer2.records.values():
        assert len(records) == 3
