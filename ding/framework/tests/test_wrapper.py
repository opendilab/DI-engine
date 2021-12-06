# In use mode
# In forward mode
# Wrapper in wrapper

import pytest
from ding.framework.task import Task
from ding.framework.wrapper import StepTimer


@pytest.mark.unittest
def test_step_timer():
    # Lazy mode (with use statment)
    step_timer = StepTimer()
    task = Task()
    task.use_step_wrapper(step_timer)
    task.use(lambda _: None)  # Step 1
    task.use(lambda _: None)  # Step 2
    task.run(3)

    assert len(step_timer.records) == 2
    for records in step_timer.records.values():
        assert len(records) == 3

    # Eager mode (with forward statment)
    step_timer = StepTimer()
    task = Task()
    task.use_step_wrapper(step_timer)
    step1 = lambda _: None
    step2 = lambda _: None
    for _ in range(3):
        task.forward(step1)  # Step 1
        task.forward(step2)  # Step 2
        task.renew()

    assert len(step_timer.records) == 2
    for records in step_timer.records.values():
        assert len(records) == 3

    # Wrapper in wrapper
    step_timer1 = StepTimer()
    step_timer2 = StepTimer()
    task = Task()
    task.use_step_wrapper(step_timer1)
    task.use_step_wrapper(step_timer2)
    task.use(lambda _: None)  # Step 1
    task.use(lambda _: None)  # Step 2
    task.run(3)

    assert len(step_timer1.records) == 2
    assert len(step_timer2.records) == 2
    for records in step_timer1.records.values():
        assert len(records) == 3
    for records in step_timer2.records.values():
        assert len(records) == 3
