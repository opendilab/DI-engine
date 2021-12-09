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
    step1 = lambda _: 1
    step2 = lambda _: 2
    step3 = lambda _: 3
    step4 = lambda _: 4
    task.use(step1)
    task.use(step2)
    task.use(task.sequence(step3, step4))
    task.run(3)

    assert len(step_timer.records) == 5
    for records in step_timer.records.values():
        assert len(records) == 3

    # Eager mode (with forward statment)
    step_timer = StepTimer()
    task = Task()
    task.use_step_wrapper(step_timer)
    step1 = lambda _: 1
    step2 = lambda _: 2
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
    step1 = lambda _: 1
    step2 = lambda _: 2
    task.use(step1)
    task.use(step2)
    task.run(3)

    assert len(step_timer1.records) == 2
    assert len(step_timer2.records) == 2
    for records in step_timer1.records.values():
        assert len(records) == 3
    for records in step_timer2.records.values():
        assert len(records) == 3
