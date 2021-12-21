# In use mode
# In forward mode
# Wrapper in wrapper

import pytest
from ding.framework.task import Task
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

    # Lazy mode (with use statment)
    step_timer = StepTimer()
    with Task() as task:
        task.use_step_wrapper(step_timer)
        task.use(step1)
        task.use(step2)
        task.use(task.sequence(step3, step4))
        task.run(3)

    assert len(step_timer.records) == 5
    for records in step_timer.records.values():
        assert len(records) == 3

    # Eager mode (with forward statment)
    step_timer = StepTimer()
    with Task() as task:
        task.use_step_wrapper(step_timer)
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
    with Task() as task:
        task.use_step_wrapper(step_timer1)
        task.use_step_wrapper(step_timer2)
        task.use(step1)
        task.use(step2)
        task.run(3)

    try:
        assert len(step_timer1.records) == 2
        assert len(step_timer2.records) == 2
    except:
        print("ExceptionStepTimer", step_timer2.records)
        raise Exception("StepTimer error")
    for records in step_timer1.records.values():
        assert len(records) == 3
    for records in step_timer2.records.values():
        assert len(records) == 3
