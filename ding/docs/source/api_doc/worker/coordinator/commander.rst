worker.coordinator.commander
=============================

base_serial_commander
--------------------------


BaseSerialCommander
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.worker.coordinator.base_serial_commander.BaseSerialCommander
    :members:  __init__, step, policy


base_parallel_commander
--------------------------


BaseCommander
~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.worker.coordinator.base_parallel_commander.BaseCommander
    :members: get_collector_task


NaiveCommander
~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.worker.coordinator.base_parallel_commander.NaiveCommander
    :members: __init__, get_collector_task, get_learner_task, finish_collector_task, finish_learner_task, notify_fail_collector_task, notify_fail_learner_task


create_parallel_commander
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ding.worker.coordinator.base_parallel_commander.create_parallel_commander


solo_parallel_commander
----------------------------------------------


SoloCommander
~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.worker.coordinator.solo_parallel_commander.SoloCommander
    :members: __init__, get_collector_task, get_learner_task, finish_collector_task, finish_learner_task, notify_fail_collector_task, notify_fail_learner_task
