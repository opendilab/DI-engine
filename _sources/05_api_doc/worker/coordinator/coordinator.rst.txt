

worker.coordinator.coordinator
--------------------------------


TaskState
~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.worker.coordinator.coordinator.TaskState
    :members: __init__


Coordinator
~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.worker.coordinator.coordinator.Coordinator
    :members:  __init__, start, close, __del__, state_dict, load_state_dict, deal_with_collector_send_data, deal_with_collector_finish_task, deal_with_learner_get_data, deal_with_learner_send_info, deal_with_learner_finish_task, info, error, system_shutdown_flag

