worker.collector
=========================

base_parallel_collector
-------------------------

Please Reference nervex/worker/collector/base_parallel_collector.py for usage

BaseCollector
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.collector.base_parallel_collector.BaseCollector
    :members: __init__, info, error, _setup_timer, _setup_logger, start, close, _iter_after_hook, _policy_inference, _env_step, _process_timestep, get_finish_info, _update_policy, _start_thread, _join_thread


base_serial_collector
-----------------------

Please Reference nervex/worker/collector/base_serial_collector.py for usage

BaseSerialCollector
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.collector.base_serial_collector.BaseSerialCollector
    :members: __init__, collect_data, _collect

CachePool
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.collector.base_serial_collector.CachePool
    :members:  __init__, update, __getitem__, reset


base_serial_evaluator
-----------------------

Please Reference nervex/worker/collector/base_serial_evaluator.py for usage

BaseSerialEvaluator
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.collector.base_serial_evaluator.BaseSerialEvaluator
    :members:   __init__, reset, close, eval
