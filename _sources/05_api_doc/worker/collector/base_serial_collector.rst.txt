worker.collector.base_serial_collector
================================================

base_serial_collector
------------------------

Please Reference ding/worker/collector/base_serial_collector.py for usage

ISerialCollector
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.collector.base_serial_collector.ISerialCollector
    :members: default_config, reset_env, reset_policy, reset, collect, envstep

create_serial_collector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: ding.worker.collector.base_serial_collector.create_serial_collector

get_serial_collector_cls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: ding.worker.collector.base_serial_collector.get_serial_collector_cls

CachePool
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.collector.base_serial_collector.CachePool
    :members: __init__, update, __getitem__, reset

TrajBuffer
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.collector.base_serial_collector.TrajBuffer
    :members: __init__, append

to_tensor_transitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: ding.worker.collector.base_serial_collector.to_tensor_transitions