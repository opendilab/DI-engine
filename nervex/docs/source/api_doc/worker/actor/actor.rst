worker.actor
=========================

base_parallel_actor
-----------------

Please Reference nervex/worker/actor/base_parallel_actor.py for usage

TickMonitor
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.base_parallel_actor.TickMonitor
    :members: __init__, fixed_time, current_time, freeze, unfreeze, register_attribute_value, __getattr__

BaseActor
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.base_parallel_actor.TickMonitor
    :members: __init__, info, error, _setup_timer, _setup_logger, start, close, _iter_after_hook, _policy_inference, _env_step, _process_timestep, _finish_task, _update_policy, _start_thread


base_serial_actor
-----------------

Please Reference nervex/worker/actor/base_serial_actor.py for usage

BaseSerialActor
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.base_serial_actor.BaseSerialActor
    :members: __init__, update, __getitem__, reset

CachePool
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.base_serial_actor.CachePool
    :members:  __init__, update, __getitem__, resetgit


base_serial_actor
-----------------

Please Reference nervex/worker/actor/base_serial_actor.py for usage

BaseSerialEvaluator
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.base_serial_evaluator.BaseSerialEvaluator
    :members:   __init__, reset, close, eval

CachePool
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.base_serial_evaluator.CachePool
    :members: __init__, update, __getitem__, reset