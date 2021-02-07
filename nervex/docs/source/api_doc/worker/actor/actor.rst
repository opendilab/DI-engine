worker.actor
=========================

base_parallel_actor
---------------------

Please Reference nervex/worker/actor/base_parallel_actor.py for usage

BaseActor
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.base_parallel_actor.BaseActor
    :members: __init__, info, error, _setup_timer, _setup_logger, start, close, _iter_after_hook, _policy_inference, _env_step, _process_timestep, _finish_task, _update_policy, _start_thread


base_serial_actor
-----------------

Please Reference nervex/worker/actor/base_serial_actor.py for usage

BaseSerialActor
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.base_serial_actor.BaseSerialActor
    :members: __init__, generate_data, _collect

CachePool
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.base_serial_actor.CachePool
    :members:  __init__, update, __getitem__, reset


base_serial_evaluator
-----------------------

Please Reference nervex/worker/actor/base_serial_evaluator.py for usage

BaseSerialEvaluator
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.base_serial_evaluator.BaseSerialEvaluator
    :members:   __init__, reset, close, eval
