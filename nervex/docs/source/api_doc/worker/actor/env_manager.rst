
worker.actor.env_manager
========================

base_env_manager
-----------------

Please Reference nervex/worker/actor/env_manager/base_env_manager.py for usage

BaseEnvManager
~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.worker.actor.env_manager.base_env_manager.BaseEnvManager
    :members: __init__, next_obs, launch, step, seed, close

subprocess_env_manager
----------------------

Please Reference nervex/worker/actor/env_manager/subprocess_env_manager.py for usage

ShmBuffer
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.env_manager.subprocess_env_manager.ShmBuffer
    :members: __init__, fill, get

ShmBufferContainer
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.env_manager.subprocess_env_manager.ShmBufferContainer
    :members: __init__, fill, get

SubprocessEnvManager
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.worker.actor.env_manager.subprocess_env_manager.SubprocessEnvManager
    :members: __init__, next_obs, launch, reset, step, seed, close, _create_state, _setup_async_args
