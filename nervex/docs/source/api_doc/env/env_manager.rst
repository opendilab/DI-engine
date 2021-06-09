
envs.env_manager
========================

base_env_manager
-----------------

Please Reference nerveX/nervex/envs/env_manager/base_env_manager.py for usage

BaseEnvManager
~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.envs.env_manager.base_env_manager.BaseEnvManager
    :members: __init__, ready_obs, launch, step, seed, close

subprocess_env_manager
-------------------------

Please Reference nerveX/nervex/envs/env_manager/subprocess_env_manager.py for usage

ShmBuffer
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_manager.subprocess_env_manager.ShmBuffer
    :members: __init__, fill, get

ShmBufferContainer
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_manager.subprocess_env_manager.ShmBufferContainer
    :members: __init__, fill, get

AsyncSubprocessEnvManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_manager.subprocess_env_manager.AsyncSubprocessEnvManager
    :members: __init__, ready_obs, launch, reset, step, seed, close, _create_state, _setup_async_args

SyncSubprocessEnvManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_manager.subprocess_env_manager.SyncSubprocessEnvManager
    :members: __init__, ready_obs, launch, reset, step, seed, close, _create_state, _setup_async_args
