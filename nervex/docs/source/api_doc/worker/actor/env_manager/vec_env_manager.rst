
worker.actor.env_manager
=========================

vec_env_manager
-----------------

ShmBuffer
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.env_manager.vec_env_manager.ShmBuffer 
    :members: __init__, fill, get

ShmBufferContainer
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.env_manager.vec_env_manager.ShmBufferContainer 
    :members: __init__, fill, get

SubprocessEnvManager
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.worker.actor.env_manager.vec_env_manager.SubprocessEnvManager 
    :members: __init__, next_obs, launch, reset, step, seed, close, wait, _create_state, _setup_async_args