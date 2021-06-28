
envs.env_manager
========================

base_env_manager
-----------------

BaseEnvManager
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_manager.base_env_manager.BaseEnvManager
    :members: reset, step, seed, close, enable_save_replay, launch, env_info, default_config, ready_obs

create_env_manager
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: nervex.envs.env_manager.base_env_manager.create_env_manager

get_env_manager_cls
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: nervex.envs.env_manager.base_env_manager.get_env_manager_cls

subprocess_env_manager
-------------------------

ShmBuffer
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_manager.subprocess_env_manager.ShmBuffer
    :members: fill, get

ShmBufferContainer
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_manager.subprocess_env_manager.ShmBufferContainer
    :members: fill, get

SyncSubprocessEnvManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_manager.subprocess_env_manager.SyncSubprocessEnvManager
    :members: reset, step, seed, close, enable_save_replay, launch, env_info, default_config, ready_obs


AsyncSubprocessEnvManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_manager.subprocess_env_manager.AsyncSubprocessEnvManager
    :members: reset, step, seed, close, enable_save_replay, launch, env_info, default_config, ready_obs
