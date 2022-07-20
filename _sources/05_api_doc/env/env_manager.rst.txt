
envs.env_manager
========================

base_env_manager
-----------------

BaseEnvManager
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_manager.base_env_manager.BaseEnvManager
    :members: reset, step, seed, close, enable_save_replay, launch, default_config, ready_obs

create_env_manager
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.envs.env_manager.base_env_manager.create_env_manager

get_env_manager_cls
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.envs.env_manager.base_env_manager.get_env_manager_cls

subprocess_env_manager
-------------------------

ShmBuffer
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_manager.subprocess_env_manager.ShmBuffer
    :members: fill, get

ShmBufferContainer
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_manager.subprocess_env_manager.ShmBufferContainer
    :members: fill, get

SyncSubprocessEnvManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_manager.subprocess_env_manager.SyncSubprocessEnvManager
    :members: reset, step, seed, close, enable_save_replay, launch, default_config, ready_obs


AsyncSubprocessEnvManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_manager.subprocess_env_manager.AsyncSubprocessEnvManager
    :members: reset, step, seed, close, enable_save_replay, launch, default_config, ready_obs
