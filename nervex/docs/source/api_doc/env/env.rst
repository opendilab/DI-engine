
envs.env
========================

base_env
-----------------

Please Reference nerveX/nervex/envs/env/base_env.py for usage

BaseEnv
~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.envs.env.base_env.BaseEnv
    :members: __init__, reset, step, close, enable_save_replay, info, create_collector_env_cfg, create_evaluator_env_cfg

get_vec_env_setting
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nervex.envs.env.base_env.get_vec_env_setting

get_env_cls
~~~~~~~~~~~
.. autofunction:: nervex.envs.env.base_env.get_env_cls
