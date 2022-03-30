
envs.env
========================

base_env
-----------------

Please Reference ding/ding/envs/env/base_env.py for usage

BaseEnv
~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.envs.env.base_env.BaseEnv
    :members: __init__, reset, step, close, enable_save_replay, random_action, create_collector_env_cfg, create_evaluator_env_cfg

get_vec_env_setting
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ding.envs.env.base_env.get_vec_env_setting

get_env_cls
~~~~~~~~~~~
.. autofunction:: ding.envs.env.base_env.get_env_cls
