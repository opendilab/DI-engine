ding.envs
----------

Env
========
Please refer to ``ding/envs/env`` for more details.

BaseEnv
~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.envs.BaseEnv
    :members: __init__, reset, step, close, enable_save_replay, random_action, create_collector_env_cfg, create_evaluator_env_cfg

get_vec_env_setting
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ding.envs.get_vec_env_setting

get_env_cls
~~~~~~~~~~~
.. autofunction:: ding.envs.get_env_cls

DingEnvWrapper
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.DingEnvWrapper
    :members: __init__, reset, step, close, seed, random_action, enable_save_replay, clone, observation_space, action_space, reward_space, create_collector_env_cfg, create_evaluator_env_cfg

get_default_wrappers
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.envs.get_default_wrappers


Env Manager
============
Please refer to ``ding/envs/env_manager`` for more details.

create_env_manager
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.envs.create_env_manager

get_env_manager_cls
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.envs.get_env_manager_cls

BaseEnvManager
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.BaseEnvManager
    :members: __init__, reset, step, launch, ready_obs, seed, close, reward_shaping, closed, done, method_name_list, ready_obs_id, ready_imgs, enable_save_replay, enable_save_figure, default_config, env_num, env_ref, observation_space, action_space, reward_space

BaseEnvManagerV2
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.BaseEnvManagerV2
    :members: __init__, reset, step, launch, ready_obs, seed, close, reward_shaping, closed, done, method_name_list, ready_obs_id, ready_imgs, enable_save_replay, enable_save_figure, default_config, env_num, env_ref, observation_space, action_space, reward_space

SyncSubprocessEnvManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.SyncSubprocessEnvManager
    :members: __init__, reset, step, ready_obs, seed, close, enable_save_replay, launch, default_config, ready_env, ready_imgs, worker_fn, worker_fn_robust

SubprocessEnvManagerV2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.SubprocessEnvManagerV2
    :members: __init__, reset, step, ready_obs, seed, close, enable_save_replay, launch, default_config, ready_env, ready_imgs, worker_fn, worker_fn_robust

AsyncSubprocessEnvManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.AsyncSubprocessEnvManager
    :members: __init__, reset, step, ready_obs, seed, close, enable_save_replay, launch, default_config, ready_env, ready_imgs, worker_fn, worker_fn_robust

GymVectorEnvManager
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.GymVectorEnvManager
    :members: __init__, reset, step, ready_obs, seed, close


Env Wrapper
=============
Please refer to ``ding/envs/env_wrappers`` for more details.

create_env_wrapper
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.envs.create_env_wrapper

update_shape
~~~~~~~~~~~~~
.. autofunction:: ding.envs.update_shape

NoopResetWrapper
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.NoopResetWrapper
    :members: __init__, reset

MaxAndSkipWrapper
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.MaxAndSkipWrapper
    :members: __init__, step

FireResetWrapper
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.FireResetWrapper
    :members: __init__, reset 

EpisodicLifeWrapper
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.EpisodicLifeWrapper
    :members: __init__, step

ClipRewardWrapper
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.ClipRewardWrapper
    :members: __init__, reward

FrameStackWrapper
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.FrameStackWrapper
    :members: __init__, reset, step

ScaledFloatFrameWrapper
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.ScaledFloatFrameWrapper
    :members: __init__, observation

WarpFrameWrapper
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.WarpFrameWrapper
    :members: __init__, observation

ActionRepeatWrapper
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.ActionRepeatWrapper
    :members: __init__, step

DelayRewardWrapper
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.DelayRewardWrapper
    :members: __init__, step, reset

ObsTransposeWrapper
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.ObsTransposeWrapper
    :members: __init__, observation

ObsNormWrapper
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.ObsNormWrapper
    :members: __init__, observation, step, reset

StaticObsNormWrapper
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.StaticObsNormWrapper
    :members: __init__, observation

RewardNormWrapper
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.RewardNormWrapper
    :members: __init__, reward, step, reset


RamWrapper
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.RamWrapper
    :members: __init__, step, reset

EvalEpisodeReturnWrapper
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.EvalEpisodeReturnWrapper
    :members: __init__, step, reset

GymHybridDictActionWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.GymHybridDictActionWrapper
    :members: __init__, step

ObsPlusPrevActRewWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.ObsPlusPrevActRewWrapper
    :members: __init__, step, reset


TimeLimitWrapper
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.TimeLimitWrapper
    :members: __init__, step, reset


FlatObsWrapper
~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.FlatObsWrapper
    :members: __init__, observation, step, reset, observation 

GymToGymnasiumWrapper
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.GymToGymnasiumWrapper
    :members: __init__, seed, reset

AllinObsWrapper
~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.AllinObsWrapper
    :members: __init__, seed, reset, step
