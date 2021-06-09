
envs.env_wrappers 
========================

env.env_wrappers
-----------------

Please Reference nerveX/nervex/envs/env_wrappers/env_wrappers.py for usage

Some descriptions referred to  `openai atari wrappers <https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py>`_

NoopResetEnv
~~~~~~~~~~~~~~~~~~

.. autoclass:: nervex.envs.env_wrappers.NoopResetEnv
    :members: __init__,reset,new_shape



MaxAndSkipEnv
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_wrappers.MaxAndSkipEnv
    :members: __init__, step,new_shape


WarpFrame
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_wrappers.WarpFrame
    :members: __init__, observation,new_shape


ScaledFloatFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_wrappers.ScaledFloatFrame
    :members: __init__, observation,new_shape


ClipRewardEnv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_wrappers.ScaledFloatFrame
    :members: __init__, reward,new_shape



FrameStack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_wrappers.FrameStack
    :members: __init__, reset, step, _get_ob,new_shape


ObsTransposeWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_wrappers.ObsTransposeWrapper
    :members: __init__, observation,new_shape

RunningMeanStd
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_wrappers.RunningMeanStd
    :members: __init__, update, reset,mean, std,new_shape


ObsNormEnv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_wrappers.ObsNormEnv
    :members: __init__, step,observation, reset,new_shape



RewardNormEnv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_wrappers.RewardNormEnv
    :members: __init__, step,reward, reset,new_shape
    

RamWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_wrappers.RamWrapper
    :members: __init__,  reset, step,new_shape
    

EpisodicLifeEnv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_wrappers.EpisodicLifeEnv
    :members: __init__,step,reset,new_shape
    

FireResetEnv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.envs.env_wrappers.FireResetEnv
    :members: __init__,reset,new_shape


update_shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.envs.env_wrappers.update_shape