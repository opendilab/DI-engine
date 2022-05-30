
envs.env_wrappers 
========================

env.env_wrappers
-----------------

Please Reference ding/ding/envs/env_wrappers/env_wrappers.py for usage

Some descriptions referred to  `openai atari wrappers <https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py>`_

NoopResetWrapper
~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.envs.env_wrappers.NoopResetWrapper
    :members: __init__,reset



MaxAndSkipWrapper
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_wrappers.MaxAndSkipWrapper
    :members: __init__, step


WarpFrameWrapper
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_wrappers.WarpFrameWrapper
    :members: __init__, observation


ScaledFloatFrameWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_wrappers.ScaledFloatFrameWrapper
    :members: __init__, observation


ClipRewardWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_wrappers.ClipRewardWrapper
    :members: __init__, reward



FrameStackWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_wrappers.FrameStackWrapper
    :members: __init__, reset, step, _get_ob


ObsTransposeWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_wrappers.ObsTransposeWrapper
    :members: __init__, observation

RunningMeanStd
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_wrappers.RunningMeanStd
    :members: __init__, update, reset,mean, std


ObsNormWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_wrappers.ObsNormWrapper
    :members: __init__, step,observation, reset



RewardNormWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_wrappers.RewardNormWrapper
    :members: __init__, step,reward, reset
    

RamWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_wrappers.RamWrapper
    :members: __init__,  reset, step
    

EpisodicLifeWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_wrappers.EpisodicLifeWrapper
    :members: __init__,step,reset
    

FireResetWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.envs.env_wrappers.FireResetWrapper
    :members: __init__,reset
