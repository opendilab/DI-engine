How to Customize an Env Wrapper
==================================

Env module is one of the most important modules in the domain of reinforcement learnig.
In some common reinforcement learning tasks, for instance, atari, mujoco, we train our agents
to explore and learn in these environments. 

Usually, defining an environment needs to start from the input and the output of the environment, and fully
consider possible obervation spaces and action spaces. The module Gym open-sourced by OpenAI has helped us define 
most common environments for the use in both academia and industry. DI-engine also follows the definition of Gym.env, and 
further adds some convenient functions for a better use experience.  

Gym.env.wrapper, as a subclass of Gym.wrapper, enables users to facilitate the manipulation or adaptation of the input and output of the environment class.  
Wrapper is a very convient and effective tool. Env wrapper only wraps some of the commonly used gym wrappers.

DI-engine provides the following env wrappers (Many of which are borrowed from openai baselines):


- NoopResetEnv. Add reset method for the env. Reset the env after some no-operations.

- MaxAndSkipEnv. Return only every `skip`-th frame (frameskipping) using most  
  recent raw observations (for max pooling across time steps).

- WarpFrame. Warp frames to 84x84 as done in the Nature paper and later work.

- ScaledFloatFrame. Normalize observations to 0~1.

- ClipRewardEnv. Clips the reward to {+1, 0, -1} by its sign.

- FrameStack. Stack n_frames last frames.

- ObsTransposeWrapper. Wrapper to transpose env, usually used in atari environments

- RunningMeanStd. Wrapper to update new variable, new mean, and new count

- ObsNormEnv. Normalize observations according to running mean and std.

- RewardNormEnv. Normalize reward according to running std.

- RamWrapper. Wrapper ram env into image-like env

- EpisodicLifeEnv. Make end-of-life == end-of-episode, but only reset on true game over. It helps \
  the value estimation.

- FireResetEnv.  Take action on reset for environments that are fixed until firing.

- update_shape. This is a function that helps recognise the shape of observations, actiions
  and reward after applying env wrappers.


The reason we need to customize an env wrapper
-----------------------------------------------

We often need to change different environments based on our needs. For instance, normalizing observations is usually very common for \
different environments so that the training stage is stabler and faster. Therefore, we extract this common part and put this feature in env_wrapper \
so that repeated developments are avoided and we only need to make a change here instead of every piece of codes if we want to modify the ways of normalistion of observations in the future. \
The reason that we use running mean and std instead of fixed mean and std is due to the fact that the distribution of samples will be different if different policies are applied, i.e., the \
distribution of sampled data are highly correlated with a policy.\

We show an implementation of ObsNormEnv below in DI-engine to explain\
how to customize an env wrapper.\


To customize an env wrapper
-------------------------------
Take the ObsNormEnv wrapper as an example. In order to normalize observations, \
we merely need to change the two functions in the original environment -- the step function and the reset function while keeping the rest functions \
almost the same. Note that sometimes, info is also needed to be modified as the boundaries of observations have been changed. \
Note also that the nature of ObsNormEnv wrapper is to add additional features to the original environment and this is exactly what a wrapper means. \

The structure of ObsNormEnv are as follows:

.. code:: python

   class ObsNormEnv(gym.ObservationWrapper):
        """
        Overview:
        Normalize observations according to running mean and std.
        Interface:
            ``__init__``, ``step``, ``reset``, ``observation``, ``new_shape``
        Properties:
            - env (:obj:`gym.Env`): the environment to wrap.

            - ``data_count``, ``clip_range``, ``rms``
        """

        def __init__(self, env):
            ...

        def step(self, action):
            ...

        def observation(self, observation):
            ...

        def reset(self, **kwargs):
            ...


- ``__init__``: Initialize ``episode count``, ``clip_range``, and ``running mean/std``

- ``step``: Step the environment with the given action. Repeat action, sum reward,  
  and update ``count of episodes``, and also update the ``running mean and std`` property  
  once after integrating with the input ``action``.

- ``observation``: Get obeservations. Return the original one or the normalized one if the ``count of episodes`` exceeds 30 (default)

- ``reset``: Resets the state of the environment and reset ``count of episodes``, ``running mean/std``.\



In general, an env wrapper can be customized as follows:

To customize an general env wrapper
------------------------------------
Users should follow the following steps to customize a model wrapper:

1. Define your env wrapper class like other wrappers in
   ``ding/envs/env_wrappers/env_wrappers.py``;


2. Wrap your env with `env_wrap` function.

.. code:: python

   wrapped_env = env_wrap(origin_env, *args, **kwargs)

Examples: \
-------------
env = gym.make(evn_id) \

env.NoopResetEnv(env, noop_max = 30) \

env = MaxAndSkipEnv(env, skip = 4) \

More details of env wrappers can be found in
``ding/envs/env_wrappers/env_wrappers.py``