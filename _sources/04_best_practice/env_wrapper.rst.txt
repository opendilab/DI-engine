How to construct environments easily with Env Wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Why do we need Env Wrapper
------------------------------------------------------
Environment module is one of the most vital modules in reinforcement learning。 We train our agents in these environmnets and we allow them to explore and learn in these envirnments. In addition to a number of benchmark environments (such as atari or mujoco) reinforcement learning may also include a variety of custom environments.Overall，The essence of the Env Wrapper is to add certain generic additional features to our custom environment.
For instance：When we are training agents, we usually need to change the definition of the environment in order to achieve better training results, and these processing techniques are somewhat universal.For some environments, normalising the observed state is a very common pre-processing method. This processing makes training faster and more stable. If we extract this common part and put this preprocessing in an Env Wrapper, we avoid duplicate development. That is, if we want to change the way we normalise the observation state in the future, we can simply change it in this Env Wrapper.

Therefore, if the original environment is not perfectly adapted to our needs, we need to add functional modules to it to extend the functionality of the original environment and make it easy for the user to manipulate or adapt the environment's inputs and outputs. Env Wrapper is a simple solution for adding functionality.


Env Wrapper offered by DI-engine
==============================================

DI-engine provides a large number of pre-defined and generic Env Wrapper. The user can wrap it directly on top of the environment they need to use according to their needs.In the process of implementation，we refered  `OpenAI Baselines <https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py>`_ ，and follow the form of gym.Wrapper，which is `Gym.Wrapper <https://www.gymlibrary.dev/api/wrappers/>`_ ，In total, these include the following:

- NoopResetEnv：add a reset method to the environment. Reset the environment after some no-operations..

- MaxAndSkipEnv： Each ``skip`` frame（doing the same action）returns the maximum value of the two most recent frames.(for max pooling across time steps)。

- WarpFrame： Convert the size of the image frame to 84x84, according to  `Nature‘s paper <https://www.deepmind.com/publications/human-level-control-through-deep-reinforcement-learning>`_  and it's following work. (Note that this registrar also converts RGB images to GREY images)

- ScaledFloatFrame： Normalize status values to 0~1。

- ClipRewardEnv： Clip the reward to {+1, 0, -1} by the positive or negative of the reward.

- FrameStack： Set the current state to be the stacked nearest n_frames.

- ObsTransposeWrapper：The dimensions of the observed state are adjusted to place the channel dimension on the first dimension of the state. It is typically used in atari environments.

- RunningMeanStd：a wrapper for updating variances, means and counts.

- ObsNormEnv：The observed states are normalised according to the running mean and std.

- RewardNormEnv： The environmental rewards are normalised according to the standard deviation of the runs (running mean and std).

- RamWrapper： Converting the ram state of the original environment into an image-like state by extending the dimensionality of the observed state.

- EpisodicLifeEnv：Let the death of an agent in the environment mark the end of an episode (game over), and only reset the game when the real game is over. In general, this helps the algorithm to estimate the value.

- FireResetEnv：  Take ``fire`` action when the environment is reset. For more information please click `here <https://github.com/openai/baselines/issues/240>`_

.. tip::
    ``update_shape`` in Env Wrapper: This is a function that helps to identify the shape of observed states, actions and rewards after the env wrapper has been applied.

How to use Env Wrapper
------------------------------------
The next question is how to wrap the environment with Env Wrapper. One solution is to wrap the environment manually and explicitly：

.. code:: python

    env = gym.make(env_id)  # 'PongNoFrameskip-v4'
    env.NoopResetEnv(env, noop_max = 30)
    env = MaxAndSkipEnv(env, skip = 4)

If it is necessary to convert an environment in gym format to DI-engine environment format and use the corresponding multiple Env Wrapper, this can be done as shown below：

.. code:: python

    from ding.envs import DingEnvWrapper
    env = DingEnvWrapper(
        gym.make(env_id),
        cfg={
            'env_wrapper': [
                lambda env: MaxAndSkipWrapper(env, skip=4),
                lambda env: ScaledFloatFrameWrapper(env)
            ]
        }
    )


In particular, the Env Wrappers in the list are wrapped outside the environment in order. In the example above, the Env Wrapper wraps a layer of MaxAndSkipWrapper and then a layer of ScaledFloatFrameWrapper, while the Env Wrapper serves to add functionality but does not change the original functionality.


How to customise an Env Wrapper (Example)
-------------------------------------------
Taking ObsNormEnv wrapper as an example. In order to normalise the observed state，we only need to change two methods in the original environment class:step method and reset method, The rest of the method remains the same.
Note that in some cases, as the normalised bounds of the observed state change, info is needed to be modified accordingly.Please also note that the essence of the ObsNormEnv wrapper is to add additional functionality to the original environment, which is what the wrapper is all about. \

In addition, since the distribution of the sampled data is highly correlated with the strategy, i.e., the distribution of the samples can vary significantly from strategy to strategy, we use running means and standard deviations to normalize the observed states, rather than fixed means and standard deviations.

The structure of ObsNormEnv as below：

.. code:: python

   class ObsNormEnv(gym.ObservationWrapper):
        """
        Overview:
            Normalize observations according to running mean and std.
        Interface:
            ``__init__``, ``step``, ``reset``, ``observation``, ``new_shape``
        Properties:
            - env (:obj:`gym.Env`): The environment to wrap.
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


- ``__init__``: initialize ``data_count``, ``clip_range``, and ``running mean/std``.

- ``step``: use the given action to advance the environment，and update ``data_count`` and  ``running mean and std``.

- ``observation``: obtain the result observed. if ``data_count`` Returns the normalised version if the total number exceeds 30.

- ``reset``: Reset the state of the environment and reset ``data_count``, ``running mean/std``.

If the data pre-processing operations to be added is not in the Env Wrapper we provide, the user can also follow the example presented above and refer to the `Related Documentation <https://www.gymlibrary.dev/api/wrappers/>`_ on Wrappers in the gym to customise a wrapper to meet the requirements.

For more details about env wrapper，please see the code implementation in ``ding/envs/env_wrappers/env_wrappers.py``
