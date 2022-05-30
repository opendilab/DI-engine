如何自定义一个 Env Wrapper
==============================================

环境模块是强化学习领域中最重要的模块之一。 我们在这些环境中训练我们的智能体，让他们在这些环境中探索和学习。强化学习中除了一些基准环境，例如 atari，mujoco 外，还可能包括各种各样自定义的环境。那么用户如何在 DI-engine 中自定义环境呢？

定义一个环境通常需要定义环境的输入和输出，并且充分考虑可行的状态空间（obervation spaces）和动作空间（action spaces）。 一个标准的环境一般包括状态空间（obervation spaces）, 动作空间（action spaces），奖励空间（reward spaces）等。 OpenAI 开源的 Gym 模块帮助我们定义了
学术界和工业界最常见的环境。 DI-engine 也遵循了 Gym.env 的定义，并且进一步增加了一些方便的功能，以提供更好的用户体验。

wrapper 是一个非常方便有效的工具。 Gym.wrapper 使用户能够方便地对环境类的输入和输出进行操作或适配。 


DI-engine 提供以下 env wrapper(以下很多都是从 `OpenAI Baselines <https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py>`_ 借鉴的):

- NoopResetEnv：为环境添加重置方法。在一些无操作（no-operations）后重置环境.

- MaxAndSkipEnv： 每`skip`帧（做同样的action）返回最近的两帧的最大值。(为了跨时间步的最大池化 max pooling)。

- WarpFrame： 将图像帧的大小转换为84x84, 如 `Nature 论文 <https://www.deepmind.com/publications/human-level-control-through-deep-reinforcement-learning>`_ 和后来的工作中所做的那样。(注意此注册器也将RGB图像转换为GREY图像)

- ScaledFloatFrame： 将状态值标准化为 0~1。

- ClipRewardEnv： 通过奖励的正负将奖励裁剪为 {+1, 0, -1}。

- FrameStack： 将堆叠好的n_frames个最近的状态帧设置为当前状态。

- ObsTransposeWrapper：对观测状态的各个维度进行调整，将通道维（channel）放置在状态的第一维上。通常用于 atari 环境。
- 
- RunningMeanStd： 用于更新方差、均值和计数的 wrapper。

- ObsNormEnv：根据运行均值和标准差（running mean and std）对观测状态进行归一化。

- RewardNormEnv： 根据运行的标准差（running mean and std）对环境奖励进行归一化。

- RamWrapper： 通过扩展观测状态的维度，将原始环境的ram状态转换成类似图像的状态

- EpisodicLifeEnv： 让环境中的智能体的死亡来标志一个episode结束（游戏结束）, 并且只有在真正的游戏结束时才会重置游戏。一般来讲， 这样有助于算法的价值估计。

- FireResetEnv：  在环境重置时采取`fire`行动。 相关的讨论查阅 `这里 <https://github.com/openai/baselines/issues/240>`_

- update_shape： 这是一个有助于在应用 env wrapper 后识别观测状态、动作和奖励的形状的函数。


我们为什么需要自定义一个 Env Wrapper
------------------------------------------------------

总的来说装饰器的本质是向原始环境添加某些通用的附加功能。
在训练智能体时，我们经常需要改变环境的定义以追求更好的训练效果， 而这些处理技巧也具备一定的普适性。例如，对于一些环境，归一化观测状态是非常常见的预处理方式。这样处理会让训练更快并且更加稳定。 如果我们将这个共同的部分提取出来，并将这个预处理放在环境装饰器（Env Wrapper）中，这样就避免了重复的开发。即如果我们以后想修改观测状态归一化的方式，我们只需要在这个环境装饰器进行更改即可。

由于采样得到数据的分布与策略高度相关，即不同的策略，样本的分布会有很大不同，所以我们使用运行均值和标准差来归一化观测状态，而不是固定均值和标准差 。

下面我们在DI-engine中展示 ObsNormEnv 的实现，以解释如何自定义 env wrapper。


Env Wrapper 结构举例
-----------------------------------------
以 ObsNormEnv wrapper 为例。为了归一化观测状态，我们只需要改变原始环境类中的两个方法：step方法和 reset方法，其余方法保持不变。
注意有些时候, 由于观测状态经过归一化后的界限改变了，info 也需要做相应的修改。 另请注意，ObsNormEnv wrapper 的本质是向原始环境添加附加功能，这正是包装器的含义. \

ObsNormEnv的结构如下：

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


- ``__init__``: 初始化 ``data_count``, ``clip_range``, 和 ``running mean/std``。

- ``step``: 使用给定的动作推进环境，并更新 ``data_count``和 ``running mean and std``。

- ``observation``: 获取观察结果. 如果 ``data_count`` 总数超过30，则返回归一化的版本。

- ``reset``: 重置环境状态并重置 ``data_count``, ``running mean/std``。



通常，可以按如下方式自定义 env wrapper：

自定义一个 Env Wrapper
------------------------------------
用户应按照以下步骤自定义 env wrapper：

1. 在 ``ding/envs/env_wrappers/env_wrappers.py`` 里像其他包装器一样定义您的 env 包装器类。


2. 用 `env_wrap` 函数包装你的环境。

.. code:: python

   wrapped_env = env_wrap(origin_env, *args, **kwargs)

举例: \
-------------
env = gym.make(evn_id) \

env.NoopResetEnv(env, noop_max = 30) \

env = MaxAndSkipEnv(env, skip = 4) \

更多关于 env wrapper 的信息，可以查看该链接
``ding/envs/env_wrappers/env_wrappers.py``