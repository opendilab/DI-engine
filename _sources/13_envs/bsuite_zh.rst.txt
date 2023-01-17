
Bsuite
~~~~~~~

概述
============

``bsuite``  是一个精心设计的环境集合，研究强化学习智能体的在不同方面的核心能力。其主要有两个目标：

    1. 收集明确的、信息量大的、可扩展的问题，试图抓住高效和通用学习算法设计中的关键问题。
    2. 通过他们在这些共享基准上的表现来研究智能体行为。

.. figure:: ./images/bsuite.png
   :align: center
   :scale: 70%

   图片选自: https://github.com/deepmind/bsuite

这里我们以 *Memory Length* 为示例环境。 它的目的是测试智能体能够记住一个比特的连续步骤的数量。底层环境是基于一个风格化的 `T-maze <https://en.wikipedia.org/wiki/T-maze>`__ 问题， 以一个长度 :math:`N \in \mathbb{N}` 为参数。 
每个 episode lasts N steps，观察空间 :math:`o_t=\left(c_t, t / N\right)` and 
action space :math:`\mathcal{A}=\{-1,+1\}`.

   - 在 episode 开始时，提供给智能体一个 +1 或 -1的上下文， 这意味着 :math:`c_1 \sim {Unif}(\mathcal{A})`。
   - 在所有未来的时间步骤中，上下文等于零，并倒计时直到 episode 结束，这意味着所有 :math:`t>2` 都有  :math:`c_t=0` 。
   - 在 episode 结束时，智能体必须选择与环境相对应的正确行动来获得奖励。对于所有 :math:`t<N` ， 奖励 :math:`r_t=0` 并且 :math:`r_N={Sign}\left(a_N=c_1\right)`。


.. figure:: ./images/bsuite_memory_length.png
   :align: center
   :scale: 70%

   图片选自 `Behaviour Suite for Reinforcement Learning <https://arxiv.org/abs/1908.03568>`__ 一文

安装
=============

安装方法
-----------------

你需要使用下列 ``pip`` 命令来安装 bsuite。

.. code:: shell

   # Method1: Install Directly
   pip install bsuite

验证安装
--------------------

一旦安装完毕，你可以通过在 Python 命令行上运行以下命令来验证安装是否成功。

.. code:: python

   import bsuite
   env = bsuite.load_from_id('memory_len/0') # this environment configuration is 'memory steps' long
   timestep = env.reset()
   print(timestep)

原始环境空间
===========================

观察空间
-------------------

-  智能体的观察是一个三维的向量。数据类型是 ``float32``。它们的具体含义如下：

  -  obs[0] 显示当前时间，范围是 [0, 1]。
  -  obs[1] 在最后一步将查询显示为0和num之间的整数位。在内存长度实验中它总是0，因为只有一个比特。(在内存大小实验中它是有用的)。
  -  obs[2] 在第一步显示了+1或-1的条件。在以后的所有时间步骤中，上下文都等于0，并且有一个倒计时，直到 episode 结束。

动作空间
---------------

-  动作空间是一个大小为 2 的离散空间，它是 {-1,1}。 数据类型为 ``int``。

奖励空间
-------------

-   奖励空间是一个大小为 3 的离散空间，是一个 ``float`` 值。

  -  如果不是最后一步 (t<N), 奖励为 0。
  -  如果是最后一步，智能体选择了正确的行动，那么奖励就是 1。
  -  如果是最后一步，智能体选择了一个错误的行动，那么奖励是 -1。

其他 
-------

-  环境一旦达到其最大步数N，就会终止。


关键事实
==========

1. 我们可以改变记忆的长度N，使其逐渐变得更具挑战性。

2. 离散行动空间。

3. 每个环境都被设计用来测试RL策略的一个特定的泛化性，包括：概括、探索、信用分配、缩放、噪音、记忆。


其他
=======

以”OpenAI Gym“ 格式使用 bsuite 
------------------------------------

我们的实现使用bsuite Gym包装器来使bsuite代码库在OpenAI Gym接口下运行。因此，需要安装 ``gym`` 来使bsuite正常工作。

.. code:: python

   import bsuite
   from bsuite.utils import gym_wrapper
   env = bsuite.load_and_record_to_csv('memory_len/0', results_dir='/path/to/results')
   gym_env = gym_wrapper.GymFromDMEnv(env)

配置
-----------------------

配置的设计是为了提高环境的难度水平。例如，在一个五臂老虎机的环境中，配置被用来调节噪音水平以扰乱奖励。给定一个特定的环境，所有可能的配置都可以通过下面的代码片断进行可视化。

.. code:: python

   from bsuite import sweep  # this module contains information about all the environments
   for bsuite_id in sweep.BANDIT_NOISE:
   env = bsuite.load_from_id(bsuite_id)
   print('bsuite_id={}, settings={}, num_episodes={}' .format(bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes))

.. image:: ./images/bsuite_config.png
   :align: center

使用DI-engine，你可以简单地用你想要的配置的名字来创建一个bsuite环境。

.. code:: python

   from easydict import EasyDict
   from dizoo.bsuite.envs import BSuiteEnv
   cfg = {'env': 'memory_len/15'}
   cfg = EasyDict(cfg)
   memory_len_env = BSuiteEnv(cfg)


DI-zoo 可运行代码示例
=======================
完整的训练配置可以在 `github
链接 <https://github.com/opendilab/DI-engine/tree/main/dizoo/bsuite/config/serial>`__ 中找到。
在下面的部门，我们展示了一个配置文件的例子，``memory_len_0_dqn_config.py``\ ，你可以用下面的代码来运行这个演示：

.. code:: python

    from easydict import EasyDict

    memory_len_0_dqn_config = dict(
        exp_name='memory_len_0_dqn',
        env=dict(
            collector_env_num=8,
            evaluator_env_num=1,
            n_evaluator_episode=10,
            env_id='memory_len/0',
            stop_value=1.,
        ),
        policy=dict(
            load_path='',
            cuda=True,
            model=dict(
                obs_shape=3,
                action_shape=2,
                encoder_hidden_size_list=[128, 128, 64],
                dueling=True,
            ),
            nstep=1,
            discount_factor=0.97,
            learn=dict(
                batch_size=64,
                learning_rate=0.001,
            ),
            collect=dict(n_sample=8),
            eval=dict(evaluator=dict(eval_freq=20, )),
            other=dict(
                eps=dict(
                    type='exp',
                    start=0.95,
                    end=0.1,
                    decay=10000,
                ),
                replay_buffer=dict(replay_buffer_size=20000, ),
            ),
        ),
    )
    memory_len_0_dqn_config = EasyDict(memory_len_0_dqn_config)
    main_config = memory_len_0_dqn_config
    memory_len_0_dqn_create_config = dict(
        env=dict(
            type='bsuite',
            import_names=['dizoo.bsuite.envs.bsuite_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(type='dqn'),
    )
    memory_len_0_dqn_create_config = EasyDict(memory_len_0_dqn_create_config)
    create_config = memory_len_0_dqn_create_config

    if __name__ == '__main__':
        from ding.entry import serial_pipeline
        serial_pipeline((main_config, create_config), seed=0)


基准算法性能
===============================

   - memory_len/15 + R2D2

   .. figure:: ./images/bsuite_momery_len_15_r2d2.png
      :align: center 
      :scale: 70%
