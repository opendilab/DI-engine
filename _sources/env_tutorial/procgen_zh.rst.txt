Procgen
~~~~~~~

概述
=======

Procgen Benchmark是OpenAI发布的一组利用16种利用程序随机生成的环境（CoinRun，StarPilot，CaveFlyer，Dodgeball，FruitBot，Chaser
，Miner，Jumper，Leaper，Maze，BigFish，Heist，Climber，Plunder，Ninja和BossFight），procgen的全称是Procedural Generation，表示程序化生成。对于procgen环境，它可以生成同一难度但是采用不同地图的游戏，也可以生成采用同一地图但是不同难度的游戏，可以用来衡量模型学习通用技能的速度，从而判断算法对于环境的泛化能力。下图所示为其中的Coinrun游戏。


.. image:: ./images/coinrun.gif
   :align: center

以下三张图片分别表示了coinrun环境下level1到level3的不同输入：

.. image:: ./images/coinrun_level1.png
   :align: center
.. image:: ./images/coinrun_level2.png
   :align: center
.. image:: ./images/coinrun_level3.png
   :align: center


安装
====

安装方法
--------

可以通过pip一键安装或结合DI-engine安装，只需要安装gym和gym[procgen]两个库即可完成

.. code:: shell

   # Method1: Install Directly
   pip install gym
   pip install gym[procgen]
   # Method2: Install with DI-engine requirements
   cd DI-engine
   pip install ".[procgen_env]"

验证安装
--------

安装完成后，可以通过在Python命令行中运行如下命令验证安装成功：

.. code:: python

   import gym
   env = gym.make('procgen:procgen-maze-v0', start_level=0, num_levels=1)
   # num_levels=0 - The number of unique levels that can be generated. Set to 0 to use unlimited levels.
   # start_level=0 - The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels.
   obs = env.reset()
   print(obs.shape)  # (64, 64, 3)


.. _变换前的空间原始环境）:

变换前的空间（原始环境）
========================

.. _观察空间-1:

观察空间
--------

-  实际的游戏画面，RGB三通道图片，具体尺寸为\ ``(64, 3, 3)``\ ，数据类型为\ ``float32``\

.. _动作空间-1:

动作空间
--------

-  游戏操作按键空间，一般是大小为N的离散动作空间（N随具体子环境变化），数据类型为\ ``int``\ ，需要传入python数值（或是0维np数组，例如动作3为\ ``np.array(3)``\ ）

-  如在Coinrun环境中，N的大小为5，即动作在0-4中取值，具体的含义是：

   -  0：NOOP

   -  1：LEFT

   -  2：RIGHT

   -  3：UP

   -  4：DOWN


.. _奖励空间-1:

奖励空间
--------

-  游戏得分，根据具体游戏内容不同会有一定的差异，一般是一个\ ``float``\ 数值， 如在Coinrun环境中， 吃到硬币则奖励10.0分，除此以外没有其它奖励。

.. _其他-1:

其他
----

-  游戏结束即为当前环境episode结束，例如在coinrun中，智能体吃到硬币或者游戏时间超过了允许的最长游戏时间，则游戏结束。

关键事实
========

1. 2D
   RGB三通道图像输入，三维np数组，尺寸为\ ``(3, 64, 64)``\ ，数据类型为\ ``np.float32``\ ，取值为 \ ``[0, 255]``\

2. 离散动作空间

3. 奖励具有稀疏性，例如在coinrun中，只有吃到硬币才有得分。

4. 环境的泛化性，对于同一环境，有不同等级，它们的输入、奖励空间、动作空间是相同的，但游戏难度却不同。

变换后的空间（RL环境）
======================

.. _观察空间-2:

观察空间
--------

-  变换内容：将尺寸由\ ``（64 ,64 ,3）``\调整为\ ``(3, 64, 64)``\

-  变换结果：三维np数组，尺寸为\ ``(3, 84, 84)``\ ，数据类型为\ ``np.float32``\ ，取值为 \ ``[0, 255]``\

.. _动作空间-2:

动作空间
--------

-  基本无变换，依然是大小为N的离散动作空间，但一般为一维np数组，尺寸为\ ``(1, )``\ ，数据类型为\ ``np.int64``

.. _奖励空间-2:

奖励空间
--------

-  基本无变换

上述空间使用gym环境空间定义则可表示为：

.. code:: python

   import gym


   obs_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.float32)
   act_space = gym.spaces.Discrete(5)
   rew_space = gym.spaces.Box(low=0, high=10, shape=(1, ), dtype=np.float32)

.. _其他-2:

其他
----

-  环境\ ``step``\ 方法返回的\ ``info``\ 必须包含\ ``final_eval_reward``\ 键值对，表示整个episode的评测指标，在Procgen中为整个episode的奖励累加和

.. _其他-3:

其他
====

惰性初始化
----------

为了便于支持环境向量化等并行操作，环境实例一般实现惰性初始化，即\ ``__init__``\ 方法不初始化真正的原始环境实例，只是设置相关参数和配置值，在第一次调用\ ``reset``\ 方法时初始化具体的原始环境实例。

随机种子
--------

-  环境中有两部分随机种子需要设置，一是原始环境的随机种子，二是各种环境变换使用到的随机库的随机种子（例如\ ``random``\ ，\ ``np.random``\ ）

-  对于环境调用者，只需通过环境的\ ``seed``\ 方法进行设置这两个种子，无需关心具体实现细节

-  环境内部的具体实现：对于原始环境的种子，在调用环境的\ ``reset``\ 方法内部，具体的原始环境\ ``reset``\ 之前设置

-  环境内部的具体实现：对于随机库种子，则在环境的\ ``seed``\ 方法中直接设置该值

训练和测试环境的区别
--------------------

-  训练环境使用动态随机种子，即每个episode的随机种子都不同，都是由一个随机数发生器产生，但这个随机数发生器的种子是通过环境的\ ``seed``\ 方法固定的；测试环境使用静态随机种子，即每个episode的随机种子相同，通过\ ``seed``\ 方法指定。

存储录像
--------

在环境创建之后，重置之前，调用\ ``enable_save_replay``\ 方法，指定游戏录像保存的路径。环境会在每个episode结束之后自动保存本局的录像文件。（默认调用\ ``gym.wrapper.Monitor``\ 实现，依赖\ ``ffmpeg``\ ），下面所示的代码将运行一个环境episode，并将这个episode的结果保存在形如\ ``./video/xxx.mp4``\ 这样的文件中：

.. code:: python

   from easydict import EasyDict
   from dizoo.procgen.coinrun.envs import CoinRunEnv

   env = CoinRunEnv(EasyDict({'env_id': 'procgen:procgen-coinrun-v0'}))
   env.enable_save_replay(replay_path='./video')
   obs = env.reset()

   while True:
       action = env.random_action()
       timestep = env.step(action)
       if timestep.done:
           print('Episode is over, final eval reward is: {}'.format(timestep.info['final_eval_reward']))
           break

DI-zoo可运行代码示例
====================

完整的训练配置文件在 `github
link <https://github.com/opendilab/DI-engine/tree/main/dizoo/procgen/coinrun/entry>`__
内，对于具体的配置文件，例如\ ``coinrun_dqn_config.py``\ ，使用如下的demo即可运行：

.. code:: python

   from easydict import EasyDict

   coinrun_dqn_default_config = dict(
       env=dict(
           collector_env_num=4,
           evaluator_env_num=4,
           n_evaluator_episode=4,
           stop_value=10,
       ),
       policy=dict(
           cuda=False,
           model=dict(
               obs_shape=[3, 64, 64],
               action_shape=5,
               encoder_hidden_size_list=[128, 128, 512],
               dueling=False,
           ),
           discount_factor=0.99,
           learn=dict(
               update_per_collect=20,
               batch_size=32,
               learning_rate=0.0005,
               target_update_freq=500,
           ),
           collect=dict(n_sample=100, ),
           eval=dict(evaluator=dict(eval_freq=5000, )),
           other=dict(
               eps=dict(
                   type='exp',
                   start=1.,
                   end=0.05,
                   decay=250000,
               ),
               replay_buffer=dict(replay_buffer_size=100000, ),
           ),
       ),
   )
   coinrun_dqn_default_config = EasyDict(coinrun_dqn_default_config)
   main_config = coinrun_dqn_default_config

   coinrun_dqn_create_config = dict(
       env=dict(
           type='coinrun',
           import_names=['dizoo.procgen.coinrun.envs.coinrun_env'],
       ),
       env_manager=dict(type='subprocess', ),
       policy=dict(type='dqn'),
   )
   coinrun_dqn_create_config = EasyDict(coinrun_dqn_create_config)
   create_config = coinrun_dqn_create_config

   if __name__ == '__main__':
       from ding.entry import serial_pipeline
       serial_pipeline((main_config, create_config), seed=0)

基准算法性能
============

-  Coinrun（平均奖励等于10视为较好的Agent）

   - Coinrun + DQN
   .. image:: images/coinrun_dqn.svg
     :align: center

-  Maze（平均奖励等于10视为较好的Agent）

   - Maze + DQN
   .. image:: images/maze_dqn.svg
     :align: center
