BipedalWalker
~~~~~~~~~~~~~~~

概述
=======

BipedalWalker是Gym中经典的一个双足四关节机器人环境，机器人需要在1600个时间步中得到300分方可通关这一环境。在这个环境中，机器人需要与环境不断交互，并最终习得跑步、跳跃、处理不同地形的运动方式等一系列技能。

.. image:: ./images/bipedal_walker.gif
   :align: center

安装
====

安装方法
--------

安装 gym 和 box2d 两个库即可，可以通过 pip 一键安装或结合 DI-engine 安装

.. code:: shell

   # Method1: Install Directly
   pip install gym
   pip install box2d
   # Method2: Install with DI-engine requirements
   cd DI-engine
   pip install ".[common_env]"

验证安装
--------

安装完成后，可以通过在 Python 命令行中运行如下命令验证安装成功：

.. code:: python

   import gym
   env = gym.make('BipedalWalker-v3')
   obs = env.reset()
   print(obs.shape)  # (24,)

镜像
----

DI-engine 的镜像包含自有框架和 Atari 环境，可通过\ ``docker pull opendilab/ding:nightly``\ 获取. 如何获取更多镜像? 访问\ `docker
hub <https://hub.docker.com/r/opendilab/ding>`__\


变换前的空间（原始环境）
========================


观察空间
--------

-  机器人的状态是由船体角速度（hull angle speed）、角速度、水平速度、垂直速度、关节位置和关节角速度、腿与地面的接触标记以及 10 次激光雷达测距仪的测量值组成的24维向量。需要注意的是 **该状态向量中不包含机器人的坐标**。


动作空间
--------

-  环境动作空间为 4 维的连续向量，每个维度的值在 [-1,1] 之间。

-  这四维的连续向量分别控制机器人四个腿关节的扭矩。机器人共有 2 条腿，每条腿有两个关节(腰关节和膝关节), 一共 4 个关节需要控制。

奖励空间
--------

-  机器人驱动关节转动将得到少量的负奖励，前进则获得少量的正奖励，成功移动到最远端累计可以得到超过 300 分的奖励。如果机器人途中摔倒，会得到 -100 的奖励，且游戏结束。 奖励是一个\ float\ 数值，范围是 [-400, 300]。


其他
======


随机种子
--------

-  环境中有两部分随机种子需要设置，一是原始环境的随机种子，二是各种环境变换使用到的随机库的随机种子（例如\ ``random``\ ，\ ``np.random``\ ）

-  对于环境调用者，只需通过环境的\ ``seed``\ 方法进行设置这两个种子，无需关心具体实现细节

-  环境内部的具体实现：对于原始环境的种子，在调用环境的\ ``reset``\ 方法内部，具体的原始环境\ ``reset``\ 之前设置

-  环境内部的具体实现：对于随机库种子，则在环境的\ ``seed``\ 方法中直接设置该值


存储录像
--------

在环境创建之后，重置之前，调用\ ``enable_save_replay``\ 方法，指定游戏录像保存的路径。环境会在每个 episode 结束之后自动保存本局的录像文件。（默认调用\ ``gym.wrappers.RecordVideo``\ 实现 ），下面所示的代码将运行一个环境 episode，并将这个episode 的结果保存在\ ``./video/``\ 中：

.. code:: python

    from easydict import EasyDict
    from dizoo.box2d.bipedalwalker.envs import BipedalWalkerEnv
    import numpy as np

    env = BipedalWalkerEnv(EasyDict({'act_scale': True, 'rew_clip': True, 'replay_path': './video'}))
    obs = env.reset()

    while True:
       action = np.random.rand(24)
       timestep = env.step(action)
       if timestep.done:
           print('Episode is over, eval episode return is: {}'.format(timestep.info['eval_episode_return']))
           break

DI-zoo 可运行代码示例
======================

完整的训练配置文件在 `github
link <https://github.com/opendilab/DI-engine/tree/main/dizoo/box2d/bipedalwalker/config>`__
内，对于具体的配置文件，例如 `bipedalwalker_td3_config.py <https://github.com/opendilab/DI-engine/blob/main/dizoo/box2d/bipedalwalker/config/bipedalwalker_td3_config.py>`__ ，使用如下命令即可运行：

.. code:: shell

    python3 ./DI-engine/dizoo/bipdalwalker/config/box2dbipedalwalker_td3_config.py
    
基准算法性能
============

-  平均奖励大于等于 300 视为表现较好的智能体

    - BipedalWalker + TD3

    .. image:: images/bipedalwalker_td3.png
     :align: center
