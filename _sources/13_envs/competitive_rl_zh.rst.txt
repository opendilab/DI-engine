Competitive RL
~~~~~~~~~~~~~~~

概述
=======

Competitive RL 是由香港中文大学（CUHK）开发的一个对抗性游戏环境，环境中允许两个玩家分别控制双方进行对抗。游戏中双方玩家均可以观测到完整的环境信息（例如相同的球局、相同的赛车跑道），并以战胜对方、取得游戏胜利为最终目标。链接为 `github repo <https://github.com/cuhkrlcourse/competitive-rl>`_

Competitive RL 目前提供两种游戏环境：

   - Competitive Pong (cPong): 是将 Atari 中的 Pong 修改为对抗式的环境，即允许对抗的双方均为玩家操控，且均都是可被训练的agent。
      
      .. image:: ./images/c_pong.gif
         :scale: 60%
   
   - Competitive Car-Racing (cCarRacing): 赛车环境允许两辆车在同一个赛道地图中进行竞速。
      
      .. image:: ./images/c_car_racing.gif
         :scale: 60%


在每个环境中，都有 single-agent 和 double-agent 两个版本。
   
   - single-agent 即只有一个 agent 可以被玩家控制，另一个则由内置 bot 控制，single-agent 版本的 pong 环境和 Atari pong 是一样的
   - double-agent 就是指两个 agent 都可以被玩家控制。

安装
======

安装方法
------------

可以通过 pip 一键安装。


.. code:: shell

   pip install git+https://github.com/cuhkrlcourse/competitive-rl.git


验证安装
--------

安装完成后，可以通过在 Python 命令行中运行如下命令验证安装成功：

.. code:: python

    import gym
    import competitive_rl

    competitive_rl.register_competitive_envs()

    pong_single_env = gym.make("cPong-v0")  # single-agent pong env
    pong_double_env = gym.make("cPongDouble-v0")  # double-agent pong env

    racing_single_env = gym.make("cCarRacing-v0")  # single-agent car_racing env
    racing_double_env = gym.make("cCarRacingDouble-v0")  # double-agent car_racing env

    pong_single_env.reset()
    pong_single_env.step(0)

    pong_double_env.reset()
    pong_double_env.step((0, 0))


以下说明均以 **cPong** 为例。

变换前的空间（原始环境）
========================

观察空间
--------

-  若为 single agent env，则为实际的游戏画面，RGB 三通道图片，具体尺寸为\ ``(210, 160, 3)``\ ，数据类型为\ ``float32``，取值范围为 ``[0.0,  255.0]``
-  若为 double agent env，则为一个元组\ ``Tuple(Box(210, 160, 3), Box(210, 160, 3))``\ ，并将右侧玩家的原始游戏画面沿纵轴对称，使其看起来是左侧玩家的视角。数据类型和取值范围不变。


动作空间
--------

-  若为 single agent env，则为游戏操作按键空间，即大小为 3 的离散动作空间\ ``Discrete(3)``\ ，数据类型为\ ``int``\ ，需要传入 python 数值（或是 0 维 np 数组，例如动作 3 为\ ``np.array(3)``\ ）。动作的具体的含义是：（与 Atari Pong 略有不同）
   -  0：NOOP
   -  1：UP
   -  2：DOWN
-  若为 double agent env，则为一个元组\ ``Tuple(Discrete(3), Discrete(3))``\ ，数据类型与动作含义不变。


奖励空间
--------

-  奖励代表游戏得分，是一个\ ``float``\ 数值，范围是\ ``(-inf, inf)``\ 。
-  虽然对战有双方，但是由于双方的奖励之和必定为 0 ，故只返回 **左侧玩家** 的奖励。

其他
----

-  游戏结束即为当前环境episode结束

关键事实
========

1. 若为 double agent env，则会对右侧玩家的观察进行翻转，使得看来两个玩家的视角相同，都是同一边的，因此只需要学习一种策略。
2. 类似 Atari Pong。一，由于单帧图像蕴含的信息不足（比如运动方向），需要堆叠多帧图像来解决。二，动作空间为离散。
3. 奖励空间取值实际应当在\ ``[-21, +21]``\ ，因为一局比赛只有 21 个球。


变换后的空间（RL环境）
======================

观察空间
--------

-  变换内容：将左右两侧玩家的观察堆叠合并，然后类似Atari Pong操作（灰度图，空间尺寸缩放，最大最小值归一化，堆叠相邻 4 个游戏帧）

-  变换结果：四维np数组，尺寸为\ ``(2， 4, 84, 84)``\ ，数据类型为\ ``np.float32``\ ，取值为 ``[0, 1]``


动作空间
--------

-  基本无变换，但允许传回的双方动作为堆叠的 ``np.ndarry`` 或 ``list``
-  任一方的动作依然是大小为3的离散动作空间，数据类型为\ ``int``


奖励空间
--------

-  无变换

训练环境与测试环境不同
-----------------------------

-  训练时允许同时控制左右两侧的两个玩家，所以需要启动 **double** agent env，上述内容都是针对此类环境的。
-  测试时在只允许控制一个玩家（一般为左侧玩家），另一边由内置AI作出动作，所以需要启动 **single** agent env，它的观察空间和动作空间都为 double agent env 的一半，和 Atari Pong 十分类似（除了动作空间）。


其他
----

-  如果一个 episode 结束，环境\ ``step``\ 方法返回的\ ``info``\ 必须包含\ ``eval_episode_return``\ 键值对，表示整个 episode 的评测指标，即整个 episode 的奖励累加和
-  和奖励空间相同，只需要传左侧玩家的\ ``eval_episode_return``\


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

-  训练采用 double agent 环境，测试采用 single agent 环境。
-  训练环境使用动态随机种子，即每个 episode 的随机种子都不同，都是由一个随机数发生器产生，但这个随机数发生器的种子是通过环境的\ ``seed``\ 方法固定的；测试环境使用静态随机种子，即每个 episode 的随机种子相同，通过\ ``seed``\ 方法指定。

存储录像
--------

在环境创建之后，重置之前，调用\ ``enable_save_replay``\ 方法，指定游戏录像保存的路径。环境会在每个 episode 结束之后自动保存本局的录像文件。（默认调用\ ``gym.wrappers.RecordVideo``\ 实现 ），下面所示的代码将运行一个环境 episode，并将这个 episode 的结果保存在形如\ ``./video/``\ 中：


DI-zoo 可运行代码示例
======================

完整的训练配置文件在 `github
link <https://github.com/opendilab/DI-engine/tree/main/dizoo/competitive_rl/config/cpong_dqn_config.py>`__ 
内，对于具体的配置文件，例如\ ``pong_dqn_config.py``\ ，使用如下的 demo 即可运行：

.. code:: python

   from easydict import EasyDict
   from dizoo.competitive_rl.config.cpong_dqn_config import cpong_dqn_config, cpong_dqn_config, cpong_dqn_system_config
   
   if __name__ == '__main__':
       from ding.entry import parallel_pipeline
       parallel_pipeline((cpong_dqn_config, cpong_dqn_config, cpong_dqn_system_config), seed=0)


基准算法性能
============

TBD
