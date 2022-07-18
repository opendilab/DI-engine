Gym-Soccer (HFO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

概述
=======
HFO (Half-field Offense, 半场进攻) 是机器人世界杯2D足球比赛中的一个子任务。DI-engine 中的 Gym-Soccer 环境是HFO的简化版本，用来验证应用于混合动作空间的RL算法的性能。
如下图所示, 白色小圆点即足球，黄黑色圆圈是进攻球员，紫黑色圆圈是守门员。进攻球员的目标是在守门员的防守下完成射门。

.. image:: ./images/hfo.gif
   :align: center

安装
====

安装方法
--------

第一步: 由于HFO的源码是C++实现的，所以我们需要先安装含有 python 接口的 hfo_py。

.. code:: shell

    # To be done by niuyazhe

第二步: 安装 gym_soccer_env

.. code:: shell

    pip install git+https://github.com/LikeJulia/gym-soccer@dev-install-packages#egg=gym-soccer

注意，在安装过程中可能会遇到由于环境依赖问题引发的报错，根据报错提示安装对应的库即可，一般需要 CMake, Boost, Qt4, zlib, flex等等。具体要求请看 `官方手册 <https://github.com/LARG/HFO/blob/master/doc/manual.pdf>`__ 


验证安装
--------

方法一: 运行如下命令, 如果能正常显示版本信息，即顺利安装完成。

.. code:: shell 

    pip show gym-soccer


方法二: 运行如下Python程序，如果没有报错则证明安装成功。

.. code:: shell 

    import gym
    import gym_soccer
    env = gym.make('Soccer-v0')
    obs = env.reset()
    print(obs)  

镜像
----

DI-engine准备好了配备有框架本身和soccer环境的镜像，可通过\ ``to do by niuyazhe``\ 获取，或访问\ `docker
hub <https://hub.docker.com/repository/docker/opendilab/ding>`__\ 获取更多镜像

环境介绍
=========

动作空间
----------

Gym-Soccer 的动作空间属于离散连续混合动作空间，有3个离散动作，每个离散动作有n个连续参数（n>=0)。

-  \ ``TURN (degree)`` \: 表示让agent朝degree方向转身。 \ ``degree`` \的取值范围是 \ ``[-180，180]`` \。当 \ ``degree`` \= 0时，表示正前方向；当 \ ``degree`` \= 90时，表示正右方向。
  
-  \ ``DASH (power, degree)`` \: 表示让agent以 \ ``power`` \ 大小的力气向 \ ``degree`` \方向移动。 \ ``degree`` \的取值范围是 \ ``[-180，180]`` \。\ ``power`` \的取值范围是\ ``[0，100]`` \。注意：DASH并不会TURN智能体。
  
-  \ ``KICK (power, degree)`` \: 表示让agent以 \ ``power`` \ 大小的力气向 \ ``degree`` \方向击球。当agent手里没球时，动作不生效。


使用gym环境空间定义则可表示为：

.. code:: python

    action_space = spaces.Tuple((spaces.Discrete(3),
                                    spaces.Box(low=0, high=100, shape=(1,)),
                                    spaces.Box(low=-180, high=180, shape=(1,)),
                                    spaces.Box(low=-180, high=180, shape=(1,)),
                                    spaces.Box(low=0, high=100, shape=(1,)),
                                    spaces.Box(low=-180, high=180, shape=(1,))))

状态空间
----------

Gym-Soccer 的状态空间描述了当前游戏的状态，分为\ ``High Level Feature Set`` \和 \ ``Low Level Feature Set`` \，包含agent当前的坐标，球的坐标，agent的朝向等等。数值型的feature被统一scale到\ ``[-1,1]`` \的范围。具体请查阅官方手册中的 State Spaces 章节。

内置环境
-----------

-  \ ``"Soccer-v0"`` \: 最简单的设定, 奖励稀疏。该环境含有1个进攻球员，0个防守球员。进球得1分，否则得0分。
  
-  \ ``"SoccerEmptyGoal-v0"`` \: 奖励相对更稠密。进攻球员向足球靠近、将足球向目标方向踢、进球都会得到奖励.
  
-  \ ``"SoccerAgainstKeeper-v0"`` \: 奖励设定与 \ ``"SoccerEmptyGoal-v0"`` \ 相同。增加1个守门员（由规则控制的bot）。进攻球员需要学习如何与守门员周旋并进球得分。

用户自定义环境(TBD)
-------------------

其他
====

存储录像
--------

存储录像依赖 Qt4 库，需提前安装好。存储录像的具体方法请查阅Gym-Soccer环境的 `README <https://github.com/opendilab/DI-engine/tree/main/dizoo/gym_soccer/envs>`__ 文档。

DI-zoo 可运行代码示例
=====================

（TBD）待添加PDQN算法。

参考资料
====================
- HFO `源码 <https://github.com/LARG/HFO>`__
- Open-AI Gym-soccer `源码 <https://github.com/openai/gym-soccer>`__ 
  














