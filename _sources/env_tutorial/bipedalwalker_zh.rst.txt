BipedalWalker
~~~~~~~~~~~~~~~

概述
=======

在BipedalWalker 环境里，智能体需要输出4维的连续动作，控制2D的双足机器人在崎岖的地形上前进，在每一步应用电机扭矩会得到小的负的奖励，每前进一步会得到小的正的奖励，成功移动到最远端累计可以得到超过300分的奖励。如果机器人途中摔倒，会得到 -100的奖励，且游戏结束。 智能体的状态是24维连续向量，包括船体角速度(hull angle speed)、角速度、水平速度、垂直速度、关节位置和关节角速度、腿与地面的接触标记以及10次激光雷达测距仪的测量值。注意的是该状态向量中不包含机器人的坐标。

.. image:: ./images/bipedal_walker.gif
   :align: center

安装
====

安装方法
--------

安装gym和box2d两个库即可，可以通过pip一键安装或结合DI-engine安装

.. code:: shell

   # Method1: Install Directly
   pip install gym
   pip install box2d
   # Method2: Install with DI-engine requirements
   cd DI-engine
   pip install ".[common_env]"

验证安装
--------

安装完成后，可以通过在Python命令行中运行如下命令验证安装成功：

.. code:: python

   import gym
   env = gym.make('BipedalWalker-v3')
   obs = env.reset()
   print(obs.shape)  # (24,)

镜像
----

DI-engine的镜像包含自有框架和Atari环境，可通过\ ``docker pull opendilab/ding:nightly``\ 获取. 如何获取更多镜像? 访问\ `docker
hub <https://hub.docker.com/repository/docker/opendilab/ding>`__\


变换前的空间（原始环境）
========================


观察空间
--------

-  智能体的状态是24维连续向量，包括船体角速度(hull angle speed)、角速度、水平速度、垂直速度、关节位置和关节角速度、腿与地面的接触标记以及10次激光雷达测距仪的测量值。注意的是该状态向量中不包含机器人的坐标。


动作空间
--------

-  环境动作空间为4维的连续向量，每个维度的值在[-1,1]之间。

-  这四维的连续向量分别控制机器人四个腿关节的扭矩。机器人共有2条腿，每条腿有两个关节(腰关节和膝关节), 一共4个关节需要控制。

奖励空间
--------

-  机器人在每一步应用电机扭矩会得到小的负的奖励，每前进一步会得到小的正的奖励，成功移动到最远端累计可以得到超过300分的奖励。如果机器人途中摔倒，会得到 -100的奖励，且游戏结束。 奖励是一个\ float\ 数值，范围是[-400, 300]。

关键事实
========


其他
====


随机种子
--------

-  环境中有两部分随机种子需要设置，一是原始环境的随机种子，二是各种环境变换使用到的随机库的随机种子（例如\ ``random``\ ，\ ``np.random``\ ）

-  对于环境调用者，只需通过环境的\ ``seed``\ 方法进行设置这两个种子，无需关心具体实现细节

-  环境内部的具体实现：对于原始环境的种子，在调用环境的\ ``reset``\ 方法内部，具体的原始环境\ ``reset``\ 之前设置

-  环境内部的具体实现：对于随机库种子，则在环境的\ ``seed``\ 方法中直接设置该值


存储录像
--------

在环境创建之后，重置之前，调用\ ``enable_save_replay``\ 方法，指定游戏录像保存的路径。环境会在每个episode结束之后自动保存本局的录像文件。（默认调用\ ``gym.wrapper.Monitor``\ 实现，依赖\ ``ffmpeg``\ ），下面所示的代码将运行一个环境episode，并将这个episode的结果保存在形如\ ``./video/xxx.mp4``\ 这样的文件中：

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
           print('Episode is over, final eval reward is: {}'.format(timestep.info['final_eval_reward']))
           break

DI-zoo可运行代码示例
====================

完整的训练配置文件在 `github
link <https://github.com/opendilab/DI-engine/tree/main/dizoo/box2d/bipedalwalker/config>`__
内，对于具体的配置文件，例如\ ``bipedalwalker_td3_config.py``\ ，使用如下的demo即可运行：

.. code:: python

    bipedalwalker_td3_config = dict(
        env=dict(
            collector_env_num=1,
            evaluator_env_num=5,
            # (bool) Scale output action into legal range.
            act_scale=True,
            n_evaluator_episode=5,
            stop_value=300,
            rew_clip=True,
            replay_path=None,
        ),
        policy=dict(
            cuda=True,
            priority=False,
            model=dict(
                obs_shape=24,
                action_shape=4,
                twin_critic=True,
                actor_head_hidden_size=400,
                critic_head_hidden_size=400,
                actor_head_type='regression',
            ),
            learn=dict(
                update_per_collect=4,
                discount_factor=0.99,
                batch_size=128,
                learning_rate_actor=0.001,
                learning_rate_critic=0.001,
                target_theta=0.005,
                ignore_done=False,
                actor_update_freq=2,
                noise=True,
                noise_sigma=0.2,
                noise_range=dict(
                    min=-0.5,
                    max=0.5,
                ),
            ),
            collect=dict(
                n_sample=256,
                noise_sigma=0.1,
                collector=dict(collect_print_freq=1000, ),
            ),
            eval=dict(evaluator=dict(eval_freq=100, ), ),
            other=dict(replay_buffer=dict(replay_buffer_size=50000, ), ),
        ),
    )
    bipedalwalker_td3_config = EasyDict(bipedalwalker_td3_config)
    main_config = bipedalwalker_td3_config

    bipedalwalker_td3_create_config = dict(
        env=dict(
            type='bipedalwalker',
            import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(type='td3'),
    )
    bipedalwalker_td3_create_config = EasyDict(bipedalwalker_td3_create_config)
    create_config = bipedalwalker_td3_create_config

   if __name__ == '__main__':
       from ding.entry import serial_pipeline
       serial_pipeline((main_config, create_config), seed=0)


基准算法性能
============

-  平均奖励大于等于300视为较好的Agent

    - BipedalWalker + TD3

    .. image:: images/bipedalwalker_td3.png
     :align: center
