PyBullet
~~~~~~~

概述
=======

PyBullet是用于游戏，视觉效果，机器人和强化学习的物理模拟，常来作为连续空间强化学习算法的基准测试环境。它是一系列环境的集合（共有20个子环境），常用的子环境有locomotion(Hopper, Walker2D, Half Cheetah, Ant, Humanoid), Manipulator和Pendulum等等，下图所示为其中Hopper游戏。

.. image:: ./images/pybullet.gif
   :align: center

安装
====

安装方法
--------

可以通过pip一键安装或结合DI-engine安装，只需安装gym和pybullet-gym两个库即可

.. code:: shell
    
    # Install pybullet-gym
    pip install gym
    git clone https://github.com/benelot/pybullet-gym.git
    cd pybullet-gym
    pip install -e .

验证安装
--------

安装完成后，可以通过在Python命令行中运行如下命令验证安装成功：

.. code:: python
    import gym  # open ai gym
    import pybulletgym  # register PyBullet enviroments with open ai gym

    env = gym.make('HopperMuJoCoEnv-v0')
    # env.render() # call this before env.reset, if you want a window showing the environment
    obs = env.reset()  # should return a state vector if everything worked
    print(obs.shape)  # (11, )

镜像
----

DI-engine的镜像配备框架本身，可通过\ ``docker pull opendilab/ding:nightly-mujoco``\ 获取，或访问\ `docker
hub <https://hub.docker.com/repository/docker/opendilab/ding>`_  获取更多镜像

.. _变换前的空间原始环境）:

变换前的空间（原始环境）
========================

.. _观察空间-1:

观察空间
--------

-  物理信息组成的向量(3D position, orientation, and joint angles etc. )，具体尺寸为\ ``(N, )``\ ，其中\ ``N``\ 根据环境决定，数据类型为\ ``float64``

.. _动作空间-1:

动作空间
--------

-  物理信息组成的向量(torque etc.)，一般是大小为N的连续动作空间（N随具体子环境变化），数据类型为\ ``np.float32``\ ，需要传入np数组（例如动作为\ ``array([-0.9266078 , -0.4958926 ,  0.46242517], dtype=np.float32)``\ ）

-  如在Hopper环境中，N的大小为3，动作在\ ``[-1, 1]``\中取值

.. _奖励空间-1:

奖励空间
--------

-  游戏得分，根据具体游戏内容不同会有非常大的差异，一般是一个\ ``float``\ 数值，具体的数值可以参考最下方的基准算法性能部分。

.. _其他-1:

其他
----

-  游戏结束即为当前环境episode结束

关键事实
========

1. Vector物理信息输入，由实际经验可知，在做norm时不宜减去均值。

2. 连续动作空间

3. 稠密奖励

4. 奖励取值尺度变化较大

.. _变换后的空间rl环境）:

变换后的空间（RL环境）
======================


.. _观察空间-2:

观察空间
--------

-  基本无变换

.. _动作空间-2:

动作空间
--------

-  基本无变换，依然是大小为N的连续动作空间，取值范围\ ``[-1, 1]``\，尺寸为\ ``(N, )``\ ，数据类型为\ ``np.float32``

.. _奖励空间-2:

奖励空间
--------

-  基本无变换

上述空间使用gym环境空间定义则可表示为：

.. code:: python

   import gym


   obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11, ), dtype=np.float64)
   act_space = gym.spaces.Box(low=-1, high=1, shape=(3, ), dtype=np.float32)
   rew_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)

.. _其他-2:

其他
----

-  环境\ ``step``\ 方法返回的\ ``info``\ 必须包含\ ``final_eval_reward``\ 键值对，表示整个episode的评测指标，在Pybullet中为整个episode的奖励累加和

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
    from dizoo.pybullet.envs import PybulletEnv

    env = PybulletEnv(EasyDict({'env_id': 'Hoopper-v3' }))
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
link <https://github.com/opendilab/DI-engine/tree/main/dizoo/pybullet/config/>`__
内，对于具体的配置文件，例如\ ``hopper_sac_default_config.py``\ ，使用如下的demo即可运行：

.. code:: python

    from easydict import EasyDict

    hopper_sac_default_config = dict(
        env=dict(
            env_id='HopperMuJoCoEnv-v0',
            norm_obs=dict(use_norm=False, ),
            norm_reward=dict(use_norm=False, ),
            collector_env_num=1,
            evaluator_env_num=8,
            use_act_scale=True,
            n_evaluator_episode=8,
            stop_value=6000,
        ),
        policy=dict(
            cuda=True,
            on_policy=False,
            random_collect_size=10000,
            model=dict(
                obs_shape=11,
                action_shape=3,
                twin_critic=True,
                actor_head_type='reparameterization',
                actor_head_hidden_size=256,
                critic_head_hidden_size=256,
            ),
            learn=dict(
                update_per_collect=1,
                batch_size=256,
                learning_rate_q=1e-3,
                learning_rate_policy=1e-3,
                learning_rate_alpha=3e-4,
                ignore_done=False,
                target_theta=0.005,
                discount_factor=0.99,
                alpha=0.2,
                reparameterization=True,
                auto_alpha=False,
            ),
            collect=dict(
                n_sample=1,
                unroll_len=1,
            ),
            command=dict(),
            eval=dict(),
            other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
        ),
    )

    hopper_sac_default_config = EasyDict(hopper_sac_default_config)
    main_config = hopper_sac_default_config

    hopper_sac_default_create_config = dict(
        env=dict(
            type='pybullet',
            import_names=['dizoo.pybullet.envs.pybullet_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(
            type='sac',
            import_names=['ding.policy.sac'],
        ),
        replay_buffer=dict(type='naive', ),
    )
    hopper_sac_default_create_config = EasyDict(hopper_sac_default_create_config)
    create_config = hopper_sac_default_create_config



   if __name__ == '__main__':
       from ding.entry import serial_pipeline
       serial_pipeline((main_config, create_config), seed=0)

注：对于某些特殊的算法，比如PPO，需要使用专门的入口函数，示例可以参考
`link <https://github.com/opendilab/DI-engine/blob/main/dizoo/pybullet/entry/pybullet_ppo_main.py>`__
也可以使用serial_pipeline_onpolicy一键进入

基准算法性能
============

-  Hopper-v3

   - Hopper-v3 + SAC
   .. image:: images/pybullet.png
     :align: center

