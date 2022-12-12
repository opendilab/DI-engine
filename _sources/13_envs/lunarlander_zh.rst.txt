LunarLander
~~~~~~~~~~~~

概述
=======

LunarLander，即月球登陆，任务目标是通过导航一个登陆器登录到指定的登陆地点。该环境同时具有离散动作空间和连续动作空间两个版本， 目前DI-engine 只支持离散动作空间版本， 后续会补充关于连续空间的版本及一些适配。以下主要介绍了离线动作空间版本的 lunarlander。

.. image:: ./images/lunarlander.gif
   :align: center

安装
====

安装方法
--------

安装 gym 和 Box2d 两个库即可, 用户可以选择通过 pip 一键安装

注：如果用户没有 root 权限，请在 install 的命令后面加上 ``--user``


.. code:: shell

   # Install Directly
   pip install gym
   pip install Box2D

验证安装
--------

安装完成后，可以通过在 Python 命令行中运行如下命令验证安装成功：

.. code:: python

   import gym
   env = gym.make('LunarLander-v2')
   obs = env.reset()
   print(obs.shape)  # (8,)
   env = gym.make('LunarLanderContinuous-v2')
   obs = env.reset()
   print(obs.shape)  # (8,)

镜像
----

DI-engine 的镜像配备有框架本身和 Lunarlander 环境，可通过\ ``docker pull opendilab/ding:nightly``\ 获取，或访问\ `docker hub <https://hub.docker.com/r/opendilab/ding>`__\ 获取更多镜像


变换前的空间（原始环境）
========================


观察空间
--------

-  观察空间为8纬的np数组，数据类型为\ ``float32``
-  s[0] 是横坐标
-  s[1] 是纵坐标
-  s[2] 是水平的速度
-  s[3] 是垂直的速度
-  s[4] 是与纵坐标的的弧度（向右为正，相左为负，180 度 = pi 弧度）
-  s[5] 是角速度
-  s[6] 如果一只脚着陆, 此值为1, 其余情况为 0
-  s[7] 如果第二只脚着陆, 此值为1, 其余情况为 0



动作空间
--------

-  对于 lunarlander 离散版本的游戏操作按键空间，一般是大小为 4 的离散动作空间，数据类型为\ ``int``\ 

-  在 lunarlander 离散版本中，动作在 0-3 中取值，具体的含义是：

   -  0：Do nothing

   -  1：Fire right engine

   -  2：Fire down engine

   -  3：Fire left engine


奖励空间
--------

-  一个\ ``int``\ 数值
-  从屏幕顶部移动到着陆点并且速度到零的奖励大约是 100...140 分。如果登陆器向着远离着陆台的方向行进，就会失去奖励。如果登陆器坠落或停止，episode 就会结束，获得额外的 -100 或 +100 分。每条腿的地面接触是 +10 的奖励。发射主引擎每帧为 -0.3 奖励。成功着陆到着陆点是200  分。在起落架外着陆是可能的。燃料是无限的。


其他
----

-  游戏结束即为当前环境 episode 结束, 如果登陆器坠毁或者到达了静止状态，则当前 episode 结束

关键事实
========

1. 离散和连续动作空间


变换后的空间（RL 环境）
=======================


观察空间
--------

-  无变化


动作空间
--------

-  依然是大小为 4 的离散动作空间，但数据类型由 ``int`` 转为 ``np.int64``, 尺寸为\ ``( )``\, 即 0-dim 的 array


奖励空间
--------

-  变换内容：数据结构变换

-  变换结果：变为 np 数组，尺寸为\ ``(1, )``\ ，数据类型为\ ``np.float64``\

上述空间使用gym环境空间定义则可表示为：

.. code:: python

   import gym
   obs_space = gym.spaces.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
   act_space = gym.spaces.Discrete(4)



其他
----

-  环境\ ``step``\ 方法返回的\ ``info``\ 必须包含\ ``eval_episode_return``\ 键值对，表示整个 episode 的评测指标，在lunarlander 中为整个 episode 的奖励累加和


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

-  环境内部的具体实现：对于随机库种子，则在环境的\ ``seed``\ 方法中直接设置该值; 对于原始环境的种子，在调用环境的\ ``reset``\ 方法内部，具体的原始环境\ ``reset``\ 之前设置为seed + np_seed, 其中seed为前述的随机库种子的值,
   np_seed = 100 * np.random.randint(1, 1000)。

训练和测试环境的区别
--------------------

-  训练环境使用动态随机种子，即每个 episode 的随机种子都不同，都是由一个随机数发生器产生，但这个随机数发生器的种子是通过环境的\ ``seed``\ 方法固定的；测试环境使用静态随机种子，即每个 episode 的随机种子相同，通过\ ``seed``\ 方法指定。


存储录像
--------

在环境创建之后，重置之前，调用\ ``enable_save_replay``\ 方法，指定游戏录像保存的路径。环境会在每个episode结束之后自动保存本局的录像文件。（默认调用\ ``gym.wrappers.RecordVideo``\ 实现 ），下面所示的代码将运行一个环境episode，并将这个episode的结果保存在\ ``./video/``\ 中：

.. code:: python

   from easydict import EasyDict
   from dizoo.box2d.lunarlander.envs import LunarLanderEnv

   env = LunarLanderEnv({})
   env.enable_save_replay(replay_path='./video')
   obs = env.reset()

   while True:
       action = env.random_action()
       timestep = env.step(action)
       if timestep.done:
           print('Episode is over, eval episode return is: {}'.format(timestep.info['eval_episode_return']))
           break

DI-zoo 可运行代码示例
======================

完整的训练配置文件在 `github
link <https://github.com/opendilab/DI-engine/blob/main/dizoo/box2d/lunarlander/config/>`__
内，对于具体的配置文件，例如\ ``lunarlander_dqn_config.py``\ ，使用如下的 demo 即可运行：

.. code:: python

    from easydict import EasyDict
    from ding.entry import serial_pipeline

    nstep = 3
    lunarlander_dqn_default_config = dict(
        env=dict(
            # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
            manager=dict(shared_memory=True, ),
            # Env number respectively for collector and evaluator.
            collector_env_num=8,
            evaluator_env_num=5,
            n_evaluator_episode=5,
            stop_value=200,
        ),
        policy=dict(
            # Whether to use cuda for network.
            cuda=False,
            model=dict(
                obs_shape=8,
                action_shape=4,
                encoder_hidden_size_list=[512, 64],
                # Whether to use dueling head.
                dueling=True,
            ),
            # Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # How many steps in td error.
            nstep=nstep,
            # learn_mode config
            learn=dict(
                update_per_collect=10,
                batch_size=64,
                learning_rate=0.001,
                # Frequency of target network update.
                target_update_freq=100,
            ),
            # collect_mode config
            collect=dict(
                # You can use either "n_sample" or "n_episode" in collector.collect.
                # Get "n_sample" samples per collect.
                n_sample=64,
                # Cut trajectories into pieces with length "unroll_len".
                unroll_len=1,
            ),
            # command_mode config
            other=dict(
                # Epsilon greedy with decay.
                eps=dict(
                    # Decay type. Support ['exp', 'linear'].
                    type='exp',
                    start=0.95,
                    end=0.1,
                    decay=50000,
                ),
                replay_buffer=dict(replay_buffer_size=100000, )
            ),
        ),
    )
    lunarlander_dqn_default_config = EasyDict(lunarlander_dqn_default_config)
    main_config = lunarlander_dqn_default_config

    lunarlander_dqn_create_config = dict(
        env=dict(
            type='lunarlander',
            import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(type='dqn'),
    )
    lunarlander_dqn_create_config = EasyDict(lunarlander_dqn_create_config)
    create_config = lunarlander_dqn_create_config

    if __name__ == "__main__":
        serial_pipeline([main_config, create_config], seed=0)


基准算法性能
==============

-  LunarLander（平均奖励大于等于200视为较好的 Agent）

   - Lunarlander + DQFD

   .. image:: images/lunarlander_dqfd.png
     :align: center


