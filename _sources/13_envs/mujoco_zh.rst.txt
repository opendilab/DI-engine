Mujoco
~~~~~~~

概述
=======

Mujoco 是旨在促进机器人、生物力学、图形和动画等需要快速准确模拟领域研究和开发的物理引擎，常来作为连续空间强化学习算法的基准测试环境。它是一系列环境的集合（共有 20 个子环境），常用的子环境有 Ant, Half Cheetah, Hopper, Huanmoid, Walker2D等等，下图所示为其中 Hopper 游戏。

.. image:: ./images/mujoco.gif
   :align: center
   :scale: 80%


安装
====

安装方法
--------

安装 gym, mujoco 与 mujoco-py 即可，可以通过 pip 一键安装或结合 DI-engine 安装

注：

1. mujoco 最新版目前已经开源免费，不再需要激活许可。你可以使用 Deepmind 最新的 mujoco 库，或使用 OpenAI 的 mujoco-py 。

2. 如果安装 ``mujoco-py>=2.1.0`` , 可以通过如下方法:

.. code:: shell
    
    # Installation for Linux
    # Download the MuJoCo version 2.1 binaries for Linux.
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
    # Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210.
    tar xvf mujoco210-linux-x86_64.tar.gz && mkdir -p ~/.mujoco && mv mujoco210 ~/.mujoco/mujoco210
    # Add path
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro210/bin:~/.mujoco/mujoco210/bin" >> ~/.bashrc
    source ~/.bashrc
    # Install and use mujoco-py
    pip install gym
    pip install -U 'mujoco-py<2.2,>=2.1'

    # Installation for macOS
    # Download the MuJoCo version 2.1 binaries for OSX.
    wget https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz
    # Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210.
    tar xvf mujoco210-macos-x86_64.tar.gz && mkdir -p ~/.mujoco && mv mujoco210 ~/.mujoco/mujoco210
    # Add path
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro210/bin:~/.mujoco/mujoco210/bin" >> ~/.bashrc
    source ~/.bashrc
    # Install and use mujoco-py
    pip install gym
    pip install -U 'mujoco-py<2.2,>=2.1'
    
3. 如果安装 ``mujoco-py<2.1.0`` , 可以通过如下方法:

.. code:: shell

    # Installation for Linux
    # Download the MuJoCo version 2.0 binaries for Linux.
    wget https://www.roboti.us/download/mujoco200_linux.zip
    # Extract the downloaded mujoco200 directory into ~/.mujoco/mujoco200.
    unzip mujoco200_linux.zip && mkdir -p ~/.mujoco && mv mujoco200_linux ~/.mujoco/mujoco200
    # Download unlocked activation key.
    wget https://www.roboti.us/file/mjkey.txt -O  ~/.mujoco/mjkey.txt 
    # Install and use mujoco-py
    pip install gym
    pip install -U 'mujoco-py<2.1'

    # Installation for macOS
    # Download the MuJoCo version 2.0 binaries for OSX.
    wget https://www.roboti.us/download/mujoco200_macos.zip
    # Extract the downloaded mujoco200 directory into ~/.mujoco/mujoco200.
    tar xvf mujoco200-macos-x86_64.tar.gz && mkdir -p ~/.mujoco && mv mujoco200_macos ~/.mujoco/mujoco200
    # Download unlocked activation key.
    wget https://www.roboti.us/file/mjkey.txt -O  ~/.mujoco/mjkey.txt 
    # Install and use mujoco-py
    pip install gym
    pip install -U 'mujoco-py<2.1'

4. 如果安装 ``mujoco>=2.2.0`` , 可以通过如下方法:

.. code:: shell

    # Install the MuJoCo version >=2.2.0
    pip install mujoco
    pip install gym
    

验证安装
--------

安装完成后，可以通过在 Python 命令行中运行如下命令验证安装成功：

.. code:: python

    import gym
    env = gym.make('Hopper-v3')
    obs = env.reset()
    print(obs.shape)  # (11, )

镜像
----

DI-engine 的镜像配备了框架本身和 Mujoco 环境，可通过\ ``docker pull opendilab/ding:nightly-mujoco``\ 获取，或访问\ `docker
hub <https://hub.docker.com/r/opendilab/ding>`_  获取更多镜像


变换前的空间（原始环境）
========================


观察空间
--------

-  物理信息组成的向量(3D position, orientation, and joint angles etc. )，具体尺寸为\ ``(N, )``\ ，其中\ ``N``\ 根据环境决定，数据类型为\ ``float64``


动作空间
--------

-  物理信息组成的向量(torque etc.)，一般是大小为N的连续动作空间（N随具体子环境变化），数据类型为\ ``np.float32``\ ，需要传入np数组（例如动作为\ ``array([-0.9266078 , -0.4958926 ,  0.46242517], dtype=np.float32)``\ ）

-  如在 Hopper 环境中，N 的大小为 3，动作在\ ``[-1, 1]``\ 中取值


奖励空间
--------

-  游戏得分，根据具体游戏内容不同会有非常大的差异，一般是一个\ ``float``\ 数值，具体的数值可以参考最下方的基准算法性能部分。


其他
----

-  游戏结束即为当前环境 episode 结束

关键事实
========

1. Vector 物理信息输入，由实际经验可知，在做 norm 时不宜减去均值。

2. 连续动作空间

3. 稠密奖励

4. 奖励取值尺度变化较大


变换后的空间（RL 环境）
=======================



观察空间
--------

-  基本无变换


动作空间
--------

-  基本无变换，依然是大小为N的连续动作空间，取值范围\ ``[-1, 1]``\，尺寸为\ ``(N, )``\ ，数据类型为\ ``np.float32``


奖励空间
--------

-  基本无变换

上述空间使用gym环境空间定义则可表示为：

.. code:: python

   import gym


   obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11, ), dtype=np.float64)
   act_space = gym.spaces.Box(low=-1, high=1, shape=(3, ), dtype=np.float32)
   rew_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)


其他
----

-  环境\ ``step``\ 方法返回的\ ``info``\ 必须包含\ ``eval_episode_return``\ 键值对，表示整个 episode 的评测指标，在 Mujoco 中为整个 episode 的奖励累加和


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

-  训练环境使用动态随机种子，即每个 episode 的随机种子都不同，都是由一个随机数发生器产生，但这个随机数发生器的种子是通过环境的\ ``seed``\ 方法固定的；测试环境使用静态随机种子，即每个 episode 的随机种子相同，通过\ ``seed``\ 方法指定。


存储录像
--------

在环境创建之后，重置之前，调用\ ``enable_save_replay``\ 方法，指定游戏录像保存的路径。环境会在每个 episode 结束之后自动保存本局的录像文件。（默认调用\ ``gym.wrappers.RecordVideo``\ 实现 ），下面所示的代码将运行一个环境 episode，并将这个 episode 的结果保存在\ ``./video/``\ 中：

.. code:: python

   from easydict import EasyDict
   from dizoo.mujoco.envs import MujocoEnv

   env = MujocoEnv(EasyDict({'env_id': 'Hoopper-v3' }))
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
link <https://github.com/opendilab/DI-engine/tree/main/dizoo/mujoco/config>`__
内，对于具体的配置文件，例如\ ``hopper_sac_default_config.py``\ ，使用如下的 demo 即可运行：

.. code:: python

   from easydict import EasyDict

    hopper_sac_default_config = dict(
        env=dict(
            env_id='Hopper-v3',
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
            type='mujoco',
            import_names=['dizoo.mujoco.envs.mujoco_env'],
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

注：对于某些特殊的算法，比如 PPO，需要使用专门的入口函数，示例可以参考
`链接 <https://github.com/opendilab/DI-engine/blob/main/dizoo/mujoco/entry/mujoco_ppo_main.py>`__
也可以使用 ``serial_pipeline_onpolicy`` 一键进入。

基准算法性能
============

-  Hopper-v3

   - Hopper-v3 + SAC

   .. image:: images/mujoco.png
     :align: center

