Acrobot
~~~~~~~~~~~~~~~~~~

概述
=======
Acrobot机器人系统是强化学习中的经典控制问题。该系统包括两个关节和两个连杆，其中两个连杆之间的关节可以被驱动。系统的初始状态是向下悬挂的。目标是在驱动关节上施加力矩，使连杆的自由端摆动到指定高度。如下图所示。

.. image:: ./images/acrobot.gif
   :align: center
   :scale: 80%

安装
====

安装方法
--------

Acrobot 环境内置在 gym 中，直接安装 gym 即可。其环境 id 是\ ``Acrobot-v1`` \。

.. code:: shell

    pip install gym
    
验证安装
--------

在 Python 命令行中运行如下命令验证安装成功。

.. code:: shell 

    import gym
    env = gym.make('Acrobot-v1')
    obs = env.reset()
    print(obs)
    assert env.observation_space.shape == (6,)
    assert env.action_space == gym.spaces.Discrete(3)

环境介绍
=========

动作空间
----------

Acrobot 的动作空间属于离散动作空间，有 3 个离散动作，分别是施加+1力矩, -1力矩和不施加力矩。

-  \ ``-1的力矩`` \: 0 表示向关节施加 -1 的力矩。

-  \ ``0的力矩`` \: 1 表示向关节施加 0 的力矩。

-  \ ``+1的力矩`` \: 2 表示向关节施加 +1 的力矩。

使用 gym 环境空间定义则可表示为：

.. code:: python
    
    action_space = gym.spaces.Discrete(3)

状态空间
----------

Acrobot 的状态空间有 6 个元素，分别是：


- \ ``Cosine of theta1`` \：第一个连杆和竖直方向角度的cos值，范围是 \ ``[-1, 1]`` \。
  
- \ ``Sine of theta1`` \：第一个连杆和竖直方向角度的sin值，范围是 \ ``[-1, 1]`` \。

- \ ``Cosine of theta2`` \：第二个连杆相对于第一个连杆的角度的cos值，范围是 \ ``[-1, 1]`` \。

- \ ``Sine of theta2`` \：第二个连杆相对于第一个连杆的角度的sin值，范围是 \ ``[-1, 1]`` \。

- \ ``Angular velocity of theta1`` \：第一个连杆相对于竖直方向的角速度，范围是 \ ``[-4 * pi, 4 * pi]`` \。

- \ ``Angular velocity of theta2`` \：第二个连杆相对于第一个连杆的角速度，范围是 \ ``[-9 * pi, 9 * pi]`` \。


``theta1`` 是第一个关节的角度，其中角度 ``0`` 表示第一个链接直接指向下方。

``theta2`` 是相对于第一个连杆的角度。 角度 ``0`` 对应于两个链接之间具有相同的角度。


奖励空间
-----------
目标是让自由端以尽可能少的步数达到指定的目标高度，因此所有未达到目标的步数都会产生 ``-1`` 的奖励。 达到目标高度会导致终止，奖励为 ``0`` 。


终止条件
------------
Acrobot 环境每个 episode 的终止条件是遇到以下任何一种情况：

- 自由端达到目标高度，构造形式是 \ :math:`-cos(\theta_1) - cos(\theta_1 + \theta_2) > 1.0`\ 。

- 达到 episode 的最大 step，默认为 ``500`` 。
  

DI-zoo 可运行代码示例
=====================


完整的训练配置文件在 `github
link <https://github.com/opendilab/DI-engine/tree/main/dizoo/classic_control/acrobot/config>`__
内，对于具体的配置文件，例如\ ``acrobot_dqn_config.py``\ ，使用如下的 demo 即可运行：

.. code:: python
    

    from easydict import EasyDict

    acrobot_dqn_config = dict(
        exp_name='acrobot_dqn_seed0',
        env=dict(
            collector_env_num=8,
            evaluator_env_num=8,
            n_evaluator_episode=8,
            stop_value=-60,
            env_id='Acrobot-v1',
            replay_path='acrobot_dqn_seed0/video',
        ),
        policy=dict(
            cuda=True,
            model=dict(
                obs_shape=6,
                action_shape=3,
                encoder_hidden_size_list=[256, 256],
                dueling=True,
            ),
            nstep=3,
            discount_factor=0.99,
            learn=dict(
                update_per_collect=10,
                batch_size=128,
                learning_rate=0.0001,
                target_update_freq=250,
            ),
            collect=dict(n_sample=96, ),
            eval=dict(evaluator=dict(eval_freq=2000, )),
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
    acrobot_dqn_config = EasyDict(acrobot_dqn_config)
    main_config = acrobot_dqn_config
    acrobot_dqn_create_config = dict(
        env=dict(type='acrobot', import_names=['dizoo.classic_control.acrobot.envs.acrobot_env']),
        env_manager=dict(type='subprocess'),
        policy=dict(type='dqn'),
        replay_buffer=dict(type='deque', import_names=['ding.data.buffer.deque_buffer_wrapper']),
    )
    acrobot_dqn_create_config = EasyDict(acrobot_dqn_create_config)
    create_config = acrobot_dqn_create_config

    if __name__ == "__main__":
        from ding.entry import serial_pipeline
        serial_pipeline((main_config, create_config), seed=0)


基准算法性能
=================
使用 DQN 算法的实验结果如下。横坐标是\ ``step`` \，纵坐标是\ ``reward_mean`` \。

.. image:: ./images/acrobot_dqn.png
   :align: center
   :scale: 80%


参考资料
=====================
- Acrobot `源码 <https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py>`__

