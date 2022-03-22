Gym-Hybrid 
~~~~~~~~~~~~~~~~

概述
=======
在gym-hybrid任务中, agent的任务很简单：在边长为2的正方形框内加速（Accelerate）、转向（Turn）或刹车（Break），以停留在红色目标区域。目标区域是一个半径为0.1的圆。如下图所示。

.. image:: ./images/hybrid.gif
   :align: center

安装
====

安装方法
--------

.. code:: shell

    pip install git+https://github.com/thomashirtz/gym-hybrid@master#egg=gym-hybrid

验证安装
--------

方法一: 运行如下命令, 如果能正常显示版本信息，即顺利安装完成。

.. code:: shell 

    pip show gym-hybrid


方法二: 运行如下Python程序，如果没有报错则证明安装成功。

.. code:: python 

    import gym
    import gym_hybrid
    env = gym.make('Moving-v0')
    obs = env.reset()
    print(obs)  

环境介绍
=========

动作空间
----------

Gym-hybrid 的动作空间属于离散连续动作混合空间，有3个离散动作：Accelerate，Turn，Break，其中动作Accelerate，Turn需要给出对应的1维连续参数。

-  \ ``Accelerate (Acceleration value)`` \: 表示让agent以 \ ``acceleration value`` \的大小加速。 \ ``Acceleration value`` \的取值范围是\ ``[0,1]`` \。数值类型为\ ``float32``。
  
-  \ ``Turn (Rotation value)`` \: 表示让agent朝 \ ``rotation value`` \的方向转身。 \ ``Rotation value`` \的取值范围是\ ``[-1,1]`` \。数值类型为\ ``float32``。
  
-  \ ``Break ()`` \: 表示停止。

使用gym环境空间定义则可表示为：

.. code:: python
    
    from gym import spaces

    action_space = spaces.Tuple((spaces.Discrete(3),
                                    spaces.Box(low=0, high=1, shape=(1,)),
                                    spaces.Box(low=-1, high=1, shape=(1,))))

状态空间
----------

Gym-hybrid 的状态空间由一个有10个元素的list表示，描述了当前agent的状态，包含agent当前的坐标，速度，朝向角度的正余弦值，目标的坐标，agent距离目标的距离，与目标距离相关的bool值，当前相对步数。

.. code:: python

    state = [
                agent.x,
                agent.y,
                agent.speed,
                np.cos(agent.theta),
                np.sin(agent.theta),
                target.x,
                target.y,
                distance,
                0 if distance > target_radius else 1,
                current_step / max_step
            ]

奖励空间
-----------
每一步的奖励设置为agent上一个step执行动作后距离目标的长度减去当前step执行动作后距离目标的长度，即\ ``dist_t-1 - dist_t`` \。算法内置了一个\ ``penalty`` \来激励agent更快的
达到目标。当episode结束时，如果agent在目标区域停下来，就会获得额外的reward，值为1；如果agent出界或是超过episode最大step次数，则不会获得额外奖励。用公式表示当前时刻的reward如下：

.. code:: python

    reward = last_distance - distance - penalty + (1 if goal else 0)


终止条件
------------
Gym-hybrid 环境每个episode的终止条件是遇到以下任何一种情况：

- agent 成功进入目标区域
  
- agant 出界
  
- 达到episode的最大step
  

内置环境
-----------
内置有两个环境，\ ``"Moving-v0"`` \和\ ``"Sliding-v0"`` \。前者不考虑惯性守恒，而后者考虑（所以更切合实际）。两个环境在状态空间、动作空间、奖励空间上都保持一致。

其他
====

存储录像
--------

有些环境有自己的渲染插件，但是DI-engine不支持环境自带的渲染插件，而是通过保存训练时的log日志来生成视频录像。具体方式可参考DI-engine `官方文档 <https://opendilab.github.io/DI-engine/quick_start/index.html>`__ Quick start 章节下的 Visualization & Logging 部分。

DI-zoo 可运行代码示例
=====================

下面提供一个完整的gym hybrid环境config，采用DDPG作为基线算法。请在\ ``DI-engine/dizoo/gym_hybrid`` \目录下运行\ ``gym_hybrid_ddpg_config.py`` \文件，如下。

.. code:: python

    from easydict import EasyDict
    from ding.entry import serial_pipeline

    gym_hybrid_ddpg_config = dict(
        exp_name='gym_hybrid_ddpg_seed0',
        env=dict(
            collector_env_num=8,
            evaluator_env_num=5,
            # (bool) Scale output action into legal range [-1, 1].
            act_scale=True,
            env_id='Moving-v0',  # ['Sliding-v0', 'Moving-v0']
            n_evaluator_episode=5,
            stop_value=2,  # 1.85 for hybrid_ddpg
        ),
        policy=dict(
            cuda=True,
            priority=False,
            random_collect_size=0,  # hybrid action space not support random collect now
            action_space='hybrid',
            model=dict(
                obs_shape=10,
                action_shape=dict(
                    action_type_shape=3,
                    action_args_shape=2,
                ),
                twin_critic=False,
                actor_head_type='hybrid',
            ),
            learn=dict(
                action_space='hybrid',
                update_per_collect=10,  # [5, 10]
                batch_size=32,
                discount_factor=0.99,
                learning_rate_actor=0.0003,  # [0.001, 0.0003]
                learning_rate_critic=0.001,
                actor_update_freq=1,
                noise=False,
            ),
            collect=dict(
                n_sample=32,
                noise_sigma=0.1,
                collector=dict(collect_print_freq=1000, ),
            ),
            eval=dict(evaluator=dict(eval_freq=1000, ), ),
            other=dict(
                eps=dict(
                    type='exp',
                    start=1.,
                    end=0.1,
                    decay=100000,  # [50000, 100000]
                ),
                replay_buffer=dict(replay_buffer_size=100000, ),
            ),
        ),
    )
    gym_hybrid_ddpg_config = EasyDict(gym_hybrid_ddpg_config)
    main_config = gym_hybrid_ddpg_config

    gym_hybrid_ddpg_create_config = dict(
        env=dict(
            type='gym_hybrid',
            import_names=['dizoo.gym_hybrid.envs.gym_hybrid_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(type='ddpg'),
    )
    gym_hybrid_ddpg_create_config = EasyDict(gym_hybrid_ddpg_create_config)
    create_config = gym_hybrid_ddpg_create_config


    if __name__ == "__main__":
        serial_pipeline([main_config, create_config], seed=0)


基准算法性能
============

-  Moving-v0（10M env step后停止，平均奖励大于等于1.8视为较好的Agent）

   - Moving-v0 + PDQN

   .. image:: images/gym_hybrid_Moving-v0_pdqn.png
     :align: center

   - Moving-v0 + MPDQN

   .. image:: images/gym_hybrid_Moving-v0_mpdqn.png
     :align: center

   - Moving-v0 + PADDPG

   .. image:: images/gym_hybrid_Moving-v0_paddpg.png
     :align: center


-  Sliding-v0（10M env step后停止，平均奖励大于等于1.8视为较好的Agent）

   - Sliding-v0 + PDQN

   .. image:: images/gym_hybrid_Sliding-v0_pdqn.png
     :align: center

   - Sliding-v0 + MPDQN

   .. image:: images/gym_hybrid_Sliding-v0_mpdqn.png
     :align: center

   - Sliding-v0 + PADDPG

   .. image:: images/gym_hybrid_Sliding-v0_paddpg.png
     :align: center



参考资料
=====================
- Gym-hybrid `源码 <https://github.com/thomashirtz/gym-hybrid>`__














