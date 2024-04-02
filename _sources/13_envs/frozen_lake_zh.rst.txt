FrozenLake
~~~~~~~~~~~~~~~~~~

概述
=======
FrozenLake 是强化学习中的经典控制问题。需要控制智能体在冰冻湖面上行进，从起点穿越冰冻湖到达目标点，且不会掉到任何冰洞里
。如下图所示。

.. image:: ./images/FrozenLake.gif
   :align: center
   :scale: 80%

安装
====

安装方法
--------

FrozenLake 环境内置在 gymnasium 中，直接安装 gymnasium 即可。其环境 id 是\ ``FrozenLake-v1`` \。

.. code:: shell

    pip install gymnasium
    
验证安装
--------

在 Python 命令行中运行如下命令验证安装成功。

.. code:: shell 

    import gymnasium 
    env = gymnasium.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    obs = env.reset()
    print(obs)
    assert env.observation_space.shape == gymnasium.spaces.Discrete(16)
    assert env.action_space == gymnasium.spaces.Discrete(4)

环境介绍
=========

动作空间
----------

FrozenLake 的动作空间属于离散动作空间，动作形状为 (1,) ，范围为 {0, 3} ，表示玩家移动的方向。

-  \ ``0:`` \: 向左移动

-  \ ``1:`` \: 向下移动

-  \ ``2:`` \: 向右移动

-  \ ``3:`` \: 向上移动

使用 gymnasium 环境空间定义则可表示为：

.. code:: python
    
    action_space = gymnasium.spaces.Discrete(4)

状态空间
----------

状态空间是一个代表玩家当前位置的值，即 \ ``current_row * nrows + current_col`` \ （其中行和列都从 0 开始）。


- \ ``15`` \: (4x4 地图中的目标位置通过下面方式计算:\ ``3*4+3 = 15`` \)

奖励空间
-----------
-  \ ``+1`` \: 到达目标

-  \ ``0`` \: 到达冰洞

-  \ ``0`` \: 前进


终止条件
------------
FrozenLake 环境每个 episode 的终止条件是遇到以下任何一种情况：

- 玩家掉入冰洞。
- 玩家到达终点(位置 \ ``max(nrow) * max(ncol) - 1`` \)。
- 达到 episode 的最大 step,默认为 ``100``。
  

DI-zoo 可运行代码示例
=====================


完整的训练配置文件在 `github
link <https://github.com/opendilab/DI-engine/tree/main/dizoo/frozen_lake/config>`__
内，对于具体的配置文件，例如\ ``frozen_lake_dqn_config.py``\ ，使用如下的 demo 即可运行：

.. code:: python
    

    from easydict import EasyDict
    frozen_lake_dqn_config = dict(
        exp_name='frozen_lake_seed0',
        env=dict(
            collector_env_num=8,
            evaluator_env_num=5,
            n_evaluator_episode=10,
            env_id='FrozenLake-v1',
            desc=None,
            map_name="4x4",
            is_slippery=False,
            save_replay_gif=False,
        ),
        policy=dict(
            cuda=True,
            load_path='frozen_lake_seed0/ckpt/ckpt_best.pth.tar',
            model=dict(
                obs_shape=16,
                action_shape=4,
                encoder_hidden_size_list=[128, 128, 64],
                dueling=True,
            ),
            nstep=3,
            discount_factor=0.97,
            learn=dict(
                update_per_collect=5,
                batch_size=256,
                learning_rate=0.001,
            ),
            collect=dict(n_sample=10),
            eval=dict(evaluator=dict(eval_freq=40, )),
            other=dict(
                eps=dict(
                    type='exp',
                    start=0.8,
                    end=0.1,
                    decay=10000,
                ),
                replay_buffer=dict(replay_buffer_size=20000, ),
            ),
        ),
    )

    frozen_lake_dqn_config = EasyDict(frozen_lake_dqn_config)
    main_config = frozen_lake_dqn_config

    frozen_lake_dqn_create_config = dict(
        env=dict(
            type='frozen_lake',
            import_names=['dizoo.frozen_lake.envs.frozen_lake_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(type='dqn'),
        replay_buffer=dict(type='deque', import_names=['ding.data.buffer.deque_buffer_wrapper']),
    )

    frozen_lake_dqn_create_config = EasyDict(frozen_lake_dqn_create_config)
    create_config = frozen_lake_dqn_create_config

    if __name__ == "__main__":
        # or you can enter `ding -m serial -c frozen_lake_dqn_config.py -s 0`
        from ding.entry import serial_pipeline
        serial_pipeline((main_config, create_config), max_env_step=5000, seed=0)


基准算法性能
=================
使用 DQN 算法的实验结果如下。横坐标是\ ``step`` \，纵坐标是\ ``reward_mean`` \。

.. image:: ./images/frozen_lake_dqn.jpg
   :align: center
   :scale: 80%


参考资料
=====================
- FrozenLake `源码 <https://github.com/opendilab/DI-engine/tree/main/dizoo/frozen_lake>`__

