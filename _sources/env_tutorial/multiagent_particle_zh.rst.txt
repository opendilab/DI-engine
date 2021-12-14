Multi-Agent Particle
~~~~~~~~~~~~~~~~~~~~~~

概述
============

- Multi-Agent Particle Environment (MPE) 是由 OpenAI 开源的一款多智能体仿真环境，里面涵盖了多智能体的竞争/协作/通信的场景，可用来对各类多智能体强化学习算法进行验证测试。
- MPE 作为 NIPS2017 那篇著名的多智能体强化学习算法 MADDPG (https://arxiv.org/abs/1706.02275) 的实验环境，因而被人们广泛所知。
- MPE 以 OpenAI 的 gym 为基础，使用 python 编写而成。它构造了一系列简单的多智能体粒子环境（9个子环境），粒子们可以互相合作进行抓捕，碰撞等。
- 其中，我们对环境比较关注的信息是：状态观测为连续空间，动作信息默认为离散控制 (可设置为连续控制)。另外，我们可以设置智能体的数量，选择要完成的任务。
- 下图是 MPE 一个子任务环境\ ``Simple Tag``\ ，其中有两类智能体，红球表示捕食者 (predator) ，绿球表示猎物 (prey) ，黑球表示障碍物 (landmark) 。

.. image:: ./images/mpe_simple_tag.gif
   :align: center

安装
===============

安装方法
------------------------

MPE 已经被集成在 DI-engine/dizoo 仓库中，所以只要安装 DI-engine 就可以正常使用。另外，MPE 也可以通过安装 pettingzoo 实现，后续 DI-engine 也会进一步包含容纳 pettingzoo 里其他的多智体环境。 

.. code:: shell

   # Method1: Download and Compile from DI-engine, which integrates Multi-Agent Particle
   git clone https://github.com/opendilab/DI-engine
   pip install -e .

   # Method2: Install by pettingzoo
   pip install pettingzoo[mpe]

验证安装
------------------------

安装完成后，可以通过在 Python 命令行运行如下命令验证是否安装成功，如果看到有相关环境信息打印出来，则说明安装成功。

.. code:: python

    from dizoo.multiagent_particle.envs import ParticleEnv, CooperativeNavigation
    num_agent, num_landmark = 5, 5
    env = CooperativeNavigation({'n_agents': num_agent, 'num_landmarks': num_landmark, 'max_step': 100})
    print(env.info())
    env.close()

.. _变换前的空间（原始环境）:

变换前的空间（原始环境）
========================================================

.. _观察空间-1:

观察空间
----------------------

-  每个智能体的观测状态是一个向量信息，包含四个部分：
  
  -  当前智能体的位置和速度；
  
  -  其他智能体对于当前智能体的相对位置和相对速度；
  
  -  地标和智能体的类型；
  
  -  以及智能体之间的通信信息。

-  以\ ``Simple Tag``\ 环境为例，智能体的观测状态的物理意义为：\ ``[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]``\ ，单个智能体的观测状态 shape 为\ ``(14),(16)``\ ，观测向量中每个维度的大小范围是\ ``(-inf,inf)``\ 。智能体的全局观测状态是所有单智能体观测向量的拼接。

-  注意：多智能体之间会存在部分可观测的问题，因此对于当前智能体的观察空间是不断变化的。

.. _动作空间-1:

动作空间
------------------

-  默认为离散动作空间，设置\ ``continuous_actions=True``\ 为连续动作空间。

-  以\ ``Simple Tag``\ 环境为例，智能体的动作空间的物理意义为：\ ``[no_action, move_left, move_right, move_down, move_up]``\ 。

-  离散动作向量的大小为 \ ``Discrete(5)``\，表示在上下左右的方向是否采取动作。

-  连续动作向量的大小为 \ ``Box(0.0, 1.0, (5))``\ ，将上面离散动作适配在连续空间，每一维的大小限制在\ ``[0,1]``\ 之间。

.. _奖励空间-1:

奖励空间
-----------------

-  游戏得分，不同任务环境奖励设置略有不同，包括捕食者的奖励和猎物的奖励信息。

-  以\ ``Simple Tag``\ 环境为例，对猎物来说，当被捕捉时，奖励为\ ``-10``\ ；对捕食者来说，当捕捉到猎物时，奖励为\ ``10``\。

.. _终止信息-1:

终止信息
----------

-  当游戏运行至\ ``max_step``\ 即为游戏结束，MPE 源码默认子环境\ ``max_step=100``\ 。

关键事实
==============

1. 状态为向量输入。

2. 动作默认为离散动作，可设置为连续动作。


.. _变换后的空间rl环境）:

变换后的空间（RL环境）
======================================================

为了更好地适配 QTRAN 等多智体强化学习算法，我们对原始环境进行二次改造，以\ ``Simple Tag``\ 环境为例，修改后的环境命名为\ ``Modified Predator Prey``\ 。

.. _观察空间-2:

观察空间
--------------------------

-  根据自定义的智能体数量，对观察空间的状态向量进行维度适配。例如，\ ``Modified Predator Prey``\ 默认有 2 个捕食者，1 个猎物，3 个障碍物，观测空间的向量表示\ ``[self_vel, self_pos, other_agent_rel_positions, landmark_rel_positions]``\ ，此时智能体观测状态 shape 为\ ``(14)``\。

.. _动作空间-2:

动作空间
-----------------

-  无基本变化。

.. _奖励空间-2:

奖励空间
-----------------

-  为了在合作围捕等非单调 (non-monotonicity) 任务上，测试 QTRAN 和 QMIX 性能差异，因此修改奖励规则设置：只有当所有的捕食者共同捕捉到猎物时，才有正向奖励；否则为负向奖励。

上述空间使用 gym 环境空间定义则可表示为：

.. code:: python

   import gym

   obs_space = gym.spaces.Box(low=-inf, high=inf, shape=(N, ), dtype=np.float32)
   act_space = gym.spaces.Discrete(5)
   rew_space = gym.spaces.Box(low=-inf, high=inf, shape=(1, ), dtype=np.float32)


其他
===========

惰性初始化
-------------------------

为了便于支持环境向量化等并行操作，环境实例一般实现惰性初始化，即\ ``__init__``\ 方法不初始化真正的原始环境实例，只是设置相关参数和配置值，在第一次调用\ ``reset``\ 方法时初始化具体的原始环境实例。

随机种子
------------------

-  环境中有两部分随机种子需要设置，一是原始环境的随机种子，二是各种环境变换使用到的随机库的随机种子（例如\ ``random``\ ，\ ``np.random``\ ）。

-  对于环境调用者，只需通过环境的\ ``seed``\ 方法进行设置这两个种子，无需关心具体实现细节。

-  环境内部的具体实现：对于原始环境的种子，在调用环境的\ ``reset``\ 方法内部，具体的原始环境\ ``reset``\ 之前设置。

-  环境内部的具体实现：对于随机库种子，则在环境的\ ``seed``\ 方法中直接设置该值。

训练和测试环境的区别
----------------------------------------------------------

-  训练环境使用动态随机种子，即每个episode的随机种子都不同，都是由一个随机数发生器产生，但这个随机数发生器的种子是通过环境的\ ``seed``\ 方法固定的；测试环境使用静态随机种子，即每个episode的随机种子相同，通过\ ``seed``\ 方法指定。

保存录像
----------------------------

在环境创建之后，重置之前，调用\ ``enable_save_replay``\ 方法，指定游戏录像保存的路径。环境会在每个episode结束之后自动保存本局的录像文件。（默认调用\ ``gym.wrapper.Monitor``\ 实现，依赖\ ``ffmpeg``\ ），下面所示的代码将运行一个环境episode，并将这个episode的结果保存在形如\ ``./video/xxx.mp4``\ 这样的文件中：

.. code:: python

   from easydict import EasyDict
   from dizoo.multiagent_particle.envs import ModifiedPredatorPrey

   env = ModifiedPredatorPrey(EasyDict({'env_id': 'modified_predator_prey', 'is_train': False}))
   env.enable_save_replay(replay_path='./video')
   obs = env.reset()

   while True:
       action = env.random_action()
       timestep = env.step(action)
       if timestep.done:
           print('Episode is over, final eval reward is: {}'.format(timestep.info['final_eval_reward']))
           break

DI-zoo 代码示例
=======================

Complete training configuration is at `github
link <https://github.com/opendilab/DI-engine/tree/main/dizoo/multiagent_particle/config>`_ .
For specific configuration file, e.g. ``modified_predator_prey_qtran_config.py``, you can run the demo as shown below:

.. code:: python

    from copy import deepcopy
    from ding.entry import serial_pipeline
    from easydict import EasyDict

    n_predator = 2
    n_prey = 1
    n_agent = n_predator + n_prey
    num_landmarks = 1

    collector_env_num = 4
    evaluator_env_num = 5
    main_config = dict(
        env=dict(
            max_step=100,
            n_predator=n_predator,
            n_prey=n_prey,
            num_landmarks=num_landmarks,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            n_evaluator_episode=5,
            stop_value=100,
            num_catch=2,
            reward_right_catch=10,
            reward_wrong_catch=-2,
            collision_ratio=2
        ),
        policy=dict(
            model=dict(
                agent_num=n_predator,
                obs_shape=2 + 2 + (n_agent - 1) * 2 + num_landmarks * 2,
                global_obs_shape=n_agent * 2 + num_landmarks * 2 + n_agent * 2,
                action_shape=5,
                hidden_size_list=[128],
                embedding_size=64,
                lstm_type='gru',
                dueling=False,
            ),
            learn=dict(
                update_per_collect=100,
                batch_size=32,
                learning_rate=0.0005,
                double_q=True,
                target_update_theta=0.001,
                discount_factor=0.99,
                td_weight=1,
                opt_weight=0.1,
                nopt_min_weight=0.0001,
            ),
            collect=dict(
                n_sample=600,
                unroll_len=16,
                env_num=collector_env_num,
            ),
            eval=dict(env_num=evaluator_env_num, ),
            other=dict(
                eps=dict(
                    type='exp',
                    start=1.0,
                    end=0.05,
                    decay=100000,
                ),
                replay_buffer=dict(
                    replay_buffer_size=15000,
                    # (int) The maximum reuse times of each data
                    max_reuse=1e+9,
                    max_staleness=1e+9,
                ),
            ),
        ),
    )
    main_config = EasyDict(main_config)
    create_config = dict(
        env=dict(
            import_names=['dizoo.multiagent_particle.envs.particle_env'],
            type='modified_predator_prey',
        ),
        env_manager=dict(type='base'),
        policy=dict(type='qtran'),
    )
    create_config = EasyDict(create_config)

    modified_predator_prey_qtran_config = main_config
    modified_predator_prey_qtran_create_config = create_config


    def train(args):
        config = [main_config, create_config]
        serial_pipeline(config, seed=args.seed)


    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=0)
        args = parser.parse_args()

        train(args)


基准算法性能
=======================

-  Modified Predator Prey

   - QTRAN 和 QMIX 算法结果对比
  
   .. image:: images/ModifiedPredatorPrey_qtran_vs_qmix_penalty2.png
     :align: center
