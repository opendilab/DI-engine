Cartpole
~~~~~~~~~~~~~~~~~~

概述
=======
倒立摆问题是强化学习中的经典控制问题。Cartpole 是倒立摆问题中的一个离散控制任务。在游戏中有一个小车，上有竖着一根杆子。小车在一个光滑无摩擦的轨道上左右滑行，目的是使杆子保持竖直。如下图所示。

.. image:: ./images/cartpole.gif
   :align: center

安装
====

安装方法
--------

Cartpole 环境内置在 gym 中，直接安装 gym 即可。其环境 id 是\ ``CartPole-v0`` \。

.. code:: shell

    pip install gym
    
验证安装
--------

在 Python 命令行中运行如下命令验证安装成功。

.. code:: shell 

    import gym
    env = gym.make('CartPole-v0')
    obs = env.reset()
    print(obs)  

环境介绍
=========

动作空间
----------

Cartpole 的动作空间属于离散动作空间，有 2 个离散动作，分别是左移和右移。

-  \ ``左移`` \: 0 表示让 agent 向左移动。

-  \ ``右移`` \: 1 表示让 agent 向右移动。

使用 gym 环境空间定义则可表示为：

.. code:: python
    
    action_space = spaces.Discrete(2)

状态空间
----------

Cartpole 的状态空间有 4 个元素，分别是：

- \ ``Cart Position`` \：小车的位置，范围是 \ ``[-4.8, 4.8]`` \。
  
- \ ``Cart Velocity`` \：小车的速度，范围是 \ ``[-inf, inf]`` \。

- \ ``Pole Angle`` \：杆的角度，范围是 \ ``[-24 deg, 24 deg]`` \。

- \ ``Pole Angular Velocity`` \：杆的角速度，范围是 \ ``[-inf, inf]`` \。


奖励空间
-----------
每一步操作，都将获得值为 1 的奖励，直到 episode 终止（终止状态也将获得值为 1 的奖励）。


终止条件
------------
Cartpole 环境每个 episode 的终止条件是遇到以下任何一种情况：

- 杆偏移的角度超过 12 度。
  
- 小车出界，通常把边界距离设置为 2.4。
  
- 达到 episode 的最大 step，默认为 200。
  

Cartpole 任务在什么情况下视为胜利
-----------------------------------

当100次试验的平均 episode 奖励达到 195 以上时，视作游戏胜利。


DI-zoo 可运行代码示例
=====================

下面提供一个完整的 cartpole 环境 config，采用 DQN 算法作为 policy。请在\ ``DI-engine/dizoo/classic_control/cartpole/entry`` \ 目录下运行\ ``cartpole_dqn_main.py`` \ 文件，如下。

.. code:: python

    import os
    import gym
    from tensorboardX import SummaryWriter

    from ding.config import compile_config
    from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
    from ding.envs import BaseEnvManager, DingEnvWrapper
    from ding.policy import DQNPolicy
    from ding.model import DQN
    from ding.utils import set_pkg_seed
    from ding.rl_utils import get_epsilon_greedy_fn
    from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config


    # Get DI-engine form env class
    def wrapped_cartpole_env():
        return DingEnvWrapper(gym.make('CartPole-v0'))


    def main(cfg, seed=0):
        cfg = compile_config(
            cfg,
            BaseEnvManager,
            DQNPolicy,
            BaseLearner,
            SampleSerialCollector,
            InteractionSerialEvaluator,
            AdvancedReplayBuffer,
            save_cfg=True
        )
        collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
        collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
        evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

        # Set random seed for all package and instance
        collector_env.seed(seed)
        evaluator_env.seed(seed, dynamic_seed=False)
        set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

        # Set up RL Policy
        model = DQN(**cfg.policy.model)
        policy = DQNPolicy(cfg.policy, model=model)

        # Set up collection, training and evaluation utilities
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
        learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
        collector = SampleSerialCollector(
            cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
        )
        evaluator = InteractionSerialEvaluator(
            cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
        )
        replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

        # Set up other modules, etc. epsilon greedy
        eps_cfg = cfg.policy.other.eps
        epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

        # Training & Evaluation loop
        while True:
            # Evaluating at the beginning and with specific frequency
            if evaluator.should_eval(learner.train_iter):
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break
            # Update other modules
            eps = epsilon_greedy(collector.envstep)
            # Sampling data from environments
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            # Training
            for i in range(cfg.policy.learn.update_per_collect):
                train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
                if train_data is None:
                    break
                learner.train(train_data, collector.envstep)
        # evaluate
        evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)
        evaluator_env.enable_save_replay(cfg.env.replay_path)  # switch save replay interface
        evaluator = InteractionSerialEvaluator(
            cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
        )
        evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)


    if __name__ == "__main__":
        main(cartpole_dqn_config)

实验结果
=================
使用 DQN 算法的实验结果如下。横坐标是\ ``episode`` \，纵坐标是\ ``reward_mean`` \。

.. image:: ./images/cartpole_dqn.png
   :align: center
   :scale: 20%


参考资料
=====================
- Cartpole `源码 <https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py>`__

