更多强化学习项目 (定制化环境)
========================================================

.. toctree::
   :maxdepth: 2

任天堂系统（NES）上的 `超级马里奥兄弟 & 超级马里奥兄弟2  <https://github.com/Kautenja/gym-super-mario-bros>`_ 是1980年代最受欢迎的游戏之一。 \
来设计一个基于 DRL 的 AI 来探索这款经典游戏怎么样? 在本教程中，我们将在 DI-engine 中实现一个由人工智能控制的马里奥（使用 DQN 算法）。

.. image::
    images/mario.gif
    :width: 400
    :align: center

使用配置文件
------------------------------

DI-engine 使用全局配置文件来控制环境和策略的所有变量，每个变量都有对应的默认配置可以在 `mario_dqn_config <https://github.com/opendilab/DI-engine/blob/main/dizoo/mario/mario_dqn_config.py>`_ 中找到，在本次教程中我们直接使用默认配置：

.. code-block:: python

    from dizoo.mario.mario_dqn_config import main_config, create_config
    from ding.config import compile_config

    cfg = compile_config(main_config, create_cfg=create_config, auto=True)

初始化环境
------------------------------

``超级马里奥兄弟`` 是一个 **图像输入** 观察环境, 所以我们不只是通过 ``DingEnvWrapper`` 封装原始的 gym 环境，而是需要添加一些额外的 Wrapper 在发送给DQN    Policy 之前对观测进行预处理。
在本教程中，我们使用以下5个 Wrapper 来预处理数据并转换为 DI-engine 的环境格式，下列是一些基本描述，你可以在这里找到 `完整代码实现和注释 <https://github.com/opendilab/DI-engine/blob/main/ding/envs/env_wrappers/env_wrappers.py>`_

  - ``MaxAndSkipWrapper`` : 由于连续帧变化不大，我们可以跳过n个中间帧来简化它而不会损失太多信息。
  - ``WarpFrameWrapper`` : 将原始RGB图像转换为灰度图，并将其调整为标准大小以进行DRL训练。
  - ``ScaledFloatFrameWrapper`` : 将原始图像从[0-255]归一化到[0-1]，有利于神经网络训练。
  - ``FrameStackWrapper`` : 堆叠连续的帧。由于我们无法从单帧推断方向、速度等信息，堆叠帧可以提供更多的必要信息。
  - ``EvalEpisodeReturnEnv`` : 记录最终的evaluation reward（即马里奥中的episode return），适配DI-engine的环境格式。


.. note::

    如果找不到合适的 Env Wrapper， 您可以按照 ``gym.Wrapper`` 格式定义自己的 Wrapper ，也可以根据 `Customized Env doc <https://di-engine-docs.readthedocs.io/en/latest/04_best_practice/ding_env.html>`_ 实现符合 DI-engine 的环境格式


.. code-block:: python

    # use subprocess env manager to speed up collecting 
    from ding.envs import DingEnvWrapper, BaseEnvManagerV2, SubprocessEnvManagerV2
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace


    def wrapped_mario_env():
        return DingEnvWrapper(
            # Limit the action-space to 2 dim: 0. walk right, 1. jump right
            JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v0"), [["right"], ["right", "A"]]),
            cfg={
                'env_wrapper': [
                    lambda env: MaxAndSkipWrapper(env, skip=4),
                    lambda env: WarpFrameWrapper(env, size=84),
                    lambda env: ScaledFloatFrameWrapper(env),
                    lambda env: FrameStackWrapper(env, n_frames=4),
                    lambda env: EvalEpisodeReturnEnv(env),
                ]
            }
        )

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = SubprocessEnvManagerV2(
        env_fn=[wrapped_mario_env for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    evaluator_env = SubprocessEnvManagerV2(
        env_fn=[wrapped_mario_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

.. note::

    以下内容与 ``CartPole + DQN`` 示例相同, 只需要选择策略并搭建整个训练管线

选择策略
--------------

DI-engine 涵盖了大部分强化学习策略，使用它们只需要选择正确的策略和模型。由于 DQN 是 Off-Policy 类型算法，我们还需要实例化一个缓冲区模块。

.. code-block:: python

    from ding.model import DQN
    from ding.policy import DQNPolicy
    from ding.data import DequeBuffer

    model = DQN(**cfg.policy.model)
    buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
    policy = DQNPolicy(cfg.policy, model=model)

构建管线
---------------------

借助 DI-engine 提供的各种中间件，我们可以轻松构建整个训练管线：

.. code-block:: python

    from ding.framework import task
    from ding.framework.context import OnlineRLContext
    from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, eps_greedy_handler, CkptSaver, nstep_reward_enhancer

    with task.start(async_mode=False, ctx=OnlineRLContext()):
        # Evaluating, we place it on the first place to get the score of the random model as a benchmark value
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))  # Decay probability of explore-exploit
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))  # Collect environmental data
        task.use(nstep_reward_enhancer(cfg))  # Prepare nstep reward for training
        task.use(data_pusher(cfg, buffer_))  # Push data to buffer
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))  # Train the model
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))  # Save the model
        # In the evaluation process, if the model is found to have exceeded the convergence value, it will end early here
        task.run()

运行代码
--------------

完整的示例可以在 `马里奥DQN示例 <https://github.com/opendilab/DI-engine/blob/main/dizoo/mario/mario_dqn_example.py>`_ 中找到，并且可以通过 ``python3 mario_dqn_example.py`` 运行，
这个示例可以在1～2小时内给出顺利的结果，我们的超级马里奥可以很快地通过1-1关卡且不收到任何伤害，下图是训练好的智能体的回放视频和详细的训练曲线。

.. image::
    images/mario.gif
    :width: 600
    :align: center

.. image::
    images/mario_dqn_curve.png
    :width: 800
    :align: center

现在您已经完成了使用 DI-engine 的 **定制化环境示例** , 您可以在 `示例目录 <https://github.com/opendilab/DI-engine/blob/main/ding/example>`_ 中尝试更多算法, 或继续阅读文档以更深入地了解 DI-engine 的  `算法 <../02_algo/index_zh.html>`_, `系统设计 <../03_system/index_zh.html>`_  和 `最佳实践 <../04_best_practice/index_zh.html>`_.
