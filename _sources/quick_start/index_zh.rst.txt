快速上手
===============================

.. toctree::
   :maxdepth: 3


.. image:: 
   images/cartpole_cmp.gif
   :align: center

首先，我们将说明如何使用 DI-engine 在简单的 ``CartPole`` 环境（如图所示）上运行 RL 实验。


具体来说，我们将在单个 python 文件中定义一整套训练逻辑，指定超参数、环境，神经网络和强化学习策略，以及主循环训练流程。

构建运行时配置
-------------------------------

构建训练工作流的第一步是指定训练配置。 DI-engine 推荐使用嵌套的 `dict` 对象来表示 RL 实验的所有参数和配置（``...`` 表示省略的配置内容，完整的配置文件可以参考下面的路径），例如：

.. code-block:: python

    cartpole_dqn_config = dict(
        exp_name="cartpole_dqn",
        env=dict(
            collector_env_num=8,
            evaluator_env_num=5,
        ),
        policy=dict(
            model=dict(
                encoder_hidden_size_list=[128, 128, 64],
            ),
            discount_factor=0.97,
        ),
        ...
    )

.. note ::

    具体内容可以参考：

      - 配置文件:  ``dizoo/classic_control/cartpole/config/cartpole_dqn_config.py``
      - 训练入口文件: ``dizoo/classic_control/cartpole/entry/cartpole_dqn_main.py``

    输入下方命令即可运行实验：

    .. code:: bash

        python3 -u dizoo/classic_control/cartpole/entry/cartpole_dqn_main.py

DI-engine 为所有模块提供默认配置，且有一个辅助函数 ``compile_config`` 合并各模块的默认配置和用户的自定义配置，从而产生具体训练运行的配置文件（最终的配置文件是一个 ``EasyDict`` 对象，可通过字符串键 ``cfg["env"]`` 或 ``cfg.env`` 访问）：

.. code-block:: python

    from ding.config import compile_config
    from ding.envs import BaseEnvManager, DingEnvWrapper
    from ding.model import DQN
    from ding.policy import DQNPolicy
    from ding.worker import BaseLearner, SampleCollector, BaseSerialEvaluator, AdvancedReplayBuffer
    from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config

    # compile config
    cfg = compile_config(
        cartpole_dqn_config,
        BaseEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleCollector,
        BaseSerialEvaluator,
        AdvancedReplayBuffer,
        save_cfg=True
    )

这个例子中展示了在入口文件中指定配置的过程。在下一节中，我们根据指定的配置在同一个入口文件中构建 RL 训练流程。

请注意，DI-engine 还支持根据给定的配置文件直接运行 RL 实验，对应命令行如下：

.. code:: bash

    ding -m serial -c cartpole_dqn_config.py -s 0

有关更多设计细节，请参阅 `配置 <../key_concept/index.html#config>`_ 以及 `入口 <../key_concept/index.html#entry>`_ 模块


初始化环境
---------------------------

RL策略与环境交互以收集训练数据或测试其性能。DI-engine 继承广泛使用的 `OpenAI Gym <https://github.com/openai/gym>`_ RL 环境接口，并扩展了部分功能。您可以使用 ``DingEnvWrapper`` 将部分简单的 ``gym`` 环境
直接转化为DI-engine所需的格式。对于较复杂的环境，建议按照 `环境 <../key_concept/index.html#env>`_ 部分中的指南构建具体的环境类。


使用 ``Env Manager`` 管理向量化的多个环境，一般使用python多进程实现。环境管理器的界面类似于简单的 ``gym`` 环境。这里我们展示一个使用 ``BaseEnvManager`` 来构建收集和评估环境的案例。

.. code-block:: python

    import gym

    def wrapped_cartpole_env():
        return DingEnvWrapper(gym.make('CartPole-v0'))

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

设置常用库的环境的随机种子，保证实验的可复现性。

.. code-block:: python

    from ding.utils import set_pkg_seed
    # Here we select a fixed seed in order to reach convergence more quickly for this demo
    # Set seed for all package and instance
    collector_env.seed(seed=0)
    evaluator_env.seed(seed=0, dynamic_seed=False)
    set_pkg_seed(seed=0, use_cuda=cfg.policy.cuda)

设置策略和神经网络模型
-------------------------------

DI-engine 支持 RL 训练中使用的大多数常用策略。每个都定义为一个 ``Policy`` class.

优化算法、数据预处理和后处理、神经网络的使用等细节都封装在 ``Policy`` 类之中。用户只需要构建一个 `PyTorch 网络 <https://pytorch.org/docs/master/generated/torch.nn.Module.html>`_ （即继承于 
``nn.Module`` ）并将其传入策略。

DI-engine 还提供一些默认的神经网络实现，以应用于较简单的环境。

例如，可以为 ``CartPole`` 环境定义如下的 ``DQN`` 神经网络和策略：

.. code-block:: python

    model = DQN(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)


定义执行模块
----------------------------

DI-engine 需要构建一些执行组件来实际运行 RL 训练过程。``Collector`` 用于收集训练所需的数据。``Learner`` 用于接收数据并进行训练。``Evaluator`` 用于在需要的时候对策略进行评估。``Replay Buffer`` 
是一个存放数据的队列。对于不同算法，整个训练过程还可能需要其他组件。所有上述模块都可以通过配置自定义或由用户重写。

具体的创建过程如下：

.. code-block:: python

    import os
    from tensorboardX import SummaryWriter    

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = BaseSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

这只是一组执行模块示例的具体设定。用户应根据需求自行选择需要的模块。

训练主循环
-----------------------------------------------

DI-engine 中的训练循环可以任意定制。通常训练过程可能包括收集数据、更新策略和相关模块，评估策略性能。

在这里，我们提供了 ``DQN`` 针对 ``CartPole`` 环境的异策略训练示例。更多算法可以参考 ``dizoo`` 。


.. code-block:: python

    from ding.rl_utils import get_epsilon_greedy_fn
    
    # DQN training loop
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
    max_iterations = int(1e8)
    for _ in range(max_iterations):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        eps = epsilon_greedy(collector.envstep)
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)

.. note::
    您也可以参考完整的训练入口： dizoo/classic_control/cartpole/entry/cartpole_dqn_main.py。

其他工具
------------------

DI-engine 支持常见 RL 训练中的各种工具，如下所示。


可视化和日志
~~~~~~~~~~~~~~~~~~~~~~~~~

某些环境具有渲染或可视化功能，DI-engine 没有使用渲染接口，而是添加了存储可视化结果 (replay) 的开关接口。如果想开启该功能，用户只需在入口函数训练主循环收敛后添加如下几行代码。如果一切正常，您可以在 ``replay_path`` 指定的文件夹中找到一些以 ``.mp4`` 为后缀的视频。

.. code-block:: python

    evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)
    cfg.env.replay_path = './video'  # indicate save replay directory path
    evaluator_env.seed(seed=0, dynamic_seed=False)
    evaluator_env.enable_save_replay(cfg.env.replay_path)  # switch save replay interface
    evaluator = BaseSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)

.. note::

  如果用户想使用之前已经训练好的策略来进行可视化，可以参照 ``dizoo/classic_control/cartpole/entry/cartpole_dqn_eval.py`` 构建一个自定义的评测入口函数，并在config中指定 ``env.replay_path`` and ``policy.load_path`` 两个字段，config示例如下面的代码所示，``...`` 表示省略的config内容

  .. code-block:: python
  
    config = dict(
        env=dict(
            replay_path='your_replay_save_dir_path',
        ),
        policy=dict(
            ...,
            load_path='your_ckpt_path',
            ...,
        ),
    )


.. tip::

   每个新的RL环境都可以自定义自己的 ``enable_save_replay`` 方法，指定具体生成回放文件的方式。DI-engine对于几个经典环境使用 ``gym wrapper (内部调用ffmpeg)`` 进行可视化。如果在使用 ``gym wrapper`` 录制视频时遇到一些错误，请尝试安装 ``ffmpeg`` 。


和其他深度强化学习平台类似，DI-engine也使用tensorboard来记录一些训练时的关键信息和参数。除了DI-engine默认记录
的信息之外，用户可以按照下文中的方式添加自己想记录的信息。


.. code-block:: python

    tb_logger.add_scalar('epsilon_greedy', eps, learner.train_iter)

如果你想了解DI-engine默认记录的参数信息的含义，可以参考DQN训练实验的 `tensorboard和日志demo <./tb_demo.html>`_ 。

