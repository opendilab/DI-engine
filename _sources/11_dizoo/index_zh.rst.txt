从 DI-zoo 开始学习
===============================

什么是 DI-zoo
-------------------------------

DI-zoo 是一个使用 DI-engine 封装的强化学习环境集合。它覆盖了绝大多数强化学习环境，既包括基础的 `OpenAI Gym <https://gym.openai.com/>`_ ，也包括 `SMAC <https://github.com/oxwhirl/smac>`_ 等更为复杂的环境。除此之外，针对每个环境，DI-zoo 都提供了不同算法的运行入口，以及每个算法对应的最优配置。

DI-zoo 的结构
-------------------------------

为了在 DI-engine 中进行强化学习训练，对于某一环境/策略组合，DI-zoo 主要提供了如下两个文件： ``config.py`` 文件，包括运行该环境/策略组合所需的关键配置，以及对训练管线的调用，作为算法的运行入口； ``env.py`` 文件，包括为了使用 DI-engine 运行该环境，而对其进行的封装。

.. note::
    
    除此之外，某些环境/策略组合还包括一个 ``main.py`` 的入口文件，是之前版本所遗留下来的训练管线文件。

这里我们基于 CartPole 环境与 DQN 算法来简单展示一下 DI-zoo 的结构。

.. code-block::

  dizoo/
    classic_control/
      cartpole/
        config/cartpole_dqn_config.py # Config
        entry/cartpole_dqn_main.py  # Main 
        envs/cartpole_env.py  # Env

DI-zoo 的用法
-------------------------------
您可以直接通过执行 DI-zoo 提供的 ``config.py`` 文件，来进行某个环境/策略组合的强化学习训练。对于 CartPole/DQN ，您可以通过以下代码来轻易进行它的强化学习训练：

.. code-block:: bash

    python dizoo/classic_control/cartpole/config/cartpole_dqn_config.py

DI-engine还为用户准备了CLI工具，您可以在终端中键入以下命令：

.. code-block:: bash

   ding -v

如果终端返回正确的信息，您可以使用这个CLI工具进行常见的训练和评估，您可以键入 ``ding -h`` 查看更多帮助。

对于 CartPole/DQN 的训练，您可以直接通过在终端键入以下命令来完成：

.. code-block:: bash

   ding -m serial -c cartpole_dqn_config.py -s 0

其中 ``-m serial`` 代表您调用的训练管线是 ``serial_pipeline``。 ``-c cartpole_dqn_config.py`` 代表您使用的 ``config`` 文件是 ``cartpole_dqn_config.py``。 ``-s 0`` 代表 ``seed`` 为0。

DI-zoo 的自定义
-------------------------------

您可以通过更改 ``config.py`` 中的配置，来自定义训练流程，或者对某环境/策略组合的性能进行调优。

还是以 ``cartpole_dqn_config.py`` 为例进行演示：

.. code-block:: python

    from easydict import EasyDict

    cartpole_dqn_config = dict(
        exp_name='cartpole_dqn_seed0',
        env=dict(
            collector_env_num=8,
            evaluator_env_num=5,
            n_evaluator_episode=5,
            stop_value=195,
            replay_path='cartpole_dqn_seed0/video',
        ),
        policy=dict(
            cuda=False,
            load_path='cartpole_dqn_seed0/ckpt/ckpt_best.pth.tar',  # necessary for eval
            model=dict(
                obs_shape=4,
                action_shape=2,
                encoder_hidden_size_list=[128, 128, 64],
                dueling=True,
            ),
            nstep=1,
            discount_factor=0.97,
            learn=dict(
                batch_size=64,
                learning_rate=0.001,
            ),
            collect=dict(n_sample=8),
            eval=dict(evaluator=dict(eval_freq=40, )),
            other=dict(
                eps=dict(
                    type='exp',
                    start=0.95,
                    end=0.1,
                    decay=10000,
                ),
                replay_buffer=dict(replay_buffer_size=20000, ),
            ),
        ),
    )
    cartpole_dqn_config = EasyDict(cartpole_dqn_config)
    main_config = cartpole_dqn_config
    cartpole_dqn_create_config = dict(
        env=dict(
            type='cartpole',
            import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(type='dqn'),
        replay_buffer=dict(
            type='deque',
            import_names=['ding.data.buffer.deque_buffer_wrapper']
        ),
    )
    cartpole_dqn_create_config = EasyDict(cartpole_dqn_create_config)
    create_config = cartpole_dqn_create_config

    if __name__ == "__main__":
        # or you can enter `ding -m serial -c cartpole_dqn_config.py -s 0`
        from ding.entry import serial_pipeline
        serial_pipeline((main_config, create_config), seed=0)

其中 ``cartpole_dqn_config`` 和 ``cartpole_dqn_create_config`` 这两个字典对象，包含了 CartPole/DQN 训练需要的关键配置。您可以通过改变这里的配置，来改变训练管线的行为。比如通过更改 ``cartpole_dqn_config.policy.cuda`` ， 您可以选择是否使用 cuda 设备来运行整个训练流程。

如果想要使用 DI-engine 提供的其他训练管线，或者使用自己自定义的训练管线的话，您只需要更改 ``config`` 文件最下方， ``__main__`` 函数中调用训练管线的部分即可。比如您可以把例子中的 ``serial_pipeline`` 改成 ``parallel_pipeline``，来调用并行的训练管线。

对于CLI工具 ``ding``，您也可以把之前的cli命令改成

.. code-block:: bash

   ding -m parallel -c cartpole_dqn_config.py -s 0

来调用 ``parallel_pipeline``。

.. note ::

    如何自定义训练管线可以参考 `serial_pipeline <https://github.com/opendilab/DI-engine/blob/0fccfcb046f04767504f68220d96a6608bb38f29/ding/entry/serial_entry.py#L17>`_ 的写法，或者参考 `DQN example <https://github.com/opendilab/DI-engine/blob/main/ding/example/dqn.py>`_，使用 DI-engine 提供的 `中间件 <../03_system/middleware_zh.html>`_ 来进行搭建。

    如果您想要接入自己的环境，只需继承 DI-engine 实现的 ``BaseEnv`` 即可。这部分可以参考 `如何将自己的环境迁移到DI-engine中 <../best_practice/ding_env_zh.html>`_。

DI-zoo 已支持的算法和环境列表
-------------------------------

`DI-engine 算法文档 <../12_policies/index_zh.html>`_

`DI-engine 环境文档 <../13_envs/index_zh.html>`_

`已支持的算法 <https://github.com/opendilab/DI-engine#algorithm-versatility>`_

`已支持的环境 <https://github.com/opendilab/DI-engine#environment-versatility>`_
