配置文件系统
===============================

.. toctree::
   :maxdepth: 2

简介
-------------------------------

配置文件系统是机器学习算法工具中的重要组成部分，而由于强化学习的复杂性，相关任务中包含的配置字段相比一般任务则会更加复杂。为了应对这个问题，DI-engine 基于“约定大于配置”（Convention over Configuration，简称为CoC）的思想，设计了一套基本原则和相关工具，具体包括:

- Python 配置原则
- 配置编译原则
- 默认配置与用户配置
- 强化学习任务核心配置字段分类


.. note::

    关于约定大于配置思想，进一步地了解可以参考中文博客 `设计杂谈(0x02)——约定大于配置 <https://zhuanlan.zhihu.com/p/540714858>`_

基本原则
------------

整体示意图如下：

.. image::
   images/config.png
   :align: center


Python 配置原则
^^^^^^^^^^^^^^^^^^
由于大部分机器学习程序都极为重视编程的灵活性和易用性，因此 DI-engine 中使用 Python 文件（即使用多级dict）来作为默认配置文件，便于兼容各种特殊需求。之前使用 yaml 和 json 的用户，也可以非常简单地迁移到 Python 配置。
不过，对于非核心模块之外的部分，例如强化学习模拟环境中的通信接口，使用原生的配置也可，并无特殊限制。

配置编译原则
^^^^^^^^^^^^^^^^^
对于整个训练程序，DI-engine 以 ``compile_config``, ``get_instance_cfg`` 等类似的工具函数为分界线，在分界线之前，用户可通过使用默认配置，代码中修改配置，命令行输入配置等多种手段自定义所需的配置内容，
但在分界线之后，所有的配置文件将被固定，后续不允许任何改动，训练程序也将依照此时的配置创建相应的功能模块和训练 pipeline。这种规则即为配置编译原则，即分界线之前为配置的编译生成期，而之后则会配置的运行使用期。
另外，在工具函数中，也会将最终生成的配置文件导出存储在实验目录文件夹下（一般命名为 ``total_config.py`` ），用户可通过这个文件来检查配置文件设定的有效性，或是直接使用该文件复现原来的实验。

.. note::
   一般来说，实验完成后会在以 ``exp_name`` 配置字段命名的路径文件夹存储相关信息，其中宏观配置信息文件为 ``total_config.py`` ，经过格式转换直接可以用于训练的配置文件为 ``formatted_total_config.py`` ，用户可以
   直接从这个文件中 import 相应的 ``main_config`` 和 ``create_config`` 传递给训练入口函数，从而开启实验。

.. tip::
   为了保证配置编译原则的有效性，使用者尽量不要在模块代码中使用例如 ``cfg.get('field_xxx', 'xxx')`` 这样的操作，这样的操作会破坏编译原则，并且无法记录和追踪相应的修改。

默认配置与用户配置
^^^^^^^^^^^^^^^^^^^^^
另一方面，由于强化学习训练程序环境 MDP 建模设置，算法调优设置，训练系统效率设置等多方面的配置内容，往往会产生大量可修改的配置文件字段。但不同的配置字段有着不同的使用频率，甚至说在大部分情况下，多数配置字段
都是不需要用户修改的，因此，DI-engine 中引入了默认配置的概念，即强化学习中的核心概念（例如 Policy，Env，Buffer）往往都具有自己的默认配置（default config），完整的定义和解释说明都可以在相应类的定义中找到。具体
使用时，用户只需要指定当前情况必须要修改或添加的配置字段即可，而在上文提到的分界线函数中（例如 ``compile_config`` ），会将默认配置和用户配置进行编译融合，构成最终的配置文件，具体融合的规则如下：

  - 默认配置中有，用户配置中没有的字段，使用默认值
  - 默认配置中没有，用户配置中有的字段，使用用户指定值
  - 默认配置和用户配置中同时存在的字段，使用用户指定值

强化学习任务核心配置字段分类
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
整体来说，DI-engine 中的强化学习任务所需的配置字段主要分类两大类：

- 第一类是在训练 pipeline 中被各个 worker/middleware 共享的对象，例如算法策略（Policy），环境（Env），奖励模型（Reward Model），数据队列（Buffer）等等，它们各自拥有自己的默认配置。在整体配置文件中，它们也
  保持平级关系并行放置，每个对象所有配置字段的含义可以参考相应类的定义，类变量 ``config`` 和类方法 ``default_config`` 指明了默认配置的内容和调用方法。

- 第二类是在训练 pipeline 中具体执行各类任务的 worker/middleware，例如学习器（Learner），数据收集器（Collector）等等，他们一般对应的参数较少，可在训练入口函数中直接指定相关参数值，或是依附在整体配置文件的全局
  区域中，具体实现并无特殊要求，保持代码清晰易用即可。

不过，在主体配置 ``main_config`` 之外，还存在帮助快速创建训练入口的创建配置 ``create_config`` ，这部分创建配置仅在 ``serial_pipeline_xxx`` 系列快速训练入口中使用，如果是自定义训练入口函数的用户可选择忽略。创建
配置中需要指定相应的类别名（type）和引入模块所需的路径（import_names）。

.. note::
   由于历史版本原因，在 Policy 字段中还定义了许多子域，例如 learn，collect，eval，other 等等，在最新版本的 DI-engine (>=0.4.7) 中已经去除了对于这些定义的强制依赖，使用或不使用这些结构均可，只要在策略中将配置
   字段和相应的生效代码段对应起来即可。


其他工具
^^^^^^^^^^
DI-engine 还提供了一些关于配置文件存储，格式化的相关工具，具体信息可以参考代码 `ding/config <https://github.com/opendilab/DI-engine/tree/main/ding/config>`_

配置文件示例解析
--------------------

下方是一个具体的 DI-engine 中的配置示例，其含义是在 CartPole 环境上训练 DQN 智能体（即快速上手文档中用到的例子），具体配置内容和相关字段解释如下：

.. code:: python

    from easydict import EasyDict
    

    cartpole_dqn_config = dict(
        exp_name='cartpole_dqn_seed0',
        env=dict(
            collector_env_num=8,
            evaluator_env_num=5,
            n_evaluator_episode=5,
            stop_value=195,
        ),
        policy=dict(
            cuda=False,
            model=dict(
                obs_shape=4,
                action_shape=2,
            ),
            nstep=1,
            discount_factor=0.97,
            learn=dict(
                update_per_collect=5,
                batch_size=64,
                learning_rate=0.001,
            ),
            collect=dict(n_sample=8),
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
    )
    cartpole_dqn_create_config = EasyDict(cartpole_dqn_create_config)
    create_config = cartpole_dqn_create_config

    if __name__ == "__main__":
        # or you can enter `ding -m serial -c cartpole_dqn_config.py -s 0`
        from ding.entry import serial_pipeline
        serial_pipeline((main_config, create_config), seed=0)


整个配置文件可分为两部分，``main_config`` 和 ``create_config`` ，主体配置（main_config) 中包含实验名（exp_name），环境（env）和策略（policy）三部分，分别指定了跟当前任务最相关，以及最常修改的部分配置字段。
而创建配置中则指定了环境、环境管理器和策略的类型，从而使得可以直接使用下方的快速训练入口 ``serial_pipeline`` 运行这个配置文件。如果是使用 task/middleware 方式自定义训练入口，则直接加载这份配置文件即可。
