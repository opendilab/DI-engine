欢迎来到DI-engine中文文档
=====================================

`English </en/latest/index.html>`_

.. image::
   images/di_engine_logo.svg
   :width: 300
   :align: center


概述
------------
DI-engine是一个通用决策智能平台。它支持大多数常用的深度强化学习算法，例如DQN，PPO，SAC以及许多研究子领域的相关算法——多智能体强化学习
中的QMIX，逆强化学习中的GAIL，探索问题中的RND。所有现已支持的算法和相关算法性能介绍可以查看 `算法概述 <./feature/algorithm_overview.html>`_

为了在各种计算尺度上的通用性和扩展性，DI-engine支持3种不同的训练模式：

  - ``单机串行``

    - 特点：单台机器，学习器(learner)和数据收集器(collector)串行交替执行
    - 用途：学术研究和算法验证
  - ``单机并行``

    - 特点：单台机器，学习器(learner)和数据收集器(collector)异步并行执行
    - 用途：加速串行训练，并作为分布式训练的介绍和过渡
  - ``分布式并行``

    - 特点：GPU和CPU混合计算集群，学习器(learner)和数据收集器(collector)异步并行执行
    - 用途：大规模决策AI训练系统，例如针对星际争霸2的智能体训练 ``DI-star``



.. image::
   images/system_layer.png

核心特点
--------------

  * DI-zoo：高性能深度强化学习算法库，具体信息可以参考 `传送门 <feature/algorithm_overview.html>`_
  * 最全最广的决策AI算法实现：深度强化学习算法族，逆强化学习算法族，多智能体强化学习算法族，基于搜索的算法（例如蒙特卡洛树搜索）等等
  * 支持各种定制化算法实现，例如强化学习/逆强化学习混合训练；多数据队列训练；联盟自对战博弈训练
  * 支持大规模深度强化学习训练和评测
  * 多种效率优化组件：``DI-hpc`` 高性能算子库，``DI-store`` 多机共享内存商店，并行环境管理器，数据加载器
  * 支持k8s容器虚拟化，``DI-orchestrator`` 提供了一整套强化学习训练的相关支持服务，支持资源管理和动态调度

作为初学者，可以首先参考 `快速开始 <./quick_start/index.html>`_ 来完成第一个决策AI智能体的训练入门，并可查阅 `API documentation <./api_doc/index.html>`_ 了解具体模块信息。
对于想了解强化学习算法原理和实现的使用者，建议详细阅读 `动手学RL <hands_on/index.html>`_ 部分了解更多细节。
如果你想深度定制化自己的算法和应用，可以查看 `核心概念 <./key_concept/index.html>`_ 和 `特性介绍 <./feature/index.html>`_ 两个部分的文档。

.. toctree::
   :maxdepth: 2
   :caption: 使用者指南

   installation/index_zh
   quick_start/index_zh
   key_concept/index
   intro_rl/index_zh
   hands_on/index_zh
   env_tutorial/index_zh
   distributed/index_zh
   best_practice/index_zh
   api_doc/index

   faq/index_zh
   feature/index_zh

.. toctree::
   :maxdepth: 2
   :caption: 开发者指南

   guide/index_zh
   tutorial_dev/index
   architecture/index
   specification/index_zh
