Welcome to DI-engine's documentation!
=====================================

`中文 </zh_CN/latest/index_zh.html>`_

.. image::
   images/di_engine_logo.svg
   :width: 300
   :align: center


Overview
------------
DI-engine is a generalized Decision Intelligence engine. It supports most basic deep reinforcement learning (DRL) algorithms,
such as DQN, PPO, SAC, and domain-specific algorithms like QMIX in multi-agent RL, GAIL in inverse RL, and RND in exploration problems.
The whole supported algorithms introduction can be found in `Algorithm <./feature/algorithm_overview.html>`_.

For scalability, DI-engine supports three different training pipelines:

  - ``serial``

    - feature: single-machine, learner-collector loop executes sequentially
    - usage: academic research
  - ``parallel``

    - feature: single-machine, learner and collector execute in parallel
    - usage: speed up serial pipeline and introduction to the whole distributed training
  - ``dist``

    - feature: for GPU and CPU mixed computing clusters, learner-collector distributed execution
    - usage: large scale AI decision application, such as AlphaStar league training



.. image::
   images/system_layer.png

Main Features
--------------

  * DI-zoo: High-performance DRL algorithm zoo, algorithm support list. `Link <feature/algorithm_overview.html>`_
  * Generalized decision intelligence algorithms: DRL family, IRL family, MARL family, searching family(MCTS) and etc.
  * Customized DRL demand implementation, such as Inverse RL/RL hybrid training; Multi-buffer training; League self-play training
  * Large scale DRL training demonstration and application
  * Various efficiency optimization modules: DI-hpc, DI-store, EnvManager, DataLoader
  * k8s support, DI-orchestrator k8s cluster scheduler for dynamic collectors and other services


To get started, take a look over the `quick start <./quick_start/index.html>`_ and `API documentation <./api_doc/index.html>`_.
For RL beginners, DI-engine advises you to refer to `hands-on RL <hands_on/index.html>`_ for more discussion.
If you want to deeply customize your algorithm and application with DI-engine, also checkout `key concept <./key_concept/index.html>`_ and `Feature <./feature/index.html>`_.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation/index
   quick_start/index
   key_concept/index
   intro_rl/index
   hands_on/index
   env_tutorial/index
   distributed/index
   best_practice/index
   api_doc/index

   faq/index
   feature/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   guide/index
   tutorial_dev/index
   architecture/index
   specification/index
