Introduction
===============================

What is DI-engine?
-------------------------------

DI-engine is a decision intelligence engine for PyTorch and JAX built by a group of enthusiastic researchers and engineers.

It provides python-first and asynchronous-native task and middleware abstractions, and modularly integrates several of the most important decision-making concepts: Env, Policy and Model. \
Based on the above mechanisms, DI-engine supports various deep reinforcement learning (DRL) algorithms with superior performance, high efficiency, well-organized documentation and unittest, \
which will provide you with the most professional and convenient assistance for your reinforcement learning algorithm research and development work, mainly including:

1. Comprehensive algorithm support, such as DQN, PPO, SAC, and many related algorithms for research subfields - \
   QMIX for multi-intelligent reinforcement learning, GAIL for inverse reinforcement learning, RND for exploration problems, etc.

2. User-friendly interface, we abstract most common objects in RL tasks, such as environments, policies, \
   and encapsulate complex reinforcement learning processes into middleware, allowing you to build your own learning process as you wish.

3. Flexible scalability, using the integrated messaging components and event programming interfaces within the framework, \
   you can flexibly scale your basic research work to industrial-grade large-scale training clusters, \
   such as StarCraft Intelligence `DI-star <https://github.com/opendilab/DI-star>`_.


DI-engine aims to standardize different Decision Intelligence environments and applications, supporting both academic research and prototype applications. \
Various training pipelines and customized decision AI applications are also supported, such as `StarCraftII <https://github.com/opendilab/DI-star>`_, `Auto-driving <https://github.com/opendilab/DI-drive>`_, `Traffic Light Control <https://github.com/opendilab/DI-smartcross>`_ and `Biological Sequence Prediction <https://github.com/opendilab/DI-bioseq>`_.

On the low-level end, DI-engine comes with a set of highly re-usable modules, including `RL optimization functions <https://github.com/opendilab/DI-engine/tree/main/ding/rl_utils>`_, `PyTorch utilities <https://github.com/opendilab/DI-engine/tree/main/ding/torch_utils>`_ and `auxiliary tools <https://github.com/opendilab/DI-engine/tree/main/ding/utils>`_.

.. image::
   ../images/system_layer.png

Key Concepts
-------------------------------

If you are not familiar with reinforcement learning, you can go to our `reinforcement learning tutorial <../10_concepts/index_zh.html>`_ \
for a glimpse into the wonderful world of reinforcement learning.

If you have already been exposed to reinforcement learning, you will already be familiar with the basic interaction objects of reinforcement learning: \
**environments** and **agents (or the policies that make them up)**.

Instead of creating more concepts, the DI-engine abstracts the complex interaction logic between the two into declarative middleware, \
such as **collect**, **train**, **evaluate**, and **save_ckpt**. You can adapt each part of the process in the most natural way.

Using the DI-engine will be very easy, in the `quickstart <... /01_quickstart/index_zh.html>`_, \
we will show you how to quickly build a classic reinforcement learning process using DI-engine with a simple example.
