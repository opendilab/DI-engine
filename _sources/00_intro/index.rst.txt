Introduction
===============================

What is DI-engine?
-------------------------------

DI-engine is a decision intelligence platform built by a group of enthusiastic researchers and engineers, \
that will provide you with the most professional and convenient assistance for your reinforcement learning algorithm research \
and development work, mainly including:

1. Comprehensive algorithm support, such as DQN, PPO, SAC, and many related algorithms for research subfields - \
   QMIX for multi-intelligent reinforcement learning, GAIL for inverse reinforcement learning, RND for exploration problems, etc.

2. User-friendly interface, we abstract most common objects in reinforcement learning tasks, such as environments, policies, \
   and encapsulate complex reinforcement learning processes into middleware, allowing you to build your own learning process as you wish.

3. Flexible scalability, using the integrated messaging components and event programming interfaces within the framework, \
   you can flexibly scale your basic research work to industrial-grade large-scale training clusters, \
   such as StarCraft Intelligence `DI-star <https://github.com/opendilab/DI-star>`_.

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
