Averaged-DQN
^^^^^^^^^^^^

Overview
---------
Averaged-DQN was proposed in `Averaged-DQN: Variance Reduction and Stabilization for Deep Reinforcement Learning <https://arxiv.org/abs/1611.01929>`_. Averaged-DQN is a simple extension to the DQN algorithm, based on averaging previously learned Q-values estimates, which leads to a more stable training procedure and improved performance by reducing approximation error variance in the target values. Compare to the Double DQN which also tackles with overestimation proiblem, averaged-DQN tries for a different road but reaches the same goal, provides more analysis to the reason behinds.

Quick Facts
-------------
1. Averaged-DQN is a **model-free** and **value-based** RL algorithm.

2. Averaged-DQN only support **discrete** action spaces.

3. Averaged-DQN is an **off-policy** algorithm.

4. Usually, Averaged-DQN uses **eps-greedy** for exploration.

Key Equations or Key Graphs
---------------------------
The Q function update difference can be divided into three parts:

.. math::

   \begin{aligned}
   \Delta_{i} &= Q(s,a; \theta_{i})- Q^{*}(s,a) \\
   &= \underbrace{Q(s,a; \theta_{i} )- y_{s,a}^{i}}_{Target Approximation Error}+ \underbrace{y_{s,a}^{i}- \hat{y}_{s,a}^{i}}_{Overestimation Error}+ \underbrace{\hat{y}^{i}_{s,a} - Q^{*}(s,a)}_{Optimality Difference}
   \end{aligned}


where the target network :math:`Q(s,a; \theta_{i})` is the *value function* of DQN at iteration :math:`i`,  :math:`y_{s,a}^i` is the *DQN target*, and :math:`\hat{y}_{s,a}^i` is the *true target*. Despite the optimality difference, both kinds of errors take a place in boosting overestimation.

The target approximation error (TAE) is the error when minimizling DQN loss between learned :math:`Q(s,a; \theta_i)` and :math:`y_{s,a}^i`. It could be caused by sub-optimality of :math:`\theta_i` due to the inexact minimization, the limited reprezentation power of DQN (mode error), and generalization error from unseen state-action pairs out of the finite ER buffer. Finally, it would cause a deviations from a policy to a worse one.

The overestimation error (OE) is first derived from TAE or random factors such as noise or initialization, but severely magnifies this start error by booststrap in TD updateing process. The Double DQN itackles the overestimation problem by breaking this booststrap mechanism and brings OE down.

Averaged-DQN focus on the original error TAE and try to control it into a minimum limits, which would also disadvantage the developments of OE. By averaging DQN's results with its meta parameter :math:`K` previous version, the value variance could be brought down to :math:`\frac{1}{K}` of DQN's.

Pseudo-code
---------------
.. image:: images/averaged-dqn.png
   :align: center

Compared to DQN, averaged-DQN needs :math:`K`-fold more forward passes through a Q-network and its previous :math:`K` versions while minimizing the DQN loss (line 7), but back-propagation updates remains the same as in DQN. The output of the algorithm is the averaged over the last :math:`K` previously learned Q-networks.

Extensions
-----------
Averaged-DQN can be combined with:
  - PER (Prioritized Experience Replay)
  - Double (target) Network

Implementation
----------------
The default config of AveragedDQNPolicy is defined as follows:

.. autoclass:: ding.policy.averaged_dqn.AveragedDQNPolicy
   :noindex:

The network interface AveragedDQN used is defined as follows:

.. autoclass:: ding.model.template.q_learning.DQN
   :members: forward
   :noindex:

Benchmark
-----------


.. .. list-table:: Benchmark and comparison of averaged-DQN algorithm
..    :widths: 25 15 30 15 15
..    :header-rows: 1

..    * - environment
..      - best mean reward
..      - evaluation results
..      - config link
..      - comparison
..    * - | Pong 
..        | (PongNoFrameskip-v4)
..      - 20
..      - .. image:: images/benchmark/pong_dqn.png
..      - `config_link_p <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_dqn_config.py>`_
..      - | Tianshou(20) Sb3(20)
..    * - | Qbert
..        | (QbertNoFrameskip-v4)
..      - 17966
..      - .. image:: images/benchmark/qbert_dqn.png
..      - `config_link_q <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/qbert/qbert_dqn_config.py>`_
..      - | Tianshou(7307) Rllib(7968) Sb3(9496)
..    * - | SpaceInvaders
..        | (SpaceInvadersNoFrameskip-v4)
..      - 2403
..      - .. image:: images/benchmark/spaceinvaders_dqn.png
..      - `config_link_s <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/spaceinvaders/spaceinvaders_dqn_config.py>`_
..      - | Tianshou(812) Rllib(1001) Sb3(622)

.. P.S.：

.. 1. The above results are obtained by running the same configuration on five different random seeds (0, 1, 2, 3, 4)
.. 2. For the discrete action space algorithm like DQN, the Atari environment set is generally used for testing (including sub-environments Pong), and Atari environment is generally evaluated by the highest mean reward training 10M ``env_step``. For more details about Atari, please refer to `Atari Env Tutorial <../env_tutorial/atari.html>`_ .


Reference
----------

- Oron Anschel, Nir Baram, and Nahum Shimkin. 2017. Averaged-DQN: variance reduction and stabilization for deep reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning - Volume 70 (ICML'17). JMLR.org, 176–185.

- Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." nature 518.7540 (2015): 529-533.

- Van Hasselt, H., Guez, A., & Silver, D. (2016, March). Deep reinforcement learning with double q-learning. In Proceedings of the AAAI conference on artificial intelligence (Vol. 30, No. 1).

