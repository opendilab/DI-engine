TD3BC
^^^^^^^

Overview
---------

TD3BC, proposed in the 2021 paper `A Minimalist Approach to Offline Reinforcement Learning <https://arxiv.org/abs/2106.06860>`_,
is a simple approach to offline RL where **only two changes** are made to TD3: a weighted behavior cloning loss is added to the policy update and the states are normalized.
Unlike competing methods there are no changes to architecture or underlying hyperparameters.
The resulting algorithm is a simple baseline that is easy to implement and tune, while more than halving the overall run time by removing the additional computational overhead of previous methods.

.. figure:: images/td3bc_paper_table1.png
   :align: center

   Implementation changes offline RL algorithms make to the underlying base RL algorithm. † corresponds
   to details that add additional hyperparameter(s), and ‡ corresponds to ones that add a computational cost.
   `Ref <https://arxiv.org/abs/2106.06860>`_

Quick Facts
-----------
1. TD3BC is an **offline** RL algorithm.

2. TD3BC is based on **TD3** and **behavior cloning**.

Key Equations or Key Graphs
---------------------------
TD3BC simply consists to add a behavior cloning term to TD3 in order to regularize the policy:

.. math::
    \begin{aligned}
    \pi = \arg\max_{\pi} \mathbb{E}_{(s, a) \sim D} [ \lambda Q(s, \pi(s)) - (\pi(s)-a)^2 ]
    \end{aligned}

:math:`(\pi(s)-a)^2` is the behavior cloning term acts as a regularizer and aims to push the policy towards favoring
actions contained in the dataset. The hyperparameter :math:`\lambda` is used to control the strength of the
regularizer.

Assuming an action range of [−1, 1], the BC term is at most 4, however the range of Q will be a function of the scale of
the reward. Consequently, the scalar :math:`\lambda` can be defined as:

.. math::
    \begin{aligned}
    \lambda = \frac{\alpha}{\frac{1}{N}\sum_{s_i, a_i}|Q(s_i, a_i)|}
    \end{aligned}

which is simply a normalization term based on the average absolute value of Q over mini-batches. This formulation has
also the benefit of normalizing the learning rate across tasks since it is dependent on the scale of Q. The default
value for :math:`\alpha` is 2.5.

Additionally, all the states in each mini-batch are normalized, such that they have mean 0 and standard deviation 1.
This normalization improves the stability of the learned policy.


Implementations
----------------
The default config is defined as follows:

.. autoclass:: ding.policy.td3_bc.TD3BCPolicy

Model
~~~~~~~~~~~~~~~~~
Here we provide examples of `ContinuousQAC` model as default model for `TD3BC`.

.. autoclass:: ding.model.ContinuousQAC
    :members: forward, compute_actor, compute_critic
    :noindex:


Benchmark
-----------


+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_ha <https:// |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|                     |                 |                                                     |DI-engine/blob/main/dizoo/| d3rlpy(12124)        |
|Halfcheetah          |  13037          |.. image:: images/benchmark/halfcheetah_td3bc.png    |d4rl/config/halfcheetah_  |                      |
|                     |                 |                                                     |td3bc_medium_expert       |                      |
|(Medium Expert)      |                 |                                                     |_config.py>`_             |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_w <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|Walker2d             |                 |                                                     |DI-engine/blob/main/dizoo/| d3rlpy(5108)         |
|                     |  5066           |.. image:: images/benchmark/walker2d_td3bc.png       |d4rl/config/walker2d_     |                      |
|(Medium Expert)      |                 |                                                     |td3bc_medium_expert       |                      |
|                     |                 |                                                     |_config.py>`_             |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_ho <https:// |                      |
|                     |                 |                                                     |github.com/opendilab/     | d3rlpy(3690)         |
|Hopper               |                 |                                                     |DI-engine/blob/main/dizoo/|                      |
|                     |  3653           |.. image:: images/benchmark/hopper_td3bc.png         |d4rl/config/hopper_sac_   |                      |
|(Medium Expert)      |                 |                                                     |td3bc_medium_expert       |                      |
|                     |                 |                                                     |_config.py>`_             |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+


+---------------------+-----------------+----------------+---------------+----------+----------+
| environment         |random           |medium replay   |medium expert  |medium    |expert    |
+=====================+=================+================+===============+==========+==========+
|                     |                 |                |               |          |          |
|Halfcheetah          |1592             |5192            |13037          |5257      |13247     |
|                     |                 |                |               |          |          |
+---------------------+-----------------+----------------+---------------+----------+----------+
|                     |                 |                |               |          |          |
|Walker2d             |345              |1724            |3653           |3268      |3664      |
|                     |                 |                |               |          |          |
+---------------------+-----------------+----------------+---------------+----------+----------+
|                     |                 |                |               |          |          |
|Hopper               |985              |2317            |5066           |3826      |5232      |
|                     |                 |                |               |          |          |
+---------------------+-----------------+----------------+---------------+----------+----------+

**Note**: the D4RL environment used in this benchmark can be found `here <https://github.com/rail-berkeley/d4rl>`_.

References
-----------
- Scott Fujimoto, Shixiang Shane Gu: “A Minimalist Approach to Offline Reinforcement Learning”, 2021; [https://arxiv.org/abs/2106.06860 arXiv:2106.06860].

- Scott Fujimoto, Herke van Hoof, David Meger: “Addressing Function Approximation Error in Actor-Critic Methods”, 2018; [http://arxiv.org/abs/1802.09477 arXiv:1802.09477].

Other Public Implementations
----------------------------

- `Official implementation`_
- d3rlpy_

.. _`Official implementation`: https://github.com/sfujim/TD3_BC
.. _d3rlpy: https://github.com/takuseno/d3rlpy
