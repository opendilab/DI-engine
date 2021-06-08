A2C
^^^^^^^

Overview
---------
A2C(advantage actor critic) is a actor-critic RL algorithm, where the policy gradient is combined with advantage function to reduce the variance.

Quick Facts
-----------
1. A2C is a **model-free** and **policy-based** RL algorithm.

2. A2C supports both **discrete** and **continuous action spaces**.

3. A2C supports **off-policy** mode and **on-policy** mode.

4. A2C can be equipped with RNN.

5. The nerveX implementation of DQN supports **multi-discrete** action space.

Key Equations or Key Graphs
-------------
A2C use advantage esitimation in the policy gradient:
.. image:: images/a2c_pg.png
where the n-step advantage function is defined:
.. image:: images/nstep_adv.png

Pseudo-code
-----------
.. image:: images/A2C.png

.. note::
   Different from Q-learning, A2C(and other actor critic methods) alternates between policy estimation and policy improvement.

Extensions
-----------
A2C can be combined with:
    - multi-step learning
    - RNN
    - GAE
        GAE is proposed in `High-Dimensional Continuous Control Using Generalized Advantage Estimation <https://arxiv.org/abs/1506.02438>`, it uses exponentially-weighted average of different steps of advantage estimators, to make trade-off between the variance and bias of the estimation of the advantage:
        .. image:: images/gae.png
           :scale: 70 %


Implementation
------------
The default config is defined as follows:

    .. autoclass:: nervex.policy.a2c.A2CPolicy

The network interface DQN used is defined as follows:

    * TODO

The Benchmark result of A2C implemented in nerveX is shown in `Benchmark <../feature/algorithm_overview.html>`_
