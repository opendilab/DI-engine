PPG
^^^^^^^

Overview
---------
PPG was proposed in `Phasic Policy Gradient <https://arxiv.org/abs/2009.04416>`_. In prior methods, one must choose between using a shared network or separate networks to represent the policy and value function. Using separate networks avoids interference between objectives, while using a shared network allows useful features to be shared. PPG is able to achieve the best of both worlds by splitting optimization into two phases, one that advances training and one that distills features.

Quick Facts
-----------
1. PPG is a **model-free** and **policy-based** RL algorithm.

2. PPG supports both **discrete** and **continuous action spaces**.


3. PPG supports **off-policy** mode and **on-policy** mode.

4. PPG can be equipped with RNN.

Key Graphs
----------
PPG uses disjoint policy and value networks to reduce interference between objectives. The policy network includes an auxiliary value head which is used to distill the knowledge of value into the policy network.

.. image:: images/ppg_net.png
   :align: center
   :height: 300

Key Equations
-------------
The optimization of PPG alternates between two phases, a policy phase and an auxiliary phase. During the policy phase, the policy network and the value network are updated similar to PPO. During the auxiliary phase, the value knowledge is distilled into the policy network with the joint loss:

.. image:: images/ppg_joint.png
   :align: center
   :height: 40

The joint loss optimizes the auxiliary objective while preserves the original policy with the KL-divergence restriction. The auxiliary loss is defined as:

.. image:: images/ppg_aux.png
   :align: center
   :height: 70

Pseudo-code
-----------
The following flow charts shows how PPG alternates between the policy phase and the auxiliary phase.

.. image:: images/PPG.png
   :align: center
   :width: 800

.. note::
   During the auxiliary phase, PPG also takes the opportunity to perform additional training on the value network.

Extensions
-----------
- PPG can be combined with:
    * multi-step learning
    * RNN
    * GAE

Implementation
---------------
  The default config is defined as follows:

  .. autoclass:: ding.policy.ppg.PPGPolicy

  The network interface PPG used is defined as follows:

      * TODO

  The Benchmark result of PPG implemented in DI-engine is shown in `Benchmark <../feature/algorithm_overview.html>`_
