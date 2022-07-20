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

4. There are two value networks in PPG.

5. In the implementation of DI-engine, we use two buffers for off-policy PPG, which are only different from maximum data usage limit (data ``max_use`` ).

Key Graphs
----------
PPG utilizes disjoint policy and value networks to reduce interference between objectives. The policy network includes an auxiliary value head which is used to distill the knowledge of value into the policy network, the concrete network architecture is shown as follows:

.. image:: images/ppg_net.png
   :align: center
   :height: 250

Key Equations
-------------
The optimization of PPG alternates between two phases, a policy phase and an auxiliary phase. During the policy phase, the policy network and the value network are updated similar to PPO. During the auxiliary phase, the value knowledge is distilled into the policy network with the joint loss:

.. math::

    L^{j o i n t}=L^{a u x}+\beta_{c l o n e} \cdot \hat{\mathbb{E}}_{t}\left[K L\left[\pi_{\theta_{o l d}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]\right]

The joint loss optimizes the auxiliary objective (distillation) while preserves the original policy with the KL-divergence restriction (i.e. the second item). And the auxiliary loss is defined as:

.. math::

    L^{a u x}=\frac{1}{2} \cdot \hat{\mathbb{E}}_{t}\left[\left(V_{\theta_{\pi}}\left(s_{t}\right)-\hat{V}_{t}^{\mathrm{targ}}\right)^{2}\right]


Pseudo-code
-----------

on-policy training procedure
==============================


The following flow charts show how PPG alternates between the policy phase and the auxiliary phase

.. image:: images/PPG.png
   :align: center
   :width: 600

.. note::

   During the auxiliary phase, PPG also takes the opportunity to perform additional training on the value network.

off-policy training procedure
==============================
DI-engine also implements off-policy PPG with two buffers with different data use constraint (``max_use``), which policy buffer offers data for policy phase while value buffer provides auxiliary phase's data. The whole training procedure is similar to off-policy PPO but execute additional auxiliary phase with a fixed frequency.

Extensions
-----------
- PPG can be combined with:

    * GAE or other advantage estimation method
    * Multi-buffer, different ``max_use``

- PPO (or PPG) + UCB-DrAC + PLR is one of the most powerful methods in procgen environment.

    * `PLR github repo <https://github.com/facebookresearch/level-replay>`_
    * `UCB-DrAC repo <https://github.com/rraileanu/auto-drac>`_

Implementation
---------------
The default config is defined as follows:

.. autoclass:: ding.policy.ppg.PPGPolicy
    :noindex:

The network interface PPG used is defined as follows:

.. autoclass:: ding.model.template.ppg.PPG
   :members: compute_actor_critic, compute_actor, compute_critic
   :noindex:



Benchmark
--------------

.. list-table:: Benchmark and comparison of PPG algorithm
   :widths: 25 15 30 15 15
   :header-rows: 1

   * - environment
     - best mean reward
     - evaluation results
     - config link
     - comparison
   * - | Pong
       | (PongNoFrameskip-v4)
     - 20
     - .. image:: images/benchmark/ppg_pong.png
     - `config_link_p <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_ppg_config.py>`_
     - | DI-engine PPO off-policy(20)
   * - | Qbert
       | (QbertNoFrameskip-v4)
     - 17775
     - .. image:: images/benchmark/ppg_qbert.png
     - `config_link_q <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/qbert/qbert_ppg_config.py>`_
     - | DI-engine PPO off-policy(16400)
   * - | SpaceInvaders
       | (SpaceInvadersNoFrame skip-v4)
     - 1213
     - .. image:: images/benchmark/ppg_spaceinvaders.png
     - `config_link_s <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/spaceinvaders/spaceinvaders_ppg_config.py>`_
     - | DI-engine PPO off-policy(1200)

References
-----------

Karl Cobbe, Jacob Hilton, Oleg Klimov, John Schulman: “Phasic Policy Gradient”, 2020; arXiv:2009.04416.


Other Public Implementations
------------------------------

- [openai](https://github.com/openai/phasic-policy-gradient)
