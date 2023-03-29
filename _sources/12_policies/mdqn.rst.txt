MDQN
^^^^^^^

Overview
---------
MDQN was proposed in `Munchausen Reinforcement Learning <https://arxiv.org/abs/2007.14430>`_. They call this general approach “Munchausen Reinforcement Learning”
(M-RL), as a reference to a famous passage of The Surprising Adventures of Baron Munchausen by Raspe, where the Baron pulls himself out of a swamp by pulling on his own hair.
From a practical point of view, the key difference between MDQN and DQN is that MDQN adding a scaled log-policy to the immediate reward on the Soft-DQN which is is an extension of the traditional DQN algorithm with max entropy.

Quick Facts
-------------
1. MDQN is a **model-free** and **value-based** RL algorithm.

2. MDQN only support **discrete** action spaces.

3. MDQN is an **off-policy** algorithm.

4. MDQN uses **eps-greedy** for exploration.

5. MDQN increased the **action gap**, and has implicit **KL regularization**.


Key Equations or Key Graphs
---------------------------
The target Q value used in MDQN is:

.. math::

   \hat{q}_{\mathrm{m} \text {-dqn }}\left(r_t, s_{t+1}\right)=r_t+\alpha \tau \ln \pi_{\bar{\theta}}\left(a_t \mid s_t\right)+\gamma \sum_{a^{\prime} \in A} \pi_{\bar{\theta}}\left(a^{\prime} \mid s_{t+1}\right)\left(q_{\bar{\theta}}\left(s_{t+1}, a^{\prime}\right)-\tau \ln \pi_{\bar{\theta}}\left(a^{\prime} \mid s_{t+1}\right)\right)
   

For the log policy  :math:`\alpha \tau \ln \pi_{\bar{\theta}}\left(a_t \mid s_t\right)`  we used the following formula to calculate 

.. math::

   \tau \ln \pi_{k}=q_k-v_k-\tau \ln \left\langle 1, \exp \frac{q_k-v_k}{\tau}\right\rangle

where  :math:`q_k`  is the  `target_q_current` in our code. For the max entropy part  :math:`\tau \ln \pi_{\bar{\theta}}\left(a^{\prime} \mid s_{t+1}\right)` , we use the same formula to calculate where the  where  :math:`q_{k+1}` is the  `target_q`  in our code

And we replace  :math:`\tau \ln \pi(a \mid s)` by :math:`[\tau \ln \pi(a \mid s)]_{l_0}^0`` because log-policy term is not bounded, and can cause numerical issues if the policy becomes too close to
deterministic. 

And we replace  :math:`\pi_{\bar{\theta}}\left(a^{\prime} \mid s_{t+1}\right)` by :math:`softmax(q-v)` which official implementations used but not mentationed in their paper.

And we test action at asterix and get the same result as paper that MDQN could increase the action gap.

.. image:: images/action_gap.png
   :align: center

Pseudo-code
---------------
.. image:: images/mdqn.png
   :align: center

Extension
---------------
- TBD


Implementations
----------------
The default config of MDQNPolicy is defined as follows:

.. autoclass:: ding.policy.mdqn.MDQNPolicy
   :noindex:

The td error interface MDQN used is defined as follows:

.. autofunction:: ding.rl_utils.td.m_q_1step_td_error
   :noindex:


Benchmark
-----------

.. list-table:: Benchmark and comparison of mdqn algorithm
   :widths: 25 15 30 15 15
   :header-rows: 1

   * - environment
     - best mean reward
     - evaluation results
     - config link
     - comparison
   * - | Asterix 
       | (Asterix-v0)
     - 8963
     - .. image:: images/benchmark/mdqn_asterix.png 
     - `config_link_asterix <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/asterix/asterix_mdqn_config.py>`_
     - | sdqn(3513) paper(1718) dqn(3444)
   * - | SpaceInvaders
       | (SpaceInvaders-v0)
     - 2211
     - .. image:: images/benchmark/mdqn_spaceinvaders.png
     - `config_link_spaceinvaders <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/spaceinvaders/spaceinvaders_mdqn_config.py>`_
     - | sdqn(1804) paper(2045) dqn(1228)
   * - | Enduro
       | (Enduro-v4)
     - 1003
     - .. image:: images/benchmark/mdqn_enduro.png
     - `config_link_enduro <https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/enduro/enduro_mdqn_config.py>`_
     - | sdqn(986.1) paper(1171) dqn(986.4)



Key difference between our config and paper config: 

-  we collect 100 samples, train 10 times. In the paper, they collect 4 samples, train 1 time.
-  we update target network for every 500 iterations, they update target network for every 2000 iterations.
-  the epsilon we used for exploration is from 1 to 0.05, their epsilon is from 0.01 to 0.001.

P.S.:

-  The above results are obtained by running the same configuration on seed 0
-  For the discrete action space algorithm like DQN, the Atari environment set is generally used for testing, and Atari environment is generally evaluated by the highest mean reward training 10M ``env_step``. For more details about Atari, please refer to `Atari Env Tutorial <../env_tutorial/atari.html>`_ .


Reference
----------

- Vieillard, Nino, Olivier Pietquin, and Matthieu Geist. "Munchausen reinforcement learning." Advances in Neural Information Processing Systems 33 (2020): 4235-4246.

