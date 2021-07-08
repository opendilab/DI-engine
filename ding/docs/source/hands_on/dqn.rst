DQN
^^^^^^^

Overview
---------
DQN was first proposed in `Playing Atari with Deep Reinforcement Learning <https://arxiv.org/abs/1312.5602>`_, which combines Q-learning with deep neural network. Different from the previous methods, DQN use a deep neural network to evaluate the q-values, which is updated via TD-loss along with gradient decent.

Quick Facts
-------------
1. DQN is a **model-free** and **value-based** RL algorithm.

2. DQN only support **discrete** action spaces.

3. DQN is an **off-policy** algorithm.

4. Usually, DQN use **eps-greedy** or **multinomial sample** for exploration.

5. DQN + RNN = DRQN.

6. The DI-engine implementation of DQN supports **multi-discrete** action space.

Key Equations or Key Graphs
---------------------------
The TD-loss used in DQN is:

.. math::

   L(w)=\mathbb{E}\left[(\underbrace{r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}, w\right)}_{\text {Target }}-Q(s, a, w))^{2}\right]

Pseudo-code
---------------
.. image:: images/DQN.png
   :align: center
   :scale: 55%

.. note::
   Compared with the vanilla version, DQN has been dramatically improved in both algorithm and implementation. In the algorithm part, **n-step TD-loss, PER, target network and dueling head** are widely used. For the implementation details, the value of epsilon anneals from a high value to zero during the training rather than keeps constant, according to env step(the number of policy interaction with env).

Extensions
-----------
DQN can be combined with:

    - PER (Prioritized Experience Replay)

        `PRIORITIZED EXPERIENCE REPLAY <https://arxiv.org/abs/1511.05952>`_ replaces the uniform sampling in replay buffer with a kind of special defined ``priority``, which is defined by various metrics, such as absolute TD error, the novelty of observation and so on. By this priority sampling, the convergence speed and performance of DQN can be improved a lot.

        One of implementation of PER is described:

        .. image:: images/PERDQN.png
           :align: center
           :scale: 60%

    - Multi-step TD-loss

        .. note::
           In the one-step setting, Q-learning learns :math:`Q(s,a)` with the Bellman update: :math:`r(s,a)+\gamma \mathop{max}\limits_{a^*}Q(s',a^*)`. While in the n-step setting the equation is :math:`\sum_{t=0}^{n-1}\gamma^t r(s_t,a_t) + \gamma^n \mathop{max}\limits_{a^*}Q(s_n,a^*)`. An issue about n-step for Q-learning is that, when epsilon greedy is adopted, the q value estimation is biased because the :math:`r(s_t,a_t)` at t>=1 are sampled under epsilon greedy rather than the policy itself. However, multi-step along with epsilon greedy generally improves DQN practically.

    - Double (target) network

      Double DQN, proposed in `Deep Reinforcement Learning with Double Q-learning <https://arxiv.org/abs/1509.06461>`_, is a kind of common variant of DQN. This method maintaines another Q-network, named target network, which is updated by the current netowrk by a fixed frequency(update times).

        Double DQN doesn't select the maximum q_value in the total discrete action space from the current network, but **first finds the action whose q_value is highest in the current network, then gets the q_value from the target network according to this selected action**. This variant can surpass the over estimation problem of target q_value, and reduce upward bias.

        .. note::
            The over estimation can be caused by the error of function approximation(neural network for q table), environment noise, numerical instability and other reasons.

    - Dueling head

      In `Dueling Network Architectures for Deep Reinforcement Learning <https://arxiv.org/abs/1511.06581>`_, dueling head architecture is utilized to implement the decomposition of state-value and advantage for taking each action, and use these two parts to construct the final q_value, which is better for evaluating the value of some states not related to action selection.

        The specific architecture is shown in the following graph:

        .. image:: images/Dueling_DQN.png
           :align: center
           :height: 300

    - RNN (DRQN, R2D2)

Implementations
----------------
The default config of DQNPolicy is defined as follows:

.. autoclass:: ding.policy.dqn.DQNPolicy
   :noindex:

The network interface DQN used is defined as follows:

.. autoclass:: ding.model.template.q_learning.DQN
   :members: __init__, forward
   :noindex:

The Benchmark result of DQN implemented in DI-engine is shown in `Benchmark <../feature/algorithm_overview_en.html>`_


Reference
----------

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller: “Playing Atari with Deep Reinforcement Learning”, 2013; arXiv:1312.5602. https://arxiv.org/abs/1312.5602
