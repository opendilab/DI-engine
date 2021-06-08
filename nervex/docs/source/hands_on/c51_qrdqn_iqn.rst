C51
^^^^^^^

Overview
---------
C51 was first proposed in `A Distributional Perspective on Reinforcement Learning <https://arxiv.org/abs/1707.06887>`_, different from previous works, C51 evaluates the complete distribution of a q-value rather than only the expectation.

Quick Facts
-----------
1. C51 is a **model-free** and **value-based** RL algorithm.

2. C51 only **support discrete action spaces**.

3. C51 is an **off-policy** algorithm.

4. Usually, C51 use **eps-greedy** or **multinomial sample** for exploration.

5. C51 can be equipped with RNN.

6. The nerveX implementation of C51 supports **multi-discrete** action space.

Pseudo-code
------------
.. image:: images/C51.png
   :scale: 50 %

.. note::
   C51 models the value distribution using a discrete distribution, whose support set are N atoms: :math:`z_i = V_min + i * delta, i = 0,1,...,N-1` and :math:`delta = (V_\max - V_\min) / N`. Each atom :math:`z_i` has a parameterized probability :math:`p_i`. The Bellman update of C51 projects the distribution of :math:`r + \gamma * z_j^(t+1)` onto the distribution :math:`z_i^t`.

Extensions
-----------
- C51s can be combined with:
   - PER(Prioritized Experience Replay)
   - multi-step TD-loss
   - double(target) network
   - dueling head
   - RNN

Implementation
-----------------
The default config is defined as follows:

.. autoclass:: nervex.policy.c51.C51Policy

The bellman updates of C51 is implemented as:

    * TODO

The Benchmark result of C51 implemented in nerveX is shown in `Benchmark <../feature/algorithm_overview.html>`_

QRDQN
^^^^^^^

Overview
---------
QR(Quantile Regression)DQN was proposed in `Distributional Reinforcement Learning with Quantile Regression <https://arxiv.org/pdf/1710.10044>`_ and inherits the idea of learning the distribution of a q-value. Instead of approximate the distribution density function with discrete atoms, QRDQN, direct regresses a discrete set of quantiles of a q-value.

Quick Facts
-----------
1. QRDQN is a **model-free** and **value-based** RL algorithm.

2. QRDQN only support **discrete action spaces**.

3. QRDQN is an **off-policy** algorithm.

4. Usually, QRDQN use **eps-greedy** or **multinomial sample** for exploration.

5. QRDQN can be equipped with RNN.

6. The nerveX implementation of QRDQN supports **multi-discrete** action space.

Key Equations or Key Graphs
----------------------------
The quantile regression loss, for a quantile tau in :math:`[0, 1]`, is an asymmetric convex loss function that penalizes overestimation errors with weight tau and underestimation errors with weight 1−tau. For a distribution Z, and a given quantile tau, the value of the quantile function :math:`F_Z^−1(tau)` may be characterized
as the minimizer of the quantile regression loss:

.. image:: images/QR.png

Pseudo-code
-------------
.. image:: images/QRDQN.png

.. note::
   The quantile huber loss is applied during the Bellman update of QRDQN.

Extensions
-----------
QRDQN can be combined with:
  - PER(Prioritized Experience Replay)
  - multi-step TD-loss
  - double(target) network
  - RNN

Implementation
----------------
The default config is defined as follows:

.. autoclass:: nervex.policy.qrdqn.QRDQNPolicy

The bellman updates of QRDQN is implemented as:

    * TODO

The Benchmark result of QRDQN implemented in nerveX is shown in `Benchmark <../feature/algorithm_overview.html>`_

IQN
^^^^^^^

Overview
---------
IQN was proposed in `Implicit Quantile Networks for Distributional Reinforcement Learning <https://arxiv.org/pdf/1806.06923>`_. The key difference between IQN and QRDQN is that IQN introduces the implicit quantile network (IQN), a deterministic parametric function trained to re-parameterize samples from a base distribution, e.g. tau in U([0, 1]), to the respective quantile values of a target distribution, while QRDQN direct learns a fixed set of pre-defined quantiles.

Quick Facts
-----------
1. IQN is a **model-free** and **value-based** RL algorithm.

2. IQN only support **discrete action spaces**.

3. IQN is an **off-policy** algorithm.

4. Usually, IQN use **eps-greedy** or **multinomial sample** for exploration.

5. IQN can be equipped with RNN.

6. The nerveX implementation of IQN supports **multi-discrete** action space.

Key Equations
-------------
In implicit quantile networks, a sampled quantile tau is first encoded into an embedding vector via:

.. image:: images/IQN.png

Then the quantile embedding is element-wise multiplied by the embedding of the observation of the environment, and the subsequent fully-connected layers map the resulted product vector to the respective quantile value.


Extensions
-----------
IQN can be combined with:
  - PER(Prioritized Experience Replay)

    .. tip::
        Whether PER improves IQN depends on the task and the training strategy.
  - multi-step TD-loss
  - double(target) Network
  - RNN

Implementation
------------------
The default config is defined as follows:

.. autoclass:: nervex.policy.iqn.IQNPolicy

The bellman updates of IQN used is defined as follows:

    * TODO

The Benchmark result of IQN implemented in nerveX is shown in `Benchmark <../feature/algorithm_overview.html>`_
