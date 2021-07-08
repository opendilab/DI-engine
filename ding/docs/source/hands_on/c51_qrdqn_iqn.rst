C51
^^^^^^^

Overview
---------
C51 was first proposed in `A Distributional Perspective on Reinforcement Learning <https://arxiv.org/abs/1707.06887>`_, different from previous works, C51 evaluates the complete distribution of a q-value rather than only the expectation. The authors designed a distributional Bellman operator, which preserves multimodality in value distributions and is believed to achieve more stable learning and mitigates the negative effects of learning from a non-stationary policy.

Quick Facts
-----------
1. C51 is a **model-free** and **value-based** RL algorithm.

2. C51 only **support discrete action spaces**.

3. C51 is an **off-policy** algorithm.

4. Usually, C51 use **eps-greedy** or **multinomial sample** for exploration.

5. C51 can be equipped with RNN.

Pseudo-code
------------
.. image:: images/C51.png
   :align: center
   :scale: 30%

.. note::
   C51 models the value distribution using a discrete distribution, whose support set are N atoms: :math:`z_i = V_min + i * delta, i = 0,1,...,N-1` and :math:`delta = (V_\max - V_\min) / N`. Each atom :math:`z_i` has a parameterized probability :math:`p_i`. The Bellman update of C51 projects the distribution of :math:`r + \gamma * z_j^(t+1)` onto the distribution :math:`z_i^t`.

Key Equations or Key Graphs
----------------------------
The Bellman target of C51 is derived by projecting the returned distribution :math:`r + \gamma * z_j` onto the current distribution :math:`z_i`. Given a sample transition :math:`(x, a, r, x')`, we compute the Bellman update :math:`Tˆz_j := r + \gamma z_j` for each atom :math:`z_j`, then distribute its probability :math:`p_{j}(x', \pi(x'))` to the immediate neighbors :math:`p_{i}(x, \pi(x))`:

.. math::

   \left(\Phi \hat{T} Z_{\theta}(x, a)\right)_{i}=\sum_{j=0}^{N-1}\left[1-\frac{\left|\left[\hat{\mathcal{T}} z_{j}\right]_{V_{\mathrm{MIN}}}^{V_{\mathrm{MAX}}}-z_{i}\right|}{\Delta z}\right]_{0}^{1} p_{j}\left(x^{\prime}, \pi\left(x^{\prime}\right)\right)

Extensions
-----------
- C51s can be combined with:
   - PER (Prioritized Experience Replay)
   - Multi-step TD-loss
   - Double (target) network
   - Dueling head
   - RNN

Implementation
-----------------

.. tip::
      Our benchmark result of C51 uses the same hyper-parameters as DQN except the exclusive `n_atom` of C51, which is empirically set as 51.


The default config of C51 is defined as follows:

.. autoclass:: ding.policy.c51.C51Policy
   :noindex:

The network interface C51 used is defined as follows:

.. autoclass:: ding.model.template.q_learning.C51DQN
   :members: __init__, forward
   :noindex:

The bellman updates of C51 is implemented as:

The Benchmark result of C51 implemented in DI-engine is shown in `Benchmark <../feature/algorithm_overview_en.html>`_

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

Key Equations or Key Graphs
----------------------------
The quantile regression loss, for a quantile tau in :math:`[0, 1]`, is an asymmetric convex loss function that penalizes overestimation errors with weight :math:`\tau` and underestimation errors with weight :math:`1−\tau`. For a distribution Z, and a given quantile tau, the value of the quantile function :math:`F_Z^−1(\tau)` may be characterized as the minimizer of the quantile regression loss:

.. math::

   \begin{array}{r}
   \mathcal{L}_{\mathrm{QR}}^{\tau}(\theta):=\mathbb{E}_{\hat{z} \sim Z}\left[\rho_{\tau}(\hat{Z}-\theta)\right], \text { where } \\
   \rho_{\tau}(u)=u\left(\tau-\delta_{\{u<0\}}\right), \forall u \in \mathbb{R}
   \end{array}

Pseudo-code
-------------
.. image:: images/QRDQN.png
   :align: center
   :scale: 25%

.. note::
   The quantile huber loss is applied during the Bellman update of QRDQN.

Extensions
-----------
QRDQN can be combined with:
  - PER (Prioritized Experience Replay)
  - Multi-step TD-loss
  - Double (target) network
  - RNN

Implementation
----------------

.. tip::
      Our benchmark result of QRDQN uses the same hyper-parameters as DQN except the QRDQN's exclusive hyper-parameter, `the number of quantiles`, which is empirically set as 32.

The default config of QRDQN is defined as follows:

.. autoclass:: ding.policy.qrdqn.QRDQNPolicy
   :noindex:

The network interface QRDQN used is defined as follows:

.. autoclass:: ding.model.template.q_learning.QRDQN
   :members: __init__, forward
   :noindex:

The bellman updates of QRDQN is implemented in the function ``qrdqn_nstep_td_error`` of ``ding/rl_utils/td.py``.

The Benchmark result of QRDQN implemented in DI-engine is shown in `Benchmark <../feature/algorithm_overview_en.html>`_

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

Key Equations
-------------
In implicit quantile networks, a sampled quantile tau is first encoded into an embedding vector via:

    .. math::

        \phi_{j}(\tau):=\operatorname{ReLU}\left(\sum_{i=0}^{n-1} \cos (\pi i \tau) w_{i j}+b_{j}\right)

Then the quantile embedding is element-wise multiplied by the embedding of the observation of the environment, and the subsequent fully-connected layers map the resulted product vector to the respective quantile value.

Key Graphs
-------------
The comparison among DQN, C51, QRDQN and IQN is shown as follows:

.. image:: images/dis_reg_compare.png
   :align: center
   :width: 800

Extensions
-----------
IQN can be combined with:
  - PER (Prioritized Experience Replay)

    .. tip::
        Whether PER improves IQN depends on the task and the training strategy.
  - Multi-step TD-loss
  - Double (target) Network
  - RNN

Implementation
------------------

.. tip::
      Our benchmark result of IQN uses the same hyper-parameters as DQN except the IQN's exclusive hyper-parameter, ``the number of quantiles``, which is empirically set as 32. The number of quantiles are not recommended to set larger than 64, which brings marginal gain and much more forward latency.

The default config of IQN is defined as follows:

.. autoclass:: ding.policy.iqn.IQNPolicy
   :noindex:

The network interface IQN used is defined as follows:

.. autoclass:: ding.model.template.q_learning.IQN
   :members: __init__, forward
   :noindex:

The bellman updates of IQN used is defined in the function ``iqn_nstep_td_error`` of ``ding/rl_utils/td.py``.

The Benchmark result of IQN implemented in DI-engine is shown in `Benchmark <../feature/algorithm_overview_en.html>`_


References
------------

(C51) Marc G. Bellemare, Will Dabney, Rémi Munos: “A Distributional Perspective on Reinforcement Learning”, 2017; arXiv:1707.06887. https://arxiv.org/abs/1707.06887

(QRDQN) Will Dabney, Mark Rowland, Marc G. Bellemare, Rémi Munos: “Distributional Reinforcement Learning with Quantile Regression”, 2017; arXiv:1710.10044. https://arxiv.org/pdf/1710.10044

(IQN) Will Dabney, Georg Ostrovski, David Silver, Rémi Munos: “Implicit Quantile Networks for Distributional Reinforcement Learning”, 2018; arXiv:1806.06923. https://arxiv.org/pdf/1806.06923
