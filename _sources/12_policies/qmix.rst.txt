QMIX
^^^^^^^

Overview
---------
QMIX is proposed by `Rashid et al.(2018) <https://arxiv.org/abs/1803.11485>`_ for learning joint action value function conditioned on global state information in multi-agent centralized training,
and extracting decentralized policies from the centralized end-to-end framework.
QMIX employs a centralized neural network to estimate joint action values as a complex non-linear combination of per-agent action values based on local observations,
which provides a novel representation of centralized action-value functions and guarantees consistency between the centralized and decentralized policies.

QMIX is a non-linear extension of `VDN (Sunehag et al. 2017) <https://arxiv.org/abs/1706.05296>`_.
Compared to VDN, QMIX can represent more extra state information during training and a much richer class of action-value functions.

Quick Facts
-------------
1. QMIX uses the paradigm of **centralized training with decentralized execution**.

2. QMIX is a  **model-free**, **value-based**, **off-policy**, **multi-agent** RL method.

3. QMIX only support **discrete** action spaces.

4. QMIX considers a **partially observable** scenario in which each agent only obtains individual observations.

5. QMIX accepts **DRQN** as individual value network to tackle the **partially observable** issue.

6. QMIX represents the joint value function using an architecture consisting of **agent networks, a mixing network, a hyper-network**.
   The mixing network is a feed-forward neural network that takes the agent network outputs as input and mixes them monotonically, producing joint action values.
   The weights of the mixing network are produced by a separate hypernetwork.

Key Equations or Key Graphs
---------------------------


VDN and QMIX are representative methods that use the idea of factorization of the joint action-value function :math:`Q_{tot}` into individual ones :math:`Q_a` for decentralized execution.

In order to achieve centralized training with decentralized execution (CTDE), we need to ensure that a global :math:`argmax` performed on :math:`Q_{tot}` yields the same result as a set of individual :math:`argmax` operations performed on each :math:`Q_a`:

.. math::
  $\arg \max _{\boldsymbol{u}} Q_{\mathrm{tot}}(\boldsymbol{\tau}, \boldsymbol{u})=\left(\begin{array}{c}\arg \max _{u_{1}} Q_{1}\left(\tau_{1}, u_{1}\right) \\ \vdots \\ \arg \max _{u_{N}} Q_{N}\left(\tau_{n}, u_{N}\right)\end{array}\right)$

VDN factorizes the joint action-value function into a sum of individual action-value functions.  :math:`$Q_{\mathrm{tot}}(\boldsymbol{\tau}, \boldsymbol{u})=\sum_{i=1}^{N} Q_{i}\left(\tau_{i}, u_{i}\right)$`


QMIX extends this additive value factorization to represent the joint action-value function as a monotonic function. QMIX is based on monotonicity, a constraint on the relationship between joint action values :math:`Q_{tot}` and individual action values :math:`Q_a`.

.. math::
   \frac{\partial Q_{tot}}{\partial Q_{a}} \geq 0， \forall a \in A


The overall QMIX architecture including individual agent networks, the mixing network, the hyper-network:

.. image:: images/marl/qmix.png

QMIX trains the mixing network via minimizing the following loss:

.. math::
   y^{tot} = r + \gamma \max_{\textbf{u}^{’}}Q_{tot}(\tau^{'}, \textbf{u}^{'}, s^{'}; \theta^{-})

.. math::
   \mathcal{L}(\theta) = \sum_{i=1}^{b} [(y_{i}^{tot} - Q_{tot}(\tau, \textbf{u}, s; \theta)^{2}]

Each weight of the mixing network is produced by a independent hyper-network, which takes the global state as input and outputs the weight of one layer of the mixing network. More details can be found in the original paper `Rashid et al.(2018) <https://arxiv.org/abs/1803.11485>`_.

VDN and QMIX are methods that attempt to factorize  :math:`Q_tot` assuming additivity and monotonicity, respectively. Thus, joint action value functions satisfying those conditions would be well factorized by VDN and QMIX. However, there exist tasks
whose joint action-value functions do not meet the said conditions. `QTRAN (Son et al. 2019) <https://arxiv.org/abs/1905.05408>`_, proposes a factorization method, which is free from such structural constraints via transforming the original joint action-value function into an easily factorizable one.
QTRAN guarantees more general factorization than VDN or QMIX.

Implementations
----------------
The default config is defined as follows:

    .. autoclass:: ding.policy.qmix.QMIXPolicy
        :noindex:

The network interface QMIX used is defined as follows:

    .. autoclass:: ding.model.template.QMix
        :members: forward
        :noindex:


Benchmark
-----------


..
    +---------------------+-----------------------------------------------------+----------------------------------+
    | SMAC Map            | evaluation results                                  | config link                      ｜
    +=====================+=====================================================+==================================+
    |                     |                                                     |`config_link_MMM <https://        |
    |                     |                                                     |github.com/opendilab/             |
    |                     |                                                     | DI-engine/tree/main/dizoo/       |
    |MMM                  |.. image:: images/benchmark/QMIX_MMM.png             |smac/config/smac_MMM_qmix         |
    |                     |                                                     |_config.py>`_                     |
    |                     |                                                     |                                  |
    +---------------------+-----------------------------------------------------+----------------------------------+
    |                     |                                                     |`config_link_3s5z <https://       |
    |                     |                                                     |github.com/opendilab/             |
    |3s5z                 |.. image:: images/benchmark/QMIX_3s5z.png            | DI-engine/tree/main/dizoo/       |
    |                     |                                                     |smac/config/smac_3s5z_qmix        |
    |                     |                                                     |_config.py>`_                     |
    +---------------------+-----------------------------------------------------+----------------------------------+
    |                     |                                                     |`config_link_MMM2 <https://       |
    |                     |                                                     |github.com/opendilab/             |
    |MMM2                 |.. image:: images/benchmark/QMIX_MMM2.png            | DI-engine/tree/main/dizoo/       |
    |                     |                                                     |smac/config/smac_MMM2_qmix        |
    |                     |                                                     |_config.py>`_                     |
    +---------------------+-----------------------------------------------------+----------------------------------+
    |                     |                                                     |`config_link_5m6m <https://       |
    |                     |                                                     |github.com/opendilab/             |
    |5m6m                 |.. image:: images/benchmark/QMIX_5m6m.png            | DI-engine/tree/main/dizoo/       |
    |                     |                                                     |smac/config/smac_5m6m_qmix        |
    |                     |                                                     |_config.py>`_                     |
    +---------------------+-----------------------------------------------------+----------------------------------+
    |                     |                                                     |`config_link_2c64zg <https://     |
    |                     |                                                     |github.com/opendilab/             |
    |2c_vs_64zg           |.. image:: images/benchmark/QMIX_2c_vs_64zg.png      | DI-engine/tree/main/dizoo/       |
    |                     |                                                     |smac/config/smac_2c64zg_qmix      |
    |                     |                                                     |_config.py>`_                     |
    +---------------------+-----------------------------------------------------+----------------------------------+
    |                     |                                                     |`config_link_3s5z3s6z <https://   |
    |                     |                                                     |github.com/opendilab/             |
    |3s5z_vs_3s6z         |.. image:: images/benchmark/QMIX_3s5z_vs_3s6z.png    | DI-engine/tree/main/dizoo/       |
    |                     |                                                     |smac/config/smac_3s5z3s6z_qmix    |
    |                     |                                                     |_config.py>`_                     |
    +---------------------+-----------------------------------------------------+----------------------------------+



.. list-table:: Benchmark and comparison of qmix algorithm
   :widths: 25 15 30 15 15
   :header-rows: 1

   * - environment
     - best mean reward
     - evaluation results
     - config link
     - comparison
   * - | MMM
       |
     - 1
     - .. image:: images/benchmark/QMIX_MMM.png
     - `config_link_MMM <https://github.com/opendilab/DI-engine/tree/main/dizoo/smac/config/smac_MMM_qmix_config.py>`_
     - | Pymarl(1)
   * - | 3s5z
       |
     - 1
     - .. image:: images/benchmark/QMIX_3s5z.png
     - `config_link_3s5z <https://github.com/opendilab/DI-engine/tree/main/dizoo/smac/config/smac_3s5z_qmix_config.py>`_
     - | Pymarl(1)
   * - | MMM2
       |
     - 0.8
     - .. image:: images/benchmark/QMIX_MMM2.png
     - `config_link_MMM2 <https://github.com/opendilab/DI-engine/tree/main/dizoo/smac/config/smac_MMM2_qmix_config.py>`_
     - | Pymarl(0.7)
   * - | 5m6m
       |
     - 0.6
     - .. image:: images/benchmark/QMIX_5m6m.png
     - `config_link_5m6m <https://github.com/opendilab/DI-engine/tree/main/dizoo/smac/config/smac_5m6m_qmix_config.py>`_
     - | Pymarl(0.76)
   * - | 2c_vs_64zg
       |
     - 1
     - .. image:: images/benchmark/QMIX_2c_vs_64zg.png
     - `config_link_2c_vs_64zg <https://github.com/opendilab/DI-engine/tree/main/dizoo/smac/config/smac_2c64zg_qmix_config.py>`_
     - | Pymarl(1)

..
   * - | 3s5z_vs_3s6z
       |
     - 0.03
     - .. image:: images/benchmark/QMIX_3s5z_vs_3s6z.png
     - `config_link_3s5z_vs_3s6z <https://github.com/opendilab/DI-engine/tree/main/dizoo/smac/config/smac_3s5z3s6z_qmix_config.py>`_
     - | Pymarl(0.03)


P.S.：


1. The above results are obtained by running the same configuration on five different random seeds (0, 1, 2).

2. For the multi-agent discrete action space algorithm like QMIX, the SMAC environment set is generally used for testing,
   and is generally evaluated by the highest mean reward training 10M ``env_step``.
   For more details about SMAC, please refer to `SMAC Env Tutorial <../env_tutorial/smac_zh.html>`_ .


References
----------------
Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob Foerster, Shimon Whiteson. Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning. International Conference on Machine Learning. PMLR, 2018.

Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zambaldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo, Karl Tuyls, Thore Graepel. Value-decomposition networks for cooperative multi-agent learning. arXiv preprint arXiv:1706.05296, 2017.

Kyunghwan Son, Daewoo Kim, Wan Ju Kang, David Earl Hostallero, Yung Yi. QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning. International Conference on Machine Learning. PMLR, 2019. 

Mikayel Samvelyan, Tabish Rashid, Christian Schroeder de Witt, Gregory Farquhar, Nantas Nardelli, Tim G. J. Rudner, Chia-Man Hung, Philip H. S. Torr, Jakob Foerster, Shimon Whiteson. The StarCraft Multi-Agent Challenge. arXiv preprint arXiv:1902.04043, 2019.


Other Public Implementations
----------------------------

- pymarl_

.. _pymarl: https://github.com/oxwhirl/pymarl/blob/master/src/config/algs/qmix.yaml

