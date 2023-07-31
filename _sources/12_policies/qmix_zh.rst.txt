QMIX
^^^^^^^

概述
---------
QMIX 是由 `Rashid et al.(2018) <https://arxiv.org/abs/1803.11485>`_ 提出的，用于在多智能体集中式训练中学习基于全局状态信息的联合动作价值函数，并从集中式端到端框架中提取分布式执行策略。
QMIX 使用集中式神经网络来估计联合动作值，作为基于局部观察的每个智能体动作值的复杂非线性组合，这为集中式动作价值函数提供了一种新的表示，并保证了集中式和分散式策略之间的一致性。

QMIX 是 `VDN(Sunehag et al. 2017) <https://arxiv.org/abs/1706.05296>`_ 的非线性扩展。
与 VDN(Value-Decomposition Networks For Cooperative Multi-Agent Learning
) 相比，QMIX 在训练过程中可以通过超网络(hyper-network)输入的全局信息表示更多的额外状态信息（智能体观测范围外），并且可以表示更丰富的动作价值函数类。

核心要点
-------------
1. QMIX 使用 **集中式训练与分散式执行(centralized training with decentralized execution)** 的范式。

2. QMIX 是一种 **无模型(model-free)、基于价值(value-based)、异策略(off-policy)、多智能体(multi-agent)** 的强化学习方法。

3. QMIX 仅支持 **离散(discrete)** 动作空间。

4. QMIX 考虑了一种 **部分可观察(partially observable)** 的情景，其中每个智能体只获得个体观察。

5. QMIX 接受 **DRQN** 作为个体价值网络来解决 **部分可观察** 问题。

6. QMIX 使用由 **智能体网络(agent networks)、混合网络(mixing network)、超网络(hyper-network)** 组成的架构来表示联合价值函数。 混合网络是一个前馈神经网络，它将智能体网络的输出作为输入并单调地混合它们，产生联合动作值。 混合网络的权重由单独的超网络产生。

关键方程或关键图形
---------------------------
VDN 和 QMIX 是使用将联合动作价值函数 :math:`Q_{tot}` 分解为用于分散执行的个体函数 :math:`Q_a` 的思想的代表性方法。

为了实现集中式训练与分散式执行 (centralized training with decentralized execution CTDE)，我们需要确保在 :math:`Q_{tot}` 上执行的全局 :math:`argmax` 与在每个 :math:`Q_a` 上执行的一组单独的 :math:`argmax` 操作产生相同的结果：

.. math::
  $\arg \max _{\boldsymbol{u}} Q_{\mathrm{tot}}(\boldsymbol{\tau}, \boldsymbol{u})=\left(\begin{array}{c}\arg \max _{u_{1}} Q_{1}\left(\tau_{1}, u_{1}\right) \\ \vdots \\ \arg \max _{u_{N}} Q_{N}\left(\tau_{n}, u_{N}\right)\end{array}\right)$

VDN 将联合动作价值函数分解为个体动作价值函数之和。 :math:`$Q_{\mathrm{tot}}(\boldsymbol{\tau}, \boldsymbol{u})=\sum_{i=1}^{N} Q_{i}\left(\tau_{i}, u_{i}\right)$`

QMIX 扩展了这种加法值分解，将联合动作价值函数表示为一个单调函数。QMIX 基于单调性，即对联合动作值 :math:`Q_{tot}` 和个体动作值 :math:`Q_a` 之间关系的约束。

.. math::
   \frac{\partial Q_{tot}}{\partial Q_{a}} \geq 0， \forall a \in A

QMIX 的整体架构包括个体智能体网络、混合网络和超网络：

.. image:: images/marl/qmix.png

QMIX 通过最小化下面的损失函数来训练混合网络：

.. math::
   y^{tot} = r + \gamma \max_{\textbf{u}^{’}}Q_{tot}(\tau^{'}, \textbf{u}^{'}, s^{'}; \theta^{-})

.. math::
   \mathcal{L}(\theta) = \sum_{i=1}^{b} [(y_{i}^{tot} - Q_{tot}(\tau, \textbf{u}, s; \theta))^{2}]

混合网络的每个权重都是由独立的超网络产生的，它以全局状态作为输入并输出混合网络一层的权重。更多细节可以在原始论文 `Rashid et al.(2018) <https://arxiv.org/abs/1803.11485>`_ 中找到。

VDN 和 QMIX 是试图分解 :math:`Q_tot` 的方法，分别假设可加性和单调性。因此，满足这些条件的联合动作价值函数将被 VDN 和 QMIX 很好地分解。
然而，存在一些任务，其联合动作价值函数不满足所述条件。 `QTRAN (Son et al. 2019) <https://arxiv.org/abs/1905.05408>`_ 提出了一种通过将原始联合动作价值函数转换为容易分解的函数来摆脱这种结构约束的分解方法。
QTRAN (QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning) 保证了比 VDN 或 QMIX 更一般的分解。

实现
----------------
算法的默认设置如下：

    .. autoclass:: ding.policy.qmix.QMIXPolicy
        :noindex:

QMIX 使用的网络接口定义如下：

    .. autoclass:: ding.model.template.QMix
        :members: forward
        :noindex:

Benchmark
-----------
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

P.S.：

1. 上述结果是通过在五个不同的随机种子 (0, 1, 2, 3, 4) 上运行相同的配置获得的。

2. 对于像 QMIX 这样的多智能体离散动作空间算法，通常使用 SMAC 环境集进行测试，并通常通过最高平均奖励训练 10M ``env_step`` 进行评估。
有关 SMAC 的更多详细信息，请参阅 SMAC Env 教程 `SMAC Env Tutorial <../13_envs/smac_zh.html>`_ 。

引用
-----------
- Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob Foerster, Shimon Whiteson. Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning. International Conference on Machine Learning. PMLR, 2018.

- Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zambaldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo, Karl Tuyls, Thore Graepel. Value-decomposition networks for cooperative multi-agent learning. arXiv preprint arXiv:1706.05296, 2017.

- Kyunghwan Son, Daewoo Kim, Wan Ju Kang, David Earl Hostallero, Yung Yi. QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning. International Conference on Machine Learning. PMLR, 2019. 

- Mikayel Samvelyan, Tabish Rashid, Christian Schroeder de Witt, Gregory Farquhar, Nantas Nardelli, Tim G. J. Rudner, Chia-Man Hung, Philip H. S. Torr, Jakob Foerster, Shimon Whiteson. The StarCraft Multi-Agent Challenge. arXiv preprint arXiv:1902.04043, 2019.

其他开源实现
----------------------------
- pymarl_

.. _pymarl: https://github.com/oxwhirl/pymarl/blob/master/src/config/algs/qmix.yaml
